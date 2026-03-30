"""
Top-level entry point for validation and training.

Usage:
    python main.py validate --dungeon 1 --variant 1
    python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion

Legacy validation usage without a subcommand is preserved:
    python main.py --dungeon 1 --variant 1
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
    torch = None

from src.config_system import (
    CONFIG_FIELDS,
    apply_runtime_environment,
    cli_name_for_path,
    cli_overrides_from_namespace,
    configure_logging,
    merge_config,
    save_reproducibility_snapshot,
    seed_everything,
)
from src.train_diffusion import (
    DiffusionTrainingConfig,
    diffusion_training_kwargs_from_resolved_config,
    train_diffusion,
)
from src.train_lcm import FastSamplerTrainingConfig, train_fast_sampler
from src.train_masked_room import (
    MaskedRoomTrainingConfig,
    masked_room_training_kwargs_from_resolved_config,
    train_masked_room,
)
from src.train_vqvae import train_vqvae
from src.utils.distributed import get_env_rank, maybe_launch_with_torchrun
from src.zelda_data.zelda_core import (
    Dungeon,
    DungeonSolver,
    DungeonStitcher,
    StitchedDungeon,
    ValidationMode,
    ZeldaDungeonAdapter,
    convert_dungeon_to_dungeondata,
    test_all_dungeons,
    visualize_semantic_grid,
)


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent


def load_dungeon(dungeon_num: int, variant: int = 1, data_root: Optional[str] = None) -> Dungeon:
    if data_root is None:
        data_root = str(PROJECT_ROOT / "Data" / "The Legend of Zelda")
    adapter = ZeldaDungeonAdapter(data_root)
    return adapter.load_dungeon(dungeon_num, variant=variant)


def stitch_dungeon(dungeon: Dungeon, compact: bool = True) -> StitchedDungeon:
    stitcher = DungeonStitcher()
    return stitcher.stitch(dungeon, compact=compact)


def validate_dungeon(stitched: StitchedDungeon, mode: str = ValidationMode.FULL) -> dict:
    solver = DungeonSolver()
    return solver.solve(stitched, mode=mode)


def run_pipeline(
    dungeon_num: int,
    variant: int = 1,
    mode: str = ValidationMode.FULL,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"PIPELINE: Dungeon {dungeon_num} (Quest {variant})")
        print(f"{'=' * 60}")

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        if torch is not None:
            torch.manual_seed(int(seed))
        logger.info("Using deterministic seed=%s", seed)

    logger.info("[STEP 1] Loading dungeon data...")
    dungeon = load_dungeon(dungeon_num, variant)
    if verbose:
        print("\n[STEP 1] Loading dungeon data...")
        print(f"  [OK] Loaded {len(dungeon.rooms)} rooms")
        print(f"  [OK] Graph: {dungeon.graph.number_of_nodes()} nodes, {dungeon.graph.number_of_edges()} edges")

    logger.info("[STEP 2] Stitching rooms...")
    stitched = stitch_dungeon(dungeon)
    if verbose:
        print("\n[STEP 2] Stitching rooms...")
        print(f"  [OK] Global grid: {stitched.global_grid.shape}")
        print(f"  [OK] Start: {stitched.start_global}")
        print(f"  [OK] Triforce: {stitched.triforce_global}")

    logger.info("[STEP 3] Validating solvability (mode: %s)...", mode)
    result = validate_dungeon(stitched, mode=mode)
    if verbose:
        if result["solvable"]:
            print("  [OK] SOLVABLE!")
            print(f"  [OK] Path length: {result.get('path_length', 'N/A')} steps")
            print(f"  [OK] Rooms traversed: {result.get('rooms_traversed', 'N/A')}")
            if "keys_available" in result:
                print(f"  [OK] Keys available: {result['keys_available']}")
                print(f"  [OK] Keys used: {result['keys_used']}")
        else:
            print("  [FAIL] NOT SOLVABLE")
            print(f"  [FAIL] Reason: {result.get('reason', 'Unknown')}")

    return {
        "dungeon_num": dungeon_num,
        "variant": variant,
        "dungeon": dungeon,
        "stitched": stitched,
        "validation": result,
        "solvable": result["solvable"],
    }


def export_dungeon_data(dungeon: Dungeon, output_path: str) -> None:
    dungeon_data = convert_dungeon_to_dungeondata(dungeon)
    room_grids = {f"room_{room_id}": room.grid for room_id, room in dungeon_data.rooms.items()}
    np.savez(
        output_path,
        dungeon_id=dungeon_data.dungeon_id,
        layout=dungeon_data.layout,
        tpe_vectors=dungeon_data.tpe_vectors,
        p_matrix=dungeon_data.p_matrix,
        node_features=dungeon_data.node_features,
        **room_grids,
    )
    logger.info("Exported dungeon data to %s", output_path)
    print(f"Exported to: {output_path}")


def _str_to_mode(mode: str) -> str:
    mode_map = {
        "strict": ValidationMode.STRICT,
        "realistic": ValidationMode.REALISTIC,
        "full": ValidationMode.FULL,
    }
    return mode_map[mode]


def _add_config_flags(parser: argparse.ArgumentParser) -> None:
    for field in CONFIG_FIELDS:
        cli_name = field.cli or cli_name_for_path(field.path)
        flag = f"--{cli_name.replace('_', '-')}"
        kwargs: Dict[str, Any] = {"dest": cli_name, "default": None, "help": field.help}
        if field.field_type is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        elif field.field_type in {list, tuple}:
            kwargs["nargs"] = "+"
            kwargs["type"] = field.sequence_item_type or str
            if field.choices is not None:
                kwargs["choices"] = tuple(field.choices)
        else:
            kwargs["type"] = field.field_type
            if field.choices is not None:
                kwargs["choices"] = tuple(field.choices)
        parser.add_argument(flag, **kwargs)


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    train_parser = subparsers.add_parser(
        "train",
        help="Run staged training with YAML config + CLI overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "zelda_hmolqd.yaml"),
        help="Path to YAML experiment config.",
    )
    _add_config_flags(train_parser)


def _build_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    validate_parser = subparsers.add_parser(
        "validate",
        help="Load, stitch, and validate Zelda dungeons.",
    )
    validate_parser.add_argument("--dungeon", "-d", type=int, choices=range(1, 10))
    validate_parser.add_argument("--variant", "-v", type=int, default=1, choices=[1, 2])
    validate_parser.add_argument("--all", "-a", action="store_true")
    validate_parser.add_argument("--mode", "-m", choices=["strict", "realistic", "full"], default="full")
    validate_parser.add_argument("--gui", "-g", action="store_true")
    validate_parser.add_argument("--export", "-e", type=str)
    validate_parser.add_argument("--data-root", type=str)
    validate_parser.add_argument("--quiet", "-q", action="store_true")
    validate_parser.add_argument("--ascii", action="store_true")
    validate_parser.add_argument(
        "--seed",
        type=int,
        default=(int(os.environ["KLTN_SEED"]) if "KLTN_SEED" in os.environ else None),
    )


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KLTN entry point for validation and training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_train_parser(subparsers)
    _build_validate_parser(subparsers)
    return parser


def _run_vqvae_stage_from_config(config: Dict[str, Any]) -> Path:
    stage = config["vqvae"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    args = SimpleNamespace(
        data_dir=dataset["data_dir"],
        epochs=stage["epochs"],
        batch_size=dataset["batch_size"],
        lr=stage["learning_rate"],
        weight_decay=stage["weight_decay"],
        grad_clip_norm=stage["grad_clip_norm"],
        latent_dim=stage["latent_dim"],
        hidden_dim=stage["hidden_dim"],
        codebook_size=stage["codebook_size"],
        num_classes=dataset["num_classes"],
        commitment_cost=stage["commitment_cost"],
        rare_tile_weight=stage["rare_tile_weight"],
        use_ema=stage["use_ema"],
        use_coordconv=stage["use_coordconv"],
        mrf_penalty_weight=stage["mrf_penalty_weight"],
        min_samples_per_epoch=dataset["min_samples_per_epoch"],
        save_dir=stage["checkpoint_dir"],
        save_every=stage["save_every"],
        num_workers=dataset["num_workers"],
        pin_memory=dataset["pin_memory"],
        drop_last=dataset["drop_last"],
        seed=runtime["seed"],
        resume=stage["resume_checkpoint"] or runtime["resume"],
        device=runtime["device"],
        verbose=runtime["verbose"],
    )
    train_vqvae(args)
    return Path(stage["checkpoint_dir"]) / "vqvae_pretrained.pth"


def _run_diffusion_stage_from_config(config: Dict[str, Any], vqvae_checkpoint: Optional[Path]) -> None:
    cfg = DiffusionTrainingConfig(
        **diffusion_training_kwargs_from_resolved_config(
            config,
            fallback_vqvae_checkpoint=(str(vqvae_checkpoint) if vqvae_checkpoint is not None else None),
        )
    )
    train_diffusion(cfg)


def _run_fast_sampler_stage_from_config(config: Dict[str, Any]) -> None:
    stage = config["fast_sampler"]
    dataset = config["dataset"]
    runtime = config["runtime"]
    base_ckpt = stage["base_diffusion_checkpoint"]
    if not base_ckpt:
        candidate = Path(config["diffusion"]["checkpoint_dir"]) / "best_model.pth"
        if candidate.exists():
            base_ckpt = str(candidate)
    if not base_ckpt:
        raise FileNotFoundError(
            "Fast sampler stage requires fast_sampler.base_diffusion_checkpoint "
            "or an existing diffusion best_model.pth."
        )
    cfg = FastSamplerTrainingConfig(
        base_diffusion_checkpoint=base_ckpt,
        data_dir=dataset["data_dir"],
        batch_size=dataset["batch_size"],
        num_workers=dataset["num_workers"],
        pin_memory=dataset["pin_memory"],
        drop_last=dataset["drop_last"],
        shuffle_train=dataset["shuffle_train"],
        shuffle_val=dataset["shuffle_val"],
        normalize=dataset["normalize"],
        room_level=dataset["room_level"],
        epochs=stage["epochs"],
        learning_rate=stage["learning_rate"],
        optimizer_weight_decay=stage["optimizer_weight_decay"],
        grad_clip_norm=stage["grad_clip_norm"],
        num_inference_steps=stage["num_inference_steps"],
        lora_rank=stage["lora_rank"],
        lora_alpha=stage["lora_alpha"],
        prediction_loss_weight=stage["prediction_loss_weight"],
        save_every=stage["save_every"],
        checkpoint_dir=stage["checkpoint_dir"],
        device=runtime["device"],
        quick=runtime["quick"],
    )
    train_fast_sampler(cfg)


def _run_masked_room_stage_from_config(config: Dict[str, Any]) -> None:
    cfg = MaskedRoomTrainingConfig(**masked_room_training_kwargs_from_resolved_config(config))
    train_masked_room(cfg)


def _log_resolved_config(config: Dict[str, Any]) -> None:
    import yaml

    logger.info("Resolved config:\n%s", yaml.safe_dump(config, sort_keys=False))


def run_training_from_args(args: argparse.Namespace) -> None:
    overrides = cli_overrides_from_namespace(args)
    config = merge_config(yaml_path=args.config, cli_overrides=overrides)
    stage = config["training"]["stage"]
    if bool(config["distributed"]["enabled"]) and int(config["distributed"]["nproc_per_node"]) > 1:
        if stage not in {"diffusion"}:
            raise ValueError(
                "distributed.enabled with nproc_per_node > 1 is currently supported only for training.stage=diffusion."
            )
    apply_runtime_environment(config)
    launched = maybe_launch_with_torchrun(
        enabled=bool(config["distributed"]["enabled"]),
        nproc_per_node=int(config["distributed"]["nproc_per_node"]),
        master_port=int(config["distributed"]["master_port"]),
        script_path=str(PROJECT_ROOT / "main.py"),
        script_args=list(sys.argv[1:]),
        extra_env={
            **(
                {"CUDA_VISIBLE_DEVICES": str(config["distributed"]["cuda_visible_devices"]).strip()}
                if str(config["distributed"]["cuda_visible_devices"]).strip()
                else {}
            ),
            "MASTER_PORT": str(config["distributed"]["master_port"]),
        },
    )
    if launched:
        return

    rank = int(get_env_rank())
    configure_logging(config, rank=rank)
    config["runtime"]["seed"] = seed_everything(config["runtime"]["seed"])
    if rank == 0:
        snapshot_paths = save_reproducibility_snapshot(config, argv=sys.argv)
        logger.info("Saved config snapshot to %s", snapshot_paths["resolved_yaml"])
        logger.info("Saved run metadata to %s", snapshot_paths["metadata_json"])
        _log_resolved_config(config)

    vqvae_ckpt: Optional[Path] = None
    if stage in {"all", "vqvae"}:
        logger.info("Stage 1/4: Training VQ-VAE")
        vqvae_ckpt = _run_vqvae_stage_from_config(config)
    if stage in {"all", "diffusion"}:
        logger.info("Stage 2/4: Training diffusion")
        _run_diffusion_stage_from_config(config, vqvae_ckpt)
    if stage in {"all", "fast_sampler"}:
        logger.info("Stage 3/4: Training fast sampler")
        _run_fast_sampler_stage_from_config(config)
    if stage in {"all", "masked_room"}:
        logger.info("Stage 4/4: Training masked-room model")
        _run_masked_room_stage_from_config(config)


def run_validation_from_args(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    mode = _str_to_mode(args.mode)
    if args.all:
        data_root = args.data_root or str(PROJECT_ROOT / "Data" / "The Legend of Zelda")
        test_all_dungeons(data_root, include_variants=True)
        return

    if args.dungeon is None:
        raise SystemExit("Either --dungeon or --all is required")

    result = run_pipeline(
        args.dungeon,
        args.variant,
        mode=mode,
        seed=args.seed,
        verbose=not args.quiet,
    )

    if args.export:
        export_dungeon_data(result["dungeon"], args.export)
    if args.ascii:
        print("\n" + "=" * 60)
        print("ASCII VISUALIZATION")
        print("=" * 60)
        print(visualize_semantic_grid(result["stitched"].global_grid))
    if args.gui:
        try:
            from src.visualization.replay_engine import DungeonReplayEngine, ReplayConfig

            stitched = result["stitched"]
            grid = stitched.global_grid
            validation = result["validation"]
            solution_path = validation.get("path", [])
            if not solution_path and validation.get("solvable"):
                from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv

                env = ZeldaLogicEnv(grid, render_mode=False)
                solver = StateSpaceAStar(env)
                _success, solution_path, _ = solver.solve()

            config = ReplayConfig(
                window_title=f"ZAVE - Dungeon {args.dungeon} Variant {args.variant}",
                show_minimap=True,
                show_path_overlay=True,
            )
            engine = DungeonReplayEngine(
                dungeon_grid=grid,
                solution_path=solution_path if solution_path else [],
                config=config,
                solver_result=validation,
            )
            engine.run()
        except ImportError as exc:
            logger.warning("New visualization not available, using legacy GUI: %s", exc)
            try:
                import pygame
                from gui_runner import ZeldaGUI

                pygame.init()
                gui = ZeldaGUI([result["stitched"]], [f"Dungeon {args.dungeon} Variant {args.variant}"])
                gui.run()
            except ImportError as exc2:
                logger.error("GUI not available - missing dependency: %s", exc2)
                print(f"\nGUI not available: {exc2}")
                print("Make sure pygame is installed: pip install pygame")


def _run_legacy_validate(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="KLTN Zelda Dungeon Pipeline - Load, Stitch, Validate")
    parser.add_argument("--dungeon", "-d", type=int, choices=range(1, 10))
    parser.add_argument("--variant", "-v", type=int, default=1, choices=[1, 2])
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--mode", "-m", choices=["strict", "realistic", "full"], default="full")
    parser.add_argument("--gui", "-g", action="store_true")
    parser.add_argument("--export", "-e", type=str)
    parser.add_argument("--data-root", type=str)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument(
        "--seed",
        type=int,
        default=(int(os.environ["KLTN_SEED"]) if "KLTN_SEED" in os.environ else None),
    )
    args = parser.parse_args(argv)
    run_validation_from_args(args)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"train", "validate"}:
        _run_legacy_validate(argv)
        return

    parser = _build_root_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_training_from_args(args)
    elif args.command == "validate":
        run_validation_from_args(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
