"""
Top-level entry point for validation and training.

Usage:
    python main.py validate --dungeon 1 --variant 1
    python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion
    python main.py topology-visualize --seed 20260406
    python main.py topology-compare-manual --run-dir outputs/zelda_hmolqd_semantic_anchor_retrain_v1 --output-dir outputs/manual_compare
    python main.py topology-audit-fixed-graph --run-dir outputs/zelda_hmolqd_semantic_anchor_retrain_v1 --output-dir outputs/fixed_graph_audit

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
except ImportError:
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
from src.train_lcm import (
    FastSamplerTrainingConfig,
    fast_sampler_training_kwargs_from_resolved_config,
    train_fast_sampler,
)
from src.train_masked_room import (
    MaskedRoomTrainingConfig,
    masked_room_training_kwargs_from_resolved_config,
    train_masked_room,
)
from src.train_vqvae import train_vqvae, vqvae_training_kwargs_from_resolved_config
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
from scripts.export_manual_rich_topology_compare import run_from_args as run_manual_topology_compare_from_args
from scripts.run_fixed_graph_multi_seed_audit import run_from_args as run_fixed_graph_audit_from_args
from scripts.run_fast_sampler_visual_audit import add_generation_override_args as add_generation_export_override_args
from scripts.visualize_block_i_graphs import run_from_args as run_topology_visualize_from_args


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


def _build_topology_visualize_parser(subparsers: argparse._SubParsersAction) -> None:
    topology_parser = subparsers.add_parser(
        "topology-visualize",
        help="Generate Block I topology graph image galleries and descriptor scatter plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    topology_parser.add_argument("--seed", type=int, default=42)
    topology_parser.add_argument("--num-generated", type=int, default=12)
    topology_parser.add_argument("--num-show", type=int, default=12)
    topology_parser.add_argument("--reference-limit", type=int, default=18)
    topology_parser.add_argument("--population-size", type=int, default=24)
    topology_parser.add_argument("--generations", type=int, default=24)
    topology_parser.add_argument("--min-rooms", type=int, default=8)
    topology_parser.add_argument("--max-rooms", type=int, default=16)
    topology_parser.add_argument("--rule-space", choices=["core", "full"], default="full")
    topology_parser.add_argument("--search-strategy", choices=["ga", "cvt_emitter"], default="ga")
    topology_parser.add_argument("--qd-archive-cells", type=int, default=128)
    topology_parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    topology_parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    topology_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data") / "The Legend of Zelda",
    )
    topology_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "topology_visuals",
    )


def _build_topology_compare_manual_parser(subparsers: argparse._SubParsersAction) -> None:
    compare_parser = subparsers.add_parser(
        "topology-compare-manual",
        help="Run diffusion / fast-sampler / masked-room on one fixed manual topology and export aligned comparison PNGs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare_parser.add_argument("--run-dir", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    compare_parser.add_argument(
        "--mission-graph",
        type=Path,
        default=None,
        help="Optional path to a user-authored mission_graph.json. If omitted, the built-in rich manual topology is used.",
    )
    compare_parser.add_argument("--seed", type=int, default=20260406)
    compare_parser.add_argument(
        "--variants",
        type=str,
        default="diffusion_cfg3_logic0_steps50,fast_cfg3_logic0_steps4,masked_room_full",
        help="Comma-separated subset of manual-comparison variants to run.",
    )
    compare_parser.add_argument(
        "--reuse-existing-variants",
        action="store_true",
        help="Reuse existing per-variant summary.json files when present.",
    )
    add_generation_export_override_args(compare_parser)


def _build_topology_fixed_graph_audit_parser(subparsers: argparse._SubParsersAction) -> None:
    fixed_parser = subparsers.add_parser(
        "topology-audit-fixed-graph",
        help="Re-run diffusion / fast-sampler / masked-room on one fixed mission graph across multiple seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    fixed_parser.add_argument("--run-dir", type=Path, required=True)
    fixed_parser.add_argument(
        "--mission-graph",
        type=Path,
        default=None,
        help="Optional path to a fixed mission_graph.json. If omitted, the built-in rich manual topology is used.",
    )
    fixed_parser.add_argument("--output-dir", type=Path, required=True)
    fixed_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260404, 20260405, 20260406],
        help="Seeds to audit on the same fixed mission graph.",
    )
    fixed_parser.add_argument(
        "--include-no-fallback-ablations",
        action="store_true",
        help="Also export strict no-fallback and pure-neural no-overlay variants for branch-audit runs.",
    )
    fixed_parser.add_argument(
        "--include-puzzle-ablations",
        action="store_true",
        help="Also export puzzle-off variants with puzzle_room_scaffold_enabled=False.",
    )
    add_generation_export_override_args(fixed_parser)


def _build_root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="KLTN entry point for training, validation, and topology-driven export workflows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    _build_train_parser(subparsers)
    _build_validate_parser(subparsers)
    _build_topology_visualize_parser(subparsers)
    _build_topology_compare_manual_parser(subparsers)
    _build_topology_fixed_graph_audit_parser(subparsers)
    return parser


def _run_vqvae_stage_from_config(config: Dict[str, Any]) -> Path:
    stage = config["vqvae"]
    args = SimpleNamespace(**vqvae_training_kwargs_from_resolved_config(config))
    train_vqvae(args)
    return Path(stage["checkpoint_dir"]) / "vqvae_pretrained.pth"


def _resolve_diffusion_stage_vqvae_checkpoint(
    config: Dict[str, Any],
    *,
    fallback_vqvae_checkpoint: Optional[Path],
) -> Path:
    explicit = config["diffusion"]["vqvae_checkpoint"]
    if explicit:
        candidate = Path(str(explicit))
        if not candidate.exists():
            raise FileNotFoundError(
                f"diffusion.vqvae_checkpoint points to a missing file: {candidate}"
            )
        return candidate

    if fallback_vqvae_checkpoint is not None:
        candidate = Path(fallback_vqvae_checkpoint)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Fresh VQ-VAE stage reported checkpoint {candidate}, but the file does not exist."
        )

    canonical = Path(config["vqvae"]["checkpoint_dir"]) / "vqvae_pretrained.pth"
    if canonical.exists():
        return canonical

    raise FileNotFoundError(
        "Diffusion training requires a trained VQ-VAE checkpoint. "
        f"Expected diffusion.vqvae_checkpoint or {canonical}."
    )


def _run_diffusion_stage_from_config(config: Dict[str, Any], vqvae_checkpoint: Optional[Path]) -> None:
    resolved_vqvae_checkpoint = _resolve_diffusion_stage_vqvae_checkpoint(
        config,
        fallback_vqvae_checkpoint=vqvae_checkpoint,
    )
    cfg = DiffusionTrainingConfig(
        **diffusion_training_kwargs_from_resolved_config(
            config,
            fallback_vqvae_checkpoint=str(resolved_vqvae_checkpoint),
        )
    )
    train_diffusion(cfg)


def _warn_if_full_retrain_may_resume(config: Dict[str, Any]) -> None:
    if config["training"]["stage"] != "all" or not bool(config["runtime"]["auto_resume"]):
        return

    existing = []
    for section_name in ("vqvae", "diffusion", "fast_sampler", "masked_room"):
        checkpoint_dir = Path(config[section_name]["checkpoint_dir"])
        latest_resume = checkpoint_dir / "latest_resume.pth"
        if latest_resume.exists():
            existing.append(str(latest_resume))

    if existing:
        logger.warning(
            "Full-stack retrain requested with runtime.auto_resume=true, and existing stage checkpoints were found: %s. "
            "This run may resume previous work instead of starting fresh. Use a new runtime.output_dir or disable auto-resume for a clean retrain.",
            existing,
        )


def _run_fast_sampler_stage_from_config(config: Dict[str, Any]) -> None:
    stage = config["fast_sampler"]
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
    kwargs = fast_sampler_training_kwargs_from_resolved_config(config)
    kwargs["base_diffusion_checkpoint"] = base_ckpt
    cfg = FastSamplerTrainingConfig(**kwargs)
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
    config["runtime"]["seed"] = seed_everything(
        config["runtime"]["seed"],
        cudnn_benchmark=bool(config["runtime"].get("cudnn_benchmark", True)),
        cudnn_deterministic=bool(config["runtime"].get("cudnn_deterministic", False)),
    )
    if rank == 0:
        snapshot_paths = save_reproducibility_snapshot(config, argv=sys.argv)
        logger.info("Saved config snapshot to %s", snapshot_paths["resolved_yaml"])
        logger.info("Saved run metadata to %s", snapshot_paths["metadata_json"])
        _log_resolved_config(config)
        _warn_if_full_retrain_may_resume(config)

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
    if argv and argv[0].startswith("-") and argv[0] not in {"-h", "--help"}:
        _run_legacy_validate(argv)
        return

    parser = _build_root_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        run_training_from_args(args)
    elif args.command == "validate":
        run_validation_from_args(args)
    elif args.command == "topology-visualize":
        run_topology_visualize_from_args(args)
    elif args.command == "topology-compare-manual":
        run_manual_topology_compare_from_args(args)
    elif args.command == "topology-audit-fixed-graph":
        run_fixed_graph_audit_from_args(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
