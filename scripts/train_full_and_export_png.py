"""
Train full KLTN pipeline from scratch and export generated full-level PNG.

This utility does three things in one run:
1) Stage training from scratch (VQ-VAE + diffusion)
2) Split the combined diffusion checkpoint into component checkpoints
3) Generate a full dungeon level and save PNG visualizations

Example:
    python scripts/train_full_and_export_png.py --epochs-vqvae 1 --epochs-diffusion 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import Counter

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import semantic_to_vglc_char
from src.utils.checkpoint import safe_torch_load


def _run(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_from_scratch(
    data_dir: str,
    checkpoint_dir: Path,
    epochs_vqvae: int,
    epochs_diffusion: int,
    epochs_fast_sampler: int,
    epochs_masked_room: int,
    batch_size: int,
    seed: int,
    graph_conditioning_mode: str,
    condition_gnn_type: str,
    topology_refinement_mode: str,
    train_fast_sampler: bool,
    train_masked_room: bool,
    fast_sampler_steps: int,
    masked_steps: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    vqvae_cmd = [
        sys.executable,
        "-m",
        "src.train_vqvae",
        "--data-dir",
        data_dir,
        "--save-dir",
        str(checkpoint_dir),
        "--epochs",
        str(int(epochs_vqvae)),
        "--batch-size",
        str(int(batch_size)),
        "--save-every",
        "1",
        "--seed",
        str(int(seed)),
    ]
    _run(vqvae_cmd)

    diffusion_cmd = [
        sys.executable,
        "-m",
        "src.train_diffusion",
        "--data-dir",
        data_dir,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(int(epochs_diffusion)),
        "--batch-size",
        str(int(batch_size)),
        "--vqvae-checkpoint",
        str(checkpoint_dir / "vqvae_pretrained.pth"),
        "--graph-conditioning-mode",
        str(graph_conditioning_mode),
        "--condition-gnn-type",
        str(condition_gnn_type),
        "--topology-refinement-mode",
        str(topology_refinement_mode),
        "--no-auto-resume",
    ]
    _run(diffusion_cmd)

    if train_fast_sampler:
        fast_cmd = [
            sys.executable,
            "-m",
            "src.train_lcm",
            "--base-diffusion-checkpoint",
            str(checkpoint_dir / "best_model.pth"),
            "--data-dir",
            data_dir,
            "--batch-size",
            str(int(batch_size)),
            "--epochs",
            str(int(epochs_fast_sampler)),
            "--num-inference-steps",
            str(int(fast_sampler_steps)),
            "--checkpoint-dir",
            str(checkpoint_dir / "fast_sampler"),
            "--device",
            "auto",
        ]
        _run(fast_cmd)

    if train_masked_room:
        masked_cmd = [
            sys.executable,
            "-m",
            "src.train_masked_room",
            "--data-dir",
            data_dir,
            "--batch-size",
            str(int(batch_size)),
            "--epochs",
            str(int(epochs_masked_room)),
            "--graph-conditioning-mode",
            str(graph_conditioning_mode),
            "--condition-gnn-type",
            str(condition_gnn_type),
            "--masked-steps",
            str(int(masked_steps)),
            "--checkpoint-dir",
            str(checkpoint_dir / "masked_room"),
            "--device",
            "auto",
        ]
        _run(masked_cmd)

    final_ckpt = checkpoint_dir / "final_model.pth"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {final_ckpt}")
    return final_ckpt


def split_component_checkpoints(final_ckpt: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = safe_torch_load(final_ckpt, map_location="cpu")

    vqvae_source = out_dir / "vqvae_pretrained.pth"

    mapping = {
        "vqvae_best.pth": ckpt.get("vqvae_state_dict"),
        "diffusion_best.pth": ckpt.get("diffusion_state_dict"),
        "condition_encoder_best.pth": ckpt.get("condition_encoder_state_dict"),
        "logic_net_best.pth": ckpt.get("logic_net_state_dict"),
    }

    out_paths: Dict[str, Path] = {}
    for name, state in mapping.items():
        path = out_dir / name
        if state is None:
            if name == "vqvae_best.pth" and vqvae_source.exists():
                shutil.copy2(vqvae_source, path)
                meta_source = Path(f"{vqvae_source}.meta.json")
                if meta_source.exists():
                    shutil.copy2(meta_source, Path(f"{path}.meta.json"))
            else:
                raise KeyError(f"Missing state in combined checkpoint: {name}")
        else:
            torch.save({"model_state_dict": state}, path)
        out_paths[name] = path

    return out_paths


def _tile_color(tile: int) -> Tuple[int, int, int]:
    # Hand-tuned semantic colors for readability.
    palette = {
        0: (8, 8, 12),      # VOID
        1: (210, 205, 185), # FLOOR
        2: (55, 60, 70),    # WALL
        3: (120, 96, 72),   # BLOCK
        10: (80, 180, 240), # DOOR_OPEN
        11: (240, 200, 70), # DOOR_LOCKED
        12: (230, 125, 65), # DOOR_BOMB
        13: (165, 135, 240),# DOOR_PUZZLE
        14: (220, 70, 70),  # DOOR_BOSS
        15: (100, 220, 180),# DOOR_SOFT
        20: (210, 75, 145), # ENEMY
        21: (90, 220, 110), # START
        22: (255, 245, 115),# TRIFORCE
        23: (170, 35, 35),  # BOSS
        30: (255, 220, 90), # KEY_SMALL
        31: (255, 165, 60), # KEY_BOSS
        32: (100, 220, 220),# KEY_ITEM
        33: (170, 235, 130),# ITEM_MINOR
        40: (85, 85, 195),  # ELEMENT
        41: (95, 145, 205), # ELEMENT_FLOOR
        42: (175, 175, 255),# STAIR
        43: (200, 115, 225),# PUZZLE
    }
    if tile in palette:
        return palette[tile]
    # Stable fallback for unknown IDs.
    return ((97 * (tile + 3)) % 255, (61 * (tile + 7)) % 255, (151 * (tile + 11)) % 255)


def save_grid_png(grid: np.ndarray, out_path: Path, tile_px: int = 16) -> None:
    h, w = grid.shape
    canvas = np.zeros((h * tile_px, w * tile_px, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = _tile_color(int(grid[r, c]))
            y0 = r * tile_px
            x0 = c * tile_px
            canvas[y0:y0 + tile_px, x0:x0 + tile_px] = color
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def _tile_to_vglc_char(tile: int) -> str:
    return semantic_to_vglc_char(int(tile))


def _infer_unknown_vglc_mapping(ids: np.ndarray) -> Dict[int, str]:
    """Infer dataset-style chars for unknown tile IDs from local context."""
    known = {tile: _tile_to_vglc_char(tile) for tile in range(44)}
    unique_ids = {int(v) for v in np.unique(ids)}
    unknown_ids = sorted(unique_ids - set(known.keys()))
    if not unknown_ids:
        return {}

    h, w = ids.shape
    inferred: Dict[int, str] = {}
    for tile_id in unknown_ids:
        votes: Counter[str] = Counter()
        positions = np.argwhere(ids == tile_id)
        for r, c in positions:
            r_i = int(r)
            c_i = int(c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r_i + dr
                    cc = c_i + dc
                    if rr < 0 or rr >= h or cc < 0 or cc >= w:
                        continue
                    neigh = int(ids[rr, cc])
                    neigh_char = known.get(neigh)
                    if neigh_char is not None and neigh_char != "-":
                        votes[neigh_char] += 1

        if votes:
            inferred[tile_id] = votes.most_common(1)[0][0]
        else:
            inferred[tile_id] = "F"
    return inferred


def save_grid_txt(grid: np.ndarray, out_path: Path, mode: str = "ids") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ids = grid.astype(int)
    inferred_unknown_map: Dict[int, str] = {}
    if mode != "ids":
        inferred_unknown_map = _infer_unknown_vglc_mapping(ids)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in range(ids.shape[0]):
            row = ids[r]
            if mode == "ids":
                f.write(" ".join(str(int(v)) for v in row))
            else:
                chars = []
                for v in row:
                    tile = int(v)
                    if tile in inferred_unknown_map:
                        chars.append(inferred_unknown_map[tile])
                    else:
                        chars.append(_tile_to_vglc_char(tile))
                f.write("".join(chars))
            f.write("\n")


def generate_and_export(
    checkpoint_dir: Path,
    output_dir: Path,
    num_rooms: int,
    seed: int,
    num_diffusion_steps: int,
    topology_population: int,
    topology_generations: int,
    diffusion_cfg_schedule_mode: str,
    diffusion_cfg_schedule_min_scale: float,
    diffusion_cfg_schedule_power: float,
    room_generator_mode: str,
    masked_room_checkpoint: Optional[Path],
    fast_sampling_checkpoint: Optional[Path],
    use_fast_sampling: bool,
    enable_map_elites: bool = False,
    device: str = "auto",
) -> None:
    from src.pipeline.dungeon_pipeline import create_pipeline

    pipeline = create_pipeline(
        checkpoint_dir=str(checkpoint_dir),
        device=device,
        diffusion_cfg_schedule_mode=diffusion_cfg_schedule_mode,
        diffusion_cfg_schedule_min_scale=diffusion_cfg_schedule_min_scale,
        diffusion_cfg_schedule_power=diffusion_cfg_schedule_power,
        room_generator_mode=str(room_generator_mode),
        masked_room_checkpoint=(str(masked_room_checkpoint) if masked_room_checkpoint is not None else None),
        masked_sampling_steps=max(1, min(12, int(num_diffusion_steps))),
        fast_sampling_checkpoint=(str(fast_sampling_checkpoint) if fast_sampling_checkpoint is not None else None),
        fast_sampling_steps=max(1, int(num_diffusion_steps)),
    )

    result = pipeline.generate_dungeon(
        mission_graph=None,
        generate_topology=True,
        num_rooms=int(num_rooms),
        population_size=int(topology_population),
        generations=int(topology_generations),
        seed=int(seed),
        guidance_scale=2.0,
        logic_guidance_scale=1.5,
        num_diffusion_steps=int(num_diffusion_steps),
        use_fast_sampling=bool(use_fast_sampling),
        apply_repair=True,
        enable_map_elites=bool(enable_map_elites),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "dungeon_grid.npy", result.dungeon_grid)
    save_grid_png(result.dungeon_grid, output_dir / "dungeon_full.png", tile_px=16)
    save_grid_txt(result.dungeon_grid, output_dir / "dungeon_grid_ids.txt", mode="ids")
    save_grid_txt(result.dungeon_grid, output_dir / "dungeon_grid_vglc.txt", mode="vglc")

    rooms_dir = output_dir / "rooms_png"
    rooms_txt_ids_dir = output_dir / "rooms_txt_ids"
    rooms_txt_vglc_dir = output_dir / "rooms_txt_vglc"
    for room_id, room in result.rooms.items():
        save_grid_png(room.room_grid, rooms_dir / f"room_{room_id}.png", tile_px=20)
        save_grid_txt(room.room_grid, rooms_txt_ids_dir / f"room_{room_id}.txt", mode="ids")
        save_grid_txt(room.room_grid, rooms_txt_vglc_dir / f"room_{room_id}.txt", mode="vglc")

    serializable_metrics = {
        "generation_time_sec": float(result.generation_time),
        "metrics": result.metrics,
        "map_elites_score": result.map_elites_score,
        "num_rooms": len(result.rooms),
        "dungeon_shape": list(result.dungeon_grid.shape),
    }
    with open(output_dir / "generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train full architecture and export dungeon PNG")
    p.add_argument("--data-dir", type=str, default="Data/The Legend of Zelda")
    p.add_argument("--run-dir", type=Path, default=Path("outputs/full_train_png"))
    p.add_argument("--epochs-vqvae", type=int, default=1)
    p.add_argument("--epochs-diffusion", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-rooms", type=int, default=8)
    p.add_argument("--num-diffusion-steps", type=int, default=50)
    p.add_argument("--topology-population", type=int, default=24)
    p.add_argument("--topology-generations", type=int, default=24)
    p.add_argument("--graph-conditioning-mode", type=str, default="node_sequence")
    p.add_argument("--condition-gnn-type", type=str, default="gcn")
    p.add_argument("--topology-refinement-mode", type=str, default="gat2")
    p.add_argument("--diffusion-cfg-schedule-mode", type=str, default="constant")
    p.add_argument("--diffusion-cfg-schedule-min-scale", type=float, default=1.0)
    p.add_argument("--diffusion-cfg-schedule-power", type=float, default=1.0)
    p.add_argument("--train-fast-sampler", action="store_true")
    p.add_argument("--epochs-fast-sampler", type=int, default=5)
    p.add_argument("--fast-sampler-steps", type=int, default=4)
    p.add_argument("--train-masked-room", action="store_true")
    p.add_argument("--epochs-masked-room", type=int, default=20)
    p.add_argument("--masked-steps", type=int, default=8)
    p.add_argument("--room-generator-mode", type=str, default="latent_diffusion")
    p.add_argument("--use-fast-sampling", action="store_true")
    p.add_argument("--enable-map-elites", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"

    if args.skip_train:
        final_ckpt = ckpt_dir / "final_model.pth"
        if not final_ckpt.exists():
            raise FileNotFoundError(
                f"--skip-train set but checkpoint is missing: {final_ckpt}"
            )
    else:
        final_ckpt = train_from_scratch(
            data_dir=args.data_dir,
            checkpoint_dir=ckpt_dir,
            epochs_vqvae=args.epochs_vqvae,
            epochs_diffusion=args.epochs_diffusion,
            epochs_fast_sampler=args.epochs_fast_sampler,
            epochs_masked_room=args.epochs_masked_room,
            batch_size=args.batch_size,
            seed=args.seed,
            graph_conditioning_mode=args.graph_conditioning_mode,
            condition_gnn_type=args.condition_gnn_type,
            topology_refinement_mode=args.topology_refinement_mode,
            train_fast_sampler=bool(args.train_fast_sampler),
            train_masked_room=bool(args.train_masked_room),
            fast_sampler_steps=args.fast_sampler_steps,
            masked_steps=args.masked_steps,
        )

    split_component_checkpoints(final_ckpt=final_ckpt, out_dir=ckpt_dir)

    fast_sampler_checkpoint = ckpt_dir / "fast_sampler" / "fast_sampler_best_reselected.pth"
    if not fast_sampler_checkpoint.exists():
        fast_sampler_checkpoint = ckpt_dir / "fast_sampler" / "fast_sampler_best.pth"
    if not fast_sampler_checkpoint.exists():
        fast_sampler_checkpoint = None
    masked_room_checkpoint = ckpt_dir / "masked_room" / "masked_room_best.pth"
    if not masked_room_checkpoint.exists():
        masked_room_checkpoint = None

    generate_and_export(
        checkpoint_dir=ckpt_dir,
        output_dir=run_dir,
        num_rooms=args.num_rooms,
        seed=args.seed,
        num_diffusion_steps=args.num_diffusion_steps,
        topology_population=args.topology_population,
        topology_generations=args.topology_generations,
        diffusion_cfg_schedule_mode=args.diffusion_cfg_schedule_mode,
        diffusion_cfg_schedule_min_scale=args.diffusion_cfg_schedule_min_scale,
        diffusion_cfg_schedule_power=args.diffusion_cfg_schedule_power,
        room_generator_mode=args.room_generator_mode,
        masked_room_checkpoint=masked_room_checkpoint,
        fast_sampling_checkpoint=fast_sampler_checkpoint,
        use_fast_sampling=bool(args.use_fast_sampling),
        enable_map_elites=bool(args.enable_map_elites),
        device=str(args.device),
    )

    print("[DONE] Outputs:")
    print(f"  - {run_dir / 'dungeon_full.png'}")
    print(f"  - {run_dir / 'rooms_png'}")
    print(f"  - {run_dir / 'dungeon_grid_ids.txt'}")
    print(f"  - {run_dir / 'dungeon_grid_vglc.txt'}")
    print(f"  - {run_dir / 'rooms_txt_ids'}")
    print(f"  - {run_dir / 'rooms_txt_vglc'}")
    print(f"  - {run_dir / 'generation_summary.json'}")


if __name__ == "__main__":
    main()
