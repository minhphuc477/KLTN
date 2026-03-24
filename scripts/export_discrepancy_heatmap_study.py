"""
Export neuro-symbolic discrepancy heatmap statistics and thesis-ready plots.

This script runs multi-seed paired experiments across topology refinement modes
and exports:
- per-room discrepancy metrics
- per-run aggregated metrics
- summary-by-mode tables
- PNG plots for thesis figures
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.evolutionary_director import mission_graph_to_networkx
from src.generation.grammar import Difficulty, MissionGrammar
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.utils.explainability import (
    compute_neuro_symbolic_discrepancy_heatmap,
    save_discrepancy_heatmap,
)

logger = logging.getLogger(__name__)


def _build_mission_graph(seed: int, num_rooms: int):
    grammar = MissionGrammar(seed=int(seed))
    mission = grammar.generate(
        difficulty=Difficulty.MEDIUM,
        num_rooms=int(num_rooms),
        max_keys=max(1, int(num_rooms) // 4),
        validate_all=True,
    )
    return mission_graph_to_networkx(mission)


def _normalize_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    if m == "upgraded":
        return "gat2"
    if m not in {"none", "lightweight", "gat2"}:
        raise ValueError(f"Invalid mode={mode!r}. Expected none|lightweight|gat2")
    return m


def _plot_box(df: pd.DataFrame, metric: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 4), dpi=140)
        ax = fig.add_subplot(111)
        modes = sorted(df["mode"].unique().tolist())
        values = [df[df["mode"] == m][metric].dropna().to_numpy(dtype=np.float64) for m in modes]
        ax.boxplot(values, tick_labels=modes, patch_artist=True)
        ax.set_title(f"Discrepancy {metric} by mode")
        ax.set_xlabel("Topology refinement mode")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True
    except Exception as exc:
        logger.warning("Skipping box plot %s: %s", out_path.name, exc)
        return False


def _plot_seed_lines(df: pd.DataFrame, metric: str, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 4), dpi=140)
        ax = fig.add_subplot(111)

        modes = sorted(df["mode"].unique().tolist())
        for mode in modes:
            sub = df[df["mode"] == mode].sort_values("seed")
            ax.plot(
                sub["seed"].to_numpy(dtype=np.int64),
                sub[metric].to_numpy(dtype=np.float64),
                marker="o",
                linewidth=1.8,
                label=mode,
            )

        ax.set_title(f"Discrepancy {metric} across seeds")
        ax.set_xlabel("Seed")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return True
    except Exception as exc:
        logger.warning("Skipping seed-line plot %s: %s", out_path.name, exc)
        return False


def run_study(
    *,
    output_dir: Path,
    seeds: Sequence[int],
    modes: Sequence[str],
    num_rooms: int,
    num_diffusion_steps: int,
    vqvae_checkpoint: str | None,
    diffusion_checkpoint: str | None,
    logic_net_checkpoint: str | None,
    condition_encoder_checkpoint: str | None,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "room_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = output_dir / "room_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    pipeline = NeuralSymbolicDungeonPipeline(
        vqvae_checkpoint=vqvae_checkpoint,
        diffusion_checkpoint=diffusion_checkpoint,
        logic_net_checkpoint=logic_net_checkpoint,
        condition_encoder_checkpoint=condition_encoder_checkpoint,
        enable_logging=False,
        topology_refinement_mode="gat2",
    )

    room_rows: List[Dict[str, float]] = []
    run_rows: List[Dict[str, float]] = []

    for seed in seeds:
        base_graph = _build_mission_graph(seed=int(seed), num_rooms=int(num_rooms))

        for mode in modes:
            mode = _normalize_mode(mode)
            pipeline.diffusion.set_topology_refinement_mode(mode)

            result = pipeline.generate_dungeon(
                mission_graph=copy.deepcopy(base_graph),
                generate_topology=False,
                num_rooms=int(num_rooms),
                num_diffusion_steps=int(num_diffusion_steps),
                guidance_scale=7.5,
                logic_guidance_scale=1.0,
                apply_repair=True,
                seed=int(seed),
                enable_map_elites=False,
            )

            run_stats: Dict[str, List[float]] = {
                "changed_ratio": [],
                "mean_changed_heat": [],
                "max_heat": [],
                "changed_tiles": [],
            }

            for room_id, room in result.rooms.items():
                neural_probs = getattr(room, "neural_probs", None)
                if neural_probs is None:
                    continue

                heatmap, stats = compute_neuro_symbolic_discrepancy_heatmap(
                    neural_probs=np.asarray(neural_probs),
                    neural_grid=np.asarray(room.neural_grid),
                    symbolic_grid=np.asarray(room.room_grid),
                )

                prefix = heatmap_dir / f"seed_{int(seed)}" / f"{mode}_room_{int(room_id)}"
                saved = save_discrepancy_heatmap(heatmap, str(prefix))

                matrix_prefix = matrix_dir / f"seed_{int(seed)}" / f"{mode}_room_{int(room_id)}"
                matrix_prefix.parent.mkdir(parents=True, exist_ok=True)
                neural_grid_path = Path(f"{matrix_prefix}_neural_grid.npy")
                symbolic_grid_path = Path(f"{matrix_prefix}_symbolic_grid.npy")
                changed_mask_path = Path(f"{matrix_prefix}_changed_mask.npy")

                neural_grid = np.asarray(room.neural_grid)
                symbolic_grid = np.asarray(room.room_grid)
                changed_mask = (neural_grid != symbolic_grid).astype(np.uint8)

                np.save(neural_grid_path, neural_grid)
                np.save(symbolic_grid_path, symbolic_grid)
                np.save(changed_mask_path, changed_mask)

                row = {
                    "seed": int(seed),
                    "mode": mode,
                    "room_id": int(room_id),
                    "changed_tiles": float(stats.get("changed_tiles", 0.0)),
                    "changed_ratio": float(stats.get("changed_ratio", 0.0)),
                    "mean_changed_heat": float(stats.get("mean_changed_heat", 0.0)),
                    "max_heat": float(stats.get("max_heat", 0.0)),
                    "heatmap_npy": str(saved.get("npy", "")),
                    "heatmap_png": str(saved.get("png", "")),
                    "neural_grid_npy": str(neural_grid_path),
                    "symbolic_grid_npy": str(symbolic_grid_path),
                    "changed_mask_npy": str(changed_mask_path),
                }
                room_rows.append(row)

                for key in ["changed_ratio", "mean_changed_heat", "max_heat", "changed_tiles"]:
                    run_stats[key].append(float(row[key]))

            run_rows.append(
                {
                    "seed": int(seed),
                    "mode": mode,
                    "n_rooms": int(len(run_stats["changed_ratio"])),
                    "changed_ratio": float(np.mean(run_stats["changed_ratio"])) if run_stats["changed_ratio"] else 0.0,
                    "mean_changed_heat": float(np.mean(run_stats["mean_changed_heat"])) if run_stats["mean_changed_heat"] else 0.0,
                    "max_heat": float(np.mean(run_stats["max_heat"])) if run_stats["max_heat"] else 0.0,
                    "changed_tiles": float(np.mean(run_stats["changed_tiles"])) if run_stats["changed_tiles"] else 0.0,
                }
            )
            logger.info(
                "Seed=%d mode=%s done (rooms=%d)",
                int(seed),
                mode,
                int(len(run_stats["changed_ratio"])),
            )

    room_df = pd.DataFrame(room_rows)
    run_df = pd.DataFrame(run_rows)
    summary_df = (
        run_df.groupby("mode", as_index=False)
        .agg(
            n_runs=("seed", "count"),
            changed_ratio_mean=("changed_ratio", "mean"),
            changed_ratio_std=("changed_ratio", "std"),
            mean_changed_heat_mean=("mean_changed_heat", "mean"),
            mean_changed_heat_std=("mean_changed_heat", "std"),
            max_heat_mean=("max_heat", "mean"),
            max_heat_std=("max_heat", "std"),
        )
    )

    room_csv = output_dir / "discrepancy_room_metrics.csv"
    run_csv = output_dir / "discrepancy_run_metrics.csv"
    summary_csv = output_dir / "discrepancy_summary_by_mode.csv"

    room_df.to_csv(room_csv, index=False)
    run_df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated_plots = []
    for metric in ["changed_ratio", "mean_changed_heat", "max_heat"]:
        box_path = plots_dir / f"box_{metric}.png"
        line_path = plots_dir / f"seed_lines_{metric}.png"
        if _plot_box(run_df, metric, box_path):
            generated_plots.append(str(box_path))
        if _plot_seed_lines(run_df, metric, line_path):
            generated_plots.append(str(line_path))

    report = {
        "seeds": [int(s) for s in seeds],
        "modes": list(modes),
        "num_rooms": int(num_rooms),
        "num_diffusion_steps": int(num_diffusion_steps),
        "outputs": {
            "room_metrics_csv": str(room_csv),
            "run_metrics_csv": str(run_csv),
            "summary_csv": str(summary_csv),
            "plots": generated_plots,
            "room_matrices_dir": str(matrix_dir),
        },
    }
    report_path = output_dir / "discrepancy_study_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "room_metrics_csv": str(room_csv),
        "run_metrics_csv": str(run_csv),
        "summary_csv": str(summary_csv),
        "report_json": str(report_path),
        "plots_dir": str(plots_dir),
        "room_matrices_dir": str(matrix_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export discrepancy heatmap metrics and plots across seeds.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/discrepancy_heatmap_study"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--modes", type=str, default="gat2,lightweight")
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--num-diffusion-steps", type=int, default=25)
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--logic-net-checkpoint", type=str, default=None)
    parser.add_argument("--condition-encoder-checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    modes = [_normalize_mode(x) for x in str(args.modes).split(",") if str(x).strip()]
    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]

    outputs = run_study(
        output_dir=args.output_dir,
        seeds=seeds,
        modes=modes,
        num_rooms=args.num_rooms,
        num_diffusion_steps=args.num_diffusion_steps,
        vqvae_checkpoint=args.vqvae_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        logic_net_checkpoint=args.logic_net_checkpoint,
        condition_encoder_checkpoint=args.condition_encoder_checkpoint,
    )

    logger.info("Discrepancy study complete")
    for k, v in outputs.items():
        logger.info("  %s: %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
