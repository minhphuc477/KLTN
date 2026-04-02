"""Run a matched-budget room-branch benchmark over latent and masked generators."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import pandas as pd

from scripts.run_ablation_study import AblationStudy, ExperimentConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused matched-budget benchmark for the room-generation branch.")
    parser.add_argument("--output", type=Path, default=Path("results") / "room_branch_benchmark")
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--target-curve", type=float, nargs="*", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--cbs-timeout", type=int, default=1000)
    parser.add_argument("--evolution-population", type=int, default=24)
    parser.add_argument("--evolution-generations", type=int, default=30)
    parser.add_argument("--vqvae-checkpoint", type=str, default=None)
    parser.add_argument("--diffusion-checkpoint", type=str, default=None)
    parser.add_argument("--masked-room-checkpoint", type=str, default=None)
    parser.add_argument("--logic-net-checkpoint", type=str, default=None)
    parser.add_argument("--condition-encoder-checkpoint", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _build_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(name="LATENT_REF_ON", room_generator_mode="latent_diffusion", use_reference_room_maps=True),
        ExperimentConfig(name="LATENT_REF_OFF", room_generator_mode="latent_diffusion", use_reference_room_maps=False),
        ExperimentConfig(name="MASKED_REF_ON", room_generator_mode="discrete_masked", use_reference_room_maps=True),
        ExperimentConfig(name="MASKED_REF_OFF", room_generator_mode="discrete_masked", use_reference_room_maps=False),
    ]


def _fmt(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> int:
    args = parse_args()
    if bool(args.quick):
        args.num_samples = min(int(args.num_samples), 4)
        args.diffusion_steps = min(int(args.diffusion_steps), 10)
        args.evolution_population = min(int(args.evolution_population), 12)
        args.evolution_generations = min(int(args.evolution_generations), 12)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    study = AblationStudy(
        output_dir=out_dir,
        data_root=args.data_root,
        num_rooms=int(args.num_rooms),
        target_curve=list(float(v) for v in args.target_curve),
        diffusion_steps=int(args.diffusion_steps),
        cbs_timeout=int(args.cbs_timeout),
        evolution_population=int(args.evolution_population),
        evolution_generations=int(args.evolution_generations),
        vqvae_checkpoint=args.vqvae_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        masked_room_checkpoint=args.masked_room_checkpoint,
        logic_net_checkpoint=args.logic_net_checkpoint,
        condition_encoder_checkpoint=args.condition_encoder_checkpoint,
    )

    configs = _build_configs()
    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]
    raw_df, summary_df = study.run(configs=configs, seeds=seeds)
    sig_df = study.significance_report(raw_df, baseline="LATENT_REF_ON", seed=int(args.seed) + 900)

    raw_path = out_dir / "room_branch_raw.csv"
    summary_path = out_dir / "room_branch_summary.csv"
    sig_path = out_dir / "room_branch_significance.csv"
    json_path = out_dir / "room_branch_benchmark_report.json"
    md_path = out_dir / "room_branch_benchmark_report.md"

    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    sig_df.to_csv(sig_path, index=False)

    payload = {
        "configs": [asdict(cfg) for cfg in configs],
        "seeds": seeds,
        "summary": summary_df.to_dict(orient="records"),
        "significance": sig_df.to_dict(orient="records"),
        "notes": [
            "This benchmark isolates the room branch under matched topology/search budgets.",
            "It is an internal matched-budget benchmark harness, not a substitute for external layout-baseline comparisons.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Room Branch Benchmark",
        "",
        "## Configurations",
    ]
    lines.extend([f"- `{cfg.name}`: {asdict(cfg)}" for cfg in configs])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            _fmt(summary_df),
            "",
            "## Significance vs LATENT_REF_ON",
            "",
            _fmt(sig_df) if not sig_df.empty else "_No paired comparisons available_",
            "",
            "## Notes",
            "",
            "- This script closes the reproducibility gap for room-branch matched-budget comparisons inside the repo.",
            "- External HouseDiffusion/LayoutDM-style baseline runs remain a separate experiment layer.",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
