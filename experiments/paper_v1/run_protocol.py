"""Create fresh, validated configs for the paper-v1 training matrix.

This command plans training; it never starts a training job.  It deliberately
refuses a non-empty destination so publication evidence cannot be mixed with
old checkpoints or ad-hoc artifacts.

Example::

    python -m experiments.paper_v1.run_protocol \
        --base-config configs/zelda_hmolqd.yaml \
        --output-root outputs/paper_v1
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import yaml

from src.config_system import merge_config, validate_config
from src.publication.protocol import (
    PUBLICATION_VARIANTS,
    ModelVariant,
    OuterFold,
    apply_publication_variant,
    build_leave_one_dungeon_out_folds,
    build_protocol_manifest,
    payload_sha256,
    seal_protocol_manifest,
)
from src.publication.dataset_provenance import verify_dataset_manifest


DEFAULT_TRAINING_SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

# Every other resolved field must be byte-for-byte equal within a fold/seed.
# This turns accidental base-config drift into a planning error instead of a
# hidden experimental confound.
ALLOWED_MODEL_VARIANT_CONFIG_DIFFERENCES = frozenset(
    {
        "runtime.experiment_name",
        "runtime.output_dir",
        "diffusion.checkpoint_dir",
        "diffusion.condition_global_graph_enabled",
        "diffusion.graph_conditioning_mode",
        "diffusion.logic_net_enabled",
        "diffusion.logic_net_trainable",
        "diffusion.alpha_logic",
        "diffusion.logic_topology_trace_weight",
        "diffusion.logic_topology_anchor_weight",
        "diffusion.logic_global_reach_weight",
        "diffusion.logic_global_room_weight",
    }
)


@dataclass(frozen=True)
class MaterializedProtocol:
    """Paths and counts produced by one protocol materialization."""

    root: Path
    manifest_path: Path
    command_plan_path: Path
    tokenizer_config_count: int
    diffusion_config_count: int


def _set(config: Dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = copy.deepcopy(value)


def differing_config_paths(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    prefix: str = "",
) -> set[str]:
    """Return leaf paths whose resolved values differ."""
    paths: set[str] = set()
    missing = object()
    keys = set(left).union(right)
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        left_value = left.get(key, missing)
        right_value = right.get(key, missing)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            paths.update(differing_config_paths(left_value, right_value, prefix=path))
        elif left_value != right_value:
            paths.add(path)
    return paths


def assert_model_variant_config_parity(configs: Sequence[Mapping[str, Any]]) -> None:
    """Reject resolved arm differences outside the predeclared causal factors."""
    if len(configs) < 2:
        return
    reference = configs[0]
    for candidate in configs[1:]:
        unexpected = differing_config_paths(reference, candidate).difference(
            ALLOWED_MODEL_VARIANT_CONFIG_DIFFERENCES
        )
        if unexpected:
            raise ValueError(
                "Publication model variants differ outside the frozen fairness "
                f"contract: {sorted(unexpected)}"
            )


def _fresh_run_overrides(
    config: Dict[str, Any],
    *,
    stage: str,
    run_id: str,
    output_dir: Path,
) -> None:
    """Remove every supported route to inherited model state."""
    values = {
        "training.stage": stage,
        "runtime.experiment_name": run_id,
        "runtime.output_dir": str(output_dir),
        "runtime.auto_resume": False,
        "runtime.resume": None,
        "vqvae.resume_checkpoint": None,
        "diffusion.warm_start_checkpoint": None,
    }
    for path, value in values.items():
        _set(config, path, value)


def _build_tokenizer_config(
    base_config: Mapping[str, Any],
    *,
    fold: OuterFold,
    training_seed: int,
    output_root: Path,
) -> tuple[Dict[str, Any], Path]:
    config = copy.deepcopy(dict(base_config))
    run_id = f"{fold.fold_id}__shared_vqvae__seed_{training_seed}"
    output_dir = output_root / "runs" / fold.fold_id / "shared_vqvae" / f"seed_{training_seed}"
    _fresh_run_overrides(
        config,
        stage="vqvae",
        run_id=run_id,
        output_dir=output_dir,
    )
    for path, value in {
        "runtime.seed": int(training_seed),
        "runtime.cudnn_benchmark": False,
        "runtime.cudnn_deterministic": True,
        "runtime.deterministic_algorithms": True,
        "dataset.train_dungeons": list(fold.train_dungeons),
        "dataset.test_dungeons": list(fold.test_dungeons),
        "dataset.variants": list(fold.variants),
        "dataset.topology_supervision_mode": "runtime_aligned",
        "vqvae.checkpoint_dir": str(output_dir / "checkpoints" / "vqvae"),
        # This stage cannot consume a tokenizer checkpoint, but keeping the
        # diffusion handoff null prevents the config from implying reuse.
        "diffusion.vqvae_checkpoint": None,
    }.items():
        _set(config, path, value)
    validated = validate_config(config)
    checkpoint = Path(validated["vqvae"]["checkpoint_dir"]) / "vqvae_pretrained.pth"
    return validated, checkpoint


def _build_diffusion_config(
    base_config: Mapping[str, Any],
    *,
    fold: OuterFold,
    variant: ModelVariant,
    training_seed: int,
    output_root: Path,
    tokenizer_checkpoint: Path,
) -> Dict[str, Any]:
    config = apply_publication_variant(
        base_config,
        fold=fold,
        variant=variant,
        training_seed=training_seed,
        output_root=str(output_root / "runs"),
    )
    run_id = str(config["runtime"]["experiment_name"])
    output_dir = Path(str(config["runtime"]["output_dir"]))
    _fresh_run_overrides(
        config,
        stage="diffusion",
        run_id=run_id,
        output_dir=output_dir,
    )
    _set(config, "diffusion.vqvae_checkpoint", str(tokenizer_checkpoint))
    _set(config, "diffusion.checkpoint_dir", str(output_dir / "checkpoints" / "diffusion"))
    return validate_config(config)


def build_run_plan(
    *,
    base_config: Mapping[str, Any],
    output_root: Path,
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    folds: Sequence[OuterFold] | None = None,
    variants: Sequence[ModelVariant] = PUBLICATION_VARIANTS,
    dataset_provenance: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], list[tuple[Path, Dict[str, Any]]]]:
    """Build the complete plan in memory and validate every config."""
    resolved_folds = tuple(
        build_leave_one_dungeon_out_folds() if folds is None else folds
    )
    resolved_variants = tuple(variants)
    seeds = tuple(int(seed) for seed in training_seeds)
    manifest = build_protocol_manifest(
        training_seeds=seeds,
        folds=resolved_folds,
        variants=resolved_variants,
    )
    configs: list[tuple[Path, Dict[str, Any]]] = []
    tokenizer_jobs: list[Dict[str, Any]] = []
    diffusion_jobs: list[Dict[str, Any]] = []

    for fold in resolved_folds:
        for seed in seeds:
            tokenizer_config, tokenizer_checkpoint = _build_tokenizer_config(
                base_config,
                fold=fold,
                training_seed=seed,
                output_root=output_root,
            )
            tokenizer_path = (
                output_root
                / "configs"
                / "vqvae"
                / fold.fold_id
                / f"seed_{seed}.yaml"
            )
            configs.append((tokenizer_path, tokenizer_config))
            tokenizer_jobs.append(
                {
                    "fold_id": fold.fold_id,
                    "training_seed": seed,
                    "config": str(tokenizer_path),
                    "expected_checkpoint": str(tokenizer_checkpoint),
                    "config_sha256": payload_sha256(tokenizer_config),
                    "depends_on": [],
                }
            )

            variant_configs: list[Dict[str, Any]] = []
            for variant in resolved_variants:
                diffusion_config = _build_diffusion_config(
                    base_config,
                    fold=fold,
                    variant=variant,
                    training_seed=seed,
                    output_root=output_root,
                    tokenizer_checkpoint=tokenizer_checkpoint,
                )
                diffusion_path = (
                    output_root
                    / "configs"
                    / "diffusion"
                    / fold.fold_id
                    / variant.model_id
                    / f"seed_{seed}.yaml"
                )
                configs.append((diffusion_path, diffusion_config))
                variant_configs.append(diffusion_config)
                diffusion_jobs.append(
                    {
                        "run_id": diffusion_config["runtime"]["experiment_name"],
                        "fold_id": fold.fold_id,
                        "model_id": variant.model_id,
                        "training_seed": seed,
                        "config": str(diffusion_path),
                        "config_sha256": payload_sha256(diffusion_config),
                        "depends_on": [str(tokenizer_checkpoint)],
                    }
                )
            assert_model_variant_config_parity(variant_configs)

    manifest["base_config_sha256"] = payload_sha256(dict(base_config))
    manifest["dataset_provenance"] = dict(dataset_provenance or {
        "verification_status": "not_verified_in_memory_plan",
    })
    manifest["output_root"] = str(output_root)
    manifest["tokenizer_run_count"] = len(tokenizer_jobs)
    manifest["diffusion_run_count"] = len(diffusion_jobs)
    manifest["jobs"] = {
        "tokenizer": tokenizer_jobs,
        "diffusion": diffusion_jobs,
    }
    return seal_protocol_manifest(manifest), configs


def _assert_empty_destination(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Publication output root is not empty: {output_root}. "
            "Choose a new directory; old artifacts and checkpoints must not be mixed "
            "with confirmatory evidence."
        )


def _powershell_quote(value: str | Path) -> str:
    """Quote one literal PowerShell argument without allowing interpolation."""
    return "'" + str(value).replace("'", "''") + "'"


def materialize_protocol(
    *,
    base_config_path: str | Path,
    output_root: str | Path,
    training_seeds: Sequence[int] = DEFAULT_TRAINING_SEEDS,
    folds: Sequence[OuterFold] | None = None,
    variants: Sequence[ModelVariant] = PUBLICATION_VARIANTS,
) -> MaterializedProtocol:
    """Validate then write a fresh publication training plan."""
    root = Path(output_root).resolve()
    _assert_empty_destination(root)
    base_path = Path(base_config_path).resolve()
    base_config = merge_config(yaml_path=str(base_path))
    dataset_provenance = verify_dataset_manifest(base_config["dataset"]["data_dir"])
    manifest, configs = build_run_plan(
        base_config=base_config,
        output_root=root,
        training_seeds=training_seeds,
        folds=folds,
        variants=variants,
        dataset_provenance=dataset_provenance,
    )

    root.mkdir(parents=True, exist_ok=True)
    for path, config in configs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

    manifest_path = root / "protocol_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    command_plan_path = root / "commands.ps1"
    command_lines = [
        "$ErrorActionPreference = 'Stop'",
        "# Run tokenizer jobs first. Diffusion configs depend on their fresh checkpoints.",
    ]
    for job in manifest["jobs"]["tokenizer"]:
        command_lines.append(
            f"python main.py train --config {_powershell_quote(job['config'])}"
        )
    command_lines.append("# Run diffusion jobs only after all tokenizer jobs above succeed.")
    for job in manifest["jobs"]["diffusion"]:
        command_lines.append(
            f"python main.py train --config {_powershell_quote(job['config'])}"
        )
    command_plan_path.write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    return MaterializedProtocol(
        root=root,
        manifest_path=manifest_path,
        command_plan_path=command_plan_path,
        tokenizer_config_count=len(manifest["jobs"]["tokenizer"]),
        diffusion_config_count=len(manifest["jobs"]["diffusion"]),
    )


def _parse_seeds(raw: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be comma-separated integers.") from exc
    if not values:
        raise argparse.ArgumentTypeError("At least one training seed is required.")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/zelda_hmolqd.yaml",
        help="Validated base YAML used only for hyperparameters.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/paper_v1",
        help="New or empty directory for the publication plan and future runs.",
    )
    parser.add_argument(
        "--training-seeds",
        type=_parse_seeds,
        default=DEFAULT_TRAINING_SEEDS,
        help="Comma-separated independent training seeds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = materialize_protocol(
        base_config_path=args.base_config,
        output_root=args.output_root,
        training_seeds=args.training_seeds,
    )
    print(
        f"Materialized {result.tokenizer_config_count} tokenizer configs and "
        f"{result.diffusion_config_count} diffusion configs under {result.root}."
    )
    print(f"Manifest: {result.manifest_path}")
    print(f"Command plan: {result.command_plan_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
