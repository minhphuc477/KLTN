"""Seal every primary sampling request from a completed training plan.

Mission-bank JSON maps each fold ID to {mission_id: graph_json_path}.
Graph files use NetworkX node-link JSON with the edge collection named links.
This command hashes input files; it neither loads model weights nor samples.
"""

from __future__ import annotations

import argparse
import itertools
import json
from numbers import Integral
from pathlib import Path

import yaml

from src.config_system import validate_config
from src.publication.attempt_runner import AttemptSpec
from src.publication.dataset_provenance import file_sha256, verify_dataset_manifest
from src.publication.protocol import canonical_json, payload_sha256, seal_protocol_manifest
from src.publication.execution_provenance import verify_execution_provenance
from src.publication.evaluation_design import seal_evaluation_design

from .run_protocol import (
    ALLOWED_MASKED_ROOM_VARIANT_CONFIG_DIFFERENCES,
    ALLOWED_MODEL_VARIANT_CONFIG_DIFFERENCES,
    assert_model_variant_config_parity,
)


def build_sampling_plan(*, training_manifest, protocol_sha256, mission_bank,
                        sampling_seeds, generation_kwargs, git_commit):
    """Expand the full paired matrix; reject omitted arms, folds, or inputs."""
    if (training_manifest.get("protocol_sha256") != protocol_sha256
            or seal_protocol_manifest(training_manifest)["protocol_sha256"] != protocol_sha256):
        raise ValueError("Training manifest differs from its sealed protocol digest.")
    seeds = tuple(sampling_seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
        raise ValueError("Sampling seeds must be nonempty nonnegative integers.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Sampling seeds must be unique.")
    if "seed" in generation_kwargs or "mission_graph" in generation_kwargs or generation_kwargs.get("generate_topology"):
        raise ValueError("Seeds and mission graphs come only from the sealed matrix.")
    if float(generation_kwargs.get("logic_guidance_scale", 0.0)) != 0.0:
        raise ValueError("Primary model comparisons require unguided sampling.")
    folds = {fold["fold_id"]: fold for fold in training_manifest["folds"]}
    if set(mission_bank) != set(folds) or any(not missions for missions in mission_bank.values()):
        raise ValueError("The mission bank must cover every planned fold with a nonempty mission set.")
    expected = set(itertools.product(folds, training_manifest["training_seeds"],
                                    (variant["model_id"] for variant in training_manifest["model_variants"])))
    jobs = training_manifest["jobs"]["diffusion"]
    actual = [(job["fold_id"], job["training_seed"], job["model_id"]) for job in jobs]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Training jobs do not cover the complete declared model/fold/seed matrix.")
    execution = training_manifest.get("execution_provenance")
    verify_execution_provenance(execution)
    if git_commit != execution["git_commit"]:
        raise ValueError("Supplied Git commit differs from the sealed execution provenance.")
    design = seal_evaluation_design(mission_bank=mission_bank, sampling_seeds=seeds,
                                    generation_kwargs=generation_kwargs, fold_ids=folds)
    if design != training_manifest.get("evaluation_design"):
        raise ValueError("Sampling requests differ from the evaluation design frozen before training.")
    provenance = training_manifest["dataset_provenance"]
    if provenance.get("verification_status") != "verified":
        raise ValueError("Sampling requires verified training-data provenance.")
    file_hashes, dataset_hashes, configs_by_pair = {}, {}, {}

    def artifact(value):
        path = Path(value).resolve()
        if path not in file_hashes:
            file_hashes[path] = file_sha256(path)
        return {"path": str(path), "sha256": file_hashes[path]}

    attempts = []
    for job in jobs:
        config = validate_config(yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8")))
        if payload_sha256(config) != job["config_sha256"]:
            raise ValueError("Training configuration changed after protocol sealing.")
        fold = folds[job["fold_id"]]
        if any(config["dataset"][key] != fold[key] for key in ("train_dungeons", "test_dungeons", "variants")):
            raise ValueError("Training configuration differs from its declared data split.")
        if config["runtime"]["seed"] != job["training_seed"]:
            raise ValueError("Training configuration differs from its declared seed.")
        data_root = str(Path(config["dataset"]["data_dir"]).resolve())
        if data_root not in dataset_hashes:
            dataset_hashes[data_root] = verify_dataset_manifest(data_root)["content_sha256"]
        if dataset_hashes[data_root] != provenance["content_sha256"]:
            raise ValueError("Dataset content changed since training protocol sealing.")
        pair = (job["fold_id"], job["training_seed"])
        configs_by_pair.setdefault(pair, []).append(config)
        branch = str(
            job.get("model_branch")
            or training_manifest.get("model_branch")
            or ("masked_room" if config["training"]["stage"] == "masked_room" else "diffusion")
        ).strip().lower()
        if branch == "masked_room":
            if config["diffusion"]["vqvae_checkpoint"] is not None or job["depends_on"] != []:
                raise ValueError("Masked-room branch must have no VQ-VAE dependency or tokenizer handoff.")
            shared = {
                "resolved_config": artifact(job["config"]),
                "masked_room_checkpoint": artifact(job["expected_checkpoint"]),
            }
        else:
            tokenizer = config["diffusion"]["vqvae_checkpoint"]
            if job["depends_on"] != [tokenizer]:
                raise ValueError("Tokenizer handoff differs from the training job dependency.")
            shared = {
                "resolved_config": artifact(job["config"]),
                "vqvae_checkpoint": artifact(tokenizer),
                "diffusion_checkpoint": artifact(job["expected_checkpoint"]),
            }
        for mission_id, mission_path in sorted(mission_bank[job["fold_id"]].items()):
            inputs = {**shared, "mission_graph": artifact(mission_path)}
            for seed in seeds:
                spec = AttemptSpec(
                    metadata=dict(run_id=job["run_id"], git_commit=git_commit,
                        config_sha256=job["config_sha256"],
                        dataset_manifest_sha256=provenance["content_sha256"],
                        protocol_sha256=protocol_sha256, outer_fold=job["fold_id"],
                        train_seed=job["training_seed"], sampling_seed=int(seed),
                        mission_graph_id=mission_id, model_id=job["model_id"], inference_variant="unguided"),
                    generation_kwargs={**generation_kwargs, "logic_guidance_scale": 0.0}, artifacts=inputs,
                )
                attempts.append(spec.to_dict())
    manifest_branch = str(training_manifest.get("model_branch", "diffusion")).strip().lower()
    allowed_diffs = (
        ALLOWED_MASKED_ROOM_VARIANT_CONFIG_DIFFERENCES
        if manifest_branch == "masked_room"
        else ALLOWED_MODEL_VARIANT_CONFIG_DIFFERENCES
    )
    for configs in configs_by_pair.values():
        assert_model_variant_config_parity(configs, allowed_differences=allowed_diffs)
    # Independent controls inherit shared symbolic settings from one declared
    # arm, but consume neither tokenizer nor diffusion weights. Repeat seeds
    # pair the requests; controls do not gain extra independent training runs.
    control_specs = []
    seen = set()
    for entry in attempts:
        meta = entry["metadata"]
        key = (meta["outer_fold"], meta["train_seed"], meta["sampling_seed"], meta["mission_graph_id"])
        if key in seen:
            continue
        seen.add(key)
        for model in design["independent_controls"]:
            control_specs.append(AttemptSpec(
                metadata={**meta, "model_id": model, "run_id": f"{meta['outer_fold']}__{model}__seed_{meta['train_seed']}"},
                generation_kwargs=entry["generation_kwargs"],
                artifacts={name: entry["artifacts"][name] for name in ("resolved_config", "mission_graph")},
            ).to_dict())
    attempts.extend(control_specs)
    return {"schema_version": "2.0", "execution_provenance": execution, "attempts": attempts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--mission-bank", type=Path, required=True)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--generation-kwargs", type=Path, required=True, help="JSON object of frozen sampling settings.")
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    from src.publication.data_support import preflight_training_manifest
    training_manifest = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    preflight_training_manifest(training_manifest)
    plan = build_sampling_plan(
        training_manifest=training_manifest,
        protocol_sha256=args.protocol_sha256,
        mission_bank=json.loads(args.mission_bank.read_text(encoding="utf-8")),
        sampling_seeds=args.sampling_seeds,
        generation_kwargs=json.loads(args.generation_kwargs.read_text(encoding="utf-8")),
        git_commit=args.git_commit,
    )
    with args.output_plan.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(plan))
    print(f"Sealed {len(plan['attempts'])} requests; plan SHA-256: {payload_sha256(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
