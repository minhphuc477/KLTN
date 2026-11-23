"""Reject endpoints that require tile classes absent from the training targets."""

from __future__ import annotations

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE


def require_raw_validity_target_support(room_grids) -> None:
    """Check a necessary condition for interpreting raw full-game validity.

    START and TRIFORCE must be learnable output classes. Graph attributes and
    compiler overlays do not count as labels in the frozen tokenizer targets.
    This is only a necessary support check, not a guarantee of learnability.
    """
    observed = set()
    for grid in room_grids:
        observed.update(int(value) for value in np.unique(np.asarray(grid)))
    missing = [name for name in ("START", "TRIFORCE") if int(SEMANTIC_PALETTE[name]) not in observed]
    if missing:
        raise ValueError(
            f"Raw full-game validity is unsupported by the training targets: missing {missing}. "
            "Graph metadata or later entity compilation cannot teach these classes to the frozen tokenizer. "
            "Revise the endpoint to an explicitly defined structural predicate, or introduce and audit "
            "semantic training targets before sealing a new protocol. Do not relabel compiled validity as raw."
        )


def audit_source_layouts(data, folds):
    """Seal source overlap strata before training; never deduplicate test rooms."""
    from src.zelda_data.zelda_loader import ZeldaRoomDataset
    from .room_evidence import TrainingRoomReference, heldout_overlap
    overlaps = {}
    for fold in folds:
        train = ZeldaRoomDataset(data_dir=data["data_dir"], dungeon_ids=fold["train_dungeons"],
                                 variants=fold["variants"], load_graphs=False)
        test = ZeldaRoomDataset(data_dir=data["data_dir"], dungeon_ids=fold["test_dungeons"],
                                variants=fold["variants"], load_graphs=False)
        overlaps[fold["fold_id"]] = heldout_overlap(TrainingRoomReference(train.rooms), test.rooms)
    return overlaps


def verify_training_inputs(manifest):
    """Recheck both training stages and raw corpus contents between jobs."""
    from pathlib import Path
    import yaml
    from src.config_system import validate_config
    from .dataset_provenance import verify_dataset_manifest
    from .protocol import payload_sha256, seal_protocol_manifest
    if seal_protocol_manifest(manifest)["protocol_sha256"] != manifest.get("protocol_sha256"):
        raise ValueError("Training manifest changed after sealing.")
    from .execution_provenance import verify_execution_provenance
    from .evaluation_design import verify_evaluation_design
    verify_execution_provenance(manifest.get("execution_provenance"))
    verify_evaluation_design(manifest.get("evaluation_design"), [fold["fold_id"] for fold in manifest["folds"]])
    configs = []
    checked = set()
    for stage in ("tokenizer", "diffusion"):
        for job in manifest["jobs"][stage]:
            config = validate_config(yaml.safe_load(Path(job["config"]).read_text(encoding="utf-8")))
            if payload_sha256(config) != job["config_sha256"]:
                raise ValueError("Training configuration changed after protocol sealing.")
            data_root = str(Path(config["dataset"]["data_dir"]).resolve())
            if data_root not in checked:
                provenance = verify_dataset_manifest(data_root)
                if provenance["content_sha256"] != manifest["dataset_provenance"]["content_sha256"]:
                    raise ValueError("Source corpus changed after protocol sealing.")
                checked.add(data_root)
            configs.append((stage, config))
    return configs


def preflight_training_manifest(manifest):
    """Inspect source labels and retain repeated/unseen layout strata per fold."""
    from pathlib import Path
    from src.zelda_data.zelda_loader import ZeldaRoomDataset
    from .room_evidence import TrainingRoomReference, heldout_overlap

    configs = verify_training_inputs(manifest)
    endpoint = manifest["confirmatory_analysis"]["primary_endpoint"]
    supported = {"raw_stage_exact_validity_over_all_requested_attempts",
                 "compiled_stage_exact_validity_over_all_requested_attempts"}
    if endpoint not in supported:
        raise ValueError(f"No data-support preflight is defined for endpoint {endpoint!r}.")
    checked = set()
    overlaps = {}
    for stage, config in configs:
        if stage != "diffusion":
            continue
        data = config["dataset"]
        key = (str(Path(data["data_dir"]).resolve()), tuple(data["train_dungeons"]), tuple(data["variants"]))
        if key in checked:
            continue
        dataset = ZeldaRoomDataset(data_dir=data["data_dir"], dungeon_ids=data["train_dungeons"],
                                   variants=data["variants"], load_graphs=False)
        if endpoint == "raw_stage_exact_validity_over_all_requested_attempts":
            require_raw_validity_target_support(dataset.rooms)
        elif not dataset.rooms or not any(np.any(np.asarray(room) == int(SEMANTIC_PALETTE["FLOOR"])) for room in dataset.rooms):
            raise ValueError("Compiled validity requires nonempty room targets with floor support.")
        test = ZeldaRoomDataset(data_dir=data["data_dir"], dungeon_ids=data["test_dungeons"],
                                variants=data["variants"], load_graphs=False)
        overlaps[f"holdout_dungeon_{data['test_dungeons'][0]}"] = heldout_overlap(
            TrainingRoomReference(dataset.rooms), test.rooms)
        checked.add(key)
    if "source_layout_overlap" in manifest and manifest["source_layout_overlap"] != overlaps:
        raise ValueError("Source layout strata differ from the sealed pre-training audit.")
    return overlaps
