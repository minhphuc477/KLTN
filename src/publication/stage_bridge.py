"""Bridge canonical generation outputs to stage-aware publication records."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np

from src.pipeline.room_stitching import build_stitched_room_layout_from_slots

from .records import PublicationResultRecord, StageResult, changed_tile_metrics


ExactValidator = Callable[[np.ndarray, str], Mapping[str, Any]]


def extract_publication_stage_grids(generation_result: Any) -> Dict[str, np.ndarray]:
    """Reconstruct comparable room stages and retain final graph compilation.

    Raw, constrained, and repaired rooms are placed in the exact same slots and
    receive the same boundary sealing and graph-edge carving. The compiled stage
    is the canonical pipeline output after graph semantics and entities are
    materialized. Keeping these operations separate prevents assembly from being
    mistaken for a learned-model or repair effect.
    """
    rooms = dict(getattr(generation_result, "rooms", {}) or {})
    stitched_layout = getattr(generation_result, "stitched_layout", None)
    slot_positions = dict(getattr(stitched_layout, "slot_positions", {}) or {})
    if not rooms or not slot_positions:
        raise ValueError("Generation result must retain rooms and stitched slot positions.")
    physical_graph = getattr(generation_result, "physical_mission_graph", None)
    if physical_graph is None:
        raise ValueError(
            "Generation result must retain the physical mission graph used for stitching."
        )

    raw_rooms: Dict[Any, np.ndarray] = {}
    constrained_rooms: Dict[Any, np.ndarray] = {}
    repaired_rooms: Dict[Any, np.ndarray] = {}
    for room_id, room in rooms.items():
        raw = getattr(room, "raw_neural_grid", None)
        constrained = getattr(room, "neural_grid", None)
        repaired = getattr(room, "room_grid", None)
        if raw is None or constrained is None or repaired is None:
            raise ValueError(
                f"Room {room_id!r} does not retain raw, constrained, and repaired grids."
            )
        raw_rooms[room_id] = np.asarray(raw, dtype=np.int32)
        constrained_rooms[room_id] = np.asarray(constrained, dtype=np.int32)
        repaired_rooms[room_id] = np.asarray(repaired, dtype=np.int32)

    def _stitch(room_grids: Mapping[Any, np.ndarray]) -> np.ndarray:
        return build_stitched_room_layout_from_slots(
            room_grids,
            physical_graph,
            slot_positions,
        ).dungeon_grid

    raw_grid = _stitch(raw_rooms)
    constrained_grid = _stitch(constrained_rooms)
    repaired_grid = _stitch(repaired_rooms)
    compiled_grid = np.asarray(getattr(generation_result, "dungeon_grid"), dtype=np.int32)
    shapes = {
        tuple(grid.shape)
        for grid in (raw_grid, constrained_grid, repaired_grid, compiled_grid)
    }
    if len(shapes) != 1:
        raise ValueError(f"Publication stages must share one stitched shape; got {sorted(shapes)}.")
    return {
        "raw": raw_grid.copy(),
        "constrained": constrained_grid.copy(),
        "repaired": repaired_grid.copy(),
        "compiled": compiled_grid.copy(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_grid(path: Path, grid: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(grid, dtype=np.int32), allow_pickle=False)
    temporary.replace(path)
    return _sha256_file(path)


def _validate_stage(
    *,
    stage: str,
    grid: np.ndarray,
    grid_path: Path,
    validator: ExactValidator,
) -> StageResult:
    grid_sha256 = _write_grid(grid_path, grid)
    try:
        validation = dict(validator(grid.copy(), stage) or {})
    except Exception as exc:  # noqa: BLE001 - validator failures are recorded evidence
        return StageResult(
            stage=stage,
            status="validator_error",
            grid_path=str(grid_path),
            grid_sha256=grid_sha256,
            shape=tuple(int(value) for value in grid.shape),
            exact_valid=None,
            validation={"error_type": type(exc).__name__, "error_message": str(exc)},
        )

    termination = str(validation.get("termination_status", "") or "").strip().lower()
    if termination in {"budget_exhausted", "timeout"}:
        status = "timeout"
        exact_valid = None
    elif validation.get("is_exact") is not True:
        status = "validator_error"
        exact_valid = None
        validation.setdefault("error_type", "non_exact_validator_result")
    else:
        verdict = validation.get("accepted", validation.get("solvable"))
        if not isinstance(verdict, (bool, np.bool_)):
            status = "validator_error"
            exact_valid = None
            validation.setdefault("error_type", "missing_boolean_exact_verdict")
        else:
            exact_valid = bool(verdict)
            status = "valid" if exact_valid else "invalid"

    return StageResult(
        stage=stage,
        status=status,
        grid_path=str(grid_path),
        grid_sha256=grid_sha256,
        shape=tuple(int(value) for value in grid.shape),
        exact_valid=exact_valid,
        validation=validation,
    )


def build_publication_result_record(
    generation_result: Any,
    *,
    validator: ExactValidator,
    output_dir: str | Path,
    run_id: str,
    git_commit: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    protocol_sha256: str,
    outer_fold: str,
    train_seed: int,
    sampling_seed: int,
    mission_graph_id: str,
    model_id: str,
    inference_variant: str,
) -> PublicationResultRecord:
    """Create one complete record from an already generated canonical result."""
    grids = extract_publication_stage_grids(generation_result)
    root = Path(output_dir)
    stages = {
        stage: _validate_stage(
            stage=stage,
            grid=grid,
            grid_path=root / "grids" / f"{stage}.npy",
            validator=validator,
        )
        for stage, grid in grids.items()
    }
    repair_metrics = {
        "raw_to_constrained": changed_tile_metrics(grids["raw"], grids["constrained"]),
        "constrained_to_repaired": changed_tile_metrics(
            grids["constrained"], grids["repaired"]
        ),
        "repaired_to_compiled": changed_tile_metrics(
            grids["repaired"], grids["compiled"]
        ),
        "raw_to_compiled": changed_tile_metrics(grids["raw"], grids["compiled"]),
    }
    metrics = dict(getattr(generation_result, "metrics", {}) or {})
    timing = {
        key: value
        for key, value in metrics.items()
        if "time" in str(key).lower() or "timing" in str(key).lower()
    }
    return PublicationResultRecord(
        run_id=run_id,
        git_commit=git_commit,
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        protocol_sha256=protocol_sha256,
        outer_fold=outer_fold,
        train_seed=int(train_seed),
        sampling_seed=int(sampling_seed),
        mission_graph_id=mission_graph_id,
        model_id=model_id,
        inference_variant=inference_variant,
        generation_status="completed",
        stages=stages,
        repair_metrics=repair_metrics,
        timing=timing,
    )


__all__ = [
    "build_publication_result_record",
    "extract_publication_stage_grids",
]
