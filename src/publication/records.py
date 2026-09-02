"""Stage-aware records for publication generation and validation results."""

from __future__ import annotations

import json
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


RESULT_SCHEMA_VERSION = "2.0"
STAGE_NAMES = ("raw", "constrained", "repaired", "compiled")
STAGE_STATUSES = (
    "generated",
    "invalid",
    "valid",
    "timeout",
    "validator_error",
    "not_applicable",
    "not_run",
)
GRID_PRESENT_STATUSES = ("generated", "invalid", "valid")
GENERATION_STATUSES = ("completed", "generation_error", "timeout")


def _require_sha256(name: str, value: Optional[str], *, optional: bool = False) -> None:
    if value is None and optional:
        return
    raw = str(value or "")
    if len(raw) != 64 or any(char not in string.hexdigits for char in raw):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256 digest.")


@dataclass(frozen=True)
class StageResult:
    """One immutable generation/validation stage for a dungeon sample."""

    stage: str
    status: str
    grid_path: Optional[str]
    exact_valid: Optional[bool]
    validation: Mapping[str, Any] = field(default_factory=dict)
    grid_sha256: Optional[str] = None
    shape: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.stage not in STAGE_NAMES:
            raise ValueError(
                f"Unknown result stage {self.stage!r}; expected one of {STAGE_NAMES}."
            )
        if self.status not in STAGE_STATUSES:
            raise ValueError(
                f"Unknown stage status {self.status!r}; expected one of {STAGE_STATUSES}."
            )
        if self.status == "valid" and self.exact_valid is not True:
            raise ValueError("status='valid' requires exact_valid=True.")
        if self.status == "invalid" and self.exact_valid is not False:
            raise ValueError("status='invalid' requires exact_valid=False.")
        if self.status not in {"valid", "invalid"} and self.exact_valid is not None:
            raise ValueError(
                f"status={self.status!r} requires exact_valid=None."
            )
        if self.shape is not None:
            if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
                raise ValueError(f"shape must contain two positive dimensions; got {self.shape}.")
        if self.status in GRID_PRESENT_STATUSES:
            if not str(self.grid_path or "").strip():
                raise ValueError(f"status={self.status!r} requires grid_path.")
            if self.shape is None:
                raise ValueError(f"status={self.status!r} requires shape.")
            _require_sha256("grid_sha256", self.grid_sha256)
        else:
            _require_sha256("grid_sha256", self.grid_sha256, optional=True)
        if self.grid_sha256 is not None and not str(self.grid_path or "").strip():
            raise ValueError("grid_sha256 cannot be present without grid_path.")
        if self.shape is not None and not str(self.grid_path or "").strip():
            raise ValueError("shape cannot be present without grid_path.")
        if str(self.grid_path or "").strip() and self.status not in GRID_PRESENT_STATUSES:
            if self.shape is None:
                raise ValueError("A retained grid_path requires shape metadata.")
            _require_sha256("grid_sha256", self.grid_sha256)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "grid_path": self.grid_path,
            "grid_sha256": self.grid_sha256,
            "shape": list(self.shape) if self.shape is not None else None,
            "exact_valid": self.exact_valid,
            "validation": dict(self.validation),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageResult":
        raw_shape = payload.get("shape")
        shape = None
        if raw_shape is not None:
            if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
                raise ValueError("Stage shape must be a two-item sequence or null.")
            shape = tuple(int(value) for value in raw_shape)
        return cls(
            stage=str(payload["stage"]),
            status=str(payload["status"]),
            grid_path=(
                None if payload.get("grid_path") is None else str(payload["grid_path"])
            ),
            grid_sha256=(
                None
                if payload.get("grid_sha256") is None
                else str(payload["grid_sha256"])
            ),
            shape=shape,
            exact_valid=payload.get("exact_valid"),
            validation=dict(payload.get("validation") or {}),
        )


@dataclass(frozen=True)
class PublicationResultRecord:
    """One complete generated dungeon with every observable output stage."""

    run_id: str
    git_commit: str
    config_sha256: str
    dataset_manifest_sha256: str
    protocol_sha256: str
    outer_fold: str
    train_seed: int
    sampling_seed: int
    mission_graph_id: str
    model_id: str
    inference_variant: str
    generation_status: str
    stages: Mapping[str, StageResult]
    repair_metrics: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)
    error_type: Optional[str] = None
    schema_version: str = RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported publication result schema {self.schema_version!r}; "
                f"expected {RESULT_SCHEMA_VERSION!r}."
            )
        for name, value in (
            ("run_id", self.run_id),
            ("git_commit", self.git_commit),
            ("config_sha256", self.config_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("protocol_sha256", self.protocol_sha256),
            ("outer_fold", self.outer_fold),
            ("mission_graph_id", self.mission_graph_id),
            ("model_id", self.model_id),
            ("inference_variant", self.inference_variant),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty.")
        if int(self.train_seed) < 0 or int(self.sampling_seed) < 0:
            raise ValueError("train_seed and sampling_seed must be non-negative.")
        for name, value in (
            ("config_sha256", self.config_sha256),
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("protocol_sha256", self.protocol_sha256),
        ):
            _require_sha256(name, value)
        if self.generation_status not in GENERATION_STATUSES:
            raise ValueError(
                f"Unknown generation_status {self.generation_status!r}; "
                f"expected one of {GENERATION_STATUSES}."
            )
        normalized = dict(self.stages)
        if "raw" not in normalized:
            raise ValueError("Publication result records must retain the raw stage.")
        unexpected = sorted(set(normalized) - set(STAGE_NAMES))
        if unexpected:
            raise ValueError(f"Unexpected result stages: {unexpected}.")
        for name, result in normalized.items():
            if not isinstance(result, StageResult):
                raise TypeError(f"Stage {name!r} must be a StageResult.")
            if result.stage != name:
                raise ValueError(
                    f"Stage key {name!r} does not match embedded stage {result.stage!r}."
                )
        if "repaired" in normalized and "constrained" not in normalized:
            raise ValueError(
                "A repaired publication record must preserve its constrained stage."
            )
        if "compiled" in normalized and "repaired" not in normalized:
            raise ValueError(
                "A compiled publication record must preserve its repaired stage."
            )
        if (
            self.generation_status == "completed"
            and normalized["raw"].status not in GRID_PRESENT_STATUSES
        ):
            raise ValueError("Completed generation requires an emitted raw grid stage.")
        if self.generation_status == "completed":
            missing_stages = [name for name in STAGE_NAMES if name not in normalized]
            if missing_stages:
                raise ValueError(
                    "Completed generation requires every publication stage; "
                    f"missing={missing_stages}."
                )
        if self.generation_status == "completed" and self.error_type is not None:
            raise ValueError("Completed generation cannot declare error_type.")
        if self.generation_status != "completed" and not str(self.error_type or "").strip():
            raise ValueError("Failed or timed-out generation requires error_type.")

    @property
    def record_id(self) -> str:
        return "__".join(
            (
                self.run_id,
                self.mission_graph_id,
                f"sample_{int(self.sampling_seed)}",
                self.inference_variant,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "run_id": self.run_id,
            "git_commit": self.git_commit,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "protocol_sha256": self.protocol_sha256,
            "outer_fold": self.outer_fold,
            "train_seed": int(self.train_seed),
            "sampling_seed": int(self.sampling_seed),
            "mission_graph_id": self.mission_graph_id,
            "model_id": self.model_id,
            "inference_variant": self.inference_variant,
            "generation_status": self.generation_status,
            "stages": {
                name: self.stages[name].to_dict()
                for name in STAGE_NAMES
                if name in self.stages
            },
            "repair_metrics": dict(self.repair_metrics),
            "timing": dict(self.timing),
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PublicationResultRecord":
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported publication result schema {schema_version!r}; "
                f"expected {RESULT_SCHEMA_VERSION!r}."
            )
        raw_stages = payload.get("stages")
        if not isinstance(raw_stages, Mapping):
            raise ValueError("Publication result stages must be a mapping.")
        return cls(
            schema_version=schema_version,
            run_id=str(payload["run_id"]),
            git_commit=str(payload["git_commit"]),
            config_sha256=str(payload["config_sha256"]),
            dataset_manifest_sha256=str(payload["dataset_manifest_sha256"]),
            protocol_sha256=str(payload["protocol_sha256"]),
            outer_fold=str(payload["outer_fold"]),
            train_seed=int(payload["train_seed"]),
            sampling_seed=int(payload["sampling_seed"]),
            mission_graph_id=str(payload["mission_graph_id"]),
            model_id=str(payload["model_id"]),
            inference_variant=str(payload["inference_variant"]),
            generation_status=str(payload["generation_status"]),
            stages={
                str(name): StageResult.from_dict(stage_payload)
                for name, stage_payload in raw_stages.items()
            },
            repair_metrics=dict(payload.get("repair_metrics") or {}),
            timing=dict(payload.get("timing") or {}),
            error_type=(
                None if payload.get("error_type") is None else str(payload["error_type"])
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f"{output_path.name}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(output_path)


def changed_tile_metrics(
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, Any]:
    """Compute exact stage-to-stage intervention metrics."""
    before_array = np.asarray(before)
    after_array = np.asarray(after)
    if before_array.shape != after_array.shape:
        raise ValueError(
            "Cannot compare publication stages with different shapes: "
            f"{before_array.shape} versus {after_array.shape}."
        )
    if before_array.ndim != 2:
        raise ValueError(
            f"Publication dungeon grids must be rank 2; got {before_array.ndim}."
        )
    changed = before_array != after_array
    changed_count = int(np.count_nonzero(changed))
    total_count = int(changed.size)
    return {
        "changed_tile_count": changed_count,
        "changed_tile_fraction": (
            float(changed_count / total_count) if total_count else 0.0
        ),
        "any_change": bool(changed_count),
        "total_tile_count": total_count,
    }
