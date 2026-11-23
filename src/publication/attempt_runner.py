"""Execute a sealed attempt set without dropping failures or overwriting evidence."""

from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .analysis import aggregate_publication_records
from .dataset_provenance import file_sha256
from .protocol import canonical_json, payload_sha256
from .records import PublicationResultRecord, STAGE_NAMES, StageResult
from .stage_bridge import ExactValidator, build_publication_result_record


METADATA_FIELDS = (
    "run_id", "git_commit", "config_sha256", "dataset_manifest_sha256",
    "protocol_sha256", "outer_fold", "train_seed", "sampling_seed",
    "mission_graph_id", "model_id", "inference_variant",
)
IDENTITY_FIELDS = (
    "outer_fold", "train_seed", "sampling_seed", "mission_graph_id",
    "model_id", "inference_variant",
)


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete file atomically, refusing even concurrent overwrites."""
    encoded = canonical_json(payload).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Unlike replace(), link() refuses an existing destination. Both names
        # are on the same filesystem; readers only see a fully written file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failed_record(metadata, exception, phase, elapsed):
    return PublicationResultRecord(
        **metadata,
        generation_status="timeout" if isinstance(exception, TimeoutError) else "generation_error",
        error_type=type(exception).__name__,
        stages={name: StageResult(
            stage=name, status="not_run", grid_path=None, exact_valid=None,
            validation={"failure_phase": phase, "error_message": str(exception)},
        ) for name in STAGE_NAMES},
        timing={"attempt_seconds": elapsed},
    )


@dataclass(frozen=True)
class AttemptSpec:
    """Metadata, sampling arguments, and exact input-file hashes for one attempt.

    The generator callback must use these inputs and the canonical pipeline.
    Long-running callbacks must enforce their declared budgets; the runner
    records TimeoutError but does not pretend a thread timeout can stop a GPU job.
    """

    metadata: Mapping[str, Any]
    generation_kwargs: Mapping[str, Any]
    artifacts: Mapping[str, Mapping[str, str]]

    def __post_init__(self):
        if set(self.metadata) != set(METADATA_FIELDS):
            raise ValueError("Attempt metadata must contain exactly the publication identity fields.")
        # Reuse strict identity/hash/seed validation without inventing a second schema.
        _failed_record(dict(self.metadata), RuntimeError("planned"), "planned", 0.0)
        if "seed" in self.generation_kwargs and self.generation_kwargs["seed"] != self.metadata["sampling_seed"]:
            raise ValueError("Generation seed must match the sealed sampling seed.")
        for name, artifact in self.artifacts.items():
            if set(artifact) != {"path", "sha256"}:
                raise ValueError(f"Artifact {name!r} requires path and sha256.")
            digest = artifact["sha256"]
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"Invalid artifact SHA-256 for {name!r}.")
        canonical_json(self.to_dict())  # reject NaN and non-serializable settings

    @property
    def attempt_id(self) -> str:
        return payload_sha256({key: self.metadata[key] for key in IDENTITY_FIELDS})

    def to_dict(self):
        return copy.deepcopy({"metadata": dict(self.metadata),
                              "generation_kwargs": dict(self.generation_kwargs),
                              "artifacts": dict(self.artifacts)})

    def verify_artifacts(self):
        for name, artifact in self.artifacts.items():
            path = Path(artifact["path"])
            if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                raise ValueError(f"Sealed input artifact is missing or changed: {name}")


class AttemptLedger:
    """A sealed expected-attempt manifest and one immutable record per attempt."""

    def __init__(self, root: Path, manifest_sha256: str):
        self.root = Path(root).resolve()
        self.manifest_sha256 = manifest_sha256

    @classmethod
    def create(cls, root: str | Path, specs: Sequence[AttemptSpec], *, execution_provenance=None) -> "AttemptLedger":
        if not specs:
            raise ValueError("An attempt plan must be nonempty.")
        ids = [spec.attempt_id for spec in specs]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate sealed attempt identity, regardless of run name.")
        for spec in specs:
            spec.verify_artifacts()
        paired_settings = {}
        mission_hashes = {}
        tokenizer_hashes = {}
        for spec in specs:
            meta = spec.metadata
            pair = tuple(meta[key] for key in IDENTITY_FIELDS if key != "model_id")
            settings = canonical_json(spec.generation_kwargs)
            if paired_settings.setdefault(pair, settings) != settings:
                raise ValueError("Paired models must use identical sampling settings.")
            for artifact, key, seen in (
                ("mission_graph", (meta["outer_fold"], meta["mission_graph_id"]), mission_hashes),
                ("vqvae_checkpoint", (meta["outer_fold"], meta["train_seed"]), tokenizer_hashes),
            ):
                if artifact in spec.artifacts:
                    digest = spec.artifacts[artifact]["sha256"]
                    if seen.setdefault(key, digest) != digest:
                        raise ValueError(f"Paired input drift: {artifact}")
        # Validate cross-attempt provenance before creating files.
        aggregate_publication_records([
            _failed_record(dict(spec.metadata), RuntimeError("planned"), "planned", 0.0)
            for spec in specs
        ])
        payload = {"schema_version": "1.0", "attempts": [spec.to_dict() for spec in specs]}
        if execution_provenance is not None:
            from .execution_provenance import verify_execution_provenance
            verify_execution_provenance(execution_provenance)
            payload["execution_provenance"] = execution_provenance
        digest = payload_sha256(payload)
        path = Path(root).resolve()
        path.mkdir(parents=True, exist_ok=False)
        (path / "attempts").mkdir()
        _atomic_create_json(path / "manifest.json", payload)
        return cls(path, digest)

    def specs(self) -> tuple[AttemptSpec, ...]:
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if payload_sha256(payload) != self.manifest_sha256:
            raise ValueError("Attempt manifest changed after sealing.")
        if payload.get("schema_version") != "1.0":
            raise ValueError("Unsupported attempt manifest schema.")
        return tuple(AttemptSpec(**entry) for entry in payload["attempts"])

    def complete_records(self) -> list[PublicationResultRecord]:
        """Reject missing attempts, including requests absent from both arms."""
        specs = self.specs()
        expected = {spec.attempt_id for spec in specs}
        actual = {path.name for path in (self.root / "attempts").iterdir()}
        if actual - expected:
            raise ValueError("Unexpected attempts exist outside the sealed manifest.")
        records = []
        for spec in specs:
            root = self.root / "attempts" / spec.attempt_id
            path = root / "record.json"
            if not path.is_file():
                raise ValueError(f"Incomplete attempt ledger: {spec.attempt_id}")
            record = PublicationResultRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if any(getattr(record, key) != spec.metadata[key] for key in METADATA_FIELDS):
                raise ValueError("Record identity/provenance differs from the sealed attempt.")
            receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
            if receipt.get("record_sha256") != payload_sha256(record.to_dict()):
                raise ValueError("Attempt record changed after completion.")
            for stage in record.stages.values():
                if stage.grid_path is not None:
                    grid_path = Path(stage.grid_path).resolve()
                    if not grid_path.is_relative_to(root.resolve()):
                        raise ValueError("Retained grid must belong to its attempt directory.")
                    if not grid_path.is_file() or file_sha256(grid_path) != stage.grid_sha256:
                        raise ValueError("Retained stage grid is missing or changed.")
            for room in record.repair_metrics.get("unassembled_rooms", []):
                for artifact in room["artifacts"].values():
                    path = Path(artifact["path"]).resolve()
                    if not path.is_relative_to(root.resolve()):
                        raise ValueError("Retained room must belong to its attempt directory.")
                    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
                        raise ValueError("Retained room grid is missing or changed.")
            graph_artifact = record.repair_metrics.get('graph_context')
            if graph_artifact is not None:
                path = Path(graph_artifact['path']).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Retained graph context must belong to its attempt directory')
                if not path.is_file() or file_sha256(path) != graph_artifact['sha256']:
                    raise ValueError('Retained graph context is missing or changed')
            records.append(record)
        aggregate_publication_records(records)
        return records


def execute_attempts(
    ledger: AttemptLedger,
    *,
    generate: Callable[[AttemptSpec], Any],
    validator_factory: Callable[[AttemptSpec, Any], ExactValidator],
    seed_fn: Callable[[int], None] | None = None,
) -> list[PublicationResultRecord]:
    """Run every sealed request once and retain all ordinary failure outcomes.

    Refuse reuse instead of retrying interrupted or failed attempts under the
    same key. KeyboardInterrupt/SystemExit leave an incomplete ledger, which
    cannot enter analysis. Caller exceptions never silently replace a seed.
    """
    specs = ledger.specs()
    manifest = json.loads((ledger.root / "manifest.json").read_text(encoding="utf-8"))
    execution = manifest.get("execution_provenance")
    if any((ledger.root / "attempts").iterdir()):
        raise FileExistsError("Attempt execution already started; refusing reruns.")
    if seed_fn is None:
        from src.config_system import seed_everything
        def seed_fn(seed):
            seed_everything(seed, cudnn_benchmark=False, cudnn_deterministic=True,
                            deterministic_algorithms=True)
    if execution is not None:
        from .execution_provenance import verify_execution_provenance
        verify_execution_provenance(execution)
    for spec in specs:
        root = ledger.root / "attempts" / spec.attempt_id
        root.mkdir(exist_ok=False)  # atomic claim against concurrent runners
        _atomic_create_json(root / "started.json", {"spec_sha256": payload_sha256(spec.to_dict())})
        # Input drift invalidates the protocol, rather than becoming a scored
        # generation failure. Leave an incomplete ledger and abort the batch.
        spec.verify_artifacts()
        started = time.perf_counter()
        phase = "seeding"
        try:
            seed_fn(int(spec.metadata["sampling_seed"]))
            phase = "generation"
            generated = generate(AttemptSpec(**spec.to_dict()))
            phase = "stage_capture"
            try:
                validator = validator_factory(spec, generated)
            except Exception as exc:  # noqa: BLE001 - retain generated grids on setup failure
                # Bind the exception now: Python clears except variables after
                # the handler. The bridge persists each grid before validation.
                def validator(grid, stage, failure=exc):
                    raise failure
            record = build_publication_result_record(
                generated, validator=validator, output_dir=root, **dict(spec.metadata),
            )
        except Exception as exc:  # noqa: BLE001 - failures are part of the experiment
            record = _failed_record(dict(spec.metadata), exc, phase, time.perf_counter() - started)
        _atomic_create_json(root / "record.json", record.to_dict())
        _atomic_create_json(root / "receipt.json", {"record_sha256": payload_sha256(record.to_dict())})
    if execution is not None:
        verify_execution_provenance(execution)
    return ledger.complete_records()
