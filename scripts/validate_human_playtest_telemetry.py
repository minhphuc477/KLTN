#!/usr/bin/env python3
"""Validate provenance for human playtest telemetry before persona calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(
                child
                for child in path.rglob("*")
                if child.suffix.lower() in {".json", ".jsonl"}
            )
        else:
            yield path


def _load_sessions(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [
            payload
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and isinstance((payload := json.loads(line)), dict)
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("sessions"), list):
        return [item for item in payload["sessions"] if isinstance(item, dict)]
    return [payload] if isinstance(payload, dict) else []


def validate_human_session(payload: Mapping[str, Any]) -> List[str]:
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    errors: List[str] = []
    if context.get("evidence_source") != "human_playtest":
        errors.append("context.evidence_source must equal 'human_playtest'")
    if context.get("consent_recorded") is not True:
        errors.append("context.consent_recorded must be true")
    if not str(context.get("participant_id", "")).strip():
        errors.append("context.participant_id must be a pseudonymous non-empty identifier")
    if not str(context.get("study_id", "")).strip():
        errors.append("context.study_id must be non-empty")
    if not str(payload.get("session_id", "")).strip():
        errors.append("session_id must be non-empty")
    if not isinstance(payload.get("events"), list) or not payload.get("events"):
        errors.append("events must contain at least one telemetry event")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "playtest" / "human_playtest_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records: List[Dict[str, Any]] = []
    for path in _iter_files(args.telemetry):
        if not path.exists():
            raise SystemExit(f"Telemetry path does not exist: {path}")
        for session in _load_sessions(path):
            errors = validate_human_session(session)
            records.append(
                {
                    "source_file": str(path),
                    "session_id": str(session.get("session_id", "")),
                    "participant_id": str((session.get("context") or {}).get("participant_id", "")),
                    "valid": not errors,
                    "errors": errors,
                }
            )
    valid = [record for record in records if record["valid"]]
    invalid = [record for record in records if not record["valid"]]
    manifest = {
        "evidence_scope": "human_playtest_telemetry_provenance",
        "valid_sessions": len(valid),
        "invalid_sessions": len(invalid),
        "unique_participants": len({record["participant_id"] for record in valid}),
        "calibration_ready": bool(valid) and not invalid,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), **manifest}, indent=2))
    return 0 if manifest["calibration_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
