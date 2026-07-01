"""Validate the non-experiment publication contract for this repository.

This script does not decide whether a result is good. It checks whether the
supporting card is complete enough that a result can be interpreted without
guessing the research question, data split, artifact provenance, metric
definitions, baselines, failure taxonomy, or claim boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parent.parent

REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "research_question",
    "publishable_direction",
    "system_boundary",
    "claim_language",
    "data_card",
    "artifact_manifest",
    "metric_contract",
    "baseline_taxonomy",
    "failure_taxonomy",
    "human_calibration",
    "reproducibility",
    "ethics_ip",
    "literature_basis",
)

REQUIRED_METRICS = (
    "raw_oracle_solved_rate",
    "post_oracle_solved_rate",
    "raw_pcbs_valid_rate",
    "post_pcbs_valid_rate",
    "repair_rate",
    "tiles_repaired_mean",
    "generation_time_sec",
    "diversity",
    "controllability_error",
)

FORBIDDEN_CLAIM_TERMS = (
    "state-of-the-art",
    "sota",
    "surpasses publications",
    "human-like",
    "humanlike",
    "paper-faithful lcm-lora",
)


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(f"{k} {_stringify(v)}" for k, v in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return " ".join(_stringify(v) for v in value)
    return str(value)


def publication_card_template() -> dict[str, Any]:
    """Return a conservative, repository-specific publication-card template."""

    return {
        "schema_version": "1.0",
        "research_question": (
            "Can a graph-conditioned neural-symbolic PCG pipeline generate Zelda-like "
            "dungeons while separating raw neural validity, symbolic repair, hard "
            "oracle validation, and bounded-agent diagnostics?"
        ),
        "publishable_direction": {
            "primary": "repair-aware graph-conditioned neural-symbolic PCG system",
            "not_primary": [
                "generic dungeon generator",
                "standalone SOTA diffusion model",
                "validated human-likeness model",
                "paper-faithful LCM-LoRA implementation",
            ],
        },
        "system_boundary": {
            "proposed_method_blocks": [
                "Block 0 data and graph extraction",
                "Block I mission-graph generation",
                "Block II tokenizer",
                "Block III condition encoder",
                "Block IV room generator",
                "Block V LogicNet diagnostics/guidance",
                "Block VI symbolic repair",
                "Block VII hard oracle and P-CBS diagnostics",
            ],
            "baselines_or_ablations": [
                "random topology",
                "grammar-only topology",
                "flat-prior WFC",
                "weighted-prior WFC",
                "no graph conditioning",
                "no LogicNet",
                "no repair",
                "categorical/codebook prior",
                "masked-token generator",
                "few-step consistency adapter",
            ],
        },
        "claim_language": {
            "allowed": [
                "graph-conditioned",
                "repair-aware",
                "bounded-agent proxy",
                "matched-budget ablation",
                "raw/pre-repair validity",
                "post-repair validity",
            ],
            "forbidden_without_extra_evidence": list(FORBIDDEN_CLAIM_TERMS),
        },
        "data_card": {
            "dataset": "VGLC The Legend of Zelda",
            "dataset_manifest_path": "Data/The Legend of Zelda/dataset_manifest.json",
            "train_dungeon_ids": [1, 2, 3, 4, 5, 6, 7, 8],
            "test_dungeon_ids": [9],
            "room_shape": [16, 11],
            "tile_classes": 44,
            "leakage_policy": "Dungeon 9 is held out from training and model selection.",
            "augmentation_policy": "Document exact augmentation commands before use.",
        },
        "artifact_manifest": {
            "required_status_values": ["current", "blocked", "stale", "invalid", "smoke_only"],
            "artifacts": [
                {
                    "name": "vqvae",
                    "path": "",
                    "sha256": "",
                    "status": "blocked",
                    "notes": "Must be a valid tokenizer checkpoint, not a diffusion bundle.",
                },
                {
                    "name": "diffusion",
                    "path": "",
                    "sha256": "",
                    "status": "blocked",
                    "notes": "Must be trained on the locked split and paired with the tokenizer above.",
                },
            ],
        },
        "metric_contract": {
            "required_metrics": list(REQUIRED_METRICS),
            "raw_vs_repaired_policy": (
                "Raw/pre-repair and post-repair oracle/P-CBS rates must be reported separately."
            ),
            "timeout_policy": "Timeout, unsolved, and invalid-input outcomes are separate labels.",
        },
        "baseline_taxonomy": {
            "symbolic": ["random graph", "grammar-only", "flat WFC", "weighted WFC"],
            "neural": ["categorical prior", "masked token", "latent diffusion"],
            "pipeline": ["no graph", "no LogicNet", "no repair", "full stack"],
            "external_alignment": ["PCG Benchmark quality", "diversity", "controllability"],
        },
        "failure_taxonomy": [
            "disconnected_graph",
            "missing_key_or_token",
            "door_mismatch",
            "raw_room_unsolvable",
            "repair_failed",
            "oracle_timeout",
            "pcbs_timeout",
            "teacher_fallback_used",
            "invalid_checkpoint",
        ],
        "human_calibration": {
            "status": "proxy_only",
            "allowed_claim": "P-CBS is a bounded-agent diagnostic.",
            "blocked_claim": "P-CBS is human-like or human-calibrated.",
            "required_for_human_claims": "Consent-marked human telemetry plus persona calibration.",
        },
        "reproducibility": {
            "required_per_table": [
                "command",
                "config",
                "seed_list",
                "git_commit",
                "checkpoint_hashes",
                "hardware",
                "runtime",
                "output_paths",
            ],
        },
        "ethics_ip": {
            "data_provenance_note_required": True,
            "zelda_copyright_note_required": True,
            "recommended_scope": "research-use evaluation on VGLC-derived data",
        },
        "literature_basis": [
            {
                "topic": "PCGML and repair/evaluation framing",
                "source": "Summerville et al., Procedural Content Generation via Machine Learning",
                "url": "https://arxiv.org/abs/1702.00539",
            },
            {
                "topic": "quality/diversity/controllability benchmark axes",
                "source": "Khalifa et al., The Procedural Content Generation Benchmark",
                "url": "https://arxiv.org/abs/2503.21474",
            },
            {
                "topic": "WFC as constraint solving",
                "source": "Karth and Smith, WaveFunctionCollapse is Constraint Solving in the Wild",
                "url": "https://doi.org/10.1145/3102071.3110566",
            },
            {
                "topic": "procedural personas as synthetic playtesters",
                "source": "Holmgard et al., Automated Playtesting with Procedural Personas",
                "url": "https://arxiv.org/abs/1802.06881",
            },
        ],
    }


def validate_card(card: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in card:
            issues.append(f"missing top-level key: {key}")

    claim_language = card.get("claim_language", {})
    claim_text = _stringify(claim_language) + " " + _stringify(card.get("publishable_direction", ""))
    lowered_claim_text = claim_text.lower()
    if isinstance(claim_language, Mapping):
        allowed_terms = {str(term).lower() for term in claim_language.get("forbidden_without_extra_evidence", [])}
    else:
        allowed_terms = set()
    for term in FORBIDDEN_CLAIM_TERMS:
        if term in lowered_claim_text and term not in allowed_terms:
            issues.append(f"forbidden unbounded claim term appears outside the forbidden list: {term}")

    data_card = card.get("data_card", {})
    if not isinstance(data_card, Mapping):
        issues.append("data_card must be an object")
    else:
        manifest_path = data_card.get("dataset_manifest_path")
        if manifest_path:
            resolved = _resolve(str(manifest_path))
            if not resolved.exists():
                issues.append(f"dataset manifest does not exist: {manifest_path}")
        for key in ("train_dungeon_ids", "test_dungeon_ids", "room_shape", "tile_classes"):
            if key not in data_card:
                issues.append(f"data_card missing key: {key}")

    metric_contract = card.get("metric_contract", {})
    if not isinstance(metric_contract, Mapping):
        issues.append("metric_contract must be an object")
    else:
        metrics = set(str(v) for v in metric_contract.get("required_metrics", []))
        missing_metrics = sorted(set(REQUIRED_METRICS) - metrics)
        if missing_metrics:
            issues.append(f"metric_contract missing required metrics: {', '.join(missing_metrics)}")

    artifact_manifest = card.get("artifact_manifest", {})
    if not isinstance(artifact_manifest, Mapping):
        issues.append("artifact_manifest must be an object")
    else:
        artifacts = artifact_manifest.get("artifacts", [])
        if not isinstance(artifacts, list) or not artifacts:
            issues.append("artifact_manifest.artifacts must be a non-empty list")
        else:
            for idx, artifact in enumerate(artifacts):
                if not isinstance(artifact, Mapping):
                    issues.append(f"artifact_manifest.artifacts[{idx}] must be an object")
                    continue
                name = str(artifact.get("name", f"#{idx}"))
                status = str(artifact.get("status", "")).strip().lower()
                path_raw = str(artifact.get("path", "")).strip()
                expected_hash = str(artifact.get("sha256", "")).strip().lower()
                if status == "current":
                    if not path_raw:
                        issues.append(f"current artifact {name} has no path")
                        continue
                    resolved = _resolve(path_raw)
                    if not resolved.exists():
                        issues.append(f"current artifact {name} path does not exist: {path_raw}")
                        continue
                    if expected_hash:
                        actual_hash = _sha256_file(resolved)
                        if actual_hash.lower() != expected_hash:
                            issues.append(f"current artifact {name} sha256 mismatch")

    human_calibration = card.get("human_calibration", {})
    if isinstance(human_calibration, Mapping):
        status = str(human_calibration.get("status", "")).strip().lower()
        if status not in {"proxy_only", "calibrated", "blocked"}:
            issues.append("human_calibration.status must be proxy_only, calibrated, or blocked")
    else:
        issues.append("human_calibration must be an object")

    return issues


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(publication_card_template(), indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=Path("docs/publication_guidance_card.json"))
    parser.add_argument("--init-template", action="store_true", help="Write a conservative card template before validation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON validation report path.")
    args = parser.parse_args(argv)

    if args.init_template:
        write_template(args.card)

    if not args.card.exists():
        raise FileNotFoundError(f"Publication card does not exist: {args.card}")

    card = json.loads(args.card.read_text(encoding="utf-8"))
    issues = validate_card(card)
    report: MutableMapping[str, Any] = {
        "card": str(args.card),
        "valid": not issues,
        "issues": issues,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not issues else 2


if __name__ == "__main__":
    raise SystemExit(main())
