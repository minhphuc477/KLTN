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
    "confirmatory_research_question",
    "research_gaps",
    "problem_statements",
    "research_objectives",
    "research_subquestions",
    "hypotheses",
    "scientific_traceability",
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
            "Can G-LMDM (Graph-Conditioned Logic-Regularized Masked Discrete Model) "
            "synthesize Zelda-like dungeons from unseen mission graphs while separating "
            "raw neural validity, symbolic repair burden, hard oracle proofs, and "
            "bounded-agent diagnostics?"
        ),
        "confirmatory_research_question": (
            "On held-out Dungeon-9 mission graphs, do causal GraphGPS+RRWP conditioning, "
            "absorbing-state discrete masked synthesis on 16x11 tokens, and direct Soft "
            "Bellman-Ford LogicNet supervision jointly increase pre-repair oracle "
            "solvability and reduce symbolic repair burden relative to unconditioned, "
            "continuous VQ-VAE diffusion/flow, and external symbolic baselines?"
        ),
        "research_gaps": {
            "RG1": {
                "title": "(FG1) Macro-to-Micro Semantic-Geometric Disconnect in Hierarchical Zelda Generation",
                "summary": "Unconditional or PCA-based room synthesizers lack causal GraphGPS+RRWP mission-graph and incident-door conditioning.",
                "literature_anchors": ["Dormans 2010", "Gutierrez and Schrum 2020", "Ma et al. 2023"],
            },
            "RG2": {
                "title": "(FG2) Continuous-Latent Bottleneck and Rarity Erasure in Small-Data Categorical Grids",
                "summary": "Continuous VQ-VAE compression blurs rare interactive tiles (<0.7%) on 349 VGLC training rooms (43 unique 12x7 interiors) compared to absorbing-state discrete masked token synthesis.",
                "literature_anchors": ["Austin et al. 2021", "Sahoo et al. 2024", "Schrum et al. 2025"],
            },
            "RG3": {
                "title": "(FG3) Traversability-Blind Likelihood Training and Unattributed Post-Hoc Repair",
                "summary": "Cross-entropy ignores 1-tile corridor cut-sets unless coupled with direct differentiable Bellman-Ford LogicNet supervision and 4-stage repair attribution.",
                "literature_anchors": ["Tamar et al. 2016", "Xu et al. 2018", "Zhang et al. 2020"],
            },
            "RG4": {
                "title": "(FG4) Dungeon-Split Leakage, External Baseline Deficit, and Proxy-vs-Oracle Conflation",
                "summary": "Room-level splits leak dungeon style, omitted external baselines weaken comparisons, and bounded P-CBS proxies cannot substitute for StateSpaceAStar proofs.",
                "literature_anchors": ["Withington et al. 2024", "Khalifa et al. 2025"],
            },
        },
        "problem_statements": {
            "PS1": {
                "research_gap": ["RG1"],
                "title": "Causal Multi-Scale Conditioning Deficit",
                "summary": "Uncoupled synthesizers break boundary door continuity and lock-key placement prior to repair.",
            },
            "PS2": {
                "research_gap": ["RG2"],
                "title": "Continuous Latent Compression Pathology vs. Discrete Token Generation",
                "summary": "High-parameter VQ-VAE compression erases rare gameplay tiles on small categorical corpora.",
            },
            "PS3": {
                "research_gap": ["RG3"],
                "title": "Traversability-Blind Objective and Unattributed Repair Burden",
                "summary": "Post-repair reporting without raw -> constrained -> repaired -> compiled attribution conceals neural failures.",
            },
            "PS4": {
                "research_gap": ["RG4"],
                "title": "Dungeon-Level Leakage, Omitted External Baselines, and Proof-Proxy Conflation",
                "summary": "Non-holdout evaluation and proxy-oracle conflation invalidate empirical claims.",
            },
        },
        "research_objectives": {
            "RO1": {
                "problem_statement": "PS1",
                "summary": "Isolate causal Block III GraphGPS+RRWP conditioning against N and No-RRWP ablations.",
            },
            "RO2": {
                "problem_statement": "PS2",
                "summary": "Benchmark discrete absorbing-state G-LMDM against continuous VQ-VAE + UNet[1,2] and DiT+Flow baselines.",
            },
            "RO3": {
                "problem_statement": "PS3",
                "summary": "Apply direct 16x11 Soft Bellman-Ford LogicNet supervision and report 4-stage repair attribution.",
            },
            "RO4": {
                "problem_statement": "PS4",
                "summary": "Enforce Dungeon-9 holdout, external symbolic baselines, paired Holm-corrected tests, and proof-proxy separation.",
            },
        },
        "research_subquestions": {
            "RQ1": {
                "objective": "RO1",
                "question": "Does causal GraphGPS+RRWP conditioning improve pre-repair door alignment and role consistency over N and No-RRWP?",
            },
            "RQ2": {
                "objective": "RO2",
                "question": "Does discrete G-LMDM improve rare-tile recall and pre-repair validity at lower cost than VQ-VAE latent diffusion/flow?",
            },
            "RQ3": {
                "objective": "RO3",
                "question": "Does direct 16x11 LogicNet supervision increase raw solvability and cut symbolic repair edits without diversity collapse?",
            },
            "RQ4": {
                "objective": "RO4",
                "question": "Do G-LMDM gains hold on Dungeon-9 paired tests against external symbolic baselines while separating StateSpaceAStar proofs from P-CBS proxies?",
            },
        },
        "hypotheses": {
            "H1": {
                "rq": "RQ1",
                "statement": "Causal GraphGPS+RRWP conditioning improves pre-repair door alignment and role consistency over N and No-RRWP.",
            },
            "H2": {
                "rq": "RQ2",
                "statement": "Discrete absorbing-state G-LMDM outperforms continuous VQ-VAE diffusion/flow on rare-tile retention and parameter efficiency.",
            },
            "H3": {
                "rq": "RQ3",
                "statement": "Direct 16x11 LogicNet supervision increases raw oracle solvability and reduces repair_rate and tiles_repaired_mean.",
            },
            "H4": {
                "rq": "RQ4",
                "statement": "Paired Dungeon-9 inference preserves G-LMDM gains over external baselines while separating formal proofs from P-CBS diagnostics.",
            },
        },
        "scientific_traceability": {
            "RG1->PS1->RO1->RQ1->H1": {"blocks": ["Block_III", "Block_IV"]},
            "RG2->PS2->RO2->RQ2->H2": {"blocks": ["Block_II", "Block_IV"]},
            "RG3->PS3->RO3->RQ3->H3": {"blocks": ["Block_V", "Block_VI"]},
            "RG4->PS4->RO4->RQ4->H4": {"blocks": ["Block_I", "Block_VII"]},
        },
        "publishable_direction": {
            "primary": "G-LMDM: graph-conditioned, logic-regularized discrete masked neuro-symbolic PCG system with attributed 4-stage repair verification",
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

    research_gaps = card.get("research_gaps")
    rg_ids: set[str] = set()
    if not isinstance(research_gaps, Mapping) or not research_gaps:
        issues.append("research_gaps must be a non-empty object")
    else:
        for rg_id, rg_item in research_gaps.items():
            rg_ids.add(str(rg_id))
            if not isinstance(rg_item, Mapping):
                issues.append(f"research_gaps.{rg_id} must be an object")
                continue
            for field_name in ("title", "summary"):
                if not str(rg_item.get(field_name, "")).strip():
                    issues.append(f"research_gaps.{rg_id} missing non-empty {field_name}")
            anchors = rg_item.get("literature_anchors")
            if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)) or not anchors:
                issues.append(f"research_gaps.{rg_id} missing non-empty literature_anchors list")

    problem_statements = card.get("problem_statements")
    ps_ids: set[str] = set()
    if not isinstance(problem_statements, Mapping) or not problem_statements:
        issues.append("problem_statements must be a non-empty object")
    else:
        for ps_id, ps_item in problem_statements.items():
            ps_ids.add(str(ps_id))
            if not isinstance(ps_item, Mapping):
                issues.append(f"problem_statements.{ps_id} must be an object")
                continue
            rg_ref = ps_item.get("research_gap")
            if isinstance(rg_ref, str):
                rg_refs = [rg_ref] if rg_ref.strip() else []
            elif isinstance(rg_ref, Sequence) and not isinstance(rg_ref, (bytes, bytearray)):
                rg_refs = [str(item).strip() for item in rg_ref if str(item).strip()]
            else:
                rg_refs = []
            if not rg_refs:
                issues.append(f"problem_statements.{ps_id} missing non-empty research_gap reference")
            else:
                for ref in rg_refs:
                    if rg_ids and ref not in rg_ids:
                        issues.append(f"problem_statements.{ps_id} references unknown research_gap: {ref}")

    research_objectives = card.get("research_objectives")
    ro_ids: set[str] = set()
    if not isinstance(research_objectives, Mapping) or not research_objectives:
        issues.append("research_objectives must be a non-empty object")
    else:
        for ro_id, ro_item in research_objectives.items():
            ro_ids.add(str(ro_id))
            if not isinstance(ro_item, Mapping):
                issues.append(f"research_objectives.{ro_id} must be an object")
                continue
            ps_ref = str(ro_item.get("problem_statement", "")).strip()
            if not ps_ref:
                issues.append(f"research_objectives.{ro_id} missing problem_statement")
            elif ps_ids and ps_ref not in ps_ids:
                issues.append(f"research_objectives.{ro_id} references unknown problem_statement: {ps_ref}")

    research_subquestions = card.get("research_subquestions")
    rq_ids: set[str] = set()
    if not isinstance(research_subquestions, Mapping) or not research_subquestions:
        issues.append("research_subquestions must be a non-empty object")
    else:
        for rq_id, rq_item in research_subquestions.items():
            rq_ids.add(str(rq_id))
            if not isinstance(rq_item, Mapping):
                issues.append(f"research_subquestions.{rq_id} must be an object")
                continue
            ro_ref = str(rq_item.get("objective", "")).strip()
            if not ro_ref:
                issues.append(f"research_subquestions.{rq_id} missing objective")
            elif ro_ids and ro_ref not in ro_ids:
                issues.append(f"research_subquestions.{rq_id} references unknown objective: {ro_ref}")

    hypotheses = card.get("hypotheses")
    if not isinstance(hypotheses, Mapping) or not hypotheses:
        issues.append("hypotheses must be a non-empty object")
    else:
        for h_id, h_item in hypotheses.items():
            if not isinstance(h_item, Mapping):
                issues.append(f"hypotheses.{h_id} must be an object")
                continue
            rq_ref = str(h_item.get("rq", "")).strip()
            if not rq_ref:
                issues.append(f"hypotheses.{h_id} missing rq")
            elif rq_ids and rq_ref not in rq_ids:
                issues.append(f"hypotheses.{h_id} references unknown rq: {rq_ref}")

    scientific_traceability = card.get("scientific_traceability")
    if not isinstance(scientific_traceability, Mapping) or not scientific_traceability:
        issues.append("scientific_traceability must be a non-empty object")

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
