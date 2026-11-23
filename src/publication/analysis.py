"""Status-aware aggregation and paired inference for publication records.

The outer dungeon identity is the independent confirmatory unit. Repeated
training and sampling seeds are aggregated within each held-out dungeon before
inference so they cannot be mistaken for independent datasets.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from .records import PublicationResultRecord, STAGE_NAMES


def holm_adjust_pvalues(
    p_values: Dict[str, float],
    *,
    familywise_alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Apply Holm's step-down correction to one preregistered contrast family."""
    if not 0.0 < float(familywise_alpha) < 1.0:
        raise ValueError("familywise_alpha must lie strictly between zero and one.")
    checked: list[Tuple[str, float]] = []
    for name, value in p_values.items():
        p_value = float(value)
        if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"Invalid p-value for {name!r}: {value!r}.")
        checked.append((str(name), p_value))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_adjusted = 0.0
    output: Dict[str, Dict[str, Any]] = {}
    for rank, (name, p_value) in enumerate(ordered, start=1):
        multiplier = family_size - rank + 1
        running_adjusted = max(running_adjusted, min(1.0, multiplier * p_value))
        output[name] = {
            "raw_p_value": p_value,
            "holm_adjusted_p_value": running_adjusted,
            "holm_rank": rank,
            "family_size": family_size,
            "familywise_alpha": float(familywise_alpha),
            "reject_familywise_null": bool(running_adjusted <= float(familywise_alpha)),
        }
    return output


def aggregate_publication_records(
    records: Iterable[PublicationResultRecord],
) -> list[Dict[str, Any]]:
    """Aggregate every attempted record without silently dropping failures."""
    records = list(records)
    _validate_record_collection(records)
    grouped: dict[Tuple[str, str, str], list[PublicationResultRecord]] = defaultdict(list)
    for record in records:
        for stage in STAGE_NAMES:
            grouped[(record.model_id, record.inference_variant, stage)].append(record)

    rows: list[Dict[str, Any]] = []
    for (model_id, inference_variant, stage), group in sorted(grouped.items()):
        statuses = Counter(
            record.stages[stage].status if stage in record.stages else "not_run"
            for record in group
        )
        total = len(group)
        valid = int(statuses["valid"])
        invalid = int(statuses["invalid"])
        evaluable = valid + invalid
        indeterminate = total - evaluable
        rows.append(
            {
                "model_id": model_id,
                "inference_variant": inference_variant,
                "stage": stage,
                "attempted_records": total,
                "generation_completed": sum(
                    record.generation_status == "completed" for record in group
                ),
                "outer_fold_count": len({record.outer_fold for record in group}),
                "training_seed_count": len({record.train_seed for record in group}),
                "valid_count": valid,
                "invalid_count": invalid,
                "indeterminate_or_missing_count": indeterminate,
                "exact_valid_rate_all_attempts": valid / total if total else None,
                "exact_valid_rate_evaluable_only": (
                    valid / evaluable if evaluable else None
                ),
                "status_counts": dict(sorted(statuses.items())),
            }
        )
    return rows


def _record_pair_key(record: PublicationResultRecord) -> Tuple[Any, ...]:
    return (
        record.outer_fold,
        int(record.train_seed),
        int(record.sampling_seed),
        record.mission_graph_id,
        record.inference_variant,
    )


def _guidance_pair_key(record: PublicationResultRecord) -> Tuple[Any, ...]:
    return (
        record.outer_fold,
        int(record.train_seed),
        int(record.sampling_seed),
        record.mission_graph_id,
    )


def _validate_record_collection(records: Sequence[PublicationResultRecord]) -> None:
    """Reject reruns and incompatible evidence before computing any rate or test.

    Configurations legitimately differ between model arms. Corpus/split
    manifests can differ between outer folds, but must agree within a fold.
    All included records must come from the same code and sealed protocol.
    """
    seen: set[Tuple[Any, ...]] = set()
    provenance: dict[Tuple[str, str], str] = {}
    for record in records:
        key = (record.model_id, *_record_pair_key(record))
        if key in seen:
            raise ValueError(f"Duplicate publication record for attempt key={key!r}.")
        seen.add(key)
        for field in ("protocol_sha256", "git_commit", "dataset_manifest_sha256"):
            scope = record.outer_fold if field == "dataset_manifest_sha256" else "all"
            provenance_key = (field, scope)
            value = getattr(record, field)
            if provenance.setdefault(provenance_key, value) != value:
                raise ValueError(f"Mixed {field} in publication records for scope={scope!r}.")


def _conservative_stage_validity(
    record: PublicationResultRecord,
    stage: str,
) -> float:
    result = record.stages.get(stage)
    return float(result is not None and result.status == "valid" and result.exact_valid is True)


def _exact_sign_permutation_pvalue(deltas: np.ndarray) -> float:
    """Two-sided exact random-sign p-value over independent fold deltas."""
    observed = abs(float(np.mean(deltas)))
    if observed <= np.finfo(np.float64).eps:
        return 1.0
    extreme = 0
    total = 2 ** int(deltas.size)
    for signs in itertools.product((-1.0, 1.0), repeat=int(deltas.size)):
        permuted = abs(float(np.mean(deltas * np.asarray(signs, dtype=np.float64))))
        if permuted >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def _paired_fold_summary(
    *,
    result: Dict[str, Any],
    baseline: dict[Tuple[Any, ...], PublicationResultRecord],
    comparison: dict[Tuple[Any, ...], PublicationResultRecord],
    stage: str,
    bootstrap_samples: int,
    seed: int,
    inferential: bool = True,
) -> Dict[str, Any]:
    baseline_keys = set(baseline)
    comparison_keys = set(comparison)
    missing_baseline = sorted(comparison_keys - baseline_keys)
    missing_comparison = sorted(baseline_keys - comparison_keys)
    common_keys = sorted(baseline_keys & comparison_keys)
    result.update(
        {
            "paired_record_count": len(common_keys),
            "missing_baseline_count": len(missing_baseline),
            "missing_comparison_count": len(missing_comparison),
            "missing_baseline_keys": [list(key) for key in missing_baseline],
            "missing_comparison_keys": [list(key) for key in missing_comparison],
        }
    )
    if missing_baseline or missing_comparison:
        return {
            **result,
            "analysis_status": "incomplete_pairs",
            "reason": "Confirmatory inference requires identical matched attempt keys in both arms.",
        }

    by_fold: dict[str, list[float]] = defaultdict(list)
    for key in common_keys:
        fold = str(key[0])
        delta = _conservative_stage_validity(comparison[key], stage) - _conservative_stage_validity(
            baseline[key], stage
        )
        by_fold[fold].append(float(delta))
    fold_deltas = {fold: float(np.mean(values)) for fold, values in sorted(by_fold.items())}
    deltas = np.asarray(list(fold_deltas.values()), dtype=np.float64)
    result.update({"outer_fold_count": int(deltas.size), "fold_mean_deltas": fold_deltas})
    if not inferential:
        return {**result, "analysis_status": "descriptive_only",
                "estimand": "comparison minus baseline exact-valid rate across all matched attempts",
                "mean_fold_delta": float(np.mean(deltas)) if deltas.size else None,
                "reason": "Fold dependence has not been accounted for; no p-value or confidence interval is certified."}
    if deltas.size < 2:
        return {
            **result,
            "analysis_status": "insufficient_folds",
            "reason": "At least two held-out dungeon identities are required for inference.",
        }
    if deltas.size > 20:
        return {
            **result,
            "analysis_status": "unsupported_fold_count",
            "reason": "Exact sign enumeration is limited to 20 outer folds.",
        }

    rng = np.random.default_rng(int(seed))
    bootstrap_indices = rng.integers(
        0,
        int(deltas.size),
        size=(int(bootstrap_samples), int(deltas.size)),
    )
    bootstrap_means = np.mean(deltas[bootstrap_indices], axis=1)
    mean_delta = float(np.mean(deltas))
    sample_std = float(np.std(deltas, ddof=1))
    effect_size = None if sample_std <= 1e-12 else float(mean_delta / sample_std)
    return {
        **result,
        "analysis_status": "complete",
        "estimand": "comparison minus baseline exact-valid rate across all matched attempts",
        "mean_fold_delta": mean_delta,
        "bootstrap_ci_95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "exact_random_sign_p_value": _exact_sign_permutation_pvalue(deltas),
        "paired_effect_size_dz": effect_size,
        "effect_size_status": "undefined_zero_fold_variance" if effect_size is None else "defined",
    }


def paired_exact_validity_by_fold(
    records: Sequence[PublicationResultRecord],
    *,
    baseline_model_id: str,
    comparison_model_id: str,
    inference_variant: str = "unguided",
    stage: str = "raw",
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    inferential: bool = True,
) -> Dict[str, Any]:
    """Compare two models after aggregating matched attempts within each fold.

    Missing, timed-out, and validator-error stages score zero in the
    all-attempt validity estimand. Entirely missing paired records are not
    imputed: they block confirmatory inference and are reported explicitly.
    Each contrast selects one inference condition; guided and unguided records
    must never be pooled into the primary estimand.
    """
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage must be one of {STAGE_NAMES}; got {stage!r}.")
    if baseline_model_id == comparison_model_id:
        raise ValueError("baseline_model_id and comparison_model_id must differ.")
    if not inference_variant:
        raise ValueError("One explicit inference_variant must be selected.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")

    _validate_record_collection([
        record for record in records
        if record.model_id in {baseline_model_id, comparison_model_id}
        and record.inference_variant == inference_variant
    ])

    indexes: dict[str, dict[Tuple[Any, ...], PublicationResultRecord]] = {}
    for model_id in (baseline_model_id, comparison_model_id):
        index: dict[Tuple[Any, ...], PublicationResultRecord] = {}
        for record in records:
            if record.model_id != model_id or record.inference_variant != inference_variant:
                continue
            key = _record_pair_key(record)
            if key in index:
                raise ValueError(f"Duplicate publication record for model={model_id!r}, key={key!r}.")
            index[key] = record
        indexes[model_id] = index

    result: Dict[str, Any] = {
        "analysis": "paired_exact_validity_by_outer_fold",
        "inference_variant": inference_variant,
        "stage": stage,
        "baseline_model_id": baseline_model_id,
        "comparison_model_id": comparison_model_id,
    }
    return _paired_fold_summary(
        result=result,
        baseline=indexes[baseline_model_id],
        comparison=indexes[comparison_model_id],
        stage=stage,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        inferential=inferential,
    )


def paired_guidance_validity_by_fold(
    records: Sequence[PublicationResultRecord],
    *,
    model_id: str,
    baseline_inference_variant: str = "unguided",
    comparison_inference_variant: str = "logic_guided",
    stage: str = "raw",
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare paired inference conditions within one LogicNet-capable model."""
    if stage not in STAGE_NAMES:
        raise ValueError(f"stage must be one of {STAGE_NAMES}; got {stage!r}.")
    if baseline_inference_variant == comparison_inference_variant:
        raise ValueError("Guidance inference variants must differ.")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive.")

    _validate_record_collection([
        record for record in records
        if record.model_id == model_id
        and record.inference_variant in {baseline_inference_variant, comparison_inference_variant}
    ])

    indexes: dict[str, dict[Tuple[Any, ...], PublicationResultRecord]] = {}
    for inference_variant in (baseline_inference_variant, comparison_inference_variant):
        index: dict[Tuple[Any, ...], PublicationResultRecord] = {}
        for record in records:
            if record.model_id != model_id or record.inference_variant != inference_variant:
                continue
            key = _guidance_pair_key(record)
            if key in index:
                raise ValueError(
                    f"Duplicate publication record for model={model_id!r}, "
                    f"inference_variant={inference_variant!r}, key={key!r}."
                )
            index[key] = record
        indexes[inference_variant] = index

    return _paired_fold_summary(
        result={
            "analysis": "paired_guidance_validity_by_outer_fold",
            "stage": stage,
            "model_id": model_id,
            "baseline_inference_variant": baseline_inference_variant,
            "comparison_inference_variant": comparison_inference_variant,
        },
        baseline=indexes[baseline_inference_variant],
        comparison=indexes[comparison_inference_variant],
        stage=stage,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )


__all__ = [
    "aggregate_publication_records",
    "holm_adjust_pvalues",
    "paired_exact_validity_by_fold",
    "paired_guidance_validity_by_fold",
]
