"""Apply the sealed primary contrast family only to a complete attempt ledger."""

from itertools import product, combinations

from .analysis import holm_adjust_pvalues, paired_exact_validity_by_fold
from .protocol import seal_protocol_manifest


def analyze_primary_ledger(ledger, protocol, *, protocol_sha256, bootstrap_samples=10_000, seed=0):
    """Keep stage selection, completeness, and multiplicity tied to the protocol.

    Callers must retain the ledger's previously sealed digest. Reading whatever
    records happen to exist is insufficient: both arms can omit the same seed.
    """
    if (protocol.get("protocol_sha256") != protocol_sha256
            or seal_protocol_manifest(protocol)["protocol_sha256"] != protocol_sha256):
        raise ValueError("Primary analysis requires the unchanged sealed protocol.")
    design = protocol["confirmatory_analysis"]
    stages = {"raw_stage_exact_validity_over_all_requested_attempts": "raw",
              "compiled_stage_exact_validity_over_all_requested_attempts": "compiled"}
    endpoint = design["primary_endpoint"]
    if endpoint not in stages:
        raise ValueError(f"Unsupported primary endpoint: {endpoint}")
    stage = stages[endpoint]
    if design.get("primary_stage", stage) != stage:
        raise ValueError("Primary stage conflicts with the sealed endpoint.")
    all_records = [record for record in ledger.complete_records() if record.inference_variant == "unguided"]
    control_models = protocol.get("evaluation_design", {}).get("independent_controls", [])
    evaluation = protocol.get("evaluation_design")
    if evaluation is not None:
        models = [variant["model_id"] for variant in protocol["model_variants"]] + control_models
        requested = {
            (fold, train_seed, sampling_seed, mission, model)
            for fold, missions in evaluation["missions"].items()
            for train_seed, sampling_seed, mission, model in product(
                protocol["training_seeds"], evaluation["sampling_seeds"], missions, models)
        }
        observed = {(record.outer_fold, record.train_seed, record.sampling_seed,
                     record.mission_graph_id, record.model_id) for record in all_records}
        if observed != requested or len(all_records) != len(requested):
            raise ValueError("Attempt ledger does not cover the complete pre-training evaluation matrix.")
    controls = [record for record in all_records if record.model_id in control_models]
    records = [record for record in all_records if record.model_id not in control_models]
    if any(record.protocol_sha256 != protocol_sha256 for record in controls):
        raise ValueError("Control attempts belong to a different protocol.")
    expected_controls = set(product((fold["fold_id"] for fold in protocol["folds"]),
                                    protocol["training_seeds"], control_models))
    if {(record.outer_fold, record.train_seed, record.model_id) for record in controls} != expected_controls:
        raise ValueError("Independent control attempts do not cover the sealed design.")
    if any(record.protocol_sha256 != protocol_sha256 for record in records):
        raise ValueError("Attempt records belong to a different protocol.")
    expected = set(product((fold["fold_id"] for fold in protocol["folds"]), protocol["training_seeds"],
                           (variant["model_id"] for variant in protocol["model_variants"])))
    actual = {(record.outer_fold, record.train_seed, record.model_id) for record in records}
    if actual != expected:
        raise ValueError("Primary attempts do not cover every declared fold, training seed, and model arm.")
    declared_jobs = protocol.get("jobs", {}).get("diffusion")
    jobs = {(job["fold_id"], job["training_seed"], job["model_id"]): job
            for job in (declared_jobs or [])}
    if declared_jobs is not None and (set(jobs) != expected or len(jobs) != len(declared_jobs)):
        raise ValueError("Sealed diffusion job matrix must cover every declared arm exactly once.")
    if jobs and any(record.config_sha256 != jobs[(record.outer_fold, record.train_seed, record.model_id)]["config_sha256"]
                    for record in records):
        raise ValueError("Attempt training configs differ from the sealed job matrix.")
    contrasts = {}
    overlaps = [
        {"fold_a": a["fold_id"], "fold_b": b["fold_id"],
         "shared_training_dungeons": sorted(set(a["train_dungeons"]) & set(b["train_dungeons"]))}
        for a, b in combinations(protocol["folds"], 2)
        if set(a["train_dungeons"]) & set(b["train_dungeons"])
    ]
    inferential = not overlaps
    for contrast in design["primary_contrast_family"]:
        baseline, comparison = contrast["baseline"], contrast["comparison"]
        name = f"{comparison}_minus_{baseline}"
        if name in contrasts:
            raise ValueError("Duplicate primary contrast.")
        contrasts[name] = paired_exact_validity_by_fold(
            records, baseline_model_id=baseline, comparison_model_id=comparison,
            inference_variant="unguided", stage=stage, bootstrap_samples=bootstrap_samples, seed=seed,
            inferential=inferential,
        )
    complete = bool(contrasts) and all(item["analysis_status"] == "complete" for item in contrasts.values())
    adjusted = holm_adjust_pvalues(
        {name: item["exact_random_sign_p_value"] for name, item in contrasts.items()},
        familywise_alpha=design["familywise_alpha"],
    ) if complete else {}
    status = "complete" if complete else "incomplete_primary_family"
    if overlaps and contrasts and all(item["analysis_status"] == "descriptive_only" for item in contrasts.values()):
        status = "descriptive_only_dependent_folds"
    return {"analysis_status": status,
            "primary_endpoint": endpoint, "stage": stage, "contrasts": contrasts,
            "multiplicity": adjusted, "attempt_count": len(records),
            "training_overlap": overlaps, "confirmatory_inference_permitted": complete,
            "independent_control_attempt_count": len(controls),
            "independent_control_comparisons": {
                model: paired_exact_validity_by_fold(all_records, baseline_model_id=model,
                    comparison_model_id="g_lg_full_logic", inference_variant="unguided", stage=stage,
                    inferential=False, bootstrap_samples=bootstrap_samples, seed=seed)
                for model in control_models
            }}
