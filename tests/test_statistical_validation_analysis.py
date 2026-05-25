from scripts.statistical_validation_and_analysis import (
    PerSeedResult,
    compute_qd_score_decomposition,
    compute_statistical_significance,
    run_random_baseline,
)
from scripts.paired_seed_ablation import (
    PerSeedResult as PairedSeedResult,
    compute_qd_decomposition,
    compute_statistical_significance as compute_paired_seed_significance,
)


def _result(seed: int, config: str, fitness: float) -> PerSeedResult:
    return PerSeedResult(
        seed=seed,
        config=config,
        fitness=fitness,
        coverage=0.0,
        qd_score=0.0,
        num_elites=0,
        mean_elite_fitness=0.0,
        feature_diversity=0.0,
        generation_time_sec=0.0,
        evaluations_used=0,
    )


def _paired_result(seed: int, config: str, fitness: float) -> PairedSeedResult:
    return PairedSeedResult(
        seed=seed,
        config=config,
        fitness=fitness,
        coverage=0.0,
        qd_score=0.0,
        num_elites=0,
        mean_elite_fitness=0.0,
        feature_diversity=0.0,
        generation_time_sec=0.0,
        evaluations_used=0,
    )


def test_statistical_significance_requires_two_paired_seeds():
    result = compute_statistical_significance(
        {7: _result(7, "n64", 1.0)},
        {7: _result(7, "n96", 1.1)},
    )

    assert result["insufficient_data"] is True
    assert result["statistical_significance"] == "INSUFFICIENT_DATA"
    assert result["paired_seeds"] == 1
    assert result["per_seed_deltas"] == {7: 0.10000000000000009}


def test_statistical_significance_uses_paired_seed_deltas():
    result = compute_statistical_significance(
        {
            1: _result(1, "n64", 1.0),
            2: _result(2, "n64", 1.1),
            3: _result(3, "n64", 1.3),
        },
        {
            1: _result(1, "n96", 1.2),
            2: _result(2, "n96", 1.25),
            3: _result(3, "n96", 1.7),
        },
    )

    assert result["insufficient_data"] is False
    assert result["test_type"] == "paired_samples_t_test"
    assert result["paired_seeds"] == 3
    assert result["seeds_list"] == [1, 2, 3]
    assert set(result["per_seed_deltas"]) == {1, 2, 3}
    assert result["ci_lower"] < result["mean_difference"] < result["ci_upper"]
    assert result["cohens_d"] is not None


def test_statistical_significance_handles_zero_variance_deltas():
    result = compute_statistical_significance(
        {
            1: _result(1, "n64", 1.0),
            2: _result(2, "n64", 1.5),
            3: _result(3, "n64", 2.0),
        },
        {
            1: _result(1, "n96", 1.25),
            2: _result(2, "n96", 1.75),
            3: _result(3, "n96", 2.25),
        },
    )

    assert result["insufficient_data"] is False
    assert result["p_value"] == 0.0
    assert result["cohens_d"] is None
    assert result["ci_lower"] == result["ci_upper"] == result["mean_difference"]
    assert result["effect_size_interpretation"].startswith("undefined")


def test_paired_seed_ablation_significance_handles_insufficient_data():
    result = compute_paired_seed_significance({}, {})

    assert result["insufficient_data"] is True
    assert result["statistical_significance"] == "INSUFFICIENT_DATA"
    assert result["p_value"] == 1.0
    assert result["cohens_d"] is None


def test_paired_seed_ablation_significance_handles_zero_variance_deltas():
    result = compute_paired_seed_significance(
        {
            1: _paired_result(1, "n64", 1.0),
            2: _paired_result(2, "n64", 2.0),
        },
        {
            1: _paired_result(1, "n96", 1.5),
            2: _paired_result(2, "n96", 2.5),
        },
    )

    assert result["insufficient_data"] is False
    assert result["p_value"] == 0.0
    assert result["t_statistic"] is None
    assert result["cohens_d"] is None


def test_qd_decomposition_correlations_stay_finite_for_single_seed():
    result = compute_qd_score_decomposition(
        {"n64": {1: _result(1, "n64", 1.0)}}
    )

    assert result["n64"]["correlation_elite_count_qd_score"] == 0.0
    assert result["n64"]["correlation_diversity_qd_score"] == 0.0


def test_statistical_random_baseline_returns_per_seed_results(monkeypatch, tmp_path):
    class _Baseline:
        mean_fitness_random = 0.25
        mean_coverage_random = 0.5
        mean_qd_score_random = 1.5
        num_elites_random = 3
        feature_diversity_random = 0.125

    calls = []

    def _fake_random_baseline(*, num_samples, archive_cells, seed, output_dir):
        calls.append((num_samples, archive_cells, seed, output_dir))
        return _Baseline()

    monkeypatch.setattr(
        "scripts.statistical_validation_and_analysis.run_random_baseline_with_archive",
        _fake_random_baseline,
    )

    results = run_random_baseline(
        num_samples=7,
        archive_cells=16,
        output_dir=tmp_path,
        seeds=[1, 2],
    )

    assert sorted(results) == [1, 2]
    assert results[1].config == "random"
    assert results[1].fitness == 0.25
    assert results[1].coverage == 0.5
    assert results[1].qd_score == 1.5
    assert results[1].num_elites == 3
    assert results[1].feature_diversity == 0.125
    assert len(calls) == 2


def test_paired_qd_decomposition_correlations_stay_finite_for_constant_vectors():
    result = compute_qd_decomposition(
        {
            "n64": {
                1: _paired_result(1, "n64", 1.0),
                2: _paired_result(2, "n64", 1.1),
            }
        }
    )

    assert result["n64"]["correlation_elite_count_qd_score"] == 0.0
    assert result["n64"]["correlation_diversity_qd_score"] == 0.0
    assert result["n64"]["correlation_time_qd_score"] == 0.0
