"""
Paired-Seed Ablation Study
===========================

Runs MAP-Elites with both configs (n64, n96) on the same seeds individually,
capturing per-seed results for proper statistical comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


@dataclass
class PerSeedResult:
    """Per-seed result for statistical analysis."""
    seed: int
    config: str  # "n64" or "n96"
    fitness: float
    coverage: float
    qd_score: float
    num_elites: int
    mean_elite_fitness: float
    feature_diversity: float
    generation_time_sec: float
    evaluations_used: int


def run_single_seed_benchmark(
    seed: int,
    config_name: str,
    eval_budget: int,
    archive_cells: int,
    population_hint: int,
    min_rooms: int,
    max_rooms: int,
    output_base_dir: Path,
) -> Optional[PerSeedResult]:
    """
    Run matched-budget benchmark for a single seed.
    
    Returns the summary metrics extracted from the output JSON.
    """
    
    output_dir = output_base_dir / f"{config_name}_seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running {config_name} with seed {seed}...")
    
    cmd = [
        sys.executable,
        "scripts/run_matched_budget_topology_benchmark.py",
        "--output", str(output_dir),
        "--methods", "MAP_ELITES",
        "--num-samples", "1",  # One sample = one seed
        "--seed", str(seed),
        "--eval-budget", str(eval_budget),
        "--population-hint", str(population_hint),
        "--min-rooms", str(min_rooms),
        "--max-rooms", str(max_rooms),
        "--archive-cells", str(archive_cells),
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per seed
        )
        
        if result.returncode != 0:
            logger.error(f"Benchmark failed for seed {seed}:")
            logger.error(f"  stdout: {result.stdout[-500:]}")
            logger.error(f"  stderr: {result.stderr[-500:]}")
            return None
        
        # Load the output JSON
        report_path = output_dir / "matched_budget_report.json"
        if not report_path.exists():
            logger.error(f"Report not found: {report_path}")
            return None
        
        with open(report_path) as f:
            data = json.load(f)
        
        summary = data["summary"][0]
        
        return PerSeedResult(
            seed=seed,
            config=config_name,
            fitness=summary["fitness"],
            coverage=summary.get("map_elites_coverage", 0),
            qd_score=summary.get("map_elites_qd_score", 0),
            num_elites=int(summary.get("map_elites_num_elites", 0)),
            mean_elite_fitness=summary.get("map_elites_mean_fitness", 0),
            feature_diversity=summary.get("map_elites_feature_diversity", 0),
            generation_time_sec=summary["generation_time_sec"],
            evaluations_used=int(summary["evaluations_used"]),
        )
        
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout running seed {seed}")
        return None
    except Exception as e:
        logger.error(f"Exception running seed {seed}: {e}")
        return None


def run_paired_seed_ablation(
    seeds: List[int],
    output_base_dir: Path,
    num_seeds: Optional[int] = None,
) -> Tuple[Dict[int, PerSeedResult], Dict[int, PerSeedResult]]:
    """
    Run paired-seed ablation on both configs.
    
    Args:
        seeds: list of seed integers
        output_base_dir: base output directory
        num_seeds: if set, only run first N seeds for faster iteration
        
    Returns:
        (results_n64, results_n96)
    """
    
    if num_seeds:
        seeds = seeds[:num_seeds]
    
    results_n64 = {}
    results_n96 = {}
    
    logger.info(f"Running paired-seed ablation on {len(seeds)} seeds...")
    logger.info(f"Total estimated time: ~{len(seeds) * 3} seconds (3 sec per seed)")
    
    for seed in seeds:
        # Run n64 config
        result_n64 = run_single_seed_benchmark(
            seed=seed,
            config_name="n64",
            eval_budget=512,
            archive_cells=128,
            population_hint=24,
            min_rooms=18,
            max_rooms=33,
            output_base_dir=output_base_dir,
        )
        if result_n64:
            results_n64[seed] = result_n64
        
        # Run n96 config
        result_n96 = run_single_seed_benchmark(
            seed=seed,
            config_name="n96",
            eval_budget=1024,
            archive_cells=256,
            population_hint=32,
            min_rooms=18,
            max_rooms=33,
            output_base_dir=output_base_dir,
        )
        if result_n96:
            results_n96[seed] = result_n96
    
    logger.info(f"Completed: {len(results_n64)} n64 seeds, {len(results_n96)} n96 seeds")
    
    return results_n64, results_n96


def compute_statistical_significance(
    results_n64: Dict[int, PerSeedResult],
    results_n96: Dict[int, PerSeedResult],
) -> Dict[str, Any]:
    """
    Compute a paired-samples t-test and Cohen's d for paired seeds.
    """
    
    # Extract fitness for paired seeds
    paired_seeds = sorted(set(results_n64.keys()) & set(results_n96.keys()))
    
    if len(paired_seeds) < 2:
        logger.warning(f"Insufficient paired seeds: {len(paired_seeds)}")
        fitness_n64 = [float(results_n64[s].fitness) for s in paired_seeds]
        fitness_n96 = [float(results_n96[s].fitness) for s in paired_seeds]
        return {
            "test_type": "paired_samples_t_test",
            "paired_seeds": len(paired_seeds),
            "seeds_list": paired_seeds,
            "insufficient_data": True,
            "statistical_significance": "INSUFFICIENT_DATA",
            "reason": "At least two matched seeds are required for a paired-samples t-test.",
            "mean_fitness_n64": float(np.mean(fitness_n64)) if fitness_n64 else 0.0,
            "mean_fitness_n96": float(np.mean(fitness_n96)) if fitness_n96 else 0.0,
            "mean_difference": (
                float(np.mean([results_n96[s].fitness - results_n64[s].fitness for s in paired_seeds]))
                if paired_seeds
                else 0.0
            ),
            "percent_improvement": 0.0,
            "t_statistic": None,
            "p_value": 1.0,
            "cohens_d": None,
            "cohens_d_interpretation": "undefined (insufficient paired seeds)",
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "per_seed_deltas": {
                s: float(results_n96[s].fitness - results_n64[s].fitness)
                for s in paired_seeds
            },
        }
    
    fitness_n64 = np.array([results_n64[s].fitness for s in paired_seeds], dtype=float)
    fitness_n96 = np.array([results_n96[s].fitness for s in paired_seeds], dtype=float)
    
    deltas = fitness_n96 - fitness_n64
    
    # Cohen's d (paired samples: d = mean(delta) / std(delta))
    mean_diff = float(np.mean(deltas))
    std_diff = float(np.std(deltas, ddof=1))
    n = len(paired_seeds)
    eps = np.finfo(float).eps
    if std_diff <= eps:
        t_stat = None
        p_value = 1.0 if abs(mean_diff) <= eps else 0.0
        cohens_d = None
        ci_lower = ci_upper = mean_diff
        effect_size_interpretation = (
            "undefined (zero paired-difference variance)"
            if abs(mean_diff) > eps
            else "negligible"
        )
    else:
        t_stat, p_value = stats.ttest_rel(fitness_n96, fitness_n64)
        cohens_d = mean_diff / std_diff

        # 95% CI for mean difference
        se = std_diff / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n - 1)  # two-tailed
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se
        effect_size_interpretation = _interpret_cohens_d(cohens_d)
    
    # Percent improvement
    pct_improvement = (mean_diff / np.mean(fitness_n64) * 100) if np.mean(fitness_n64) > 0 else 0
    
    return {
        "test_type": "paired_samples_t_test",
        "paired_seeds": len(paired_seeds),
        "seeds_list": sorted(paired_seeds),
        "insufficient_data": False,
        "mean_fitness_n64": float(np.mean(fitness_n64)),
        "std_fitness_n64": float(np.std(fitness_n64, ddof=1)),
        "mean_fitness_n96": float(np.mean(fitness_n96)),
        "std_fitness_n96": float(np.std(fitness_n96, ddof=1)),
        "mean_difference": float(mean_diff),
        "std_difference": float(std_diff),
        "percent_improvement": float(pct_improvement),
        "t_statistic": None if t_stat is None else float(t_stat),
        "p_value": float(p_value),
        "cohens_d": None if cohens_d is None else float(cohens_d),
        "cohens_d_interpretation": effect_size_interpretation,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "effect_size_interpretation": effect_size_interpretation,
        "statistical_significance": "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)",
        "practical_significance": (
            "UNDEFINED (zero paired-difference variance)"
            if cohens_d is None
            else ("YES (d > 0.5)" if abs(cohens_d) > 0.5 else "NO (d <= 0.5)")
        ),
        "per_seed_deltas": {s: float(d) for s, d in zip(paired_seeds, deltas)},
        "per_seed_fitnesses_n64": {s: float(fitness_n64[i]) for i, s in enumerate(paired_seeds)},
        "per_seed_fitnesses_n96": {s: float(fitness_n96[i]) for i, s in enumerate(paired_seeds)},
    }


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def _safe_pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite Pearson correlation, or 0.0 when it is undefined."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 2 or y_arr.size < 2 or x_arr.size != y_arr.size:
        return 0.0
    if np.std(x_arr) <= np.finfo(float).eps or np.std(y_arr) <= np.finfo(float).eps:
        return 0.0
    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
    return corr if np.isfinite(corr) else 0.0


def compute_coverage_convergence(
    results_by_config: Dict[str, Dict[int, PerSeedResult]],
) -> Dict[str, Any]:
    """Analyze how coverage scales with evaluation budget."""
    
    convergence = {}
    
    for config_name, results in results_by_config.items():
        if not results:
            continue
        
        coverages = np.array([r.coverage for r in results.values()])
        evals_used = np.array([r.evaluations_used for r in results.values()])
        
        convergence[config_name] = {
            "mean_coverage": float(np.mean(coverages)),
            "std_coverage": float(np.std(coverages, ddof=1)) if len(coverages) > 1 else 0,
            "min_coverage": float(np.min(coverages)),
            "max_coverage": float(np.max(coverages)),
            "mean_evals_used": float(np.mean(evals_used)),
            "coverage_per_eval": float(np.mean(coverages) / np.mean(evals_used)) if np.mean(evals_used) > 0 else 0,
            "coverage_per_elite": float(np.mean(coverages) / np.mean([r.num_elites for r in results.values() if r.num_elites > 0])),
        }
    
    return convergence


def compute_qd_decomposition(
    results_by_config: Dict[str, Dict[int, PerSeedResult]],
) -> Dict[str, Any]:
    """Decompose QD-Score metrics."""
    
    decomp = {}
    
    for config_name, results in results_by_config.items():
        if not results:
            continue
        
        elite_counts = np.array([r.num_elites for r in results.values()])
        qd_scores = np.array([r.qd_score for r in results.values()])
        mean_fitnesses = np.array([r.mean_elite_fitness for r in results.values()])
        feature_divs = np.array([r.feature_diversity for r in results.values()])
        gen_times = np.array([r.generation_time_sec for r in results.values()])
        
        # Correlations are undefined for one seed or constant vectors.
        corr_elite_qd = _safe_pearson_corr(elite_counts, qd_scores)
        corr_div_qd = _safe_pearson_corr(feature_divs, qd_scores)
        corr_time_qd = _safe_pearson_corr(gen_times, qd_scores)
        
        decomp[config_name] = {
            "mean_elite_count": float(np.mean(elite_counts)),
            "std_elite_count": float(np.std(elite_counts, ddof=1)) if len(elite_counts) > 1 else 0,
            "mean_qd_score": float(np.mean(qd_scores)),
            "std_qd_score": float(np.std(qd_scores, ddof=1)) if len(qd_scores) > 1 else 0,
            "mean_elite_fitness": float(np.mean(mean_fitnesses)),
            "std_elite_fitness": float(np.std(mean_fitnesses, ddof=1)) if len(mean_fitnesses) > 1 else 0,
            "mean_feature_diversity": float(np.mean(feature_divs)),
            "std_feature_diversity": float(np.std(feature_divs, ddof=1)) if len(feature_divs) > 1 else 0,
            "mean_generation_time": float(np.mean(gen_times)),
            "correlation_elite_count_qd_score": float(corr_elite_qd),
            "correlation_diversity_qd_score": float(corr_div_qd),
            "correlation_time_qd_score": float(corr_time_qd),
            "interpretation": {
                "elite_qd_correlation": "Higher elites correlate with higher QD-score (expected)" if corr_elite_qd > 0.5 else "Weak correlation",
                "diversity_qd_correlation": "Higher diversity correlates with higher QD-score (exploration)" if corr_div_qd > 0.5 else "Weak correlation",
                "time_qd_correlation": "Generation time scales with QD-score" if corr_time_qd > 0.5 else "Weak time correlation",
            },
        }
    
    return decomp


def generate_report(
    output_dir: Path,
    results_n64: Dict[int, PerSeedResult],
    results_n96: Dict[int, PerSeedResult],
    statistical_test: Dict[str, Any],
    coverage_convergence: Dict[str, Any],
    qd_decomposition: Dict[str, Any],
) -> Path:
    """Generate final statistical report."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohens_d = statistical_test.get("cohens_d")
    effect_size_summary = (
        "Cohen's d = undefined"
        if cohens_d is None
        else f"Cohen's d = {cohens_d:.3f}"
    )
    
    report = {
        "title": "Paired-Seed Ablation Statistical Validation Report",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "num_paired_seeds": statistical_test.get("paired_seeds", 0),
            "fitness_improvement": f"{statistical_test.get('percent_improvement', 0):.2f}%",
            "statistical_significance": statistical_test.get("statistical_significance", "UNKNOWN"),
            "effect_size": effect_size_summary,
            "practical_significance": statistical_test.get("practical_significance", "UNKNOWN"),
        },
        "sections": {
            "paired_seeds_statistical_test": {
                "description": "Paired-samples t-test comparing n64 vs n96 on identical seeds",
                "results": statistical_test,
            },
            "coverage_convergence": {
                "description": "How coverage scales with evaluation budget",
                "results": coverage_convergence,
            },
            "qd_score_decomposition": {
                "description": "Elite count, fitness distribution, diversity, and time metrics",
                "results": qd_decomposition,
            },
            "per_config_summary": {
                "n64": {
                    "num_seeds_completed": len(results_n64),
                    "seeds": sorted(results_n64.keys()),
                },
                "n96": {
                    "num_seeds_completed": len(results_n96),
                    "seeds": sorted(results_n96.keys()),
                },
            },
        },
    }
    
    report_path = output_dir / "paired_seed_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved: {report_path}")
    
    # Also save CSV for easy spreadsheet import
    csv_data = []
    all_seeds = sorted(set(results_n64.keys()) | set(results_n96.keys()))
    
    for seed in all_seeds:
        row = {"seed": seed}
        if seed in results_n64:
            r = results_n64[seed]
            row.update({
                "fitness_n64": r.fitness,
                "coverage_n64": r.coverage,
                "qd_score_n64": r.qd_score,
                "num_elites_n64": r.num_elites,
            })
        if seed in results_n96:
            r = results_n96[seed]
            row.update({
                "fitness_n96": r.fitness,
                "coverage_n96": r.coverage,
                "qd_score_n96": r.qd_score,
                "num_elites_n96": r.num_elites,
            })
        if seed in results_n64 and seed in results_n96:
            row["fitness_delta"] = results_n96[seed].fitness - results_n64[seed].fitness
            row["fitness_pct_change"] = (row["fitness_delta"] / results_n64[seed].fitness * 100) if results_n64[seed].fitness > 0 else 0
        
        csv_data.append(row)
    
    csv_path = output_dir / "paired_seed_comparison.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Run paired-seed ablation study"
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=64,
        help="Number of seeds to run (default: 64 for full ablation)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="First seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/paired_seed_ablation"),
        help="Output directory",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run only first 3 seeds for quick testing",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    # Generate seed list
    num_seeds = 3 if args.quick_test else args.num_seeds
    seeds = list(range(args.start_seed, args.start_seed + num_seeds))
    
    logger.info("="*80)
    logger.info("Paired-Seed Ablation Study")
    logger.info("="*80)
    logger.info(f"Configuration: {num_seeds} seeds starting from {args.start_seed}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")
    
    # Run paired ablation
    logger.info("[1/4] Running paired-seed ablation...")
    results_n64, results_n96 = run_paired_seed_ablation(
        seeds=seeds,
        output_base_dir=args.output_dir / "per_seed_runs",
        num_seeds=num_seeds if args.quick_test else None,
    )
    
    # Compute statistics
    results_by_config = {"n64": results_n64, "n96": results_n96}
    
    logger.info("\n[2/4] Computing statistical significance...")
    statistical_test = compute_statistical_significance(results_n64, results_n96)
    
    logger.info("\n[3/4] Computing convergence and QD metrics...")
    coverage_convergence = compute_coverage_convergence(results_by_config)
    qd_decomposition = compute_qd_decomposition(results_by_config)
    
    # Generate report
    logger.info("\n[4/4] Generating report...")
    report_path = generate_report(
        args.output_dir,
        results_n64,
        results_n96,
        statistical_test,
        coverage_convergence,
        qd_decomposition,
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("PAIRED-SEED ABLATION RESULTS SUMMARY")
    logger.info("="*80)
    
    if statistical_test:
        logger.info(f"\n✓ Statistical Significance:")
        logger.info(f"    Paired seeds: {statistical_test.get('paired_seeds', 'N/A')}")
        logger.info(f"    Mean n64 fitness: {statistical_test.get('mean_fitness_n64', 0):.4f}")
        logger.info(f"    Mean n96 fitness: {statistical_test.get('mean_fitness_n96', 0):.4f}")
        logger.info(f"    Improvement: {statistical_test.get('mean_difference', 0):.4f} ({statistical_test.get('percent_improvement', 0):.2f}%)")
        t_stat = statistical_test.get("t_statistic")
        logger.info(f"    t-statistic: {'undefined' if t_stat is None else f'{t_stat:.4f}'}")
        logger.info(f"    p-value: {statistical_test.get('p_value', 0):.6f}")
        logger.info(f"    Result: {statistical_test.get('statistical_significance', 'UNKNOWN')}")
        cohens_d = statistical_test.get("cohens_d")
        if cohens_d is None:
            logger.info(f"    Cohen's d: undefined ({statistical_test.get('cohens_d_interpretation', 'N/A')})")
        else:
            logger.info(f"    Cohen's d: {cohens_d:.3f} ({statistical_test.get('cohens_d_interpretation', 'N/A')})")
        logger.info(f"    95% CI: [{statistical_test.get('ci_lower', 0):.4f}, {statistical_test.get('ci_upper', 0):.4f}]")
    
    logger.info(f"\n✓ Coverage Convergence:")
    for config, metrics in coverage_convergence.items():
        logger.info(f"    {config}: coverage={metrics['mean_coverage']:.4f} ± {metrics['std_coverage']:.4f}")
    
    logger.info(f"\n✓ Report saved to: {report_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
