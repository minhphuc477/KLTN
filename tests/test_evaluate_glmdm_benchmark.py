import numpy as np

from scripts.evaluate_glmdm_benchmark import (
    aggregate_fold_reports,
    compute_interior_distribution,
    compute_novelty_metrics,
    evaluate_non_neural_baseline,
    expand_checkpoint_dir_spec,
    jensen_shannon_divergence,
    run_falsification_checks,
    total_variation_distance,
)
from src.core.definitions import SEMANTIC_PALETTE as P


def _make_room(interior_tile: int, extra_block: bool = False) -> np.ndarray:
    grid = np.full((16, 11), int(P["WALL"]), dtype=np.int32)
    grid[2:-2, 2:-2] = int(interior_tile)
    if extra_block:
        grid[5, 5] = int(P["BLOCK"])
    return grid


def test_interior_distribution_js_and_tv_identities():
    room_a = _make_room(int(P["FLOOR"]), extra_block=True)
    room_b = _make_room(int(P["ELEMENT"]), extra_block=False)
    dist_a = compute_interior_distribution([room_a])
    dist_b = compute_interior_distribution([room_b])

    assert dist_a["BLOCK"] > 0.0
    assert dist_a["WALL"] == 0.0
    assert jensen_shannon_divergence(dist_a, dist_a) == 0.0
    assert total_variation_distance(dist_a, dist_a) == 0.0
    assert 0.0 < jensen_shannon_divergence(dist_a, dist_b) <= 1.0
    assert 0.0 < total_variation_distance(dist_a, dist_b) <= 1.0


def test_novelty_metrics_distinguish_exact_memorization_and_novel_rooms():
    train = [_make_room(int(P["FLOOR"]), extra_block=False)]
    mem_room = _make_room(int(P["FLOOR"]), extra_block=False)
    novel_room = _make_room(int(P["FLOOR"]), extra_block=True)

    mem_stats = compute_novelty_metrics([mem_room], train)
    assert mem_stats["train_exact_match_rate"] == 1.0
    assert mem_stats["mean_min_hamming_to_train"] == 0.0

    nov_stats = compute_novelty_metrics([novel_room], train)
    assert nov_stats["train_exact_match_rate"] == 0.0
    assert nov_stats["train_v4_match_rate"] == 0.0
    assert nov_stats["mean_min_hamming_to_train"] > 0.0


def test_aggregate_fold_reports_computes_mean_and_std():
    f0 = {"arms": [{"arm": "G-LG", "repair_rate": 0.1, "js_div_raw_vs_vglc": 0.04}]}
    f1 = {"arms": [{"arm": "G-LG", "repair_rate": 0.3, "js_div_raw_vs_vglc": 0.06}]}

    agg = aggregate_fold_reports([f0, f1])
    assert agg["num_folds"] == 2
    assert np.isclose(agg["arms"]["G-LG"]["repair_rate"]["mean"], 0.2)
    assert np.isclose(agg["arms"]["G-LG"]["repair_rate"]["std"], np.sqrt(0.02))
    assert agg["arms"]["G-LG"]["js_div_raw_vs_vglc"]["n"] == 2


def test_expand_checkpoint_dir_spec_finds_available_checkpoints(tmp_path):
    (tmp_path / "masked_room_best.pth").write_bytes(b"a")
    (tmp_path / "masked_room_best_rare_recall.pth").write_bytes(b"b")
    specs = expand_checkpoint_dir_spec(f"G-LG=configs/glmdm_canonical.yaml={tmp_path}")
    assert [name for name, _, _ in specs] == ["G-LG:best", "G-LG:best_rare_recall"]


def test_run_falsification_checks_flags_collapse_memorization_and_pairs_orders():
    vglc_dist = {"FLOOR": 0.81, "BLOCK": 0.086, "ELEMENT": 0.06, "ELEMENT_FLOOR": 0.02, "ENEMY": 0.02, "WALL": 0.0, "OTHER": 0.004}
    arm_results = [
        {
            "arm": "baseline:empty_floor",
            "is_baseline": True,
            "raw_rooms_with_block_rate": 0.0,
            "raw_interior_distribution": {"BLOCK": 0.0},
            "js_div_raw_vs_vglc": 0.14,
        },
        {
            "arm": "G-LG@random",
            "unmask_order": "random",
            "checkpoint_path": "ckpt_a.pth",
            "raw_rooms_with_block_rate": 0.45,
            "raw_interior_distribution": {"BLOCK": 0.08},
            "js_div_raw_vs_vglc": 0.03,
            "train_v4_match_rate": 0.10,
            "mean_min_hamming_to_train": 0.20,
            "repair_rate": 0.20,
            "repair_distribution_shift_js": 0.01,
        },
        {
            "arm": "G-LG@confidence",
            "unmask_order": "confidence",
            "checkpoint_path": "ckpt_a.pth",
            "raw_rooms_with_block_rate": 0.05,
            "raw_interior_distribution": {"BLOCK": 0.005},
            "js_div_raw_vs_vglc": 0.16,
            "train_v4_match_rate": 0.90,
            "mean_min_hamming_to_train": 0.01,
            "repair_rate": 0.65,
            "repair_distribution_shift_js": 0.08,
        },
    ]
    checks = run_falsification_checks(arm_results, vglc_dist)
    assert checks["H20_floor_shortcut_collapse"]["passed"] is False
    assert [x["arm"] for x in checks["H20_floor_shortcut_collapse"]["flagged_arms"]] == ["G-LG@confidence"]
    assert checks["H10_memorization_alarm"]["passed"] is False
    assert checks["H40_repair_dominance"]["passed"] is False
    assert checks["H40_beats_empty_floor_baseline"]["passed"] is False
    assert checks["H40_beats_empty_floor_baseline"]["per_arm"]["G-LG@random"]["passed"] is True
    assert checks["H40_beats_empty_floor_baseline"]["per_arm"]["G-LG@confidence"]["passed"] is False
    pair = checks["H30_unmask_order_delta"]["paired_arms"]["G-LG"]
    assert np.isclose(pair["block_rate_random_minus_confidence"], 0.40)


def test_evaluate_non_neural_baseline_smoke():
    train_rooms = [_make_room(int(P["FLOOR"]), extra_block=True)]
    vglc_dist = compute_interior_distribution(train_rooms)

    empty_res = evaluate_non_neural_baseline(
        "empty_floor",
        seeds=[42],
        num_rooms=5,
        train_rooms=train_rooms,
        vglc_dist=vglc_dist,
        topology_population_size=6,
        topology_generations=4,
    )
    assert empty_res["arm"] == "baseline:empty_floor"
    assert empty_res["raw_rooms_with_block_rate"] == 0.0
    assert empty_res["num_rooms"] > 0

    tmpl_res = evaluate_non_neural_baseline(
        "vglc_template",
        seeds=[42],
        num_rooms=5,
        train_rooms=train_rooms,
        vglc_dist=vglc_dist,
        topology_population_size=6,
        topology_generations=4,
    )
    assert tmpl_res["arm"] == "baseline:vglc_template"
    assert tmpl_res["raw_rooms_with_block_rate"] == 1.0
