import csv
import json

from scripts.consolidate_compute_sample_efficiency import (
    collect_compute_observations,
    summarize_compute_efficiency,
)


def test_compute_consolidation_extracts_json_and_csv_metrics(tmp_path):
    run_dir = tmp_path / "outputs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "wall_time_sec": 10.0,
                "num_samples": 5,
                "success_rate": 0.8,
                "nested": {"val_loss": 0.25},
            }
        ),
        encoding="utf-8",
    )
    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_loss", "train_loss"])
        writer.writeheader()
        writer.writerow({"epoch": 1, "val_loss": 0.4, "train_loss": 0.6})
        writer.writerow({"epoch": 2, "val_loss": 0.2, "train_loss": 0.5})

    observations, inventory = collect_compute_observations([tmp_path / "outputs"])
    metrics = {obs.metric for obs in observations}
    assert {"wall_time_sec", "num_samples", "success_rate", "val_loss", "epoch"}.issubset(metrics)
    assert inventory and inventory[0]["file_count"] == 2

    summary = summarize_compute_efficiency(observations)
    assert len(summary) == 1
    row = summary[0]
    assert row["observed_runtime_sec"] == 10.0
    assert row["observed_sample_count"] == 5.0
    assert row["samples_per_sec"] == 0.5
    assert row["best_loss_like_metric"] == 0.2
    assert row["best_success_like_metric"] == 0.8
