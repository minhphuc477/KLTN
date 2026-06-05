from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_script(script: str, tmp_path: Path, *extra: str) -> dict:
    output = tmp_path / (Path(script).stem + ".json")
    cmd = [sys.executable, script, "--dry-run", "--output", str(output), *extra]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert output.exists()
    return json.loads(output.read_text(encoding="utf-8"))


def test_ablation_vqvae_fsq_dry_run(tmp_path: Path) -> None:
    payload = _run_script(
        "experiments/ablation_vqvae_fsq.py",
        tmp_path,
        "--num-classes",
        "8",
        "--codebook-size",
        "16",
        "--latent-dim",
        "8",
        "--hidden-dim",
        "8",
        "--max-samples",
        "2",
        "--batch-size",
        "1",
    )
    assert set(payload["metrics"]) == {"vq_ema", "fsq"}


def test_ablation_logicnet_scaling_dry_run(tmp_path: Path) -> None:
    payload = _run_script(
        "experiments/ablation_logicnet_scaling.py",
        tmp_path,
        "--iterations",
        "2",
    )
    assert payload["local_chunk"]["finite_output"] is True


def test_ablation_gat_attention_dry_run(tmp_path: Path) -> None:
    payload = _run_script(
        "experiments/ablation_gat_attention.py",
        tmp_path,
        "--grid-dim",
        "16",
        "--graph-dim",
        "16",
        "--heads",
        "4",
        "--repeats",
        "1",
    )
    assert {row["attention_mode"] for row in payload["results"]} == {"softmax", "linear_hedgehog"}
    assert all(row["finite_output"] for row in payload["results"])
    topology_rows = payload["topology_costs"]
    assert {row["topology_refinement_mode"] for row in topology_rows} == {
        "gat2",
        "sparse_edge",
        "sparse_directed",
        "sparse_directed_semantic",
        "graphormer",
    }
    by_mode = {
        (row["nodes"], row["topology_refinement_mode"]): row
        for row in topology_rows
    }
    for nodes in {row["nodes"] for row in topology_rows}:
        assert by_mode[(nodes, "graphormer")]["shortest_path_bias_ops"] > 0
        assert by_mode[(nodes, "sparse_edge")]["attention_pairs"] < by_mode[(nodes, "gat2")]["attention_pairs"]


def test_ablation_pcbs_vs_astar_dry_run(tmp_path: Path) -> None:
    payload = _run_script(
        "experiments/ablation_pcbs_vs_astar.py",
        tmp_path,
        "--timeout-astar",
        "200",
        "--timeout-pcbs",
        "50",
    )
    assert payload["summary"]["count"] == 1
    assert "oracle_solvable_rate" in payload["summary"]
