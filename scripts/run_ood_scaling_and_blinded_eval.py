"""OOD scaling benchmark + blinded human-eval packet builder."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
    run_block_i_benchmark,
)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    rule_space: str
    search_strategy: str


METHODS: Dict[str, MethodConfig] = {
    "FULL_GA": MethodConfig(name="FULL_GA", rule_space="full", search_strategy="ga"),
    "FULL_CVT": MethodConfig(name="FULL_CVT", rule_space="full", search_strategy="cvt_emitter"),
    "CORE_GA": MethodConfig(name="CORE_GA", rule_space="core", search_strategy="ga"),
}


def _method_list(raw: str) -> List[MethodConfig]:
    out: List[MethodConfig] = []
    for token in [t.strip().upper() for t in str(raw).split(",") if t.strip()]:
        if token not in METHODS:
            raise ValueError(f"Unknown method '{token}'. Valid: {sorted(METHODS.keys())}")
        out.append(METHODS[token])
    if not out:
        out = [METHODS["FULL_GA"], METHODS["FULL_CVT"]]
    return out


def _room_budgets(
    refs: Sequence[nx.Graph],
    room_budget_cap: int,
) -> Dict[str, Tuple[int, int]]:
    if not refs:
        return {
            "in_dist": (8, 16),
            "ood_small": (6, 8),
            "ood_large": (18, min(int(room_budget_cap), 32)),
        }
    nodes = np.asarray([max(1, int(g.number_of_nodes())) for g in refs], dtype=np.float64)
    q25 = int(np.floor(np.quantile(nodes, 0.25)))
    q75 = int(np.ceil(np.quantile(nodes, 0.75)))
    cap = max(12, int(room_budget_cap))
    in_min = int(np.clip(q25, 6, cap))
    in_max = int(np.clip(max(q75, in_min + 2), in_min + 1, cap))
    small_min = int(np.clip(max(6, in_min - 8), 6, max(6, in_min - 1)))
    small_max = int(np.clip(max(small_min + 1, in_min - 2), small_min + 1, max(small_min + 1, in_min)))
    large_min = int(np.clip(in_max + 2, in_max + 1, cap))
    large_max = int(np.clip(max(large_min + 2, in_max + 10), large_min + 1, cap))
    return {
        "in_dist": (in_min, in_max),
        "ood_small": (small_min, small_max),
        "ood_large": (large_min, large_max),
    }


def _edge_class(attrs: Dict[str, Any]) -> str:
    t = str(attrs.get("edge_type", attrs.get("type", "")) or "").upper().strip()
    if t in {"", "PATH"}:
        return "PATH"
    if t in {"LOCKED", "BOSS_LOCKED"}:
        return "LOCKED"
    if t in {"ITEM_GATE", "ON_OFF_GATE", "STATE_BLOCK", "MULTI_LOCK", "SHUTTER"}:
        return "GATED"
    if t in {"ONE_WAY"}:
        return "ONE_WAY"
    if t in {"HAZARD"}:
        return "HAZARD"
    return "OTHER"


def _node_color(attrs: Dict[str, Any]) -> str:
    t = str(attrs.get("type", attrs.get("label", "")) or "").upper()
    if "START" in t:
        return "#2ca02c"
    if "GOAL" in t or "TRIFORCE" in t:
        return "#d4a017"
    if "KEY" in t:
        return "#1f77b4"
    if "LOCK" in t:
        return "#d62728"
    if "PUZZLE" in t:
        return "#17becf"
    if "ITEM" in t:
        return "#9467bd"
    if "ENEMY" in t or "BOSS" in t:
        return "#ff7f0e"
    return "#8c564b"


def _draw_blinded_graph(graph: nx.Graph, out_path: Path, seed: int) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.1), dpi=170)
    ax.set_axis_off()
    if graph.number_of_nodes() == 0:
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return

    pos = nx.spring_layout(graph.to_undirected(), seed=seed)
    path_edges: List[Tuple[Any, Any]] = []
    seen_undirected = set()
    directed_groups: Dict[str, List[Tuple[Any, Any]]] = {"LOCKED": [], "GATED": [], "ONE_WAY": [], "HAZARD": [], "OTHER": []}

    for u, v, attrs in graph.edges(data=True):
        cls = _edge_class(attrs)
        if cls == "PATH":
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            if (a, b) in seen_undirected:
                continue
            seen_undirected.add((a, b))
            path_edges.append((a, b))
        else:
            directed_groups.setdefault(cls, []).append((u, v))

    if path_edges:
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=path_edges,
            ax=ax,
            edge_color="#9aa0a6",
            width=1.6,
            style="solid",
            arrows=False,
            alpha=0.90,
        )

    style = {
        "LOCKED": {"color": "#d62728", "style": "solid"},
        "GATED": {"color": "#9467bd", "style": "dashdot"},
        "ONE_WAY": {"color": "#8c564b", "style": "dotted"},
        "HAZARD": {"color": "#ff7f0e", "style": "dashed"},
        "OTHER": {"color": "#6c757d", "style": "solid"},
    }
    for cls, edges in directed_groups.items():
        if not edges:
            continue
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            edgelist=edges,
            ax=ax,
            edge_color=style[cls]["color"],
            width=1.7,
            style=style[cls]["style"],
            arrows=bool(graph.is_directed()),
            arrowstyle="-|>",
            arrowsize=9,
            connectionstyle="arc3,rad=0.03",
            alpha=0.92,
        )

    nodes = list(graph.nodes())
    colors = [_node_color(dict(graph.nodes[n])) for n in nodes]
    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        nodelist=nodes,
        node_color=colors,
        node_size=280,
        linewidths=0.7,
        edgecolors="#1f2937",
        ax=ax,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OOD scaling benchmark and build blinded human-eval packet.")
    p.add_argument("--output", type=Path, default=Path("results") / "ood_blinded_eval")
    p.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    p.add_argument("--reference-limit", type=int, default=None)
    p.add_argument("--methods", type=str, default="FULL_GA,FULL_CVT")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--population-size", type=int, default=24)
    p.add_argument("--generations", type=int, default=24)
    p.add_argument("--room-budget-cap", type=int, default=42)
    p.add_argument("--blinded-per-condition", type=int, default=6)
    p.add_argument("--qd-archive-cells", type=int, default=128)
    p.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    p.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
    logging.getLogger("src.generation.evolutionary_director").setLevel(logging.ERROR)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    methods = _method_list(args.methods)
    refs = load_vglc_reference_graphs(data_root=args.data_root, limit=args.reference_limit)
    if not refs:
        raise RuntimeError(f"No reference graphs loaded from {args.data_root}")

    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in refs]
    descriptor_targets = {
        "linearity": float(np.mean([d.linearity for d in ref_desc])),
        "leniency": float(np.mean([d.leniency for d in ref_desc])),
        "progression_complexity": float(np.mean([d.progression_complexity for d in ref_desc])),
        "topology_complexity": float(np.mean([d.topology_complexity for d in ref_desc])),
        "path_length": float(np.mean([d.path_length for d in ref_desc])),
        "num_nodes": float(np.mean([d.num_nodes for d in ref_desc])),
    }
    regimes = _room_budgets(refs, room_budget_cap=args.room_budget_cap)

    summary_rows: List[Dict[str, Any]] = []
    graph_bank: List[Dict[str, Any]] = []
    benchmark_payloads: Dict[str, Any] = {}

    for regime_name, (min_rooms, max_rooms) in regimes.items():
        for m in methods:
            t0 = time.time()
            graphs, gen_times = generate_block_i_graphs(
                num_samples=int(args.num_samples),
                seed=int(args.seed) + hash((regime_name, m.name)) % 100000,
                min_rooms=int(min_rooms),
                max_rooms=int(max_rooms),
                population_size=int(args.population_size),
                generations=int(args.generations),
                rule_space=str(m.rule_space),
                descriptor_targets=descriptor_targets,
                room_count_bias=0.45,
                search_strategy=str(m.search_strategy),
                qd_archive_cells=int(args.qd_archive_cells),
                qd_init_random_fraction=float(args.qd_init_random_fraction),
                qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
            )
            bench = run_block_i_benchmark(generated_graphs=graphs, reference_graphs=refs, generation_times=gen_times)
            benchmark_payloads[f"{regime_name}:{m.name}"] = asdict(bench)
            summary_rows.append(
                {
                    "regime": regime_name,
                    "method": m.name,
                    "n": int(len(graphs)),
                    "min_rooms": int(min_rooms),
                    "max_rooms": int(max_rooms),
                    "overall_completeness": float(bench.completeness.get("overall_completeness", 0.0)),
                    "constraint_valid_rate": float(bench.completeness.get("constraint_valid_rate", 0.0)),
                    "repair_rate": float(bench.robustness.get("repair_rate", 0.0)),
                    "mean_generation_constraint_rejections": float(bench.robustness.get("mean_generation_constraint_rejections", 0.0)),
                    "linearity": float(bench.generated_descriptor_means.get("linearity", 0.0)),
                    "leniency": float(bench.generated_descriptor_means.get("leniency", 0.0)),
                    "progression_complexity": float(bench.generated_descriptor_means.get("progression_complexity", 0.0)),
                    "topology_complexity": float(bench.generated_descriptor_means.get("topology_complexity", 0.0)),
                    "path_length": float(bench.generated_descriptor_means.get("path_length", 0.0)),
                    "num_nodes": float(bench.generated_descriptor_means.get("num_nodes", 0.0)),
                    "novelty_vs_reference": float(bench.reference_comparison.get("novelty_vs_reference", 0.0)),
                    "expressive_overlap_reference": float(bench.reference_comparison.get("expressive_overlap_reference", 0.0)),
                    "generation_time_sec": float(np.mean(gen_times)) if gen_times else 0.0,
                    "wall_time_sec": float(time.time() - t0),
                }
            )
            for idx, g in enumerate(graphs):
                graph_bank.append(
                    {
                        "regime": regime_name,
                        "method": m.name,
                        "index": int(idx),
                        "graph": g,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out / "ood_scaling_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # Build blinded packet.
    rnd = random.Random(int(args.seed) + 777)
    conditioned: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in graph_bank:
        key = (str(row["regime"]), str(row["method"]))
        conditioned.setdefault(key, []).append(row)

    packet_rows: List[Dict[str, Any]] = []
    key_rows: List[Dict[str, Any]] = []
    sample_id = 1
    for key, rows in sorted(conditioned.items()):
        rnd.shuffle(rows)
        keep = rows[: max(1, int(args.blinded_per_condition))]
        for row in keep:
            sid = f"S{sample_id:03d}"
            sample_id += 1
            packet_rows.append(
                {
                    "sample_id": sid,
                    "image_file": f"images/{sid}.png",
                }
            )
            key_rows.append(
                {
                    "sample_id": sid,
                    "regime": key[0],
                    "method": key[1],
                    "index": int(row["index"]),
                }
            )

    rnd.shuffle(packet_rows)
    images_dir = out / "blinded" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    key_lookup = {r["sample_id"]: r for r in key_rows}
    graph_lookup: Dict[str, nx.Graph] = {}
    for row in packet_rows:
        sid = str(row["sample_id"])
        meta = key_lookup[sid]
        for item in conditioned[(meta["regime"], meta["method"])]:
            if int(item["index"]) == int(meta["index"]):
                graph_lookup[sid] = item["graph"]
                break

    for i, row in enumerate(packet_rows):
        sid = str(row["sample_id"])
        graph = graph_lookup.get(sid)
        if graph is None:
            continue
        _draw_blinded_graph(graph, images_dir / f"{sid}.png", seed=int(args.seed) + i)

    blinded_dir = out / "blinded"
    blinded_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = blinded_dir / "blinded_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "image_file"])
        writer.writeheader()
        for row in packet_rows:
            writer.writerow({"sample_id": row["sample_id"], "image_file": row["image_file"]})

    key_csv = blinded_dir / "blinded_key.csv"
    with key_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "regime", "method", "index"])
        writer.writeheader()
        for row in key_rows:
            writer.writerow(row)

    rating_csv = blinded_dir / "rating_sheet.csv"
    with rating_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "organization_score_1_5",
                "readability_score_1_5",
                "novelty_score_1_5",
                "fairness_score_1_5",
                "notes",
            ],
        )
        writer.writeheader()
        for row in packet_rows:
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "organization_score_1_5": "",
                    "readability_score_1_5": "",
                    "novelty_score_1_5": "",
                    "fairness_score_1_5": "",
                    "notes": "",
                }
            )

    report = {
        "output": str(out),
        "methods": [m.name for m in methods],
        "regimes": {k: {"min_rooms": int(v[0]), "max_rooms": int(v[1])} for k, v in regimes.items()},
        "summary_csv": str(summary_csv),
        "blinded_manifest_csv": str(manifest_csv),
        "blinded_key_csv": str(key_csv),
        "rating_sheet_csv": str(rating_csv),
        "payload_json": str(out / "ood_scaling_payload.json"),
    }
    (out / "ood_scaling_payload.json").write_text(json.dumps(benchmark_payloads, indent=2), encoding="utf-8")
    (out / "ood_scaling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
