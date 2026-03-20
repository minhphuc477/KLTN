"""
Rule-application visual debugger + deterministic replay checker.

Outputs:
- `trace.json` with per-gene rule outcomes,
- `replay_payload.json` for deterministic regeneration,
- `steps/*.png` snapshots for rule-by-rule inspection,
- `summary.json` including replay consistency check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.evolutionary_director import (
    GraphGrammarExecutor,
    mission_graph_to_networkx,
)
from src.generation.grammar import MissionGraph

logger = logging.getLogger(__name__)


NODE_COLORS: Dict[str, str] = {
    "START": "#2ca02c",
    "GOAL": "#d4a017",
    "KEY": "#1f77b4",
    "LOCK": "#d62728",
    "ENEMY": "#ff7f0e",
    "PUZZLE": "#17becf",
    "ITEM": "#9467bd",
    "EMPTY": "#7f7f7f",
}

TWO_WAY_EDGE_TYPES = {"PATH", "SHORTCUT", "WARP", "STAIRS", "HIDDEN"}


def _parse_genome(raw: str) -> List[int]:
    vals = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _random_genome(executor: GraphGrammarExecutor, length: int, seed: int) -> List[int]:
    rng = random.Random(int(seed))
    rule_ids = list(range(1, len(executor.rules)))
    weights = [max(1e-6, float(executor.rules[i].weight)) for i in rule_ids]
    return [int(rng.choices(rule_ids, weights=weights, k=1)[0]) for _ in range(int(length))]


def _graph_signature(graph: MissionGraph) -> str:
    graph.sanitize()
    node_rows = []
    for nid, n in sorted(graph.nodes.items(), key=lambda x: int(x[0])):
        node_rows.append(
            {
                "id": int(nid),
                "type": n.node_type.name,
                "pos": [int(n.position[0]), int(n.position[1]), int(n.position[2]) if len(n.position) > 2 else 0],
                "difficulty": round(float(n.difficulty), 5),
                "key_id": int(n.key_id) if n.key_id is not None else None,
                "item_type": n.item_type,
                "required_item": n.required_item,
                "switch_id": int(n.switch_id) if n.switch_id is not None else None,
            }
        )
    edge_rows = []
    for e in sorted(graph.edges, key=lambda x: (int(x.source), int(x.target), x.edge_type.name)):
        edge_rows.append(
            {
                "source": int(e.source),
                "target": int(e.target),
                "type": e.edge_type.name,
                "key_required": int(e.key_required) if e.key_required is not None else None,
                "item_required": e.item_required,
                "switch_id": int(e.switch_id) if e.switch_id is not None else None,
                "token_count": int(e.token_count),
            }
        )
    payload = {"nodes": node_rows, "edges": edge_rows}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _draw_graph(graph: MissionGraph, *, title: str, out_path: Path, layout_seed: int = 0) -> None:
    G = mission_graph_to_networkx(graph, directed=True)
    if G.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(G.to_undirected(), seed=int(layout_seed), k=1.0 / max(1.0, np.sqrt(G.number_of_nodes())))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5), dpi=170)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    node_groups: Dict[str, List[Any]] = {}
    for nid, attrs in G.nodes(data=True):
        t = str(attrs.get("type", "EMPTY")).upper()
        node_groups.setdefault(t, []).append(nid)

    for t, nodes in node_groups.items():
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=NODE_COLORS.get(t, "#8c564b"),
            node_size=280 if t not in {"START", "GOAL"} else 380,
            linewidths=0.7,
            edgecolors="#1f2937",
            ax=ax,
        )

    two_way = []
    one_way = []
    seen_tw: set[Tuple[Any, Any]] = set()
    for u, v, attrs in G.edges(data=True):
        e_type = str(attrs.get("edge_type", "PATH")).upper()
        if e_type in TWO_WAY_EDGE_TYPES:
            a, b = (u, v) if str(u) <= str(v) else (v, u)
            if (a, b) in seen_tw:
                continue
            seen_tw.add((a, b))
            two_way.append((a, b))
        else:
            one_way.append((u, v))

    if two_way:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=two_way,
            edge_color="#9aa0a6",
            width=1.5,
            arrows=False,
            ax=ax,
        )
    if one_way:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=one_way,
            edge_color="#374151",
            width=1.6,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=10,
            ax=ax,
        )

    labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_color="#111827", ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug grammar rule application with deterministic replay.")
    parser.add_argument("--output", type=Path, default=Path("results") / "rule_debugger")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule-space", type=str, default="full", choices=["core", "full"])
    parser.add_argument("--genome", type=str, default="", help="Comma-separated rule IDs.")
    parser.add_argument("--genome-length", type=int, default=18)
    parser.add_argument("--difficulty", type=float, default=0.5)
    parser.add_argument("--max-nodes", type=int, default=16)
    parser.add_argument("--max-lock-key-rules", type=int, default=3)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    executor = GraphGrammarExecutor(
        seed=int(args.seed),
        use_full_rule_space=(str(args.rule_space).lower() == "full"),
        max_lock_key_rules=int(args.max_lock_key_rules),
        enforce_generation_constraints=True,
        allow_candidate_repairs=False,
    )

    genome = _parse_genome(args.genome) if str(args.genome).strip() else []
    if not genome:
        genome = _random_genome(executor, length=int(args.genome_length), seed=int(args.seed) + 11)

    graph = executor.execute(
        genome=genome,
        difficulty=float(args.difficulty),
        max_nodes=int(args.max_nodes),
        allow_override=False,
        record_trace=True,
    )

    trace = list(graph.generation_stats.get("rule_trace", []))
    replay_payload = dict(graph.generation_stats.get("generation_replay", {}))
    if not replay_payload:
        replay_payload = {
            "seed": int(args.seed),
            "difficulty": float(args.difficulty),
            "max_nodes": int(args.max_nodes),
            "allow_override": False,
            "use_full_rule_space": bool(str(args.rule_space).lower() == "full"),
            "max_lock_key_rules": int(args.max_lock_key_rules),
            "enforce_generation_constraints": True,
            "allow_candidate_repairs": False,
            "rule_weight_overrides": {},
            "genome": [int(g) for g in genome],
        }

    replay_graph = GraphGrammarExecutor.replay_from_payload(replay_payload, record_trace=True)
    sig_a = _graph_signature(graph)
    sig_b = _graph_signature(replay_graph)

    if not args.no_images:
        # Step 0 (start-only baseline).
        g0 = executor.execute(
            genome=[],
            difficulty=float(args.difficulty),
            max_nodes=int(args.max_nodes),
            allow_override=False,
            record_trace=False,
        )
        _draw_graph(g0, title="Step 0: StartRule", out_path=steps_dir / "step_000.png", layout_seed=int(args.seed))

        for idx in range(len(trace)):
            prefix = genome[: idx + 1]
            g_step = executor.execute(
                genome=prefix,
                difficulty=float(args.difficulty),
                max_nodes=int(args.max_nodes),
                allow_override=False,
                record_trace=False,
            )
            row = trace[idx]
            title = f"Step {idx + 1}: {row.get('rule_name', '?')} [{row.get('status', '?')}]"
            _draw_graph(
                g_step,
                title=title,
                out_path=steps_dir / f"step_{idx + 1:03d}.png",
                layout_seed=int(args.seed) + idx + 1,
            )

    trace_path = out_dir / "trace.json"
    payload_path = out_dir / "replay_payload.json"
    summary_path = out_dir / "summary.json"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    payload_path.write_text(json.dumps(replay_payload, indent=2), encoding="utf-8")

    summary = {
        "seed": int(args.seed),
        "rule_space": str(args.rule_space),
        "genome": [int(g) for g in genome],
        "trace_steps": int(len(trace)),
        "final_nodes": int(len(graph.nodes)),
        "final_edges": int(len(graph.edges)),
        "replay_nodes": int(len(replay_graph.nodes)),
        "replay_edges": int(len(replay_graph.edges)),
        "signature_original": sig_a,
        "signature_replay": sig_b,
        "deterministic_replay_match": bool(sig_a == sig_b),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved rule debugger outputs to %s", out_dir)
    logger.info("Deterministic replay match: %s", summary["deterministic_replay_match"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

