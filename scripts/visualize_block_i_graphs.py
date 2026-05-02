"""
Visualize Block I topology graphs (generated vs VGLC reference).

Outputs publication-style figures for rapid visual inspection:
1) Generated graph gallery
2) Reference graph gallery
3) Descriptor scatter comparison
4) Run summary JSON
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.definitions import parse_edge_type_tokens, parse_node_label_tokens
from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    generate_block_i_graphs,
    load_vglc_reference_graphs,
)

logger = logging.getLogger(__name__)


# Consistent visual language across exports.
NODE_STYLE: Dict[str, Dict[str, Any]] = {
    "START": {"color": "#2ca02c", "shape": "s"},
    "GOAL": {"color": "#d4a017", "shape": "*"},
    "KEY": {"color": "#1f77b4", "shape": "o"},
    "LOCK": {"color": "#d62728", "shape": "D"},
    "ENEMY": {"color": "#ff7f0e", "shape": "o"},
    "PUZZLE": {"color": "#17becf", "shape": "o"},
    "ITEM": {"color": "#9467bd", "shape": "o"},
    "EMPTY": {"color": "#7f7f7f", "shape": "o"},
    "UNKNOWN": {"color": "#8c564b", "shape": "o"},
}

NODE_DISPLAY_NAME: Dict[str, str] = {
    "START": "Start",
    "GOAL": "Goal",
    "KEY": "Key",
    "BIG_KEY": "Big Key",
    "LOCK": "Lock",
    "BOSS_DOOR": "Boss Door",
    "BOSS": "Boss",
    "MINI_BOSS": "Mini Boss",
    "ENEMY": "Enemy",
    "PUZZLE": "Puzzle",
    "TUTORIAL_PUZZLE": "Tutorial Puzzle",
    "COMBAT_PUZZLE": "Combat Puzzle",
    "COMPLEX_PUZZLE": "Complex Puzzle",
    "ITEM": "Item",
    "SWITCH": "Switch",
    "EMPTY": "Empty",
    "STAIRS_DOWN": "Stairs Down",
    "STAIRS_UP": "Stairs Up",
    "UNKNOWN": "Room",
}

EDGE_STYLE: Dict[str, Dict[str, Any]] = {
    "PATH": {"color": "#9aa0a6", "width": 1.6, "style": "solid"},
    "LOCKED": {"color": "#d62728", "width": 2.0, "style": "solid"},
    "ITEM_GATE": {"color": "#9467bd", "width": 2.0, "style": "dashdot"},
    "SWITCH": {"color": "#17becf", "width": 2.0, "style": "dashed"},
    "ONE_WAY": {"color": "#8c564b", "width": 1.8, "style": "dotted"},
    "HAZARD": {"color": "#ff7f0e", "width": 1.9, "style": "dashed"},
    "OTHER": {"color": "#374151", "width": 1.4, "style": "dashdot"},
}

LOCK_EDGE_TOKENS = {
    "locked",
    "key_locked",
    "boss_locked",
    "item_locked",
    "switch",
    "switch_locked",
    "state_block",
    "item_gate",
}

HAZARD_EDGE_TOKENS = {
    "hazard",
    "lava",
    "spikes",
    "ice",
    "poison",
}

TWO_WAY_EDGE_TOKENS = {
    "open",
    "path",
    "shortcut",
    "stair",
    "stairs",
    "warp",
    "hidden",
}

TWO_WAY_EDGE_RAW_TYPES = {
    "PATH",
    "OPEN",
    "SHORTCUT",
    "WARP",
    "STAIR",
    "STAIRS",
    "HIDDEN",
}


def _node_tokens(attrs: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    raw_type = str(attrs.get("type", "") or "").strip().upper()
    if raw_type:
        tokens.append(raw_type.lower())
    label = str(attrs.get("label", "") or "")
    if label:
        tokens.extend(parse_node_label_tokens(label))
    if attrs.get("is_start"):
        tokens.extend(["s", "start"])
    if attrs.get("is_goal") or attrs.get("is_triforce"):
        tokens.extend(["t", "goal", "triforce"])
    out: List[str] = []
    seen = set()
    for token in tokens:
        t = str(token).strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _classify_node(attrs: Dict[str, Any]) -> str:
    tokens = set(_node_tokens(attrs))
    raw_type = str(attrs.get("type", "") or "").strip().upper()
    if raw_type in NODE_STYLE:
        return raw_type
    if "start" in tokens or "s" in tokens:
        return "START"
    if "goal" in tokens or "triforce" in tokens or "t" in tokens:
        return "GOAL"
    if "key" in tokens or "k" in tokens or "key_small" in tokens:
        return "KEY"
    if "lock" in tokens:
        return "LOCK"
    if "enemy" in tokens or "e" in tokens or "boss" in tokens or "b" in tokens:
        return "ENEMY"
    if "puzzle" in tokens or "p" in tokens:
        return "PUZZLE"
    if "item" in tokens:
        return "ITEM"
    if "switch" in tokens:
        return "SWITCH"
    if "empty" in tokens:
        return "EMPTY"
    return "UNKNOWN"


def _classify_edge(attrs: Dict[str, Any]) -> str:
    raw = str(attrs.get("edge_type", attrs.get("type", "")) or "").strip().upper()
    tokens = set(
        parse_edge_type_tokens(
            label=str(attrs.get("label", "") or ""),
            edge_type=raw.lower(),
        )
    )
    if raw in {"LOCKED", "BOSS_LOCKED"} or tokens.intersection(LOCK_EDGE_TOKENS):
        if "item_gate" in tokens or raw == "ITEM_GATE":
            return "ITEM_GATE"
        if "switch" in tokens or raw in {"ON_OFF_GATE", "STATE_BLOCK"}:
            return "SWITCH"
        return "LOCKED"
    if raw in {"ITEM_GATE"}:
        return "ITEM_GATE"
    if raw in {"ONE_WAY", "VALVE", "SOFT_LOCKED", "SHUTTER"} or {"one_way", "soft_locked"}.intersection(tokens):
        return "ONE_WAY"
    if raw in {"HAZARD"} or tokens.intersection(HAZARD_EDGE_TOKENS):
        return "HAZARD"
    if raw in TWO_WAY_EDGE_RAW_TYPES or tokens.intersection(TWO_WAY_EDGE_TOKENS):
        return "PATH"
    if raw in {"PATH", ""}:
        return "PATH"
    return "OTHER"


def _find_start_node(G: nx.Graph) -> Optional[Any]:
    candidates: List[Any] = []
    for node_id, attrs in G.nodes(data=True):
        t = _classify_node(attrs)
        if t == "START":
            candidates.append(node_id)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: str(x))[0]


def _progression_layout(G: nx.Graph, seed: int = 42) -> Dict[Any, np.ndarray]:
    """
    Layout strategy:
    1) Try Graphviz 'dot' layered layout (best for directed progression).
    2) Fallback to progression-layer + multipartite.
    3) Final fallback to deterministic spring layout.
    """
    if G.number_of_nodes() == 0:
        return {}
    try:
        return nx.nx_pydot.graphviz_layout(G, prog="dot")
    except Exception as exc:
        logger.debug("graphviz_layout unavailable; falling back to non-graphviz layout: %s", exc)

    start = _find_start_node(G)
    U = G.to_undirected()
    dist: Dict[Any, int] = {}
    if start is not None:
        try:
            dist = dict(nx.single_source_shortest_path_length(U, start))
        except Exception:
            dist = {}

    if not dist:
        return nx.spring_layout(U, seed=seed, k=1.0 / max(1.0, math.sqrt(U.number_of_nodes())))

    max_dist = max(dist.values()) if dist else 0
    H = U.copy()
    for n in H.nodes():
        H.nodes[n]["subset"] = int(dist.get(n, max_dist + 1))
    try:
        pos = nx.multipartite_layout(H, subset_key="subset", align="horizontal")
        # Light spring relaxation keeps hierarchy while reducing overlaps.
        return nx.spring_layout(
            H,
            pos=pos,
            fixed=list(H.nodes()),
            iterations=5,
            seed=seed,
        )
    except Exception:
        return nx.spring_layout(U, seed=seed, k=1.0 / max(1.0, math.sqrt(U.number_of_nodes())))


def _draw_one_graph(
    ax: plt.Axes,
    G: nx.Graph,
    title: str,
    seed: int,
    *,
    label_mode: str = "minimal",
    node_scale: float = 1.0,
    title_fontsize: int = 9,
    show_metrics: bool = True,
) -> None:
    ax.set_title(title, fontsize=title_fontsize, pad=8)
    ax.set_axis_off()
    if G.number_of_nodes() == 0:
        return

    is_directed = bool(G.is_directed())
    pos = _progression_layout(G, seed=seed)
    center = np.mean(np.asarray(list(pos.values()), dtype=np.float64), axis=0)

    for edge_class, style in EDGE_STYLE.items():
        edgelist = []
        seen_undirected = set()
        for u, v, attrs in G.edges(data=True):
            if _classify_edge(attrs) != edge_class:
                continue
            if edge_class == "PATH":
                # PATH semantics are traversable both ways in mission graphs.
                # Collapse mirrored edges and render without arrowheads.
                a, b = (u, v) if str(u) <= str(v) else (v, u)
                if (a, b) in seen_undirected:
                    continue
                seen_undirected.add((a, b))
                edgelist.append((a, b))
            else:
                edgelist.append((u, v))
        if not edgelist:
            continue
        draw_arrows = bool(is_directed and edge_class != "PATH")
        edge_kwargs: Dict[str, Any] = {
            "G": G,
            "pos": pos,
            "edgelist": edgelist,
            "ax": ax,
            "edge_color": style["color"],
            "width": style["width"],
            "style": style["style"],
            "arrows": draw_arrows,
            "alpha": 0.92,
        }
        if draw_arrows:
            edge_kwargs.update(
                {
                    "arrowstyle": "-|>",
                    "arrowsize": 10,
                    "connectionstyle": "arc3,rad=0.03",
                    "min_source_margin": 5,
                    "min_target_margin": 5,
                }
            )
        nx.draw_networkx_edges(
            **edge_kwargs,
        )

    for node_class, style in NODE_STYLE.items():
        nodes = [n for n, attrs in G.nodes(data=True) if _classify_node(attrs) == node_class]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=nodes,
            node_color=style["color"],
            node_shape=style["shape"],
            node_size=(320 if node_class not in {"START", "GOAL"} else 440) * float(node_scale),
            linewidths=0.7,
            edgecolors="#1f2937",
            ax=ax,
            alpha=0.95,
        )

    if label_mode == "minimal":
        labels: Dict[Any, str] = {}
        for n, attrs in G.nodes(data=True):
            cls = _classify_node(attrs)
            if cls in {"START", "GOAL"}:
                labels[n] = cls[0]
            elif cls in {"KEY", "LOCK"}:
                labels[n] = cls[0]
        if labels:
            nx.draw_networkx_labels(
                G,
                pos=pos,
                labels=labels,
                font_size=max(6, int(7 * float(node_scale))),
                font_weight="bold",
                font_color="#111827",
                ax=ax,
            )
    elif label_mode == "readable":
        for node_id, attrs in G.nodes(data=True):
            node_class = _classify_node(attrs)
            raw_type = str(attrs.get("type", attrs.get("label", "")) or "").strip().upper()
            display = NODE_DISPLAY_NAME.get(raw_type, NODE_DISPLAY_NAME.get(node_class, node_class.title()))
            text = display.replace(" ", "\n") if len(display) > 10 else display
            point = np.asarray(pos[node_id], dtype=np.float64)
            offset = point - center
            norm = float(np.linalg.norm(offset))
            if norm < 1e-6:
                direction = np.asarray([0.0, 1.0], dtype=np.float64)
            else:
                direction = offset / norm
            label_xy = point + direction * (0.12 + 0.05 * float(node_scale))
            ha = "left" if direction[0] >= 0.12 else ("right" if direction[0] <= -0.12 else "center")
            va = "bottom" if direction[1] >= 0.12 else ("top" if direction[1] <= -0.12 else "center")
            ax.text(
                float(label_xy[0]),
                float(label_xy[1]),
                text,
                fontsize=max(7, int(8 * float(node_scale))),
                fontweight="semibold",
                color="#111827",
                ha=ha,
                va=va,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "#ffffff",
                    "edgecolor": "#d1d5db",
                    "alpha": 0.92,
                },
                zorder=6,
            )

    desc = extract_graph_descriptor(G, grammar=None)
    if show_metrics:
        metrics = (
            f"n={desc.num_nodes}  e={desc.num_edges}  p={desc.path_length}\n"
            f"lin={desc.linearity:.2f}  topo={desc.topology_complexity:.2f}"
        )
        ax.text(
            0.02,
            0.02,
            metrics,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "alpha": 0.84, "edgecolor": "#d1d5db"},
        )


def save_single_graph_figure(
    graph: nx.Graph,
    out_path: Path,
    *,
    seed: int = 42,
    title: str = "Mission Graph",
) -> Dict[str, Any]:
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.4), dpi=220)
    _draw_one_graph(
        ax,
        graph,
        title=title,
        seed=seed,
        label_mode="readable",
        node_scale=1.22,
        title_fontsize=13,
        show_metrics=True,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    desc = extract_graph_descriptor(graph, grammar=None)
    return {
        "num_nodes": int(desc.num_nodes),
        "num_edges": int(desc.num_edges),
        "path_length": int(desc.path_length),
        "linearity": float(desc.linearity),
        "leniency": float(desc.leniency),
        "progression_complexity": float(desc.progression_complexity),
        "topology_complexity": float(desc.topology_complexity),
    }


def _save_gallery(
    graphs: Sequence[nx.Graph],
    out_path: Path,
    title_prefix: str,
    max_graphs: int = 12,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    chosen = list(graphs[: max(0, int(max_graphs))])
    if not chosen:
        return []

    if len(chosen) <= 9:
        cols = min(3, max(1, int(math.ceil(math.sqrt(len(chosen))))))
    else:
        cols = min(4, max(1, int(math.ceil(math.sqrt(len(chosen))))))
    rows = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.25), dpi=170)
    axes_list = np.array(axes).reshape(-1)

    metadata: List[Dict[str, Any]] = []
    for idx, G in enumerate(chosen):
        desc = extract_graph_descriptor(G, grammar=None)
        metadata.append(
            {
                "index": int(idx),
                "num_nodes": int(desc.num_nodes),
                "num_edges": int(desc.num_edges),
                "path_length": int(desc.path_length),
                "linearity": float(desc.linearity),
                "leniency": float(desc.leniency),
                "progression_complexity": float(desc.progression_complexity),
                "topology_complexity": float(desc.topology_complexity),
            }
        )
        _draw_one_graph(
            axes_list[idx],
            G,
            title=f"{title_prefix} {idx + 1}",
            seed=seed + idx,
            label_mode="minimal",
            node_scale=1.0,
            title_fontsize=11,
            show_metrics=True,
        )

    for idx in range(len(chosen), len(axes_list)):
        axes_list[idx].set_axis_off()

    fig.suptitle(f"{title_prefix} Graph Gallery", fontsize=15, y=0.995)
    node_handles = [
        Line2D(
            [0],
            [0],
            marker=style["shape"],
            color="none",
            markerfacecolor=style["color"],
            markeredgecolor="#1f2937",
            markersize=7,
            label=node_type,
        )
        for node_type, style in NODE_STYLE.items()
        if node_type in {"START", "GOAL", "KEY", "LOCK", "ENEMY", "PUZZLE", "ITEM"}
    ]
    edge_label_overrides = {
        "PATH": "PATH/SHORTCUT/STAIRS/WARP/HIDDEN (two-way)",
        "ONE_WAY": "ONE_WAY (arrowed)",
    }
    edge_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            lw=style["width"],
            linestyle=style["style"],
            label=edge_label_overrides.get(edge_type, edge_type),
        )
        for edge_type, style in EDGE_STYLE.items()
        if edge_type in {"PATH", "LOCKED", "ITEM_GATE", "SWITCH", "ONE_WAY", "HAZARD"}
    ]
    fig.legend(
        handles=node_handles + edge_handles,
        loc="lower center",
        ncol=7,
        fontsize=8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return metadata


def _save_descriptor_scatter(
    generated: Sequence[nx.Graph],
    reference: Sequence[nx.Graph],
    out_path: Path,
) -> Dict[str, float]:
    gen_desc = [extract_graph_descriptor(g, grammar=None) for g in generated]
    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in reference]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=220)
    ax0, ax1 = axes

    ax0.scatter(
        [d.linearity for d in ref_desc],
        [d.leniency for d in ref_desc],
        c="#2563eb",
        label="Reference",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax0.scatter(
        [d.linearity for d in gen_desc],
        [d.leniency for d in gen_desc],
        c="#dc2626",
        label="Generated",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax0.set_title("Linearity vs Leniency", fontsize=12, fontweight="bold", pad=10)
    ax0.set_xlim(-0.05, 1.05)
    ax0.set_ylim(-0.05, 1.05)
    ax0.set_xlabel("Linearity", fontsize=11)
    ax0.set_ylabel("Leniency", fontsize=11)
    ax0.grid(alpha=0.25, linestyle="--")
    ax0.legend(loc="best", fontsize=10, framealpha=0.95)

    ax1.scatter(
        [d.progression_complexity for d in ref_desc],
        [d.topology_complexity for d in ref_desc],
        c="#2563eb",
        label="Reference",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.scatter(
        [d.progression_complexity for d in gen_desc],
        [d.topology_complexity for d in gen_desc],
        c="#dc2626",
        label="Generated",
        alpha=0.7,
        s=48,
        edgecolors="white",
        linewidth=0.5,
    )
    ax1.set_title("Progression vs Topology Complexity", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel("Progression Complexity", fontsize=11)
    ax1.set_ylabel("Topology Complexity", fontsize=11)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(loc="best", fontsize=10, framealpha=0.95)

    fig.suptitle("Block I Descriptor Distribution: Generated vs Reference", fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=220)
    plt.close(fig)

    def _mean(values: Iterable[float]) -> float:
        arr = np.asarray(list(values), dtype=np.float64)
        return float(np.mean(arr)) if arr.size > 0 else 0.0

    return {
        "generated_linearity_mean": _mean(d.linearity for d in gen_desc),
        "generated_leniency_mean": _mean(d.leniency for d in gen_desc),
        "generated_progression_complexity_mean": _mean(d.progression_complexity for d in gen_desc),
        "generated_topology_complexity_mean": _mean(d.topology_complexity for d in gen_desc),
        "reference_linearity_mean": _mean(d.linearity for d in ref_desc),
        "reference_leniency_mean": _mean(d.leniency for d in ref_desc),
        "reference_progression_complexity_mean": _mean(d.progression_complexity for d in ref_desc),
        "reference_topology_complexity_mean": _mean(d.topology_complexity for d in ref_desc),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize generated vs reference Block I topology graphs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generated", type=int, default=12)
    parser.add_argument("--num-show", type=int, default=9)
    parser.add_argument("--reference-limit", type=int, default=18)
    parser.add_argument("--population-size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=24)
    parser.add_argument("--min-rooms", type=int, default=8)
    parser.add_argument("--max-rooms", type=int, default=16)
    parser.add_argument("--rule-space", choices=["core", "full"], default="full")
    parser.add_argument("--search-strategy", choices=["ga", "cvt_emitter"], default="ga")
    parser.add_argument("--qd-archive-cells", type=int, default=128)
    parser.add_argument("--qd-init-random-fraction", type=float, default=0.35)
    parser.add_argument("--qd-emitter-mutation-rate", type=float, default=0.18)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data") / "The Legend of Zelda",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "topology_visuals",
    )
    return parser


def run_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Keep CLI output concise (file paths + summary only).
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("src.generation.grammar").setLevel(logging.ERROR)
    logging.getLogger("src.generation.evolutionary_director").setLevel(logging.ERROR)

    references = load_vglc_reference_graphs(
        data_root=args.data_root,
        limit=args.reference_limit,
    )
    if not references:
        raise RuntimeError(f"No reference graphs loaded from: {args.data_root}")

    # Use reference means as Block-I descriptor targets (same benchmark practice).
    ref_desc = [extract_graph_descriptor(g, grammar=None) for g in references]
    descriptor_targets = {
        "linearity": float(np.mean([d.linearity for d in ref_desc])),
        "leniency": float(np.mean([d.leniency for d in ref_desc])),
        "progression_complexity": float(np.mean([d.progression_complexity for d in ref_desc])),
        "topology_complexity": float(np.mean([d.topology_complexity for d in ref_desc])),
        "path_length": float(np.mean([d.path_length for d in ref_desc])),
        "num_nodes": float(np.mean([d.num_nodes for d in ref_desc])),
    }

    generated, _times = generate_block_i_graphs(
        num_samples=int(args.num_generated),
        seed=int(args.seed),
        min_rooms=int(args.min_rooms),
        max_rooms=int(args.max_rooms),
        population_size=int(args.population_size),
        generations=int(args.generations),
        rule_space=str(args.rule_space),
        descriptor_targets=descriptor_targets,
        room_count_bias=0.45,
        search_strategy=str(args.search_strategy),
        qd_archive_cells=int(args.qd_archive_cells),
        qd_init_random_fraction=float(args.qd_init_random_fraction),
        qd_emitter_mutation_rate=float(args.qd_emitter_mutation_rate),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated_path = out_dir / "generated_gallery.png"
    reference_path = out_dir / "reference_gallery.png"
    scatter_path = out_dir / "descriptor_scatter.png"
    summary_path = out_dir / "summary.json"

    generated_meta = _save_gallery(
        generated,
        out_path=generated_path,
        title_prefix="Generated",
        max_graphs=int(args.num_show),
        seed=int(args.seed),
    )
    reference_meta = _save_gallery(
        references,
        out_path=reference_path,
        title_prefix="Reference",
        max_graphs=int(args.num_show),
        seed=int(args.seed) + 700,
    )
    scatter_stats = _save_descriptor_scatter(generated, references, scatter_path)

    summary = {
        "seed": int(args.seed),
        "num_generated": int(len(generated)),
        "num_reference_loaded": int(len(references)),
        "descriptor_targets": descriptor_targets,
        "scatter_means": scatter_stats,
        "files": {
            "generated_gallery": str(generated_path),
            "reference_gallery": str(reference_path),
            "descriptor_scatter": str(scatter_path),
        },
        "generated_gallery_metrics": generated_meta,
        "reference_gallery_metrics": reference_meta,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["files"], indent=2))
    print(f"Saved summary: {summary_path}")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
