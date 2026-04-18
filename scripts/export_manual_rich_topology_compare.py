"""
Export and compare all room-generation branches on one rich hand-authored topology.

This gives us a precise, repeatable testbed for room generation quality without
mixing in Block I topology variability.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw
from networkx.readwrite import json_graph


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_room_variants import compare_variant_dirs
from scripts.export_semantic_anchor_end_to_end import export_masked_variant
from scripts.run_fast_sampler_visual_audit import (
    add_generation_override_args,
    build_validation_search_stats_payload,
    export_variant,
    generation_overrides_from_namespace,
)


VARIANT_LABELS: Dict[str, str] = {
    "diffusion_cfg3_logic0_steps50": "diffusion",
    "fast_cfg3_logic0_steps4": "fast_sampler",
    "masked_room_full": "masked_room",
}


def _ensure_directed_progression_graph(graph: nx.Graph, *, source: str) -> nx.DiGraph:
    """
    Require directed mission graphs for progression-sensitive audits.

    Research and prior bugs in this repo both point in the same direction:
    topology semantics like ITEM -> TUTORIAL -> COMBAT -> COMPLEX -> GOAL are
    directional. Quietly accepting undirected graphs weakens those semantics and
    makes the audit lie about what Block I and the room models are doing.
    """
    if graph.is_directed():
        return nx.DiGraph(graph)
    raise ValueError(
        f"{source} must be a directed mission graph (nx.DiGraph). "
        "Undirected graphs flatten progression semantics and are not valid for "
        "manual topology comparison or fixed-graph audits."
    )


def build_manual_rich_topology_graph() -> nx.DiGraph:
    """
    Build a rich, explicitly positioned mission graph for controlled room testing.

    Every edge is unit-Manhattan adjacent in the supplied `pos`, so strict room
    placement is feasible and we avoid conflating room quality with relaxed
    topology stitching artifacts.
    """
    graph = nx.DiGraph()
    graph.graph.update(
        {
            "style_id": 1,
            "name": "manual_rich_topology_v4",
            "description": (
                "Hand-authored rich topology for room-branch comparison: "
                "start, enemy, key, big-key, bomb resource, complex puzzle, item, stair, combat puzzle, boss door, boss, goal. "
                "The boss gauntlet is strictly normalized so the boss room has only the boss door as predecessor, "
                "and the progression graph is now a clean DAG so room-layout batching is not degraded by a synthetic cycle."
            ),
        }
    )

    nodes: List[Dict[str, Any]] = [
        {"id": 0, "label": "START", "type": "START", "pos": (0, 0), "is_start": True},
        {"id": 1, "label": "ENEMY", "type": "ENEMY", "pos": (0, 1), "has_enemy": True, "enemy_count": 2},
        {"id": 2, "label": "KEY", "type": "KEY", "pos": (0, 2), "has_key": True, "key_id": 1},
        {"id": 3, "label": "BIG_KEY", "type": "BIG_KEY", "pos": (0, 3), "has_boss_key": True, "key_id": 2},
        {"id": 4, "label": "RESOURCE_FARM", "type": "RESOURCE_FARM", "pos": (1, 0), "drops_resource": "BOMB"},
        {"id": 5, "label": "COMPLEX_PUZZLE", "type": "COMPLEX_PUZZLE", "pos": (1, 1), "has_puzzle": True, "difficulty_rating": "HARD"},
        {"id": 6, "label": "ITEM", "type": "ITEM", "pos": (1, 2), "has_item": True, "item_type": "BOMB"},
        {"id": 7, "label": "BOSS_DOOR", "type": "BOSS_DOOR", "pos": (1, 3), "key_id": 2},
        {"id": 8, "label": "STAIR", "type": "STAIR", "pos": (2, 1)},
        {"id": 9, "label": "COMBAT_PUZZLE", "type": "COMBAT_PUZZLE", "pos": (2, 2), "has_enemy": True, "has_puzzle": True, "enemy_count": 3, "difficulty_rating": "MODERATE"},
        {"id": 10, "label": "BOSS", "type": "BOSS", "pos": (2, 3), "has_boss": True, "has_enemy": True, "enemy_count": 1},
        {"id": 11, "label": "GOAL", "type": "GOAL", "pos": (3, 3), "is_goal": True, "has_triforce": True},
    ]

    for node in nodes:
        attrs = dict(node)
        node_id = attrs.pop("id")
        graph.add_node(node_id, **attrs)

    def _add_edge(src: int, dst: int, label: str, *, key_required: int | None = None, item_required: str | None = None) -> None:
        graph.add_edge(
            int(src),
            int(dst),
            label=str(label).lower(),
            edge_type=str(label).upper(),
            key_required=key_required,
            item_required=item_required,
        )

    _add_edge(0, 1, "path")
    _add_edge(1, 2, "path")
    _add_edge(2, 3, "path")
    _add_edge(0, 4, "path")
    _add_edge(4, 5, "bombable", item_required="BOMB")
    _add_edge(1, 5, "key_locked", key_required=1)
    _add_edge(5, 6, "switch_locked")
    _add_edge(3, 7, "path")
    _add_edge(6, 7, "path")
    _add_edge(5, 8, "path")
    _add_edge(8, 9, "path")
    _add_edge(6, 9, "item_gate", item_required="BOMB")
    _add_edge(7, 10, "boss_locked", key_required=2)
    _add_edge(10, 11, "path")

    return _ensure_directed_progression_graph(graph, source="built_in_manual_rich_topology")


def _load_mission_graph(path: Path) -> nx.DiGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    graph = json_graph.node_link_graph(payload, edges="links")
    return _ensure_directed_progression_graph(graph, source=str(path))


def _write_graph_summary(graph: nx.Graph, out_dir: Path) -> None:
    summary = {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "is_directed": bool(graph.is_directed()),
        "is_dag": bool(nx.is_directed_acyclic_graph(graph)) if graph.is_directed() else False,
        "nodes": [
            {
                "id": int(node_id),
                "label": str(data.get("label", "")),
                "type": str(data.get("type", "")),
                "pos": list(data.get("pos", ())),
                "has_enemy": bool(data.get("has_enemy", False)),
                "has_key": bool(data.get("has_key", False)),
                "has_item": bool(data.get("has_item", False)),
                "has_puzzle": bool(data.get("has_puzzle", False)),
                "has_boss": bool(data.get("has_boss", False)),
                "is_goal": bool(data.get("is_goal", False) or data.get("has_triforce", False)),
            }
            for node_id, data in sorted(graph.nodes(data=True), key=lambda item: int(item[0]))
        ],
        "edges": [
            {
                "source": int(u),
                "target": int(v),
                "label": str(data.get("label", "")),
                "edge_type": str(data.get("edge_type", "")),
                "key_required": data.get("key_required"),
                "item_required": data.get("item_required"),
            }
            for u, v, data in sorted(graph.edges(data=True), key=lambda item: (int(item[0]), int(item[1])))
        ],
    }
    (out_dir / "graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_search_algorithm_comparison_payload(summaries: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for variant_name, summary in summaries.items():
        payload[str(variant_name)] = build_validation_search_stats_payload(summary.get("validation", {}))
    return payload


def _save_manual_graph_layout_png(graph: nx.Graph, out_path: Path) -> None:
    node_palette = {
        "START": "#2ca02c",
        "ENEMY": "#ff7f0e",
        "KEY": "#1f77b4",
        "BIG_KEY": "#4e79a7",
        "RESOURCE_FARM": "#8c564b",
        "PUZZLE": "#9467bd",
        "TUTORIAL_PUZZLE": "#8b5cf6",
        "COMBAT_PUZZLE": "#ec4899",
        "COMPLEX_PUZZLE": "#7c3aed",
        "ITEM": "#17becf",
        "BOSS_DOOR": "#d62728",
        "STAIR": "#bcbd22",
        "ARENA": "#e15759",
        "BOSS": "#ad494a",
        "GOAL": "#d4a017",
    }
    edge_palette = {
        "PATH": ("#9aa0a6", "solid"),
        "KEY_LOCKED": ("#1f77b4", "dashed"),
        "BOMBABLE": ("#8c564b", "dashdot"),
        "SWITCH_LOCKED": ("#9467bd", "dashed"),
        "ITEM_GATE": ("#17becf", "dashdot"),
        "BOSS_LOCKED": ("#d62728", "solid"),
    }

    pos: Dict[Any, Any] = {}
    for node_id, data in graph.nodes(data=True):
        raw = data.get("pos")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            row, col = int(raw[0]), int(raw[1])
            pos[node_id] = (float(col), float(-row))
    if len(pos) != graph.number_of_nodes():
        pos = nx.spring_layout(graph, seed=20260406)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title("Manual Rich Topology Graph", fontsize=14)
    ax.set_axis_off()

    for edge_type, (color, style) in edge_palette.items():
        edges = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if str(data.get("edge_type", "")).upper() == edge_type
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=edges,
                edge_color=color,
                style=style,
                width=2.2,
            )

    for node_type, color in node_palette.items():
        nodes = [
            node_id
            for node_id, data in graph.nodes(data=True)
            if str(data.get("type", "")).upper() == node_type
        ]
        if nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=ax,
                nodelist=nodes,
                node_color=color,
                edgecolors="#0f172a",
                linewidths=1.2,
                node_size=1050,
            )

    labels = {
        node_id: f"{node_id}:{str(data.get('label', data.get('type', 'ROOM')))}"
        for node_id, data in graph.nodes(data=True)
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight="bold", ax=ax)

    edge_labels = {
        (u, v): str(data.get("edge_type", data.get("label", "")))
        for u, v, data in graph.edges(data=True)
    }
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_comparison_sheet(
    *,
    variant_paths: Dict[str, Path],
    filename: str,
    out_path: Path,
) -> None:
    label_h = 28
    margin = 12
    opened: List[tuple[str, Image.Image]] = []
    try:
        for label, path in variant_paths.items():
            opened.append((label, Image.open(path).convert("RGB")))

        max_h = max(image.height for _, image in opened)
        total_w = sum(image.width for _, image in opened) + margin * (len(opened) + 1)
        canvas = Image.new("RGB", (total_w, max_h + label_h + 2 * margin), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        x = margin
        for label, image in opened:
            draw.text((x, margin // 2), label, fill=(15, 23, 42))
            canvas.paste(image, (x, label_h))
            draw.rectangle(
                [x - 1, label_h - 1, x + image.width, label_h + image.height],
                outline=(148, 163, 184),
                width=1,
            )
            x += image.width + margin

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
    finally:
        for _, image in opened:
            image.close()


def _pairwise_compare(output_dir: Path, variant_names: List[str]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    labels = {name: VARIANT_LABELS.get(name, str(name)) for name in variant_names}
    for baseline_name, candidate_name in itertools.combinations(variant_names, 2):
        pair_name = f"{labels[baseline_name]}_vs_{labels[candidate_name]}"
        summary_path = compare_variant_dirs(
            output_dir / baseline_name,
            output_dir / candidate_name,
            output_dir / "comparisons" / pair_name,
        )
        outputs[pair_name] = str(summary_path)
    return outputs


def _write_report(
    *,
    output_dir: Path,
    summaries: Dict[str, Any],
    comparison_outputs: Dict[str, str],
    generation_overrides: Dict[str, Any] | None = None,
) -> None:
    def _metric_line(name: str, payload: Dict[str, Any]) -> str:
        metrics = payload.get("metrics", {})
        layout = payload.get("layout", {})
        validation = payload.get("validation", {})
        astar_grid = validation.get("astar_grid", {}) if isinstance(validation, dict) else {}
        softlock = validation.get("softlock_check", {}) if isinstance(validation, dict) else {}
        cbs_balanced = validation.get("cbs_balanced", {}) if isinstance(validation, dict) else {}
        layout_metric_name = str(layout.get("primary_quality_metric_name") or "layout_quality")
        layout_metric_value = float(layout.get("primary_quality_metric_value", 0.0) or 0.0)
        return (
            f"- `{name}`: gen_time={float(metrics.get('generation_time_sec', 0.0)):.2f}s, "
            f"repair_rate={float(metrics.get('repair_rate', 0.0)):.3f}, "
            f"total_tiles_repaired={int(metrics.get('total_tiles_repaired', 0))}, "
            f"overwrite={float(metrics.get('avg_final_graph_marker_overwrite_rate', 0.0)):.3f}, "
            f"post_overlay_anchor_error={float(metrics.get('avg_final_post_overlay_semantic_anchor_error', 0.0)):.3f}, "
            f"{layout_metric_name}={layout_metric_value:.3f}, "
            f"astar_solvable={bool(astar_grid.get('solvable', False))}, "
            f"softlock_safe={bool(softlock.get('is_safe', False))}, "
            f"cbs_success={bool(cbs_balanced.get('success', False))}, "
            f"cbs_confusion_ratio={float(cbs_balanced.get('confusion_ratio_vs_astar', float('inf'))):.3f}"
        )

    pairwise_lines = (
        [f"- `{name}`: `{path}`" for name, path in comparison_outputs.items()]
        if comparison_outputs
        else ["- none"]
    )
    lines = [
        "# Manual Rich Topology Comparison",
        "",
        "This export uses one hand-authored topology with explicit semantic content and fixed room positions.",
        "That makes branch comparison much more precise than comparing on changing Block I topologies.",
        "",
        "## Runtime Overrides",
        "",
        (
            f"- `{json.dumps(generation_overrides, sort_keys=True)}`"
            if generation_overrides
            else "- none"
        ),
        "",
        "## Variants",
        "",
        *(_metric_line(name, summaries[name]) for name in summaries.keys()),
        "",
        "## Pairwise Room Diff Audits",
        "",
        *pairwise_lines,
        "",
        "## Key Artifacts",
        "",
        f"- mission graph: `{output_dir / 'mission_graph.json'}`",
        f"- mission graph PNG: `{output_dir / 'mission_graph_layout.png'}`",
        f"- graph summary: `{output_dir / 'graph_summary.json'}`",
        f"- overall summary: `{output_dir / 'summary.json'}`",
        f"- search algorithm comparison: `{output_dir / 'search_algorithm_comparison.json'}`",
        f"- dungeon alignment comparison: `{output_dir / 'dungeon_alignment_comparison.png'}`",
        f"- rooms comparison: `{output_dir / 'rooms_sheet_comparison.png'}`",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export all generation branches on one rich hand-authored topology.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mission-graph",
        type=Path,
        default=None,
        help="Optional path to a user-authored mission_graph.json. If omitted, the built-in rich topology template is used.",
    )
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument(
        "--variants",
        type=str,
        default="diffusion_cfg3_logic0_steps50,fast_cfg3_logic0_steps4,masked_room_full",
        help="Comma-separated subset of variants to export. Supports resume-friendly one-branch runs.",
    )
    parser.add_argument(
        "--reuse-existing-variants",
        action="store_true",
        help="Reuse existing per-variant summary.json files instead of regenerating those variants.",
    )
    add_generation_override_args(parser)
    return parser.parse_args()


def _parse_variant_names(raw: str) -> List[str]:
    requested = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not requested:
        raise ValueError("--variants resolved to an empty set.")
    unsupported = [name for name in requested if name not in VARIANT_LABELS]
    if unsupported:
        supported = ", ".join(sorted(VARIANT_LABELS))
        raise ValueError(f"Unsupported manual-compare variant(s): {unsupported}. Supported: {supported}")
    ordered_unique: List[str] = []
    seen = set()
    for name in requested:
        if name not in seen:
            ordered_unique.append(name)
            seen.add(name)
    return ordered_unique


def _load_variant_summary(output_dir: Path, variant_name: str) -> Dict[str, Any]:
    summary_path = output_dir / str(variant_name) / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_from_args(args: argparse.Namespace) -> Dict[str, str]:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    graph = _load_mission_graph(args.mission_graph) if args.mission_graph is not None else build_manual_rich_topology_graph()
    generation_overrides = generation_overrides_from_namespace(args)
    graph_source = str(args.mission_graph) if args.mission_graph is not None else "built_in_manual_rich_topology"
    variant_names = _parse_variant_names(args.variants)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(json_graph.node_link_data(graph, edges="links"), indent=2),
        encoding="utf-8",
    )
    _write_graph_summary(graph, args.output_dir)
    _save_manual_graph_layout_png(graph, args.output_dir / "mission_graph_layout.png")
    print(
        f"[topology-compare-manual] run_dir={run_dir} graph={graph_source} "
        f"seed={int(args.seed)} variants={variant_names} overrides={generation_overrides}",
        flush=True,
    )

    summaries: Dict[str, Any] = {}
    for variant_name in variant_names:
        summary_path = args.output_dir / variant_name / "summary.json"
        if bool(args.reuse_existing_variants) and summary_path.exists():
            print(f"[topology-compare-manual] reusing {summary_path}", flush=True)
            summaries[variant_name] = _load_variant_summary(args.output_dir, variant_name)
            continue
        if variant_name == "diffusion_cfg3_logic0_steps50":
            summaries[variant_name] = export_variant(
                run_dir=run_dir,
                mission_graph=copy.deepcopy(graph),
                variant_name=variant_name,
                out_dir=args.output_dir,
                guidance_scale=3.0,
                logic_guidance_scale=0.0,
                num_diffusion_steps=50,
                use_fast_sampling=False,
                seed=int(args.seed),
                generation_overrides=generation_overrides,
            )
        elif variant_name == "fast_cfg3_logic0_steps4":
            summaries[variant_name] = export_variant(
                run_dir=run_dir,
                mission_graph=copy.deepcopy(graph),
                variant_name=variant_name,
                out_dir=args.output_dir,
                guidance_scale=3.0,
                logic_guidance_scale=0.0,
                num_diffusion_steps=4,
                use_fast_sampling=True,
                seed=int(args.seed),
                generation_overrides=generation_overrides,
            )
        elif variant_name == "masked_room_full":
            summaries[variant_name] = export_masked_variant(
                run_dir=run_dir,
                mission_graph=copy.deepcopy(graph),
                variant_name=variant_name,
                out_dir=args.output_dir,
                seed=int(args.seed),
                generation_overrides=generation_overrides,
            )
        else:
            raise ValueError(f"Unhandled variant: {variant_name}")
    summary_payload = {
        "generation_overrides": generation_overrides,
        "variants": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    search_algorithm_payload = _build_search_algorithm_comparison_payload(summaries)
    (args.output_dir / "search_algorithm_comparison.json").write_text(
        json.dumps(search_algorithm_payload, indent=2),
        encoding="utf-8",
    )

    if len(variant_names) >= 2:
        _build_comparison_sheet(
            variant_paths={
                VARIANT_LABELS[name]: args.output_dir / name / "dungeon_grid_alignment.png"
                for name in variant_names
            },
            filename="dungeon_grid_alignment.png",
            out_path=args.output_dir / "dungeon_alignment_comparison.png",
        )
        _build_comparison_sheet(
            variant_paths={
                VARIANT_LABELS[name]: args.output_dir / name / "rooms_sheet_stylized.png"
                for name in variant_names
            },
            filename="rooms_sheet_stylized.png",
            out_path=args.output_dir / "rooms_sheet_comparison.png",
        )

    comparison_outputs = _pairwise_compare(args.output_dir, variant_names) if len(variant_names) >= 2 else {}
    _write_report(
        output_dir=args.output_dir,
        summaries=summaries,
        comparison_outputs=comparison_outputs,
        generation_overrides=generation_overrides,
    )
    print(f"[topology-compare-manual] wrote {args.output_dir / 'summary.json'}", flush=True)

    return {
        "output_dir": str(args.output_dir),
        "summary": str(args.output_dir / "summary.json"),
        "report": str(args.output_dir / "report.md"),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(run_from_args(args), indent=2))


if __name__ == "__main__":
    main()
