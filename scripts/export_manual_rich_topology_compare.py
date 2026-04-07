"""
Export and compare all room-generation branches on one rich hand-authored topology.

This gives us a precise, repeatable testbed for room generation quality without
mixing in Block I topology variability.
"""

from __future__ import annotations

import argparse
import copy
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
    export_variant,
    generation_overrides_from_namespace,
)


def build_manual_rich_topology_graph() -> nx.Graph:
    """
    Build a rich, explicitly positioned mission graph for controlled room testing.

    Every edge is unit-Manhattan adjacent in the supplied `pos`, so strict room
    placement is feasible and we avoid conflating room quality with relaxed
    topology stitching artifacts.
    """
    graph = nx.Graph()
    graph.graph.update(
        {
            "style_id": 1,
            "name": "manual_rich_topology_v1",
            "description": (
                "Hand-authored rich topology for room-branch comparison: "
                "start, enemy, key, big-key, bomb resource, puzzle, item, stair, arena, boss door, boss, goal."
            ),
        }
    )

    nodes: List[Dict[str, Any]] = [
        {"id": 0, "label": "START", "type": "START", "pos": (0, 0), "is_start": True},
        {"id": 1, "label": "ENEMY", "type": "ENEMY", "pos": (0, 1), "has_enemy": True, "enemy_count": 2},
        {"id": 2, "label": "KEY", "type": "KEY", "pos": (0, 2), "has_key": True, "key_id": 1},
        {"id": 3, "label": "BIG_KEY", "type": "BIG_KEY", "pos": (0, 3), "has_key": True, "key_id": 2},
        {"id": 4, "label": "RESOURCE_FARM", "type": "RESOURCE_FARM", "pos": (1, 0), "drops_resource": "BOMB"},
        {"id": 5, "label": "PUZZLE", "type": "PUZZLE", "pos": (1, 1), "has_puzzle": True},
        {"id": 6, "label": "ITEM", "type": "ITEM", "pos": (1, 2), "has_item": True, "item_type": "BOMB"},
        {"id": 7, "label": "BOSS_DOOR", "type": "BOSS_DOOR", "pos": (1, 3), "key_id": 2},
        {"id": 8, "label": "STAIR", "type": "STAIR", "pos": (2, 1)},
        {"id": 9, "label": "ARENA", "type": "ARENA", "pos": (2, 2), "has_enemy": True, "is_arena": True, "enemy_count": 3},
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
    _add_edge(9, 10, "path")
    _add_edge(7, 10, "boss_locked", key_required=2)
    _add_edge(10, 11, "path")

    return graph


def _load_mission_graph(path: Path) -> nx.Graph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return json_graph.node_link_graph(payload, edges="links")


def _write_graph_summary(graph: nx.Graph, out_dir: Path) -> None:
    summary = {
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
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


def _save_manual_graph_layout_png(graph: nx.Graph, out_path: Path) -> None:
    node_palette = {
        "START": "#2ca02c",
        "ENEMY": "#ff7f0e",
        "KEY": "#1f77b4",
        "BIG_KEY": "#4e79a7",
        "RESOURCE_FARM": "#8c564b",
        "PUZZLE": "#9467bd",
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


def _pairwise_compare(output_dir: Path) -> Dict[str, str]:
    variants = {
        "diffusion": output_dir / "diffusion_cfg3_logic0_steps50",
        "fast_sampler": output_dir / "fast_cfg3_logic0_steps4",
        "masked_room": output_dir / "masked_room_full",
    }
    pairs = {
        "diffusion_vs_fast_sampler": ("diffusion", "fast_sampler"),
        "diffusion_vs_masked_room": ("diffusion", "masked_room"),
        "fast_sampler_vs_masked_room": ("fast_sampler", "masked_room"),
    }
    outputs: Dict[str, str] = {}
    for pair_name, (baseline_name, candidate_name) in pairs.items():
        summary_path = compare_variant_dirs(
            variants[baseline_name],
            variants[candidate_name],
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
        return (
            f"- `{name}`: gen_time={float(metrics.get('generation_time_sec', 0.0)):.2f}s, "
            f"repair_rate={float(metrics.get('repair_rate', 0.0)):.3f}, "
            f"total_tiles_repaired={int(metrics.get('total_tiles_repaired', 0))}"
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
        _metric_line("diffusion_cfg3_logic0_steps50", summaries["diffusion_cfg3_logic0_steps50"]),
        _metric_line("fast_cfg3_logic0_steps4", summaries["fast_cfg3_logic0_steps4"]),
        _metric_line("masked_room_full", summaries["masked_room_full"]),
        "",
        "## Pairwise Room Diff Audits",
        "",
        *(f"- `{name}`: `{path}`" for name, path in comparison_outputs.items()),
        "",
        "## Key Artifacts",
        "",
        f"- mission graph: `{output_dir / 'mission_graph.json'}`",
        f"- mission graph PNG: `{output_dir / 'mission_graph_layout.png'}`",
        f"- graph summary: `{output_dir / 'graph_summary.json'}`",
        f"- overall summary: `{output_dir / 'summary.json'}`",
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
    add_generation_override_args(parser)
    return parser.parse_args()


def run_from_args(args: argparse.Namespace) -> Dict[str, str]:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir}")

    graph = _load_mission_graph(args.mission_graph) if args.mission_graph is not None else build_manual_rich_topology_graph()
    generation_overrides = generation_overrides_from_namespace(args)
    graph_source = str(args.mission_graph) if args.mission_graph is not None else "built_in_manual_rich_topology"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mission_graph.json").write_text(
        json.dumps(json_graph.node_link_data(graph, edges="links"), indent=2),
        encoding="utf-8",
    )
    _write_graph_summary(graph, args.output_dir)
    _save_manual_graph_layout_png(graph, args.output_dir / "mission_graph_layout.png")
    print(
        f"[topology-compare-manual] run_dir={run_dir} graph={graph_source} "
        f"seed={int(args.seed)} overrides={generation_overrides}",
        flush=True,
    )

    summaries = {
        "diffusion_cfg3_logic0_steps50": export_variant(
            run_dir=run_dir,
            mission_graph=copy.deepcopy(graph),
            variant_name="diffusion_cfg3_logic0_steps50",
            out_dir=args.output_dir,
            guidance_scale=3.0,
            logic_guidance_scale=0.0,
            num_diffusion_steps=50,
            use_fast_sampling=False,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        ),
        "fast_cfg3_logic0_steps4": export_variant(
            run_dir=run_dir,
            mission_graph=copy.deepcopy(graph),
            variant_name="fast_cfg3_logic0_steps4",
            out_dir=args.output_dir,
            guidance_scale=3.0,
            logic_guidance_scale=0.0,
            num_diffusion_steps=4,
            use_fast_sampling=True,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        ),
        "masked_room_full": export_masked_variant(
            run_dir=run_dir,
            mission_graph=copy.deepcopy(graph),
            variant_name="masked_room_full",
            out_dir=args.output_dir,
            seed=int(args.seed),
            generation_overrides=generation_overrides,
        ),
    }
    summary_payload = {
        "generation_overrides": generation_overrides,
        "variants": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    _build_comparison_sheet(
        variant_paths={
            "diffusion": args.output_dir / "diffusion_cfg3_logic0_steps50" / "dungeon_grid_alignment.png",
            "fast_sampler": args.output_dir / "fast_cfg3_logic0_steps4" / "dungeon_grid_alignment.png",
            "masked_room": args.output_dir / "masked_room_full" / "dungeon_grid_alignment.png",
        },
        filename="dungeon_grid_alignment.png",
        out_path=args.output_dir / "dungeon_alignment_comparison.png",
    )
    _build_comparison_sheet(
        variant_paths={
            "diffusion": args.output_dir / "diffusion_cfg3_logic0_steps50" / "rooms_sheet_stylized.png",
            "fast_sampler": args.output_dir / "fast_cfg3_logic0_steps4" / "rooms_sheet_stylized.png",
            "masked_room": args.output_dir / "masked_room_full" / "rooms_sheet_stylized.png",
        },
        filename="rooms_sheet_stylized.png",
        out_path=args.output_dir / "rooms_sheet_comparison.png",
    )

    comparison_outputs = _pairwise_compare(args.output_dir)
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
