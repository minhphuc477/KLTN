"""Generate report-facing Chapter 4 case-study figures and solver analysis."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.visualize_block_i_graphs import save_single_graph_figure
from src.evaluation.pcbs_validation import prepare_dungeon_grid_for_validation
from src.evaluation.search_benchmark_utils import path_transition_count
from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
from src.simulation.search_base import GameStateSearchConfig, SearchRepresentation
from src.simulation.search_factory import iter_game_state_algorithm_specs, run_game_state_solver
from src.simulation.validator import ZeldaLogicEnv

logger = logging.getLogger(__name__)

SOLVER_TITLE_MAP: Dict[str, str] = {
    "graph_guided_oracle": "Graph-Guided Oracle",
    "astar": "A*",
    "bfs": "BFS",
    "dijkstra": "Dijkstra",
    "greedy": "Greedy",
    "dstar_lite": "D* Lite",
    "dfs_iddfs": "DFS/IDDFS",
    "bidirectional_astar": "Bidirectional A*",
    "pcbs_balanced": "P-CBS (balanced)",
}

SOLVER_COLOR_MAP: Dict[str, str] = {
    "graph_guided_oracle": "#ef4444",
    "astar": "#2563eb",
    "greedy": "#f59e0b",
    "dstar_lite": "#8b5cf6",
    "bidirectional_astar": "#10b981",
    "pcbs_balanced": "#111827",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_case_context(case_dir: Path, variant_name: str) -> Dict[str, Any]:
    variant_dir = case_dir / variant_name
    mission_graph = json_graph.node_link_graph(_load_json(case_dir / "mission_graph.json"), edges="links")
    layout = _load_json(variant_dir / "room_layout.json")
    dungeon_grid = np.asarray(_load_json(variant_dir / "dungeon_grid_ids.json"), dtype=np.int32)
    prepared = prepare_dungeon_grid_for_validation(dungeon_grid)

    room_positions: Dict[Tuple[int, int], Tuple[int, int]] = {}
    room_to_node: Dict[Tuple[int, int], int] = {}
    node_to_room: Dict[int, Tuple[int, int]] = {}
    room_centers: Dict[int, Tuple[float, float]] = {}
    for room in layout.get("rooms", []):
        room_id = int(room["room_id"])
        slot_position = tuple(int(v) for v in room["slot_position"])
        room_offset = tuple(int(v) for v in room["room_offset"])
        center = tuple(float(v) for v in room["center"])
        room_positions[slot_position] = room_offset
        room_to_node[slot_position] = room_id
        node_to_room[room_id] = slot_position
        room_centers[room_id] = center

    return {
        "case_dir": case_dir,
        "variant_dir": variant_dir,
        "mission_graph": mission_graph,
        "layout": layout,
        "grid": prepared.grid,
        "room_positions": room_positions,
        "room_to_node": room_to_node,
        "node_to_room": node_to_room,
        "room_centers": room_centers,
        "stylized_image": variant_dir / "dungeon_grid_stylized.png",
        "branch_images": {
            "diffusion_cfg3_logic0_steps50": case_dir / "diffusion_cfg3_logic0_steps50" / "dungeon_grid_stylized.png",
            "fast_cfg3_logic0_steps4": case_dir / "fast_cfg3_logic0_steps4" / "dungeon_grid_stylized.png",
            "masked_room_full": case_dir / "masked_room_full" / "dungeon_grid_stylized.png",
        },
        "validation_summary": _load_json(variant_dir / "summary.json").get("validation", {}),
    }


def _env_kwargs(context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "render_mode": False,
        "graph": context["mission_graph"],
        "room_positions": context["room_positions"],
        "room_to_node": context["room_to_node"],
        "node_to_room": context["node_to_room"],
    }


def _run_tile_state_suite(
    context: Mapping[str, Any],
    *,
    context_name: str,
    timeout_states: int,
    pcbs_timeout: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "context_name": str(context_name),
        "tile_state_budget_states": int(timeout_states),
        "pcbs_timeout_states": int(pcbs_timeout),
        "solvers": {},
    }
    grid = np.asarray(context["grid"], dtype=np.int32)
    if str(context_name) == "full":
        env_kwargs = _env_kwargs(context)
    else:
        env_kwargs = {"render_mode": False}

    for spec in iter_game_state_algorithm_specs():
        env = ZeldaLogicEnv(grid, **env_kwargs)
        started = time.perf_counter()
        try:
            result = run_game_state_solver(
                env,
                algorithm_idx=int(spec.index),
                config=GameStateSearchConfig(
                    timeout=int(timeout_states),
                    allow_diagonals=False,
                    rules_profile="vglc_strict",
                    representation=SearchRepresentation.TILE,
                    max_depth=max(500, int(grid.size)),
                    use_iddfs=True,
                ),
            )
        finally:
            try:
                env.close()
            except Exception:
                logger.debug("Failed to close game-state solver environment.", exc_info=True)
        elapsed = float(time.perf_counter() - started)
        row = {
            "key": str(spec.key),
            "label": str(spec.label),
            "success": bool(result.success),
            "path_length": path_transition_count(result.path),
            "states_explored": int(result.states_explored or 0),
            "time_sec": elapsed,
            "metadata": dict(result.metadata or {}),
            "path": [[int(r), int(c)] for r, c in list(result.path or [])],
        }
        payload["solvers"][str(spec.key)] = row

    if str(context_name) == "full":
        env_cbs = ZeldaLogicEnv(grid, **_env_kwargs(context))
        try:
            started = time.perf_counter()
            cbs = CognitiveBoundedSearch(env_cbs, persona="balanced", timeout=int(pcbs_timeout), seed=123)
            success, path, states, metrics = cbs.solve()
            elapsed = float(time.perf_counter() - started)
        finally:
            try:
                env_cbs.close()
            except Exception:
                logger.debug("Failed to close P-CBS environment.", exc_info=True)
        payload["solvers"]["pcbs_balanced"] = {
            "key": "pcbs_balanced",
            "label": "P-CBS (balanced)",
            "success": bool(success),
            "path_length": path_transition_count(path),
            "states_explored": int(states or 0),
            "time_sec": elapsed,
            "path": [[int(r), int(c)] for r, c in list(path or [])],
            "metadata": {
                "confusion_index": float(getattr(metrics, "confusion_index", 0.0) or 0.0),
                "navigation_entropy": float(getattr(metrics, "navigation_entropy", 0.0) or 0.0),
                "room_entropy": float(getattr(metrics, "room_entropy", 0.0) or 0.0),
                "aha_latency": int(getattr(metrics, "aha_latency", 0) or 0),
                "goal_sighting_latency": int(getattr(metrics, "aha_latency", 0) or 0),
                "deliberation_events": int(getattr(metrics, "deliberation_events", 0) or 0),
                "peak_frustration": float(getattr(metrics, "peak_frustration", 0.0) or 0.0),
            },
        }

        validation = dict(context.get("validation_summary", {}) or {})
        graph_guided = dict(validation.get("graph_guided_oracle", {}) or {})
        graph_path = [int(node) for node in list(graph_guided.get("graph_path", []) or [])]
        payload["solvers"]["graph_guided_oracle"] = {
            "key": "graph_guided_oracle",
            "label": "Graph-guided oracle",
            "success": bool(graph_guided.get("solvable", False)),
            "path_length": path_transition_count(graph_path),
            "states_explored": int(graph_guided.get("room_validation_count", 0) or 0),
            "time_sec": 0.0,
            "path": graph_path,
            "metadata": {
                "connectivity_score": float(graph_guided.get("connectivity_score", 0.0) or 0.0),
                "room_traversable_count": int(graph_guided.get("room_traversable_count", 0) or 0),
            },
        }
    return payload


def _write_solver_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    fieldnames = [
        "context",
        "key",
        "label",
        "success",
        "path_length",
        "states_explored",
        "time_sec",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _solver_note(entry: Mapping[str, Any]) -> str:
    metadata = dict(entry.get("metadata", {}) or {})
    for key in ("failure_reason", "solver_status", "oracle_status"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if bool(metadata.get("fallback_used")):
        return "Fallback to A* engaged."
    if entry.get("key") == "graph_guided_oracle":
        return f"room path={entry.get('path', [])}"
    if entry.get("key") == "pcbs_balanced":
        return f"confusion_index={float(metadata.get('confusion_index', 0.0)):.3f}"
    return ""


def _tile_to_image_points(path: Sequence[Sequence[int]], image_shape: Sequence[int], grid_shape: Sequence[int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 2), dtype=np.float64)
    img_h, img_w = float(image_shape[0]), float(image_shape[1])
    grid_h, grid_w = float(grid_shape[0]), float(grid_shape[1])
    points = []
    for row, col in path:
        x = (float(col) + 0.5) * img_w / grid_w
        y = (float(row) + 0.5) * img_h / grid_h
        points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def _room_path_to_image_points(path: Sequence[int], room_centers: Mapping[int, Sequence[float]], image_shape: Sequence[int], grid_shape: Sequence[int]) -> np.ndarray:
    if not path:
        return np.zeros((0, 2), dtype=np.float64)
    img_h, img_w = float(image_shape[0]), float(image_shape[1])
    grid_h, grid_w = float(grid_shape[0]), float(grid_shape[1])
    points = []
    for room_id in path:
        center = room_centers.get(int(room_id))
        if not center or len(center) < 2:
            continue
        row, col = float(center[0]), float(center[1])
        x = (col + 0.5) * img_w / grid_w
        y = (row + 0.5) * img_h / grid_h
        points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def _draw_overlay_panel(
    ax: plt.Axes,
    background: np.ndarray,
    *,
    points: np.ndarray,
    color: str,
    title: str,
    stats_text: str,
    draw_room_centers: bool = False,
) -> None:
    ax.imshow(background)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")
    if points.shape[0] > 0:
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=3.0, alpha=0.92, solid_capstyle="round")
        ax.scatter(points[0, 0], points[0, 1], s=55, c="#22c55e", edgecolors="white", linewidths=1.2, zorder=5)
        ax.scatter(points[-1, 0], points[-1, 1], s=75, c="#facc15", marker="*", edgecolors="#111827", linewidths=0.8, zorder=5)
        if draw_room_centers:
            ax.scatter(points[:, 0], points[:, 1], s=26, c=color, edgecolors="white", linewidths=0.8, zorder=4)
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        color="#111827",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "alpha": 0.88, "edgecolor": "#d1d5db"},
    )


def _build_solver_figure(
    context: Mapping[str, Any],
    solver_payload: Mapping[str, Any],
    out_path: Path,
) -> None:
    background = mpimg.imread(str(context["stylized_image"]))
    grid_shape = np.asarray(context["grid"]).shape
    contexts = dict(solver_payload.get("contexts", {}) or {})
    full_map = dict(dict(contexts.get("full", {}) or {}).get("solvers", {}) or {})
    grid_only_map = dict(dict(contexts.get("grid_only", {}) or {}).get("solvers", {}) or {})

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.6), dpi=220)
    axes_flat = list(axes.flatten())

    base_ax = axes_flat[0]
    base_ax.imshow(background)
    base_ax.set_title("Reference Dungeon", fontsize=11, pad=8)
    base_ax.axis("off")
    base_ax.text(
        0.02,
        0.02,
        "Diffusion branch selected for solver comparison",
        transform=base_ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#ffffff", "alpha": 0.88, "edgecolor": "#d1d5db"},
    )

    panel_specs = [
        ("full", "graph_guided_oracle", "Graph-guided oracle"),
        ("full", "bidirectional_astar", "Bidirectional A*"),
        ("full", "pcbs_balanced", "P-CBS (balanced)"),
        ("grid_only", "astar", "A* (grid-only diagnostic)"),
        ("grid_only", "greedy", "Greedy (grid-only diagnostic)"),
    ]

    for ax, (context_name, solver_key, display_title) in zip(axes_flat[1:], panel_specs):
        solver_map = full_map if context_name == "full" else grid_only_map
        entry = dict(solver_map.get(solver_key, {}) or {})
        success = bool(entry.get("success", False))
        color = SOLVER_COLOR_MAP.get(solver_key, "#111827")
        if solver_key == "graph_guided_oracle":
            points = _room_path_to_image_points(
                entry.get("path", []),
                context["room_centers"],
                background.shape,
                grid_shape,
            )
            stats = (
                f"solved={int(success)}  room-path={int(entry.get('path_length', 0))}\n"
                f"coverage={float(dict(entry.get('metadata', {}) or {}).get('connectivity_score', 0.0)):.2f}"
            )
            _draw_overlay_panel(
                ax,
                background,
                points=points,
                color=color,
                title=display_title,
                stats_text=stats,
                draw_room_centers=True,
            )
            continue

        points = _tile_to_image_points(entry.get("path", []), background.shape, grid_shape)
        metadata = dict(entry.get("metadata", {}) or {})
        if solver_key == "pcbs_balanced":
            stats = (
                f"solved={int(success)}  path={int(entry.get('path_length', 0))}  states={int(entry.get('states_explored', 0))}\n"
                f"CGR={float(metadata.get('confusion_index', 0.0)):.3f}  Hnav={float(metadata.get('navigation_entropy', 0.0)):.3f}"
            )
        else:
            stats = (
                f"solved={int(success)}  path={int(entry.get('path_length', 0))}  states={int(entry.get('states_explored', 0))}\n"
                f"time={float(entry.get('time_sec', 0.0)):.2f}s"
            )
        _draw_overlay_panel(
            ax,
            background,
            points=points,
            color=color,
            title=display_title,
            stats_text=stats,
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _build_end_to_end_case_figure(context: Mapping[str, Any], out_path: Path) -> None:
    report_fig_dir = out_path.parent
    mission_graph_path = report_fig_dir / "generated_topology_seed20260418_graph.png"
    save_single_graph_figure(
        context["mission_graph"],
        mission_graph_path,
        seed=20260418,
        title="",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), dpi=220)
    panels = [
        (mission_graph_path, "(a) Mission graph Block I"),
        (context["branch_images"]["diffusion_cfg3_logic0_steps50"], "(b) Diffusion branch (selected)"),
        (context["branch_images"]["fast_cfg3_logic0_steps4"], "(c) Fast-sampler branch"),
        (context["branch_images"]["masked_room_full"], "(d) Masked-room branch"),
    ]
    for ax, (image_path, label_text) in zip(list(axes.flatten()), panels):
        ax.imshow(mpimg.imread(str(image_path)))
        ax.axis("off")
        ax.set_title(label_text, fontsize=11, pad=8)
        if "selected" in label_text:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#2563eb")
                spine.set_linewidth(2.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _sorted_solver_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    preferred_order = {
        "graph_guided_oracle": 0,
        "astar": 1,
        "bfs": 2,
        "dijkstra": 3,
        "greedy": 4,
        "dstar_lite": 5,
        "dfs_iddfs": 6,
        "bidirectional_astar": 7,
        "pcbs_balanced": 8,
    }
    rows: List[Dict[str, Any]] = []
    for context_name, context_payload in dict(payload.get("contexts", {}) or {}).items():
        for key, entry in dict(dict(context_payload or {}).get("solvers", {}) or {}).items():
            row = {
                "context": str(context_name),
                "key": key,
                "label": entry.get("label", SOLVER_TITLE_MAP.get(key, key)),
                "success": int(bool(entry.get("success", False))),
                "path_length": int(entry.get("path_length", 0) or 0),
                "states_explored": int(entry.get("states_explored", 0) or 0),
                "time_sec": round(float(entry.get("time_sec", 0.0) or 0.0), 4),
                "notes": _solver_note({**entry, "key": key}),
            }
            rows.append(row)
    context_order = {"full": 0, "grid_only": 1}
    rows.sort(key=lambda item: (context_order.get(str(item["context"]), 999), preferred_order.get(str(item["key"]), 999)))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 case-study figures and solver overlays.")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=ROOT / "results" / "ch4_generated_topology_real_pdrop035_seed20260418_fixedvalidator",
    )
    parser.add_argument(
        "--variant-name",
        type=str,
        default="diffusion_cfg3_logic0_steps50",
    )
    parser.add_argument(
        "--solver-timeout-states",
        type=int,
        default=250000,
    )
    parser.add_argument(
        "--pcbs-timeout-states",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--report-fig-dir",
        type=Path,
        default=ROOT / "REPORT_LATEX" / "figures" / "ch4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    context = _load_case_context(args.case_dir, args.variant_name)
    solver_payload = {
        "contexts": {
            "full": _run_tile_state_suite(
                context,
                context_name="full",
                timeout_states=int(args.solver_timeout_states),
                pcbs_timeout=int(args.pcbs_timeout_states),
            ),
            "grid_only": _run_tile_state_suite(
                context,
                context_name="grid_only",
                timeout_states=int(args.solver_timeout_states),
                pcbs_timeout=int(args.pcbs_timeout_states),
            ),
        }
    }

    json_path = args.case_dir / f"solver_case_study_budget{int(args.solver_timeout_states)}.json"
    csv_path = args.case_dir / f"solver_case_study_budget{int(args.solver_timeout_states)}.csv"
    _write_json(json_path, solver_payload)
    _write_solver_csv(csv_path, _sorted_solver_rows(solver_payload))

    _build_end_to_end_case_figure(context, args.report_fig_dir / "end_to_end_case_figure.png")
    _build_solver_figure(context, solver_payload, args.report_fig_dir / "solver_path_comparison.png")

    print(json.dumps({
        "solver_json": str(json_path),
        "solver_csv": str(csv_path),
        "case_figure": str(args.report_fig_dir / "end_to_end_case_figure.png"),
        "solver_figure": str(args.report_fig_dir / "solver_path_comparison.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
