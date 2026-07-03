"""Adapter I/O and batch-processing helpers extracted from zelda_core."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import networkx as nx


def load_dungeon(
    data_root: Path,
    vglc_parser: Any,
    dot_parser: Any,
    matcher: Any,
    log_virtual_node_report_fn: Callable[[Any, str], None],
    dungeon_num: int,
    variant: int = 1,
) -> Any:
    """Load one dungeon from VGLC text + DOT graph and run matching."""
    vglc_path = data_root / "Processed" / f"tloz{dungeon_num}_{variant}.txt"

    if variant == 2:
        dot_path = data_root / "Graph Processed" / f"LoZ2_{dungeon_num}.dot"
    else:
        dot_path = data_root / "Graph Processed" / f"LoZ_{dungeon_num}.dot"

    if not vglc_path.exists():
        raise FileNotFoundError(f"VGLC file not found: {vglc_path}")
    if not dot_path.exists():
        raise FileNotFoundError(f"DOT file not found: {dot_path}")

    rooms = vglc_parser.parse(str(vglc_path))
    graph = dot_parser.parse(str(dot_path))

    dungeon = matcher.match(rooms, graph)
    dungeon.dungeon_id = f"D{dungeon_num}"
    log_virtual_node_report_fn(dungeon, f"load:v{variant}")

    return dungeon


def layout_from_graph(
    data_root: Path,
    dot_parser: Any,
    hybrid_layout_engine_cls: Any,
    dungeon_num: int,
    variant: int = 1,
    **sa_kwargs,
) -> Dict[int, Tuple[int, int]]:
    """Compute topology-only layout from DOT graph."""
    if variant == 2:
        dot_path = data_root / "Graph Processed" / f"LoZ2_{dungeon_num}.dot"
    else:
        dot_path = data_root / "Graph Processed" / f"LoZ_{dungeon_num}.dot"

    graph = dot_parser.parse(str(dot_path))
    engine = hybrid_layout_engine_cls(**sa_kwargs)
    return engine.layout(graph)


def process_all_dungeons(
    data_root: Path,
    load_dungeon_fn: Callable[[int, int], Any],
    logger: Any,
    processed_dir: Optional[str] = None,
    graph_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load all quest variants and keep stable dungeon_id naming."""
    if processed_dir is None:
        processed_dir = data_root / "Processed"
    if graph_dir is None:
        graph_dir = data_root / "Graph Processed"

    processed_dir = Path(processed_dir)
    _ = Path(graph_dir)

    results: Dict[str, Any] = {}
    map_files = sorted(processed_dir.glob("*.txt"))

    for map_file in map_files:
        if map_file.name == "README.txt":
            continue

        match = re.match(r"tloz(\d+)_(\d+)\.txt", map_file.name)
        if not match:
            continue

        dungeon_num = int(match.group(1))
        quest_num = int(match.group(2))
        dungeon_id = f"zelda_{dungeon_num}_quest{quest_num}"

        try:
            dungeon = load_dungeon_fn(dungeon_num, quest_num)
            dungeon.dungeon_id = dungeon_id
            results[dungeon_id] = dungeon
            logger.info("Processed %s: %d rooms", dungeon_id, len(dungeon.rooms))
        except (AttributeError, FileNotFoundError, OSError, TypeError, ValueError):
            logger.exception("Error processing %s", dungeon_id)

    return results


def save_processed_data(
    data_root: Path,
    processed_dungeons: Dict[str, Any],
    output_path: Optional[str],
    logger: Any,
) -> str:
    """Serialize processed dungeon data to pickle."""
    if output_path is None:
        output_path = data_root / "processed_data.pkl"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data: Dict[str, Any] = {}
    for dungeon_id, dungeon in processed_dungeons.items():
        rooms_out = {}
        for rid, room in dungeon.rooms.items():
            grid = getattr(room, "grid", None)
            if grid is None:
                grid = getattr(room, "semantic_grid", None)
            rooms_out[str(rid)] = {
                "grid": grid,
                "contents": getattr(room, "contents", []),
                "doors": getattr(room, "doors", {}),
                "position": getattr(room, "position", None),
            }

        save_data[dungeon_id] = {
            "rooms": rooms_out,
            "graph_edges": list(getattr(dungeon, "graph", nx.DiGraph()).edges(data=True)),
            "graph_nodes": dict(getattr(dungeon, "graph", nx.DiGraph()).nodes(data=True)),
            "layout": getattr(dungeon, "layout", None),
            "tpe_vectors": getattr(dungeon, "tpe_vectors", None),
            "p_matrix": getattr(dungeon, "p_matrix", None),
            "node_features": getattr(dungeon, "node_features", None),
        }

    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "wb") as file_obj:
        pickle.dump(save_data, file_obj)
    tmp_path.replace(output_path)

    logger.info("Saved processed data to %s", output_path)
    return str(output_path)
