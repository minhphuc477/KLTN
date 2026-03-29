"""Stitching helpers for Zelda data processing."""

from src.zelda_data.stitching.stitch_orchestration import (
    build_global_grid_from_rooms,
    build_stitched_room_layout_from_rooms,
    build_room_node_mappings,
    place_special_markers,
    project_output_metadata,
)
from src.zelda_data.stitching.graph_placement import (
    apply_door_types_from_graph,
    find_boundary_doors,
    place_entities_from_graph,
    place_items_from_graph,
)
from src.zelda_data.stitching.connectivity import (
    connect_doors,
    ensure_room_connectivity,
    find_floor_near_door,
)
from src.zelda_data.stitching.compaction import compact_rooms

__all__ = [
    "build_global_grid_from_rooms",
    "build_stitched_room_layout_from_rooms",
    "build_room_node_mappings",
    "place_special_markers",
    "project_output_metadata",
    "apply_door_types_from_graph",
    "compact_rooms",
    "connect_doors",
    "ensure_room_connectivity",
    "find_floor_near_door",
    "find_boundary_doors",
    "place_entities_from_graph",
    "place_items_from_graph",
]
