"""Matching helpers for Zelda data processing."""

from src.zelda_data.matching.room_graph_matching import (
    deterministic_greedy_assignment,
    refine_mapping_with_swaps,
    solve_assignment_with_fallback,
)
from src.zelda_data.matching.infer_missing_mappings import (
    apply_label_hints,
    assign_pairs_from_scores,
    build_component_context,
    build_score_matrix,
    compute_normalized_room_centers,
    propagate_from_anchors,
    seed_from_special_nodes,
)
from src.zelda_data.matching.spectral_refinement import (
    edge_consistency_score,
    local_refine_assignments,
    seeded_spectral_match,
)
from src.zelda_data.matching.topology_utils import (
    build_room_adjacency,
    find_entrance_room,
    find_farthest_dead_end,
    find_room_at_distance,
    room_signature,
)

__all__ = [
    "deterministic_greedy_assignment",
    "refine_mapping_with_swaps",
    "solve_assignment_with_fallback",
    "apply_label_hints",
    "assign_pairs_from_scores",
    "build_room_adjacency",
    "build_component_context",
    "build_score_matrix",
    "compute_normalized_room_centers",
    "edge_consistency_score",
    "find_entrance_room",
    "find_farthest_dead_end",
    "find_room_at_distance",
    "local_refine_assignments",
    "propagate_from_anchors",
    "room_signature",
    "seeded_spectral_match",
    "seed_from_special_nodes",
]
