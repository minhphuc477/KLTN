"""Topology helper orchestration bridges for ZeldaGUI wrappers."""

from __future__ import annotations

from src.gui.topology.export import export_topology as _export_topology
from src.gui.topology.helpers import (
    build_room_adjacency_from_graph as _build_room_adjacency_from_graph,
    node_has_critical_content as _node_has_critical_content,
    node_has_small_key as _node_has_small_key,
    room_for_global_position as _room_for_global_position,
)


def export_topology(*, gui):
    return _export_topology(gui)


def room_for_global_position(*, pos, room_positions):
    return _room_for_global_position(pos, room_positions)


def node_has_small_key(*, attrs):
    return _node_has_small_key(attrs)


def node_has_critical_content(*, graph, node_id):
    return _node_has_critical_content(graph, node_id)


def build_room_adjacency_from_graph(*, graph, room_to_node, node_to_room):
    return _build_room_adjacency_from_graph(graph, room_to_node, node_to_room)
