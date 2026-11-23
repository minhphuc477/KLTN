"""Retain the executable graph and its exact tile mapping alongside each grid."""

import json
from pathlib import Path

import networkx as nx
import numpy as np

from .dataset_provenance import file_sha256


def _numpy_json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported graph evidence value: {type(value).__name__}')


def write_graph_evidence(result, root):
    layout = result.stitched_layout
    payload = {
        'schema_version': 1,
        'graph': nx.node_link_data(result.physical_mission_graph, edges='links'),
        'slot_positions': list(layout.slot_positions.items()),
        'room_offsets': list(getattr(layout, 'room_offsets', {}).items()),
        'connection_realizations': [
            {'source': source, 'target': target, 'record': record}
            for (source, target), record in getattr(layout, 'connection_realizations', {}).items()
        ],
        'room_puzzle_metadata': getattr(result, 'puzzle_metadata', {}) or {},
        'realization_metrics': getattr(layout, 'realization_metrics', {}),
    }
    path = Path(root) / 'graph_context.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False, default=_numpy_json)
    return {'path': str(path), 'sha256': file_sha256(path)}


def load_graph_evidence(artifact):
    path = Path(artifact['path'])
    if file_sha256(path) != artifact['sha256']:
        raise ValueError('Retained graph context is missing or changed')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['schema_version'] != 1:
        raise ValueError('Unsupported graph context schema')
    graph = nx.node_link_graph(payload['graph'], edges='links')
    slots = {node: tuple(slot) for node, slot in payload['slot_positions']}
    offsets = {node: tuple(offset) for node, offset in payload['room_offsets']}
    return {
        'graph': graph,
        'node_to_room': slots,
        'room_to_node': {slot: node for node, slot in slots.items()},
        'room_positions': {slots[node]: offset for node, offset in offsets.items()},
        'connection_realizations': {(item['source'], item['target']): item['record']
                                    for item in payload['connection_realizations']},
        'room_puzzle_metadata': payload['room_puzzle_metadata'],
    }
