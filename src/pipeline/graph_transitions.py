"""Record exact endpoint transitions when a mission cannot use flat corridors."""

from __future__ import annotations

from collections import defaultdict

from src.core.definitions import SEMANTIC_PALETTE as P
from src.pipeline.spatial_utils import stable_node_sort_key
from src.simulation.edge_logic import edge_constraints_from_data


def record_graph_transitions(layout, graph, spatial_edge_predicate, fixed_records=None):
    """Realize existing edges without inventing floors or changing gate labels.

    This policy is opt-in mission metadata, sealed with publication inputs.
    A transition has distinct entry/exit tiles and costs one graph traversal.
    It is part of the grid-plus-graph artifact, not a flat geometric corridor.
    """
    if graph.graph.get('spatial_transition_policy') != 'recorded':
        return
    records = layout.connection_realizations
    fixed_records = {edge: record for edge, record in (fixed_records or {}).items()
                     if record.get('mode') == 'graph_transition'}
    cells_to_pairs = defaultdict(set)
    for edge, record in records.items():
        for cell in record.get('cells', ()):
            cells_to_pairs[tuple(cell)].add(frozenset(edge))
    ambiguous = set().union(*(pairs for pairs in cells_to_pairs.values() if len(pairs) > 1))
    pairs = {}
    for source, target, data in graph.edges(data=True):
        if source not in layout.layout_map or target not in layout.layout_map:
            continue
        explicit_transport = 'stair' in edge_constraints_from_data(data)
        if not explicit_transport and not spatial_edge_predicate(graph, source, target, data):
            continue
        pair = frozenset((source, target))
        existing = records.get((source, target), records.get((target, source)))
        if (existing is None or pair in ambiguous or explicit_transport or source == target
                or (source, target) in fixed_records or (target, source) in fixed_records):
            pairs[pair] = ((source, target) if source == target else tuple(sorted(pair, key=stable_node_sort_key)))
        elif (source, target) not in records:
            records[(source, target)] = dict(existing)

    # Remove physical realizations which would create ambiguous junctions.
    for edge in list(records):
        if frozenset(edge) not in pairs:
            continue
        for row, col in records.pop(edge).get('cells', ()):
            owners = [box for box in layout.layout_map.values()
                      if box[0] <= col <= box[2] and box[1] <= row <= box[3]]
            if not owners:
                layout.dungeon_grid[row, col] = P['VOID']
            elif any(col in (box[0], box[2]) or row in (box[1], box[3]) for box in owners):
                layout.dungeon_grid[row, col] = P['WALL']

    reserved = {tuple(cell) for record in records.values() for cell in record.get('cells', ())}
    reserved.update(tuple(record[field]) for record in fixed_records.values()
                    for field in ('source_position', 'target_position'))

    def allocate(node):
        left, top, right, bottom = layout.layout_map[node]
        candidates = [(row, col) for row in range(top + 2, bottom - 1)
                      for col in range(left + 2, right - 1)
                      if (row, col) not in reserved
                      and int(layout.dungeon_grid[row, col]) == P['FLOOR']]
        if not candidates:
            raise ValueError(f'No unreserved interior floor for graph transition in room {node!r}')
        center = ((top + bottom) // 2, (left + right) // 2)
        position = min(candidates, key=lambda pos: (abs(pos[0] - center[0]) + abs(pos[1] - center[1]), pos))
        reserved.add(position)
        layout.dungeon_grid[position] = P['STAIR']
        return position

    for source, target in sorted(pairs.values(), key=lambda pair: tuple(map(stable_node_sort_key, pair))):
        fixed = fixed_records.get((source, target))
        reverse_fixed = fixed_records.get((target, source))
        if fixed is not None:
            source_pos, target_pos = tuple(fixed['source_position']), tuple(fixed['target_position'])
        elif reverse_fixed is not None:
            source_pos, target_pos = tuple(reverse_fixed['target_position']), tuple(reverse_fixed['source_position'])
        else:
            source_pos, target_pos = allocate(source), allocate(target)
        for pos in (source_pos, target_pos):
            reserved.add(pos)
            layout.dungeon_grid[pos] = P['STAIR']
        directions = [(source, target, source_pos, target_pos)]
        if graph.is_directed():
            directions = []
            if graph.has_edge(source, target):
                directions.append((source, target, source_pos, target_pos))
            if target != source and graph.has_edge(target, source):
                directions.append((target, source, target_pos, source_pos))
        for src, dst, entry, exit_pos in directions:
            records[(src, dst)] = {
                'mode': 'graph_transition', 'cells': [entry, exit_pos],
                'source_position': entry, 'target_position': exit_pos,
                'transition_cost': 1,
            }
