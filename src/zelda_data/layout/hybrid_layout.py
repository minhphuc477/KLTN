"""Hybrid spectral + simulated annealing layout engine extracted from zelda_core."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


class HybridLayoutEngine:
    """Compute graph-to-grid room layouts with spectral init and SA refinement."""

    W_OVERLAP = 1000.0
    W_EDGE_LEN = 5.0
    W_BOSS_FAR = 8.0
    W_COMPACT = 2.0
    W_CROSSING = 10.0

    def __init__(
        self,
        sa_iterations: int = 8000,
        sa_t0: float = 10.0,
        sa_t_min: float = 0.01,
        sa_cooling: float = 0.997,
        seed: int = 42,
    ):
        self.sa_iterations = sa_iterations
        self.sa_t0 = sa_t0
        self.sa_t_min = sa_t_min
        self.sa_cooling = sa_cooling
        self.rng = np.random.RandomState(seed)

    def layout(self, graph: nx.DiGraph) -> Dict[int, Tuple[int, int]]:
        physical_nodes = [node for node in graph.nodes() if not graph.nodes[node].get("is_start_pointer", False)]
        if len(physical_nodes) == 0:
            return {}
        if len(physical_nodes) == 1:
            return {physical_nodes[0]: (0, 0)}

        graph_undirected = graph.to_undirected()
        graph_undirected = graph_undirected.subgraph(physical_nodes).copy()

        coords = self._spectral_init(graph_undirected, physical_nodes)
        grid_pos = self._snap_to_grid(coords, physical_nodes)
        grid_pos = self._simulated_annealing(grid_pos, graph_undirected, physical_nodes, graph)
        grid_pos = self._resolve_collisions(grid_pos, physical_nodes)

        if grid_pos:
            min_r = min(pos[0] for pos in grid_pos.values())
            min_c = min(pos[1] for pos in grid_pos.values())
            grid_pos = {node: (row - min_r, col - min_c) for node, (row, col) in grid_pos.items()}

        return grid_pos

    def _spectral_init(self, graph: nx.Graph, nodes: List[int]) -> Dict[int, Tuple[float, float]]:
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n))
        for u, v in graph.edges():
            if u in node_idx and v in node_idx:
                adj[node_idx[u], node_idx[v]] = 1.0
                adj[node_idx[v], node_idx[u]] = 1.0
        deg = np.sum(adj, axis=1)
        laplacian = np.diag(deg) - adj
        try:
            _eigvals, eigvecs = np.linalg.eigh(laplacian)
            x = eigvecs[:, 1] if n > 1 else np.zeros(n)
            y = eigvecs[:, 2] if n > 2 else np.zeros(n)
        except np.linalg.LinAlgError:
            x = self.rng.randn(n)
            y = self.rng.randn(n)

        scale = max(1.0, np.sqrt(n))
        x_range = x.max() - x.min() if x.max() != x.min() else 1.0
        y_range = y.max() - y.min() if y.max() != y.min() else 1.0
        x = (x - x.min()) / x_range * scale
        y = (y - y.min()) / y_range * scale

        return {nodes[i]: (float(x[i]), float(y[i])) for i in range(n)}

    def _snap_to_grid(self, coords: Dict[int, Tuple[float, float]], nodes: List[int]) -> Dict[int, Tuple[int, int]]:
        occupied: Set[Tuple[int, int]] = set()
        result: Dict[int, Tuple[int, int]] = {}

        sorted_nodes = sorted(nodes, key=lambda node: coords[node][0] ** 2 + coords[node][1] ** 2)

        for node in sorted_nodes:
            rx, ry = int(round(coords[node][0])), int(round(coords[node][1]))
            pos = self._spiral_find(rx, ry, occupied)
            occupied.add(pos)
            result[node] = pos
        return result

    @staticmethod
    def _spiral_find(r: int, c: int, occupied: Set[Tuple[int, int]]) -> Tuple[int, int]:
        if (r, c) not in occupied:
            return (r, c)
        for radius in range(1, 200):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if abs(dr) == radius or abs(dc) == radius:
                        pos = (r + dr, c + dc)
                        if pos not in occupied:
                            return pos
        return (r + 200, c)

    def _simulated_annealing(
        self,
        pos: Dict[int, Tuple[int, int]],
        graph: nx.Graph,
        nodes: List[int],
        directed_graph: nx.DiGraph,
    ) -> Dict[int, Tuple[int, int]]:
        if len(nodes) <= 2:
            return pos

        start_node = None
        boss_node = None
        for node in nodes:
            node_data = directed_graph.nodes.get(node, {})
            if node_data.get("is_start") and not node_data.get("is_start_pointer"):
                start_node = node
            if node_data.get("is_boss"):
                boss_node = node

        if start_node is None:
            for node in directed_graph.nodes():
                if directed_graph.nodes[node].get("is_start_pointer"):
                    neighbors = list(directed_graph.successors(node)) + list(directed_graph.predecessors(node))
                    for neighbor in neighbors:
                        if neighbor in pos:
                            start_node = neighbor
                            break
                    break

        edges = list(graph.edges())
        best_pos = dict(pos)
        best_energy = self._energy(best_pos, edges, nodes, start_node, boss_node)
        current_pos = dict(pos)
        current_energy = best_energy

        temperature = self.sa_t0
        for _ in range(self.sa_iterations):
            if temperature < self.sa_t_min:
                break

            node = nodes[self.rng.randint(len(nodes))]
            old_pos = current_pos[node]
            dr, dc = self.rng.choice([-2, -1, 0, 1, 2], size=2)
            new_cell = (old_pos[0] + dr, old_pos[1] + dc)

            _collision = any(current_pos[other] == new_cell for other in nodes if other != node)

            current_pos[node] = new_cell
            new_energy = self._energy(current_pos, edges, nodes, start_node, boss_node)
            delta = new_energy - current_energy

            if delta < 0 or self.rng.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = dict(current_pos)
            else:
                current_pos[node] = old_pos

            temperature *= self.sa_cooling

        return best_pos

    def _energy(
        self,
        pos: Dict[int, Tuple[int, int]],
        edges: List[Tuple[int, int]],
        nodes: List[int],
        start_node: Optional[int],
        boss_node: Optional[int],
    ) -> float:
        energy = 0.0
        positions = list(pos.values())
        n = len(positions)

        occupied_set: Set[Tuple[int, int]] = set()
        for point in positions:
            if point in occupied_set:
                energy += self.W_OVERLAP
            occupied_set.add(point)

        for u, v in edges:
            if u in pos and v in pos:
                dist = abs(pos[u][0] - pos[v][0]) + abs(pos[u][1] - pos[v][1])
                energy += self.W_EDGE_LEN * max(0, dist - 1)

        if start_node is not None and boss_node is not None:
            if start_node in pos and boss_node in pos:
                dist_sb = abs(pos[start_node][0] - pos[boss_node][0]) + abs(pos[start_node][1] - pos[boss_node][1])
                energy -= self.W_BOSS_FAR * dist_sb

        if n > 0:
            rows = [point[0] for point in positions]
            cols = [point[1] for point in positions]
            bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
            energy += self.W_COMPACT * bbox_area

        crossings = self._count_crossings(pos, edges)
        energy += self.W_CROSSING * crossings

        return energy

    @staticmethod
    def _count_crossings(pos: Dict[int, Tuple[int, int]], edges: List[Tuple[int, int]]) -> int:
        def _segments_cross(
            p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], p4: Tuple[int, int]
        ) -> bool:
            def ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

            if p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4:
                return False
            return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

        valid_edges = [(u, v) for u, v in edges if u in pos and v in pos]
        count = 0
        for i in range(len(valid_edges)):
            for j in range(i + 1, len(valid_edges)):
                u1, v1 = valid_edges[i]
                u2, v2 = valid_edges[j]
                if _segments_cross(pos[u1], pos[v1], pos[u2], pos[v2]):
                    count += 1
        return count

    def _resolve_collisions(self, pos: Dict[int, Tuple[int, int]], nodes: List[int]) -> Dict[int, Tuple[int, int]]:
        occupied: Set[Tuple[int, int]] = set()
        result: Dict[int, Tuple[int, int]] = {}
        for node in nodes:
            point = pos.get(node, (0, 0))
            if point in occupied:
                point = self._spiral_find(point[0], point[1], occupied)
            occupied.add(point)
            result[node] = point
        return result

    def layout_to_grid_positions(
        self, graph: nx.DiGraph
    ) -> Tuple[
        Dict[int, Tuple[int, int]],
        Optional[int],
        Optional[int],
    ]:
        start_pointer_id = None
        first_room_id = None
        for node in graph.nodes():
            if graph.nodes[node].get("is_start_pointer", False):
                start_pointer_id = node
                neighbors = list(graph.successors(node)) + list(graph.predecessors(node))
                if neighbors:
                    first_room_id = neighbors[0]
                break

        positions = self.layout(graph)
        return positions, start_pointer_id, first_room_id
