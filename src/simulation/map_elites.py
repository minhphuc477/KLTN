"""MAP-Elites evaluator for Zelda dungeons.

Provides a minimal MAP-Elites evaluator that can operate on a list of
stitched dungeons (or dungeon-like objects exposing a 2D semantic grid).

API:
- MAPElitesEvaluator(resolution=20)
- run_map_elites_on_maps(maps, resolution=20, tie_breaker='path_length')

The evaluator is intentionally lightweight and dependency-tolerant for use
from the GUI (optional plotting via matplotlib when available).
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import json
import math
import logging
import pickle
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, Optional, Any, List, Iterable, Union

import networkx as nx

# Import from canonical source
from src.core.definitions import (
    SEMANTIC_PALETTE,
    parse_edge_type_tokens,
    parse_node_label_tokens,
)
from src.simulation.validation_helpers import MetricsEngine

try:
    from src.zelda_data.zelda_core import DungeonSolver
except ImportError:
    DungeonSolver = None

try:
    from src.simulation.validator import ZeldaValidator
except ImportError:
    ZeldaValidator = None

try:
    # Reuse richer QD archive implementation as an auxiliary backend.
    from src.evaluation.map_elites import CVTEliteArchive
except ImportError:
    CVTEliteArchive = None

logger = logging.getLogger(__name__)

def _clip01(value: float) -> float:
    """Clamp scalar to [0, 1] as float."""
    return float(np.clip(float(value), 0.0, 1.0))


@dataclass
class BinEntry:
    dungeon: Any
    score: float
    metrics: Dict[str, float]


class MAPElitesEvaluator:
    def __init__(
        self,
        resolution: int = 20,
        tie_breaker: str = 'path_length',
        enable_advanced_archive: bool = True,
        descriptor_mode: str = 'hybrid',
        seed: Optional[int] = None,
        archive_path: Optional[Union[str, Path]] = None,
        load_existing_archive: bool = False,
        autosave_archive: bool = False,
    ):
        self.resolution = int(resolution)
        self.grid: Dict[Tuple[int, int], BinEntry] = {}
        self.tie_breaker = tie_breaker
        self.descriptor_mode = str(descriptor_mode).strip().lower()
        self.rng = np.random.default_rng(seed)
        self.archive_path = Path(archive_path) if archive_path is not None else None
        self.autosave_archive = bool(autosave_archive)

        # Optional CVT archive (from src.evaluation.map_elites) to keep
        # grid-based and research-grade QD tracking aligned.
        self._advanced_archive = None
        if enable_advanced_archive and CVTEliteArchive is not None:
            try:
                num_cells = max(64, self.resolution * self.resolution)
                self._advanced_archive = CVTEliteArchive(
                    num_cells=num_cells,
                    feature_dims=4,
                    feature_ranges=[(0.0, 1.0)] * 4,
                    num_cvt_samples=min(4000, num_cells * 40),
                    seed=seed,
                )
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                logger.warning("Advanced CVT archive unavailable, using legacy grid archive only: %s", e)
                self._advanced_archive = None

        if load_existing_archive:
            if self.archive_path is None:
                raise ValueError("load_existing_archive=True requires archive_path.")
            if self.archive_path.exists():
                self.load_archive(self.archive_path)
            else:
                logger.info("MAP-Elites archive path does not exist yet: %s", self.archive_path)

    @staticmethod
    def calculate_linearity(
        path: Iterable[Tuple[int, int]],
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Return geometric route directness: Manhattan displacement / steps."""
        path_points = [(int(row), int(col)) for row, col in path]
        if not path_points:
            return 0.0
        return MetricsEngine.calculate_linearity(
            path_points,
            start or path_points[0],
            goal or path_points[-1],
        )

    @staticmethod
    def calculate_route_sparsity(path_len: int, playable_area: int) -> float:
        """Legacy inverse route-coverage proxy; this is not path linearity."""
        raw = 1.0 - (float(path_len) / max(1.0, float(playable_area)))
        return float(np.clip(raw, 0.0, 1.0))

    def calculate_leniency(self, grid: np.ndarray) -> float:
        if SEMANTIC_PALETTE is None:
            # Fallback heuristic: treat higher values as "more enemies"
            enemies = int((grid == 7).sum())
            floors = int((grid == 1).sum())
        else:
            enemies = int((grid == SEMANTIC_PALETTE['ENEMY']).sum())
            if 'BOSS' in SEMANTIC_PALETTE:
                enemies += int((grid == SEMANTIC_PALETTE['BOSS']).sum())
            floors = int((grid == SEMANTIC_PALETTE['FLOOR']).sum())
        return float(np.clip(1.0 - (enemies / max(1, floors)), 0.0, 1.0))

    def _discretize(self, lin: float, len_score: float) -> Tuple[int, int]:
        x = min(int(math.floor(np.clip(lin, 0.0, 1.0) * self.resolution)), self.resolution - 1)
        y = min(int(math.floor(np.clip(len_score, 0.0, 1.0) * self.resolution)), self.resolution - 1)
        return (x, y)

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _iter_locked_edge_types() -> Iterable[str]:
        # Canonical progression-gating edge types used across the codebase.
        return (
            'locked',
            'key_locked',
            'boss_locked',
            'item_locked',
            'item_gate',
            'state_block',
            'switch',
            'switch_locked',
        )

    def _extract_graph_topology_metrics(self, mission_graph: Optional[nx.Graph]) -> Dict[str, float]:
        """
        Extract progression-aware topology descriptors from mission graph.
        """
        if mission_graph is None:
            return {
                'branching_factor': 0.0,
                'cycle_density': 0.0,
                'gating_density': 0.0,
                'topology_complexity': 0.0,
            }
        try:
            n_nodes = int(mission_graph.number_of_nodes())
            n_edges = int(mission_graph.number_of_edges())
            if n_nodes <= 1:
                return {
                    'branching_factor': 0.0,
                    'cycle_density': 0.0,
                    'gating_density': 0.0,
                    'topology_complexity': 0.0,
                }

            if isinstance(mission_graph, nx.DiGraph):
                branch_nodes = sum(
                    1 for node in mission_graph.nodes()
                    if mission_graph.out_degree(node) >= 2
                )
            else:
                branch_nodes = sum(
                    1 for node in mission_graph.nodes()
                    if mission_graph.degree(node) >= 3
                )
            branching_factor = self._clip01(branch_nodes / max(1, n_nodes))

            undirected = mission_graph.to_undirected()
            cycle_rank = max(0, int(undirected.number_of_edges()) - int(undirected.number_of_nodes()) + nx.number_connected_components(undirected))
            cycle_norm = max(1, n_nodes // 2)
            cycle_density = self._clip01(cycle_rank / cycle_norm)

            locked_types = set(self._iter_locked_edge_types())
            gated_edges = 0
            for _, _, data in mission_graph.edges(data=True):
                edge_tokens = parse_edge_type_tokens(
                    label=str(data.get('label', '') or ''),
                    edge_type=str(data.get('edge_type', data.get('type', '')) or ''),
                )
                if any(tok in locked_types for tok in edge_tokens):
                    gated_edges += 1
            gating_density = self._clip01(gated_edges / max(1, n_edges))

            topology_complexity = self._clip01(
                (0.45 * branching_factor) + (0.35 * cycle_density) + (0.20 * gating_density)
            )
            return {
                'branching_factor': branching_factor,
                'cycle_density': cycle_density,
                'gating_density': gating_density,
                'topology_complexity': topology_complexity,
            }
        except (AttributeError, RuntimeError, ValueError, TypeError, nx.NetworkXException) as exc:
            logger.debug("Failed to extract graph topology metrics: %s", exc)
            return {
                'branching_factor': 0.0,
                'cycle_density': 0.0,
                'gating_density': 0.0,
                'topology_complexity': 0.0,
            }

    @staticmethod
    def _node_tokens(data: Dict[str, Any]) -> set[str]:
        tokens = set(parse_node_label_tokens(str(data.get('label', '') or '')))
        node_type = str(data.get('type', '') or '').strip().lower()
        if node_type:
            tokens.add(node_type)
        if data.get('is_start'):
            tokens.update({'s', 'start'})
        if data.get('is_goal') or data.get('is_triforce'):
            tokens.update({'t', 'goal', 'triforce'})
        return tokens

    @classmethod
    def _progression_feasible_path(
        cls,
        mission_graph: nx.Graph,
        start: Any,
        goal: Any,
        *,
        max_states: int = 100_000,
    ) -> Optional[Tuple[List[Any], Dict[str, int], int, int]]:
        """Find a shortest macro path while respecting consumable and persistent gates."""

        def normalize(value: Any) -> str:
            return str(value or '').strip().lower()

        def collect(node: Any, inventory: Dict[str, int], collected: frozenset) -> Tuple[Dict[str, int], frozenset, int]:
            if node in collected:
                return inventory, collected, 0
            data = dict(mission_graph.nodes[node])
            updated = dict(inventory)
            gained_keys = 0
            role = normalize(data.get('type', data.get('label', '')))
            key_id = normalize(data.get('key_id'))
            key_count = max(1, int(data.get('key_count_hint', data.get('key_count', 0)) or 0))
            if key_id:
                updated[key_id] = updated.get(key_id, 0) + key_count
                gained_keys += key_count
            elif data.get('has_key') or role in {'key', 'k'}:
                updated['key_generic'] = updated.get('key_generic', 0) + key_count
                gained_keys += key_count
            if role in {'big_key', 'boss_key'}:
                updated['key_boss'] = 1
            for field in ('item_type', 'drops_resource'):
                item = normalize(data.get(field))
                if item:
                    updated[item] = updated.get(item, 0) + 1
            for item in data.get('items', []) or []:
                item_id = normalize(item)
                if item_id:
                    updated[item_id] = updated.get(item_id, 0) + 1
            return updated, collected | {node}, gained_keys

        def traverse(data: Dict[str, Any], inventory: Dict[str, int]) -> Optional[Tuple[Dict[str, int], int]]:
            updated = dict(inventory)
            edge_tokens = set(parse_edge_type_tokens(
                label=str(data.get('label', '') or ''),
                edge_type=str(data.get('edge_type', data.get('type', '')) or ''),
            ))
            lock_type = normalize(data.get('lock_type'))
            required_item = normalize(data.get('item_required'))
            if required_item or 'item_gate' in edge_tokens:
                if not required_item or updated.get(required_item, 0) <= 0:
                    return None
                return updated, 1
            if lock_type == 'boss' or {'boss_locked', 'boss_lock'} & edge_tokens:
                if updated.get('key_boss', 0) <= 0:
                    return None
                return updated, 1
            if lock_type == 'bomb' or {'bomb', 'bombable', 'bomb_locked'} & edge_tokens:
                bomb_id = next((name for name in ('item_bomb', 'bomb', 'bombs') if updated.get(name, 0) > 0), None)
                if bomb_id is None:
                    return None
                updated[bomb_id] -= 1
                return updated, 1
            if lock_type in {'locked', 'key_locked'} or {'locked', 'key_locked'} & edge_tokens:
                key_id = normalize(data.get('key_required', data.get('key_id'))) or 'key_generic'
                required = max(1, int(data.get('requires_key_count', 1) or 1))
                if updated.get(key_id, 0) < required:
                    return None
                updated[key_id] -= required
                return updated, required
            return updated, 0

        initial_inventory, initial_collected, initial_keys = collect(start, {}, frozenset())
        initial = (start, tuple(sorted(initial_inventory.items())), initial_collected)
        queue = deque([initial])
        parents: Dict[Tuple[Any, Tuple[Tuple[str, int], ...], frozenset], Optional[Tuple[Any, Tuple[Tuple[str, int], ...], frozenset]]] = {initial: None}
        path_stats = {initial: (initial_keys, 0)}

        while queue and len(parents) <= int(max_states):
            state = queue.popleft()
            node, inventory_items, collected = state
            inventory = dict(inventory_items)
            if node == goal:
                path: List[Any] = []
                cursor = state
                while cursor is not None:
                    path.append(cursor[0])
                    cursor = parents[cursor]
                keys_collected, keys_consumed = path_stats[state]
                return list(reversed(path)), inventory, keys_collected, keys_consumed

            for neighbor in mission_graph.neighbors(node):
                edge_data = dict(mission_graph.get_edge_data(node, neighbor, {}) or {})
                traversed = traverse(edge_data, inventory)
                if traversed is None:
                    continue
                next_inventory, consumed = traversed
                next_inventory, next_collected, gained = collect(neighbor, next_inventory, collected)
                next_state = (neighbor, tuple(sorted((k, v) for k, v in next_inventory.items() if v > 0)), next_collected)
                if next_state in parents:
                    continue
                parents[next_state] = state
                prior_gained, prior_consumed = path_stats[state]
                path_stats[next_state] = (prior_gained + gained, prior_consumed + consumed)
                queue.append(next_state)
        return None

    def _extract_graph_behavior_metrics(self, mission_graph: Optional[nx.Graph]) -> Dict[str, float]:
        """Extract dungeon-level descriptors from the macro mission graph."""
        if mission_graph is None or mission_graph.number_of_nodes() == 0:
            return {}

        try:
            nodes = list(mission_graph.nodes())
            n_nodes = len(nodes)
            start = None
            goal = None
            key_count = 0
            enemy_count = 0
            for node, data in mission_graph.nodes(data=True):
                tokens = self._node_tokens(dict(data))
                if start is None and ({'s', 'start'} & tokens):
                    start = node
                if goal is None and ({'t', 'goal', 'triforce'} & tokens):
                    goal = node
                if 'k' in tokens or ('key' in tokens and 'boss_key' not in tokens):
                    key_count += max(1, int(data.get('key_count_hint', data.get('key_count', 0)) or 0))
                if {'e', 'enemy', 'boss', 'mini_boss'} & tokens or data.get('has_enemy'):
                    enemy_count += max(1, int(data.get('enemy_count_hint', data.get('enemy_count', 0)) or 0))

            start = nodes[0] if start is None else start
            goal = nodes[-1] if goal is None else goal
            feasible = self._progression_feasible_path(mission_graph, start, goal)
            if feasible is None:
                return {}
            critical_path, final_inventory, path_keys_collected, path_keys_consumed = feasible
            critical_path_length = max(0, len(critical_path) - 1)
            critical_path_ratio = self._clip01(len(critical_path) / max(1, n_nodes))

            locked_types = set(self._iter_locked_edge_types())
            lock_count = 0
            for _, _, data in mission_graph.edges(data=True):
                edge_tokens = parse_edge_type_tokens(
                    label=str(data.get('label', '') or ''),
                    edge_type=str(data.get('edge_type', data.get('type', '')) or ''),
                )
                if any(token in locked_types for token in edge_tokens):
                    lock_count += 1

            path_key_surplus = max(0, int(path_keys_collected) - int(path_keys_consumed))
            if path_keys_consumed == 0:
                graph_leniency = 1.0
            else:
                graph_leniency = self._clip01(
                    (path_key_surplus + 1.0) / (float(path_keys_consumed) + path_key_surplus + 1.0)
                )

            difficulty_curve = []
            for node in critical_path:
                data = mission_graph.nodes[node]
                difficulty_curve.append(
                    float(
                        data.get(
                            'difficulty',
                            data.get('difficulty_rating', data.get('tension_value', 0.0)),
                        )
                        or 0.0
                    )
                )
            difficulty_progression = 0.0
            if len(difficulty_curve) >= 2:
                steps = np.diff(np.asarray(difficulty_curve, dtype=np.float64))
                nondecreasing_fraction = float(np.mean(steps >= -1e-9))
                positive_rise = self._clip01(max(0.0, difficulty_curve[-1] - difficulty_curve[0]))
                difficulty_progression = self._clip01(
                    (0.65 * nondecreasing_fraction) + (0.35 * positive_rise)
                )

            topology = self._extract_graph_topology_metrics(mission_graph)
            progression_complexity = self._clip01(
                (0.35 * topology['gating_density'])
                + (0.25 * topology['branching_factor'])
                + (0.20 * critical_path_ratio)
                + (0.20 * difficulty_progression)
            )
            return {
                'graph_linearity': critical_path_ratio,
                'graph_leniency': graph_leniency,
                'graph_progression_complexity': progression_complexity,
                'critical_path_length': float(critical_path_length),
                'critical_path_ratio': critical_path_ratio,
                'global_difficulty_progression': difficulty_progression,
                'graph_key_count': float(key_count),
                'graph_lock_count': float(lock_count),
                'graph_enemy_count': float(enemy_count),
                'graph_descriptor_feasible': 1.0,
                'graph_path_keys_collected': float(path_keys_collected),
                'graph_path_keys_consumed': float(path_keys_consumed),
                'graph_path_key_surplus': float(path_key_surplus),
                'graph_final_inventory_size': float(sum(final_inventory.values())),
                **topology,
            }
        except (AttributeError, RuntimeError, ValueError, TypeError, nx.NetworkXException) as exc:
            logger.debug("Failed to extract graph behavior metrics: %s", exc)
            return {}

    def _build_behavior_descriptor(
        self,
        grid: np.ndarray,
        solver_result: Dict[str, Any],
        mission_graph: Optional[nx.Graph] = None,
    ) -> Tuple[Tuple[float, float, float, float], Dict[str, float]]:
        """
        Build progression-aware 4D descriptor and aligned metric dict.

        Dimensions:
        1) linearity
        2) leniency
        3) progression_complexity (gating pressure + backtracking + path pressure)
        4) topology_complexity (branching + cycles + gate density)
        """
        floor_id = int(SEMANTIC_PALETTE['FLOOR']) if SEMANTIC_PALETTE else 1
        enemy_id = int(SEMANTIC_PALETTE['ENEMY']) if SEMANTIC_PALETTE else 7
        locked_door_id = int(SEMANTIC_PALETTE['DOOR_LOCKED']) if SEMANTIC_PALETTE else 11
        key_id = int(SEMANTIC_PALETTE.get('KEY_SMALL', SEMANTIC_PALETTE.get('KEY', 8))) if SEMANTIC_PALETTE else 8

        playable_area = int((grid == floor_id).sum())
        path_len = int(solver_result.get('path_length', 0))
        route_sparsity = self.calculate_route_sparsity(path_len, playable_area)
        graph_metrics = self._extract_graph_behavior_metrics(mission_graph)
        graph_descriptor_used = bool(graph_metrics) and self.descriptor_mode != 'legacy'
        supplied_linearity = solver_result.get('linearity')
        solver_path = solver_result.get('path')
        if supplied_linearity is not None:
            linearity = self._clip01(supplied_linearity)
        elif solver_path is not None and len(solver_path) > 0:
            linearity = self._clip01(self.calculate_linearity(solver_path))
        elif graph_descriptor_used:
            # Replaced below by the mission-graph descriptor.
            linearity = 0.0
        elif self.descriptor_mode == 'legacy':
            linearity = route_sparsity
        else:
            raise ValueError(
                "Grid MAP-Elites requires solver_result['linearity'] or a concrete "
                "solver_result['path']; route sparsity is only available in legacy mode"
            )
        leniency = self._clip01(solver_result.get('leniency', self.calculate_leniency(grid)))

        total_tiles = max(1, int(grid.size))
        density = self._clip01(playable_area / total_tiles)

        enemy_count = int((grid == enemy_id).sum())
        lock_count = int(solver_result.get('lock_count', int((grid == locked_door_id).sum())))
        key_count = int(solver_result.get('key_count', int((grid == key_id).sum())))
        backtracking = self._clip01(float(solver_result.get('backtracking_score', 0.0)))

        enemy_pressure = min(1.0, enemy_count / max(1.0, playable_area * 0.2))
        lock_pressure = min(1.0, lock_count / max(1.0, float(max(1, key_count))))
        path_pressure = min(1.0, float(path_len) / max(1.0, math.sqrt(total_tiles) * 4.0))
        progression_complexity = self._clip01(
            (0.40 * lock_pressure) + (0.30 * backtracking) + (0.20 * path_pressure) + (0.10 * enemy_pressure)
        )

        if graph_descriptor_used:
            linearity = self._clip01(graph_metrics['graph_linearity'])
            leniency = self._clip01(graph_metrics['graph_leniency'])
            progression_complexity = self._clip01(graph_metrics['graph_progression_complexity'])
            topology_complexity = self._clip01(graph_metrics['topology_complexity'])
            key_count = int(graph_metrics['graph_key_count'])
            lock_count = int(graph_metrics['graph_lock_count'])
            enemy_count = int(graph_metrics['graph_enemy_count'])
        else:
            # Preserve the former stitched-grid descriptors for explicit legacy
            # ablations and for callers that genuinely have no mission graph.
            topology_complexity = density

        features = (linearity, leniency, progression_complexity, topology_complexity)
        metrics = {
            'linearity': linearity,
            'leniency': leniency,
            'path_length': float(path_len),
            'progression_complexity': progression_complexity,
            'topology_complexity': topology_complexity,
            'density': density,
            'route_sparsity': route_sparsity,
            'backtracking_score': backtracking,
            'key_count': float(key_count),
            'lock_count': float(lock_count),
            'enemy_count': float(enemy_count),
            'graph_descriptor_used': float(graph_descriptor_used),
        }
        metrics.update(graph_metrics)
        return features, metrics

    def add_dungeon(
        self,
        dungeon: Any,
        grid: np.ndarray,
        solver_result: Dict[str, Any],
        mission_graph: Optional[nx.Graph] = None,
    ) -> Optional[Dict[str, float]]:
        # solver_result expected to contain 'solvable' and 'path_length' when solvable
        if not solver_result or not solver_result.get('solvable', False):
            return None

        path_len = int(solver_result.get('path_length', 0))
        features, descriptor_metrics = self._build_behavior_descriptor(
            grid=grid,
            solver_result=solver_result,
            mission_graph=mission_graph,
        )
        lin = descriptor_metrics['linearity']
        len_score = descriptor_metrics['leniency']

        key = self._discretize(lin, len_score)
        score = float(
            solver_result.get(
                self.tie_breaker,
                solver_result.get('quality_score', path_len)
            )
        )

        entry = BinEntry(
            dungeon=dungeon,
            score=score,
            metrics=descriptor_metrics,
        )
        # Keep the better-scoring entry per tie-breaker
        existing = self.grid.get(key)
        if existing is None or score > existing.score:
            self.grid[key] = entry

        # Mirror into advanced CVT archive when available.
        if self._advanced_archive is not None:
            try:
                self._advanced_archive.add(
                    solution=dungeon,
                    fitness=score,
                    features=features,
                    metadata=descriptor_metrics,
                )
            except (AttributeError, RuntimeError, ValueError, TypeError) as e:
                logger.debug("Advanced archive add failed: %s", e)

        if self.autosave_archive and self.archive_path is not None:
            try:
                self.save_archive(self.archive_path)
            except (AttributeError, OSError, pickle.PickleError, TypeError, ValueError) as e:
                logger.warning("Failed to autosave MAP-Elites archive to %s: %s", self.archive_path, e)
        return descriptor_metrics

    def occupancy_grid(self) -> np.ndarray:
        arr = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        for (x, y) in self.grid.keys():
            arr[y, x] = 1
        return arr

    def occupied_bins(self) -> List[Tuple[int, int, BinEntry]]:
        return [(x, y, e) for (x, y), e in self.grid.items()]

    def clear(self) -> None:
        """Clear both legacy and auxiliary archives."""
        self.grid.clear()
        if self._advanced_archive is not None:
            self._advanced_archive.clear()

    def _archive_payload(self) -> Dict[str, Any]:
        return {
            'version': 1,
            'resolution': int(self.resolution),
            'tie_breaker': str(self.tie_breaker),
            'descriptor_mode': str(self.descriptor_mode),
            'grid': dict(self.grid),
            'advanced_archive': self._advanced_archive,
        }

    def save_archive(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persist the current archive for reproducible warm starts."""
        path = Path(filepath) if filepath is not None else self.archive_path
        if path is None:
            raise ValueError("save_archive requires filepath or evaluator.archive_path.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump(self._archive_payload(), f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def export_archive_json(self, filepath: Union[str, Path]) -> Path:
        """Export a portable, analysis-only JSON snapshot without pickled solutions."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        elites: List[Dict[str, Any]] = []
        if self._advanced_archive is not None:
            for elite in self._advanced_archive.get_all_elites():
                elites.append(
                    {
                        'cell': list(elite.cell),
                        'fitness': float(elite.fitness),
                        'features': [float(value) for value in elite.features],
                        'metrics': dict(elite.metadata or {}),
                    }
                )
        else:
            for (x, y), entry in self.grid.items():
                metrics = dict(entry.metrics)
                elites.append(
                    {
                        'cell': [int(x), int(y)],
                        'fitness': float(entry.score),
                        'features': [
                            float(metrics.get('linearity', 0.0)),
                            float(metrics.get('leniency', 0.0)),
                            float(metrics.get('progression_complexity', 0.0)),
                            float(metrics.get('topology_complexity', 0.0)),
                        ],
                        'metrics': metrics,
                    }
                )
        payload = {
            'version': 1,
            'descriptor_mode': str(self.descriptor_mode),
            'descriptor_names': [
                'linearity',
                'leniency',
                'progression_complexity',
                'topology_complexity',
            ],
            'elites': elites,
            'advanced_archive_stats': self.advanced_archive_stats(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return path

    def load_archive(self, filepath: Optional[Union[str, Path]] = None) -> None:
        """Load a previously persisted archive into this evaluator."""
        path = Path(filepath) if filepath is not None else self.archive_path
        if path is None:
            raise ValueError("load_archive requires filepath or evaluator.archive_path.")
        with path.open('rb') as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid MAP-Elites archive payload in {path}")

        payload_resolution = int(payload.get('resolution', self.resolution))
        if payload_resolution != self.resolution:
            raise ValueError(
                f"Archive resolution mismatch: file has {payload_resolution}, "
                f"evaluator uses {self.resolution}."
            )

        grid = payload.get('grid', {})
        if not isinstance(grid, dict):
            raise ValueError(f"Invalid MAP-Elites grid payload in {path}")
        self.grid = {
            (int(key[0]), int(key[1])): value
            for key, value in grid.items()
            if isinstance(key, tuple) and len(key) == 2 and isinstance(value, BinEntry)
        }
        self.tie_breaker = str(payload.get('tie_breaker', self.tie_breaker))
        self.descriptor_mode = str(payload.get('descriptor_mode', self.descriptor_mode)).strip().lower()

        advanced_archive = payload.get('advanced_archive')
        if advanced_archive is not None:
            self._advanced_archive = advanced_archive

    def advanced_archive_stats(self) -> Optional[Dict[str, float]]:
        """Return auxiliary CVT archive stats if enabled."""
        if self._advanced_archive is None:
            return None
        stats = self._advanced_archive.get_stats()
        return {
            'coverage': float(stats.coverage),
            'qd_score': float(stats.total_fitness),
            'mean_fitness': float(stats.mean_fitness),
            'num_elites': float(stats.num_elites),
            'feature_diversity': float(stats.feature_diversity),
        }


def _get_grid_from_dungeon(dungeon) -> Optional[np.ndarray]:
    # Accept multiple dungeon representations
    if dungeon is None:
        return None
    if isinstance(dungeon, np.ndarray):
        return dungeon
    if hasattr(dungeon, 'global_grid'):
        return getattr(dungeon, 'global_grid')
    if hasattr(dungeon, 'layout'):
        return getattr(dungeon, 'layout')
    # Some adapters store as 'grid'
    if hasattr(dungeon, 'grid'):
        return getattr(dungeon, 'grid')
    return None


def _get_mission_graph_from_dungeon(dungeon: Any) -> Optional[nx.Graph]:
    """Extract an embedded macro mission graph from common dungeon wrappers."""
    if isinstance(dungeon, nx.Graph):
        return dungeon
    for attr in ('mission_graph_physical', 'mission_graph', 'graph'):
        candidate = getattr(dungeon, attr, None)
        if isinstance(candidate, nx.Graph):
            return candidate
    return None


def _quick_solver_result_from_grid(grid: np.ndarray) -> Dict[str, Any]:
    """Deterministic fallback when no solver is available."""
    floor_id = int(SEMANTIC_PALETTE['FLOOR']) if SEMANTIC_PALETTE else 1
    playable = int((grid == floor_id).sum())
    path_length = int(max(1, np.sqrt(max(1, playable))))
    return {
        'solvable': playable > 0,
        'path_length': path_length,
        'quality_score': _clip01(path_length / max(1.0, playable)),
    }


def run_map_elites_on_maps(
    maps: List[Any],
    resolution: int = 20,
    tie_breaker: str = 'path_length',
    solver: Optional[Any] = None,
    archive_path: Optional[Union[str, Path]] = None,
    load_existing_archive: bool = False,
    autosave_archive: bool = False,
    enable_advanced_archive: bool = True,
    descriptor_mode: str = 'hybrid',
) -> Tuple[MAPElitesEvaluator, np.ndarray]:
    """Run MAP-Elites on a provided list of dungeon-like objects.

    Returns a tuple (evaluator, occupancy_grid) where occupancy_grid is a
    numpy array (resolution,resolution) with 1 for occupied bins.
    """
    if solver is None:
        solver = DungeonSolver() if DungeonSolver is not None else None

    evaluator = MAPElitesEvaluator(
        resolution=resolution,
        tie_breaker=tie_breaker,
        enable_advanced_archive=enable_advanced_archive,
        descriptor_mode=descriptor_mode,
        archive_path=archive_path,
        load_existing_archive=load_existing_archive,
        autosave_archive=autosave_archive,
    )

    for d in maps:
        grid = _get_grid_from_dungeon(d)
        if grid is None:
            continue

        # Validate solvability with best available backend.
        solver_result: Dict[str, Any] = {}
        try:
            # Case 1: A proper dungeon object + graph-aware solver.
            if solver is not None and hasattr(solver, "solve") and not isinstance(d, np.ndarray):
                solver_result = solver.solve(d)
            # Case 2: Grid-only validation via ZeldaValidator.
            elif ZeldaValidator is not None:
                validator = ZeldaValidator()
                result = validator.validate_single(np.asarray(grid))
                path_length = int(result.path_length) if result.is_solvable else 0
                solver_result = {
                    'solvable': bool(result.is_solvable),
                    'path_length': path_length,
                    'path': list(result.path),
                    'backtracking_score': float(getattr(result, 'backtracking_score', 0.0)),
                    'quality_score': _clip01(
                        (0.6 * float(getattr(result, 'reachability', 0.0)))
                        + (0.4 * float(getattr(result, 'is_valid_syntax', False)))
                    ),
                }
            # Case 3: Coarse deterministic fallback.
            else:
                solver_result = _quick_solver_result_from_grid(np.asarray(grid))
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.debug("Solver failed in run_map_elites_on_maps, using fallback: %s", e)
            solver_result = _quick_solver_result_from_grid(np.asarray(grid))

        evaluator.add_dungeon(
            d,
            grid,
            solver_result,
            mission_graph=_get_mission_graph_from_dungeon(d),
        )

    occ = evaluator.occupancy_grid()
    return evaluator, occ


# Optional plotting helper (uses matplotlib if available)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_heatmap(occ_grid: np.ndarray, output_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Plot a simple heatmap of occupancy grid. Saves to output_path if given and
    returns the image as an RGB numpy array when matplotlib is available.
    """
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(occ_grid, origin='lower', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Linearity')
    ax.set_ylabel('Leniency')
    ax.set_xticks([])
    ax.set_yticks([])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    # Return numpy rgba buffer
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return img


# ===== DIVERSITY METRICS FOR THESIS DEFENSE =====
# Addresses the concern: "How do you prove your system doesn't suffer from mode collapse?"

def calculate_diversity_score(evaluator: MAPElitesEvaluator) -> float:
    """
    Compute diversity score across archived solutions.
    
    Method: Average pairwise difference in 4D behavioral descriptors:
    (linearity, leniency, progression_complexity, topology_complexity).
    
    Target: >35% average difference indicates healthy diversity.
    
    Args:
        evaluator: MAPElitesEvaluator instance with populated archive
    
    Returns:
        Diversity score in [0, 1] where higher = more diverse
    """
    if len(evaluator.grid) < 2:
        return 0.0
    
    # Extract behavioral descriptors from archive entries.
    descriptors = []
    for entry in evaluator.grid.values():
        descriptors.append([
            float(entry.metrics.get('linearity', 0.0)),
            float(entry.metrics.get('leniency', 0.0)),
            float(entry.metrics.get('progression_complexity', 0.0)),
            float(entry.metrics.get('topology_complexity', 0.0)),
        ])
    
    descriptors = np.array(descriptors)
    
    # Calculate pairwise Euclidean distances in behavior space
    distances = []
    n = len(descriptors)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(descriptors[i] - descriptors[j])
            # Normalize by max possible distance in unit hypercube.
            normalized_dist = dist / np.sqrt(float(descriptors.shape[1]))
            distances.append(normalized_dist)
    
    # Average pairwise diversity
    diversity_score = float(np.mean(distances)) if distances else 0.0
    return diversity_score


def calculate_feature_coverage(evaluator: MAPElitesEvaluator) -> float:
    """
    Calculate percentage of feature space covered by archive.
    
    Args:
        evaluator: MAPElitesEvaluator instance
    
    Returns:
        Coverage percentage [0, 1]
    """
    resolution = evaluator.resolution
    total_bins = resolution * resolution
    filled_bins = len(evaluator.grid)
    return filled_bins / total_bins


def generate_diversity_report(
    evaluator: MAPElitesEvaluator, 
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive diversity analysis report.
    
    This addresses thesis defense questions about generalization and mode collapse.
    
    Args:
        evaluator: MAPElitesEvaluator with archived solutions
        output_path: Optional path to save JSON report
    
    Returns:
        Dictionary with diversity metrics
    """
    diversity_score = calculate_diversity_score(evaluator)
    feature_coverage = calculate_feature_coverage(evaluator)
    
    # Additional metrics
    num_solutions = len(evaluator.grid)
    
    # Performance spread (variance in scores)
    scores = [entry.score for entry in evaluator.grid.values()]
    score_std = float(np.std(scores)) if scores else 0.0
    score_range = (float(np.min(scores)), float(np.max(scores))) if scores else (0.0, 0.0)
    
    # Behavioral descriptor ranges
    linearities = [e.metrics.get('linearity', 0.0) for e in evaluator.grid.values()]
    leniencies = [e.metrics.get('leniency', 0.0) for e in evaluator.grid.values()]
    
    progression = [e.metrics.get('progression_complexity', 0.0) for e in evaluator.grid.values()]
    topology = [e.metrics.get('topology_complexity', 0.0) for e in evaluator.grid.values()]

    linearity_range = (float(np.min(linearities)), float(np.max(linearities))) if linearities else (0.0, 0.0)
    leniency_range = (float(np.min(leniencies)), float(np.max(leniencies))) if leniencies else (0.0, 0.0)
    progression_range = (float(np.min(progression)), float(np.max(progression))) if progression else (0.0, 0.0)
    topology_range = (float(np.min(topology)), float(np.max(topology))) if topology else (0.0, 0.0)
    
    report = {
        'diversity_score': float(diversity_score),
        'feature_coverage': float(feature_coverage),
        'num_solutions': int(num_solutions),
        'score_std': score_std,
        'score_range': score_range,
        'linearity_range': linearity_range,
        'leniency_range': leniency_range,
        'progression_complexity_range': progression_range,
        'topology_complexity_range': topology_range,
        'diversity_classification': _classify_diversity(diversity_score),
        'coverage_classification': _classify_coverage(feature_coverage)
    }
    
    # Print report
    print("\n" + "="*60)
    print("MAP-ELITES DIVERSITY REPORT")
    print("="*60)
    print(f"Diversity Score:        {diversity_score:.3f} (target: >0.35)")
    print(f"Feature Coverage:       {feature_coverage*100:.1f}%")
    print(f"Solutions in Archive:   {num_solutions}")
    print(f"Score Std Dev:          {score_std:.3f}")
    print(f"Score Range:            [{score_range[0]:.1f}, {score_range[1]:.1f}]")
    print(f"Linearity Range:        [{linearity_range[0]:.3f}, {linearity_range[1]:.3f}]")
    print(f"Leniency Range:         [{leniency_range[0]:.3f}, {leniency_range[1]:.3f}]")
    print(f"Progression Range:      [{progression_range[0]:.3f}, {progression_range[1]:.3f}]")
    print(f"Topology Range:         [{topology_range[0]:.3f}, {topology_range[1]:.3f}]")
    print(f"Diversity Class:        {report['diversity_classification']}")
    print(f"Coverage Class:         {report['coverage_classification']}")
    print("="*60)
    
    # Save to file
    if output_path:
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved diversity report to {output_path}")
    
    return report


def _classify_diversity(score: float) -> str:
    """Classify diversity level."""
    if score >= 0.35:
        return "HEALTHY DIVERSITY [OK]"
    elif score >= 0.20:
        return "MODERATE DIVERSITY"
    else:
        return "LOW DIVERSITY (Mode Collapse Risk) [FAIL]"


def _classify_coverage(coverage: float) -> str:
    """Classify feature space coverage."""
    if coverage >= 0.15:
        return "GOOD COVERAGE [OK]"
    elif coverage >= 0.05:
        return "MODERATE COVERAGE"
    else:
        return "LOW COVERAGE [FAIL]"
