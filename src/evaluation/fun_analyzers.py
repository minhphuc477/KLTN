"""Analyzer implementations for frustration/exploration/flow/pacing metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

from src.evaluation.fun_types import (
    ExplorabilityMetrics,
    FlowMetrics,
    FrustrationMetrics,
    PacingMetrics,
)

SECRET_EDGE_TYPES = frozenset({"soft_locked", "hidden", "secret"})


class FrustrationAnalyzer:
    """Quantifies sources of player frustration."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "backtracking": 0.4,
            "dead_ends": 0.2,
            "unclear_goals": 0.3,
            "empty_rooms": 0.1,
        }

    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
    ) -> FrustrationMetrics:
        backtracking_ratio = self._compute_backtracking(mission_graph, solution_path)
        dead_end_count = self._count_dead_ends(mission_graph, room_contents)
        unclear_goal_score = self._compute_goal_clarity(mission_graph, room_contents)
        empty_room_ratio = self._compute_empty_room_ratio(room_contents)

        total_frustration = (
            self.weights["backtracking"] * backtracking_ratio
            + self.weights["dead_ends"] * (dead_end_count / max(len(mission_graph.nodes), 1))
            + self.weights["unclear_goals"] * unclear_goal_score
            + self.weights["empty_rooms"] * empty_room_ratio
        )

        return FrustrationMetrics(
            backtracking_ratio=backtracking_ratio,
            dead_end_count=dead_end_count,
            unclear_goal_score=unclear_goal_score,
            empty_room_ratio=empty_room_ratio,
            total_frustration=total_frustration,
        )

    def _compute_backtracking(self, mission_graph: nx.Graph, solution_path: List[int]) -> float:
        del mission_graph
        if not solution_path:
            return 0.0

        visited_rooms = set()
        revisits = 0
        for room in solution_path:
            if room in visited_rooms:
                revisits += 1
            visited_rooms.add(room)

        total_visits = len(solution_path)
        return revisits / max(total_visits, 1)

    def _count_dead_ends(self, mission_graph: nx.Graph, room_contents: Dict[int, Dict]) -> int:
        dead_ends = 0
        for node in mission_graph.nodes():
            if mission_graph.degree(node) <= 1:
                content = room_contents.get(node, {})
                has_content = any(
                    [
                        content.get("keys", 0) > 0,
                        content.get("treasures", 0) > 0,
                        content.get("boss", False),
                        content.get("goal", False),
                    ]
                )
                if not has_content:
                    dead_ends += 1
        return dead_ends

    def _compute_goal_clarity(self, mission_graph: nx.Graph, room_contents: Dict[int, Dict]) -> float:
        graph_nodes = set(mission_graph.nodes)
        if not graph_nodes:
            return 0.0

        if mission_graph.is_directed():
            # A single forward choice is progression, not confusion.
            excess_choices = [
                min(max(int(mission_graph.out_degree(node)) - 1, 0) / 3.0, 1.0)
                for node in graph_nodes
            ]
        else:
            # Degree-two corridors are similarly unambiguous in undirected maps.
            excess_choices = [
                min(max(int(mission_graph.degree(node)) - 2, 0) / 2.0, 1.0)
                for node in graph_nodes
            ]
        branching_confusion = float(np.mean(excess_choices))

        goal_rooms = {
            node for node in graph_nodes
            if room_contents.get(node, {}).get("goal", False)
            or room_contents.get(node, {}).get("boss", False)
        }
        # A single final boss must not erase the layout-level branching signal.
        goal_visibility = min(len(goal_rooms) / max(len(graph_nodes), 1), 1.0)
        return branching_confusion * (1.0 - goal_visibility)

    def _compute_empty_room_ratio(self, room_contents: Dict[int, Dict]) -> float:
        if not room_contents:
            return 0.0

        empty_count = 0
        for content in room_contents.values():
            has_content = any(
                [
                    content.get("enemies", 0) > 0,
                    content.get("keys", 0) > 0,
                    content.get("treasures", 0) > 0,
                    content.get("puzzles", 0) > 0,
                ]
            )
            if not has_content:
                empty_count += 1

        return empty_count / len(room_contents)


class ExplorabilityAnalyzer:
    """Quantifies exploration potential and discovery richness."""

    def analyze(
        self,
        mission_graph: nx.Graph,
        critical_path: Set[int],
        room_contents: Dict[int, Dict],
    ) -> ExplorabilityMetrics:
        total_rooms = len(mission_graph.nodes)
        graph_nodes = set(mission_graph.nodes)
        optional_rooms = max(0, total_rooms - len(set(critical_path) & graph_nodes))
        optional_ratio = optional_rooms / max(total_rooms, 1)
        secret_count = self._count_secrets(mission_graph)
        reward_density = self._compute_reward_density(room_contents)

        discovery_potential = (
            0.4 * optional_ratio
            + 0.3 * min(secret_count / max(total_rooms, 1), 1.0)
            + 0.3 * reward_density
        )

        return ExplorabilityMetrics(
            optional_room_ratio=optional_ratio,
            secret_count=secret_count,
            reward_density=reward_density,
            discovery_potential=discovery_potential,
        )

    def _count_secrets(self, mission_graph: nx.Graph) -> int:
        secret_count = 0
        for _, _, data in mission_graph.edges(data=True):
            edge_type = data.get("type", "")
            if edge_type in SECRET_EDGE_TYPES:
                secret_count += 1
        return secret_count

    def _compute_reward_density(self, room_contents: Dict[int, Dict]) -> float:
        if not room_contents:
            return 0.0

        total_rewards = sum(
            content.get("keys", 0) + content.get("treasures", 0) + content.get("items", 0)
            for content in room_contents.values()
        )

        reward_density = total_rewards / len(room_contents)
        return min(reward_density / 3.0, 1.0)


class FlowAnalyzer:
    """Measures flow-state quality from challenge progression and variety."""

    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
    ) -> FlowMetrics:
        del mission_graph
        difficulty_progression = self._compute_difficulty_progression(solution_path, room_contents)
        skill_utilization = self._compute_skill_utilization(room_contents)
        challenge_balance = self._compute_challenge_balance(room_contents)

        flow_score = (
            0.4 * difficulty_progression + 0.3 * skill_utilization + 0.3 * challenge_balance
        )

        return FlowMetrics(
            difficulty_progression=difficulty_progression,
            skill_utilization=skill_utilization,
            challenge_balance=challenge_balance,
            flow_score=flow_score,
        )

    def _compute_difficulty_progression(self, solution_path: List[int], room_contents: Dict[int, Dict]) -> float:
        if len(solution_path) < 2:
            return 0.0

        difficulties = []
        for room in solution_path:
            content = room_contents.get(room, {})
            difficulty = self._calculate_weighted_difficulty(content)
            difficulties.append(difficulty)

        deltas = np.diff(np.asarray(difficulties, dtype=np.float32))
        progressive_steps = int(np.count_nonzero(deltas > 1e-6))
        regressive_steps = int(np.count_nonzero(deltas < -1e-6))
        return float(np.clip((progressive_steps - regressive_steps) / max(len(deltas), 1), 0.0, 1.0))

    def _calculate_weighted_difficulty(self, room_content: Dict) -> float:
        enemy_count = room_content.get("enemies", 0)
        avg_enemy_hp = room_content.get("avg_enemy_hp", 30)
        player_dps = 10.0

        if enemy_count > 0:
            combat_score = (enemy_count * avg_enemy_hp) / player_dps
            combat_score = min(combat_score / 30.0, 1.0)
        else:
            combat_score = 0.0

        shortest_path_tiles = room_content.get("path_length", 20)
        room_width = room_content.get("room_width", 11)
        room_height = room_content.get("room_height", 7)

        euclidean_distance = np.sqrt(room_width**2 + room_height**2) * 0.5
        euclidean_distance = max(euclidean_distance, 1.0)

        nav_complexity = shortest_path_tiles / euclidean_distance
        nav_complexity = min((nav_complexity - 1.0) / 2.0, 1.0)
        nav_complexity = max(nav_complexity, 0.0)

        health_drops = room_content.get("health_pickups", 1)
        expected_damage = enemy_count * 0.5

        if expected_damage > 0:
            resource_scarcity = 1.0 - min(health_drops / expected_damage, 1.0)
        else:
            resource_scarcity = 0.0

        difficulty = 0.4 * combat_score + 0.4 * nav_complexity + 0.2 * resource_scarcity
        return difficulty

    def _compute_skill_utilization(self, room_contents: Dict[int, Dict]) -> float:
        mechanic_types = set()
        for content in room_contents.values():
            if content.get("enemies", 0) > 0:
                mechanic_types.add("combat")
            if content.get("puzzles", 0) > 0:
                mechanic_types.add("puzzle")
            if content.get("keys", 0) > 0:
                mechanic_types.add("key_hunt")
            if content.get("boss", False):
                mechanic_types.add("boss")
        return min(len(mechanic_types) / 4.0, 1.0)

    def _compute_challenge_balance(self, room_contents: Dict[int, Dict]) -> float:
        difficulties = []
        for content in room_contents.values():
            difficulty = content.get("enemies", 0) * 0.5 + content.get("puzzles", 0) * 0.5
            difficulties.append(difficulty)

        if not difficulties:
            return 0.0

        avg_difficulty = np.mean(difficulties)
        ideal = 2.5
        return 1.0 - min(abs(avg_difficulty - ideal) / ideal, 1.0)


class PacingAnalyzer:
    """Estimates pacing quality from a tension curve over the solution path."""

    def __init__(self, target_peak_position: float = 0.75):
        self.target_peak_position = float(np.clip(target_peak_position, 0.0, 1.0))

    def analyze(
        self,
        mission_graph: nx.Graph,
        solution_path: List[int],
        room_contents: Dict[int, Dict],
    ) -> PacingMetrics:
        del mission_graph

        tension_curve = self._compute_tension_curve(solution_path, room_contents)
        if tension_curve.size == 0:
            return PacingMetrics(
                tension_variance=0.0,
                peak_placement=self.target_peak_position,
                rest_areas=0,
                pacing_score=0.0,
            )

        if tension_curve.size == 1:
            return PacingMetrics(
                tension_variance=0.0,
                peak_placement=1.0,
                rest_areas=int(tension_curve[0] < 0.35),
                pacing_score=0.0,
            )

        first_diff = np.diff(tension_curve)
        tension_variance = float(np.var(first_diff))

        peak_idx = int(np.argmax(tension_curve))
        peak_placement = float(peak_idx / max(1, tension_curve.size - 1))

        rest_areas = int(self._count_rest_areas(tension_curve))
        target_curve = self._target_curve(tension_curve.size)
        rmse = float(np.sqrt(np.mean((tension_curve - target_curve) ** 2)))
        curve_alignment_score = float(np.clip(1.0 - rmse, 0.0, 1.0))

        peak_score = float(np.clip(1.0 - abs(peak_placement - self.target_peak_position), 0.0, 1.0))

        target_variance = 0.02
        variance_score = float(
            np.exp(-((tension_variance - target_variance) ** 2) / (2.0 * target_variance**2))
        )

        rest_ratio = rest_areas / max(1, tension_curve.size)
        if rest_ratio < 0.1:
            rest_score = rest_ratio / 0.1
        elif rest_ratio <= 0.35:
            rest_score = 1.0
        else:
            rest_score = max(0.0, 1.0 - (rest_ratio - 0.35) / 0.35)

        pacing_score = (
            0.45 * curve_alignment_score
            + 0.25 * peak_score
            + 0.15 * variance_score
            + 0.15 * rest_score
        )
        pacing_score = float(np.clip(pacing_score, 0.0, 1.0))

        return PacingMetrics(
            tension_variance=tension_variance,
            peak_placement=peak_placement,
            rest_areas=rest_areas,
            pacing_score=pacing_score,
        )

    def _compute_tension_curve(self, solution_path: List[int], room_contents: Dict[int, Dict]) -> np.ndarray:
        if not solution_path:
            return np.array([], dtype=np.float32)

        tensions: List[float] = []
        for room_id in solution_path:
            content = room_contents.get(room_id, {})

            challenge = (
                0.45 * float(content.get("enemies", 0))
                + 0.35 * float(content.get("puzzles", 0))
                + 0.40 * float(content.get("locks", 0))
                + (1.50 if content.get("boss", False) else 0.0)
            )

            recovery = (
                0.30 * float(content.get("health_pickups", 0))
                + 0.20 * float(content.get("keys", 0))
                + 0.20 * float(content.get("items", 0))
                + 0.15 * float(content.get("treasures", 0))
                + (0.50 if content.get("safe_room", False) else 0.0)
            )

            tension_value = max(0.0, challenge - 0.6 * recovery)
            tensions.append(tension_value)

        curve = np.asarray(tensions, dtype=np.float32)
        if curve.max() > 0:
            curve = curve / curve.max()

        if curve.size >= 3:
            kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
            curve = np.convolve(np.pad(curve, (1, 1), mode="edge"), kernel, mode="valid")
            curve = np.clip(curve, 0.0, 1.0)

        return curve

    def _count_rest_areas(self, curve: np.ndarray) -> int:
        if curve.size == 0:
            return 0

        rest_threshold = min(0.35, float(np.quantile(curve, 0.35)))
        rest_areas = 0
        for i, value in enumerate(curve):
            prev_v = curve[i - 1] if i > 0 else value
            next_v = curve[i + 1] if i < curve.size - 1 else value
            if value <= rest_threshold and value <= prev_v and value <= next_v:
                rest_areas += 1
        return rest_areas

    def _target_curve(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.array([], dtype=np.float32)

        x = np.linspace(0.0, 1.0, length, dtype=np.float32)
        base_rise = 0.15 + 0.70 * x
        mid_dip = 0.18 * np.exp(-((x - 0.55) ** 2) / (2.0 * 0.08**2))
        late_peak = 0.22 * np.exp(
            -((x - self.target_peak_position) ** 2) / (2.0 * 0.06**2)
        )
        target = np.clip(base_rise - mid_dip + late_peak, 0.0, 1.0)
        return target.astype(np.float32)
