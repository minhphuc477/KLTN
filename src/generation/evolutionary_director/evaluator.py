"""Topology tension and quality evaluators."""

from __future__ import annotations

from ._shared import *
from .converters import mission_graph_to_networkx, networkx_to_mission_graph

class TensionCurveEvaluator:
    """
    Evaluates how well a graph's tension curve matches a target curve.
    
    Tension is extracted from the critical path (START -> GOAL) by
    assigning difficulty values to each node type and interpolating.
    """
    
    # Node type difficulty weights
    NODE_DIFFICULTIES = {
        'START': 0.0,
        'GOAL': 1.0,
        'ENEMY': 0.5,
        'PUZZLE': 0.6,
        'LOCK': 0.7,
        'KEY': 0.3,
        'ITEM': 0.4,
        'EMPTY': 0.1,
    }
    GATE_EDGE_TYPES = {
        EdgeType.LOCKED,
        EdgeType.BOSS_LOCKED,
        EdgeType.ITEM_GATE,
        EdgeType.ON_OFF_GATE,
        EdgeType.STATE_BLOCK,
        EdgeType.MULTI_LOCK,
        EdgeType.SHUTTER,
        EdgeType.HAZARD,
    }
    SHORTCUT_EDGE_TYPES = {
        EdgeType.SHORTCUT,
        EdgeType.WARP,
    }
    REVERSE_TRAVERSABLE_EDGE_TYPES = {
        EdgeType.PATH,
        EdgeType.SHORTCUT,
        EdgeType.WARP,
        EdgeType.STAIRS,
        EdgeType.HIDDEN,
    }
    
    def __init__(
        self,
        target_curve: List[float],
        descriptor_targets: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize evaluator with target curve.
        
        Args:
            target_curve: Desired difficulty progression (normalized 0-1)
        """
        self.target_curve = np.array(target_curve, dtype=np.float32)
        self.target_length = len(target_curve)
        self.legacy_baseline_mode = descriptor_targets is None
        provided_targets = descriptor_targets or {}
        # Descriptor targets anchor search toward VGLC-like topology.
        self.target_linearity = float(
            np.clip(float(provided_targets.get("linearity", 0.58 - (0.0020 * self.target_length))), 0.36, 0.72)
        )
        self.target_leniency = float(
            np.clip(float(provided_targets.get("leniency", 0.42)), 0.12, 0.85)
        )
        self.target_progression_complexity = float(
            np.clip(float(provided_targets.get("progression_complexity", 0.52 + (0.006 * self.target_length))), 0.45, 0.85)
        )
        self.target_topology_complexity = float(
            np.clip(float(provided_targets.get("topology_complexity", 0.28 + (0.007 * self.target_length))), 0.20, 0.75)
        )
        self.target_path_length = float(max(0.0, float(provided_targets.get("path_length", 0.0))))
        self.target_num_nodes = float(max(0.0, float(provided_targets.get("num_nodes", 0.0))))
        self.target_num_edges = float(max(0.0, float(provided_targets.get("num_edges", 0.0))))
        self.target_key_count = float(max(0.0, float(provided_targets.get("key_count", 0.0))))
        self.target_lock_count = float(max(0.0, float(provided_targets.get("lock_count", 0.0))))
        self.exact_keylock_targets_enabled = bool(
            ("key_count" in provided_targets) or ("lock_count" in provided_targets)
        )
        self.target_cycle_density = float(
            np.clip(
                float(
                    provided_targets.get(
                        "cycle_density",
                        0.70 * float(self.target_topology_complexity),
                    )
                ),
                0.05,
                0.90,
            )
        )
        self.target_shortcut_density = float(
            np.clip(
                float(
                    provided_targets.get(
                        "shortcut_density",
                        0.08 + (0.35 * float(self.target_topology_complexity)),
                    )
                ),
                0.0,
                0.80,
            )
        )
        self.target_gate_depth_ratio = float(
            np.clip(
                float(
                    provided_targets.get(
                        "gate_depth_ratio",
                        0.10 + (0.40 * float(self.target_progression_complexity)),
                    )
                ),
                0.05,
                0.85,
            )
        )
        self.target_directionality_gap = float(
            np.clip(
                float(provided_targets.get("directionality_gap", 0.0)),
                0.0,
                0.30,
            )
        )
        self.target_gating_density = float(
            np.clip(
                float(
                    provided_targets.get(
                        "gating_density",
                        0.08 + (0.22 * float(self.target_progression_complexity)),
                    )
                ),
                0.02,
                0.65,
            )
        )

        if self.target_num_nodes > 0.0:
            self.min_nodes_soft = max(3, int(round(0.78 * self.target_num_nodes)))
            self.max_nodes_soft = max(
                self.min_nodes_soft + 2,
                int(round(1.40 * self.target_num_nodes)),
            )
        else:
            # Dynamic node-count band tied to desired mission length.
            # This avoids a fixed bias toward tiny graphs when evaluating larger dungeons.
            self.min_nodes_soft = max(3, int(round(0.46 * max(1, self.target_length))))
            self.max_nodes_soft = max(18, int(round(1.80 * max(1, self.target_length))))

        if self.target_num_edges > 0.0:
            self.min_edges_soft = max(1, int(round(0.72 * self.target_num_edges)))
            self.max_edges_soft = max(
                self.min_edges_soft + 3,
                int(round(1.30 * self.target_num_edges)),
            )
        else:
            # Approximate sparse-to-mid density band for mission graphs.
            self.min_edges_soft = max(1, int(round(1.35 * self.min_nodes_soft)))
            self.max_edges_soft = max(self.min_edges_soft + 4, int(round(2.90 * self.max_nodes_soft)))

        if self.target_path_length > 0.0:
            desired_edges = self.target_path_length
            self.min_critical_edges = max(1, int(math.floor(0.55 * desired_edges)))
            self.max_critical_edges = max(
                self.min_critical_edges + 1,
                int(math.ceil(1.55 * desired_edges)),
            )
        else:
            desired_edges = 0.40 * max(1, self.target_length)
            self.min_critical_edges = max(1, int(math.floor(0.45 * desired_edges)))
            self.max_critical_edges = max(
                self.min_critical_edges + 1,
                int(math.ceil(2.00 * desired_edges)),
            )
        self.desired_critical_edges = max(2, int(round(desired_edges)))
        self.target_path_depth_ratio = float(
            np.clip(
                float(
                    provided_targets.get(
                        "path_depth_ratio",
                        float(self.desired_critical_edges) / float(max(2, self.max_nodes_soft)),
                    )
                ),
                0.08,
                0.95,
            )
        )
        self.max_directionality_gap = float(
            np.clip(self.target_directionality_gap + 0.10, 0.03, 0.45)
        )
        # Keep generation-time constraint skips low so search pressure improves
        # rule applicability instead of relying on repeated failed attempts.
        self.target_generation_rejection_ratio = float(
            np.clip(
                float(provided_targets.get("generation_rejection_ratio", 0.012)),
                0.0,
                0.80,
            )
        )
        self.max_generation_rejection_ratio = float(
            np.clip(self.target_generation_rejection_ratio + 0.045, 0.03, 0.90)
        )
        # Additional feature targets to avoid over-optimizing only key/enemy signals.
        self.target_puzzle_density = float(
            np.clip(float(provided_targets.get("puzzle_density", 0.34)), 0.02, 0.95)
        )
        self.target_item_density = float(
            np.clip(float(provided_targets.get("item_density", 0.22)), 0.01, 0.90)
        )
        self.target_gate_variety = float(
            np.clip(float(provided_targets.get("gate_variety", 0.45)), 0.0, 1.0)
        )
        self.target_bombable_ratio = float(
            np.clip(float(provided_targets.get("bombable_ratio", 0.12)), 0.0, 1.0)
        )
        self.target_soft_lock_ratio = float(
            np.clip(float(provided_targets.get("soft_lock_ratio", 0.18)), 0.0, 1.0)
        )
        self.target_switch_ratio = float(
            np.clip(float(provided_targets.get("switch_ratio", 0.02)), 0.0, 1.0)
        )
        self.target_stair_ratio = float(
            np.clip(float(provided_targets.get("stair_ratio", 0.03)), 0.0, 1.0)
        )
        # Encourage richer tutorial -> combat -> complex puzzle intent in
        # automatic topology generation instead of treating all puzzles as one
        # undifferentiated bucket.
        self.target_pedagogical_puzzle_variety = float(
            np.clip(
                float(provided_targets.get("pedagogical_puzzle_variety", 0.58 + (0.008 * self.target_length))),
                0.0,
                1.0,
            )
        )
        self.target_skill_chain_score = float(
            np.clip(
                float(provided_targets.get("skill_chain_score", 0.50 + (0.010 * self.target_length))),
                0.0,
                1.0,
            )
        )
        self.target_tutorial_climax_depth_score = float(
            np.clip(
                float(provided_targets.get("tutorial_climax_depth_score", 0.44 + (0.012 * self.target_length))),
                0.0,
                1.0,
            )
        )
        # Hard global difficulty-curve constraints (opt-in via descriptor targets).
        self.min_curve_alignment_score = float(
            np.clip(float(provided_targets.get("difficulty_curve_min_alignment", 0.0)), 0.0, 1.0)
        )
        self.min_curve_trend_corr = float(
            np.clip(float(provided_targets.get("difficulty_curve_min_trend_corr", -1.0)), -1.0, 1.0)
        )
        # Narrative beat objective/gate.
        self.narrative_beats_enabled = bool(provided_targets.get("narrative_beats_enabled", False))
        self.narrative_score_weight = float(
            np.clip(float(provided_targets.get("narrative_score_weight", 0.06)), 0.0, 0.20)
        )
        self.min_narrative_score = float(
            np.clip(float(provided_targets.get("narrative_min_score", 0.0)), 0.0, 1.0)
        )
        # Optional cognitive objective from CBS-style navigation metrics.
        self.cognitive_persona = str(provided_targets.get("cognitive_persona", "balanced"))
        self.target_cognitive_confusion_ratio = float(
            np.clip(float(provided_targets.get("cognitive_target_confusion_ratio", 1.8)), 1.0, 6.0)
        )
        self.cognitive_score_weight = float(
            np.clip(float(provided_targets.get("cognitive_score_weight", 0.0)), 0.0, 0.18)
        )
        self.min_cognitive_score = float(
            np.clip(float(provided_targets.get("cognitive_min_score", 0.0)), 0.0, 1.0)
        )

        # Minimal criteria used as explicit structural constraints.
        # Keep them as strict floors relative to target topology realism.
        self.min_cycle_density = float(max(0.04, 0.72 * self.target_cycle_density))
        if self.target_shortcut_density <= 0.01:
            self.min_shortcut_density = 0.0
        else:
            self.min_shortcut_density = float(max(0.01, 0.68 * self.target_shortcut_density))
        self.min_gate_depth_ratio = float(max(0.03, 0.70 * self.target_gate_depth_ratio))
        self.min_path_depth_ratio = float(max(0.12, 0.72 * self.target_path_depth_ratio))
        self.min_gating_density = float(max(0.03, 0.72 * self.target_gating_density))

    @staticmethod
    def _clip01(value: float) -> float:
        return float(np.clip(float(value), 0.0, 1.0))

    @staticmethod
    def _score_target(value: float, target: float, tol: float) -> float:
        """
        Symmetric target score in [0, 1], where score=1 at target and
        linearly decreases to 0 when |value-target| >= tol.
        """
        t = max(1e-6, float(tol))
        err = abs(float(value) - float(target))
        return float(np.clip(1.0 - (err / t), 0.0, 1.0))

    @staticmethod
    def _score_count_target(value: float, target: float) -> float:
        """Relative exact-count score for raw designer controls."""
        if float(target) <= 0.0:
            return 1.0
        err = abs(float(value) - float(target)) / max(1.0, abs(float(target)))
        return float(np.clip(1.0 - err, 0.0, 1.0))

    @staticmethod
    def _count_target_gap(value: float, target: float) -> float:
        """Relative count gap used by feasibility penalties."""
        if float(target) <= 0.0:
            return 0.0
        return float(np.clip(abs(float(value) - float(target)) / max(1.0, abs(float(target))), 0.0, 2.0))

    @staticmethod
    def _normalized_error(value: float, target: float, *, floor: float = 0.05) -> float:
        """
        Scale-robust absolute error used for descriptor realism penalties.
        """
        denom = max(float(floor), abs(float(target)), 1e-6)
        return float(abs(float(value) - float(target)) / denom)

    def _topology_realism_error(self, descriptor_metrics: Dict[str, float]) -> float:
        """
        Weighted normalized error over Block I realism descriptors.
        """
        cycle_err = self._normalized_error(
            descriptor_metrics.get("cycle_density", 0.0),
            self.target_cycle_density,
            floor=0.08,
        )
        shortcut_err = self._normalized_error(
            descriptor_metrics.get("shortcut_density", 0.0),
            self.target_shortcut_density,
            floor=0.04,
        )
        gate_err = self._normalized_error(
            descriptor_metrics.get("gate_depth_ratio", 0.0),
            self.target_gate_depth_ratio,
            floor=0.06,
        )
        gating_err = self._normalized_error(
            descriptor_metrics.get("gating_density", 0.0),
            self.target_gating_density,
            floor=0.05,
        )
        path_err = self._normalized_error(
            descriptor_metrics.get("path_depth_ratio", 0.0),
            self.target_path_depth_ratio,
            floor=0.10,
        )
        directionality_err = self._normalized_error(
            descriptor_metrics.get("directionality_gap", 0.0),
            self.target_directionality_gap,
            floor=0.05,
        )
        return float(
            np.clip(
                (0.29 * cycle_err)
                + (0.13 * shortcut_err)
                + (0.23 * gate_err)
                + (0.17 * gating_err)
                + (0.13 * path_err)
                + (0.05 * directionality_err),
                0.0,
                3.0,
            )
        )

    def _under_target_gap(self, descriptor_metrics: Dict[str, float]) -> float:
        """
        Mean normalized shortfall against structural realism targets.

        Unlike symmetric target scoring, this only penalizes under-target
        topology (too few cycles/shortcuts/shallow gate depth).
        """
        checks = [
            ("cycle_density", self.target_cycle_density),
            ("shortcut_density", self.target_shortcut_density),
            ("gate_depth_ratio", self.target_gate_depth_ratio),
            ("path_depth_ratio", self.target_path_depth_ratio),
            ("gating_density", self.target_gating_density),
        ]
        deficits: List[float] = []
        for key, target in checks:
            t = max(1e-6, float(target))
            value = float(descriptor_metrics.get(key, 0.0))
            deficits.append(max(0.0, t - value) / t)
        return float(np.clip(np.mean(deficits) if deficits else 0.0, 0.0, 2.0))

    def _pedagogical_progression_metrics(self, graph: MissionGraph) -> Dict[str, float]:
        """
        Measure whether the graph contains an item-gated tutorial progression.

        We care about two distinct things:
        - subtype variety: are tutorial/combat/complex puzzle roles present?
        - skill-chain quality: does at least one ITEM lead into those roles in
          escalating order?

        This follows the existing AddSkillChainRule contract instead of only
        rewarding generic puzzle density.
        """
        stage_order = (
            NodeType.TUTORIAL_PUZZLE,
            NodeType.COMBAT_PUZZLE,
            NodeType.COMPLEX_PUZZLE,
        )
        counts = {
            NodeType.TUTORIAL_PUZZLE: 0,
            NodeType.COMBAT_PUZZLE: 0,
            NodeType.COMPLEX_PUZZLE: 0,
        }
        for node in graph.nodes.values():
            if node.node_type in counts:
                counts[node.node_type] += 1

        present_types = sum(1 for value in counts.values() if value > 0)
        pedagogical_puzzle_variety = self._clip01(float(present_types) / 3.0)

        best_skill_chain_score = 0.0
        best_tutorial_climax_depth_score = 0.0
        forward_adj = graph.get_forward_adjacency_map()
        item_nodes = graph.get_nodes_by_type(NodeType.ITEM)
        climax_candidates: List[Tuple[int, int]] = []
        for priority, node_type in enumerate((NodeType.BOSS_DOOR, NodeType.BOSS, NodeType.GOAL)):
            for node in graph.get_nodes_by_type(node_type):
                climax_candidates.append((priority, int(node.id)))

        for item_node in item_nodes:
            candidate_paths: List[Tuple[int, int, List[int]]] = []
            for priority, target_id in climax_candidates:
                path = self._find_path_in_adjacency(forward_adj, item_node.id, target_id)
                if path and len(path) >= 2:
                    candidate_paths.append((len(path), priority, list(path)))
            if not candidate_paths:
                continue

            candidate_paths.sort(key=lambda entry: (entry[0], entry[1]))
            path = candidate_paths[0][2]
            ordered_stage_distances: List[int] = []
            previous_index = 0
            local_score = 0.0
            for weight, stage in zip((0.34, 0.33, 0.33), stage_order):
                matched_index = None
                for idx in range(previous_index + 1, len(path)):
                    node = graph.get_node(int(path[idx]))
                    if node is not None and node.node_type == stage:
                        matched_index = idx
                        break
                if matched_index is None:
                    break
                previous_index = matched_index
                ordered_stage_distances.append(int(matched_index))
                local_score += float(weight)
            best_skill_chain_score = max(best_skill_chain_score, float(np.clip(local_score, 0.0, 1.0)))

            if len(ordered_stage_distances) == len(stage_order):
                tutorial_dist = int(ordered_stage_distances[0])
                complex_dist = int(ordered_stage_distances[-1])
                stage_span = max(0, complex_dist - tutorial_dist)
                target_stage_span = max(2.0, 0.35 * float(max(3, self.target_length)))
                chain_depth_score = self._clip01(float(stage_span) / float(target_stage_span))

                climax_dist = max(0, int(len(path) - 1))
                climax_span = max(0, climax_dist - tutorial_dist)
                target_climax_span = max(3.0, 0.60 * float(max(4, self.target_length)))
                climax_depth_score = self._clip01(float(climax_span) / float(target_climax_span))

                local_depth_score = float(
                    np.clip((0.55 * chain_depth_score) + (0.45 * climax_depth_score), 0.0, 1.0)
                )
                best_tutorial_climax_depth_score = max(best_tutorial_climax_depth_score, local_depth_score)

        return {
            "tutorial_puzzle_count": float(counts[NodeType.TUTORIAL_PUZZLE]),
            "combat_puzzle_count": float(counts[NodeType.COMBAT_PUZZLE]),
            "complex_puzzle_count": float(counts[NodeType.COMPLEX_PUZZLE]),
            "pedagogical_puzzle_variety": float(pedagogical_puzzle_variety),
            "skill_chain_score": float(best_skill_chain_score),
            "tutorial_climax_depth_score": float(best_tutorial_climax_depth_score),
        }

    def _edge_for_step(
        self,
        graph: MissionGraph,
        source: int,
        target: int,
    ) -> Optional[MissionEdge]:
        """
        Resolve edge metadata for a traversed step on a path.

        Path traversal can use reverse movement for bidirectional edge types,
        so we also check reversed orientation where semantics allow it.
        """
        for edge in graph.edges:
            if edge.source == source and edge.target == target:
                return edge
        for edge in graph.edges:
            if (
                edge.source == target
                and edge.target == source
                and edge.edge_type in self.REVERSE_TRAVERSABLE_EDGE_TYPES
            ):
                return edge
        return None

    def _critical_path_edges(
        self,
        graph: MissionGraph,
        path: Optional[List[int]],
    ) -> List[MissionEdge]:
        if not path or len(path) < 2:
            return []
        resolved: List[MissionEdge] = []
        for i in range(len(path) - 1):
            edge = self._edge_for_step(graph, int(path[i]), int(path[i + 1]))
            if edge is not None:
                resolved.append(edge)
        return resolved

    def _structural_violation(self, descriptor_metrics: Dict[str, float]) -> float:
        """
        Normalized shortfall against explicit topology/progression criteria.
        """
        shortfalls = []
        checks = [
            ("cycle_density", self.min_cycle_density),
            ("shortcut_density", self.min_shortcut_density),
            ("gate_depth_ratio", self.min_gate_depth_ratio),
            ("path_depth_ratio", self.min_path_depth_ratio),
            ("gating_density", self.min_gating_density),
        ]
        for key, floor in checks:
            floor_v = max(1e-6, float(floor))
            value = float(descriptor_metrics.get(key, 0.0))
            shortfall = max(0.0, floor_v - value) / floor_v
            shortfalls.append(shortfall)
        if not shortfalls:
            return 0.0
        return float(np.clip(np.mean(shortfalls), 0.0, 2.0))

    def _extract_descriptor_metrics(self, graph: MissionGraph) -> Dict[str, float]:
        """
        Extract benchmark-aligned structural descriptors from a MissionGraph.
        """
        node_count = int(len(graph.nodes))
        edge_count = int(len(graph.edges))

        start = graph.get_start_node()
        goal = graph.get_goal_node()
        directed_path = self._find_path(graph, start.id, goal.id) if (start and goal) else None
        weak_path = self._find_weak_path(graph, start.id, goal.id) if (start and goal) else None
        path = directed_path
        path_len = max(0, (len(directed_path) - 1)) if directed_path else 0
        directed_path_len = int(path_len)
        weak_path_len = max(0, (len(weak_path) - 1)) if weak_path else 0
        directionality_gap = 0.0
        if weak_path is not None:
            if directed_path is None:
                directionality_gap = 1.0
            else:
                directionality_gap = self._clip01(
                    max(0.0, float(directed_path_len) - float(weak_path_len))
                    / max(1.0, float(weak_path_len))
                )
        linearity = self._clip01(float(path_len + 1) / float(max(1, node_count)))
        critical_path_edges = self._critical_path_edges(graph, path)
        gate_edges_on_critical = sum(
            1 for edge in critical_path_edges if edge.edge_type in self.GATE_EDGE_TYPES
        )
        gate_depth_ratio = self._clip01(
            float(gate_edges_on_critical) / float(max(1, len(critical_path_edges)))
        )

        key_like_types = {NodeType.KEY, NodeType.BIG_KEY, NodeType.TOKEN}
        enemy_like_types = {NodeType.ENEMY, NodeType.BOSS, NodeType.MINI_BOSS, NodeType.ARENA, NodeType.COMBAT_PUZZLE}
        puzzle_like_types = {NodeType.PUZZLE, NodeType.TUTORIAL_PUZZLE, NodeType.COMBAT_PUZZLE, NodeType.COMPLEX_PUZZLE}
        item_like_types = {NodeType.ITEM, NodeType.TREASURE, NodeType.PROTECTION_ITEM}
        key_count = 0
        enemy_count = 0
        puzzle_count = 0
        item_count = 0
        for node in graph.nodes.values():
            key_hint = int(max(0, int(getattr(node, "key_count_hint", 0) or 0)))
            enemy_hint = int(max(0, int(getattr(node, "enemy_count_hint", 0) or 0)))
            puzzle_hint = int(max(0, int(getattr(node, "puzzle_count_hint", 0) or 0)))
            item_hint = int(max(0, int(getattr(node, "item_count_hint", 0) or 0)))
            if node.node_type in key_like_types:
                key_count += max(1, key_hint)
            else:
                key_count += key_hint
            if node.node_type in enemy_like_types:
                enemy_count += max(1, enemy_hint)
            else:
                enemy_count += enemy_hint
            if node.node_type in puzzle_like_types:
                puzzle_count += max(1, puzzle_hint)
            else:
                puzzle_count += puzzle_hint
            if node.node_type in item_like_types:
                item_count += max(1, item_hint)
            else:
                item_count += item_hint
        pedagogical_metrics = self._pedagogical_progression_metrics(graph)
        lock_count = sum(1 for e in graph.edges if e.edge_type in self.GATE_EDGE_TYPES)
        small_key_supply = sum(1 for node in graph.nodes.values() if node.node_type == NodeType.KEY)
        small_key_demand = 0
        for edge in graph.edges:
            if edge.edge_type != EdgeType.LOCKED:
                continue
            if edge.requires_key_count > 0:
                small_key_demand += int(max(1, edge.requires_key_count))
            else:
                small_key_demand += 1
        boss_key_supply = sum(1 for node in graph.nodes.values() if node.node_type == NodeType.BIG_KEY)
        boss_key_demand = sum(
            1
            for edge in graph.edges
            if edge.edge_type == EdgeType.BOSS_LOCKED
        )
        if boss_key_demand <= 0:
            boss_key_demand = sum(
                1 for node in graph.nodes.values() if node.node_type == NodeType.BOSS_DOOR
            )
        small_key_surplus = max(0.0, float(small_key_supply - small_key_demand))
        boss_key_surplus = max(0.0, float(boss_key_supply - boss_key_demand))
        leniency = self._clip01(float(key_count) / float(max(1, lock_count))) if lock_count > 0 else 1.0

        directed_branch_nodes = 0
        for node_id in graph.nodes.keys():
            out_degree = int(graph.get_out_degree(node_id))
            if out_degree >= 2:
                directed_branch_nodes += 1
        branching_factor = self._clip01(float(directed_branch_nodes) / float(max(1, node_count)))

        # Undirected edge set for cycle rank estimate.
        undirected_edges: Set[Tuple[Any, Any]] = set()
        for e in graph.edges:
            a = e.source
            b = e.target
            if a == b:
                continue
            if str(a) <= str(b):
                undirected_edges.add((a, b))
            else:
                undirected_edges.add((b, a))
        u_edge_count = int(len(undirected_edges))
        components = 1
        if node_count > 0:
            U = nx.Graph()
            U.add_nodes_from(graph.nodes.keys())
            U.add_edges_from(list(undirected_edges))
            components = max(1, nx.number_connected_components(U))
        cycle_rank = max(0, u_edge_count - node_count + components) if node_count > 0 else 0
        cycle_density = self._clip01(float(cycle_rank) / float(max(1, node_count // 2)))
        cyclomatic_complexity = float(cycle_rank)

        # Raw branching factor for Pareto constraints (not normalized 0..1).
        U_branch = nx.DiGraph()
        U_branch.add_nodes_from(graph.nodes.keys())
        U_branch.add_edges_from([(e.source, e.target) for e in graph.edges])
        branching_factor_raw = float(compute_branching_factor(U_branch))

        # Raw loop complexity for hard loop constraints.
        U_loops = nx.Graph()
        U_loops.add_nodes_from(graph.nodes.keys())
        U_loops.add_edges_from(list(undirected_edges))
        cyclomatic_complexity = float(compute_cyclomatic_complexity(U_loops))

        gating_density = self._clip01(float(lock_count) / float(max(1, edge_count)))
        path_pressure = self._clip01(float(path_len) / float(max(1, node_count)))
        path_depth_ratio = self._clip01(float(path_len) / float(max(1, node_count - 1)))
        puzzle_density = self._clip01(float(puzzle_count) / float(max(1, node_count)))
        item_density = self._clip01(float(item_count) / float(max(1, node_count)))
        backtracking_proxy = self._clip01((1.0 - linearity) * 0.65 + cycle_density * 0.35)
        lock_pressure = min(1.0, float(lock_count) / float(max(1, key_count)))
        shortcut_edge_count = sum(
            1
            for edge in graph.edges
            if (
                edge.edge_type in self.SHORTCUT_EDGE_TYPES
                or int(getattr(edge, "path_savings", 0) or 0) >= 2
            )
        )
        shortcut_density = self._clip01(float(shortcut_edge_count) / float(max(1, edge_count)))

        key_lock_count = 0
        bombable_count = 0
        soft_lock_count = 0
        switch_count = 0
        item_gate_count = 0
        stair_count = 0
        for edge in graph.edges:
            constraints = []
            metadata = getattr(edge, "metadata", {}) or {}
            if isinstance(metadata, dict):
                constraints = [str(t).strip().lower() for t in metadata.get("vglc_constraints", []) if str(t).strip()]
            cset = set(constraints)
            et = edge.edge_type

            if et in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED} or ("key_locked" in cset) or ("boss_locked" in cset):
                key_lock_count += 1
            if ("bombable" in cset) or (et == EdgeType.ITEM_GATE and str(edge.item_required or "").upper() == "BOMB"):
                bombable_count += 1
            if ("soft_locked" in cset) or (et in {EdgeType.ONE_WAY, EdgeType.SHUTTER}):
                soft_lock_count += 1
            if ("switch" in cset) or (et in {EdgeType.ON_OFF_GATE, EdgeType.STATE_BLOCK}):
                switch_count += 1
            if ("item_locked" in cset) or (et == EdgeType.ITEM_GATE):
                item_gate_count += 1
            if ("stair" in cset) or (et in {EdgeType.STAIRS, EdgeType.WARP}):
                stair_count += 1

        key_lock_ratio = self._clip01(float(key_lock_count) / float(max(1, edge_count)))
        bombable_ratio = self._clip01(float(bombable_count) / float(max(1, edge_count)))
        soft_lock_ratio = self._clip01(float(soft_lock_count) / float(max(1, edge_count)))
        switch_ratio = self._clip01(float(switch_count) / float(max(1, edge_count)))
        item_gate_ratio = self._clip01(float(item_gate_count) / float(max(1, edge_count)))
        stair_ratio = self._clip01(float(stair_count) / float(max(1, edge_count)))
        gate_variety = self._clip01(
            float(
                sum(
                    1
                    for cnt in [key_lock_count, bombable_count, soft_lock_count, switch_count, item_gate_count, stair_count]
                    if cnt > 0
                )
            )
            / 6.0
        )
        feature_complexity = self._clip01(
            0.35 * puzzle_density
            + 0.20 * item_density
            + 0.15 * bombable_ratio
            + 0.15 * soft_lock_ratio
            + 0.15 * gate_variety
        )

        progression_complexity = self._clip01(
            0.34 * lock_pressure
            + 0.22 * backtracking_proxy
            + 0.18 * path_pressure
            + 0.12 * gate_depth_ratio
            + 0.14 * feature_complexity
        )
        topology_complexity = self._clip01(
            0.30 * branching_factor
            + 0.30 * cycle_density
            + 0.15 * gating_density
            + 0.15 * shortcut_density
            + 0.10 * gate_variety
        )
        gen_stats = getattr(graph, "generation_stats", {})
        if isinstance(gen_stats, dict):
            rule_applications = int(max(0, int(gen_stats.get("rule_applications", 0) or 0)))
            generation_constraint_rejections = int(
                max(0, int(gen_stats.get("generation_constraint_rejections", 0) or 0))
            )
            candidate_repairs_applied = int(max(0, int(gen_stats.get("candidate_repairs_applied", 0) or 0)))
        else:
            rule_applications = 0
            generation_constraint_rejections = 0
            candidate_repairs_applied = 0
        generation_rejection_ratio = self._clip01(
            float(generation_constraint_rejections) / float(max(1, rule_applications))
        )
        candidate_repair_ratio = self._clip01(
            float(candidate_repairs_applied) / float(max(1, rule_applications))
        )

        return {
            "linearity": float(linearity),
            "leniency": float(leniency),
            "progression_complexity": float(progression_complexity),
            "topology_complexity": float(topology_complexity),
            "path_len": float(path_len),
            "directed_path_length": float(directed_path_len),
            "weak_path_length": float(weak_path_len),
            "directionality_gap": float(directionality_gap),
            "node_count": float(node_count),
            "edge_count": float(edge_count),
            "enemy_count": float(enemy_count),
            "key_count": float(key_count),
            "small_key_supply": float(small_key_supply),
            "small_key_demand": float(small_key_demand),
            "small_key_surplus": float(small_key_surplus),
            "boss_key_supply": float(boss_key_supply),
            "boss_key_demand": float(boss_key_demand),
            "boss_key_surplus": float(boss_key_surplus),
            "puzzle_count": float(puzzle_count),
            "item_count": float(item_count),
            "tutorial_puzzle_count": float(pedagogical_metrics["tutorial_puzzle_count"]),
            "combat_puzzle_count": float(pedagogical_metrics["combat_puzzle_count"]),
            "complex_puzzle_count": float(pedagogical_metrics["complex_puzzle_count"]),
            "puzzle_density": float(puzzle_density),
            "item_density": float(item_density),
            "pedagogical_puzzle_variety": float(pedagogical_metrics["pedagogical_puzzle_variety"]),
            "skill_chain_score": float(pedagogical_metrics["skill_chain_score"]),
            "tutorial_climax_depth_score": float(pedagogical_metrics["tutorial_climax_depth_score"]),
            "feature_complexity": float(feature_complexity),
            "branching_factor": float(branching_factor),
            "branching_factor_raw": float(branching_factor_raw),
            "cycle_density": float(cycle_density),
            "cyclomatic_complexity": float(cyclomatic_complexity),
            "shortcut_density": float(shortcut_density),
            "gating_density": float(gating_density),
            "gate_depth_ratio": float(gate_depth_ratio),
            "path_depth_ratio": float(path_depth_ratio),
            "backtracking_proxy": float(backtracking_proxy),
            "key_lock_ratio": float(key_lock_ratio),
            "bombable_ratio": float(bombable_ratio),
            "soft_lock_ratio": float(soft_lock_ratio),
            "switch_ratio": float(switch_ratio),
            "item_gate_ratio": float(item_gate_ratio),
            "stair_ratio": float(stair_ratio),
            "gate_variety": float(gate_variety),
            "rule_applications": float(rule_applications),
            "generation_constraint_rejections": float(generation_constraint_rejections),
            "candidate_repairs_applied": float(candidate_repairs_applied),
            "generation_rejection_ratio": float(generation_rejection_ratio),
            "candidate_repair_ratio": float(candidate_repair_ratio),
        }
    
    def extract_tension_curve(self, graph: MissionGraph) -> np.ndarray:
        """
        Extract tension curve from graph's critical path.
        
        Process:
        1. Find shortest path from START to GOAL
        2. Assign difficulty to each node on path
        3. Interpolate to match target curve length
        
        Args:
            graph: MissionGraph to analyze
            
        Returns:
            Numpy array of tension values (normalized 0-1)
        """
        # Find START and GOAL nodes
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        if not start or not goal:
            logger.warning("Graph missing START or GOAL node")
            return np.zeros(self.target_length)
        
        # Find critical path using BFS
        path = self._find_path(graph, start.id, goal.id)
        
        if not path:
            logger.debug("No path from START to GOAL")
            return np.zeros(self.target_length)
        
        # Extract difficulty values along path
        difficulties = []
        for node_id in path:
            node = graph.nodes[node_id]
            base_difficulty = self.NODE_DIFFICULTIES.get(
                node.node_type.name,
                0.5
            )
            # Mix base difficulty with node's own difficulty
            difficulties.append(base_difficulty * 0.7 + node.difficulty * 0.3)
        
        # Interpolate to target length
        if len(difficulties) == 0:
            return np.zeros(self.target_length)
        
        curve = self._interpolate(difficulties, self.target_length)
        
        # Normalize to 0-1 range
        if curve.max() > 0:
            curve = curve / curve.max()
        
        return curve
    
    def _constraint_violation(self, critical_edges: int, node_count: int, edge_count: int) -> float:
        """
        Compute normalized constraint violation for feasibility-first selection.

        This follows a Deb-style constraint handling pattern:
        prioritize feasible individuals; among infeasible ones, minimize total
        violation magnitude.
        """
        path_low = max(0.0, float(self.min_critical_edges - critical_edges)) / float(max(1, self.min_critical_edges))
        path_high = max(0.0, float(critical_edges - self.max_critical_edges)) / float(max(1, self.max_critical_edges))
        node_low = max(0.0, float(self.min_nodes_soft - node_count)) / float(max(1, self.min_nodes_soft))
        node_high = max(0.0, float(node_count - self.max_nodes_soft)) / float(max(1, self.max_nodes_soft))
        edge_low = max(0.0, float(self.min_edges_soft - edge_count)) / float(max(1, self.min_edges_soft))
        edge_high = max(0.0, float(edge_count - self.max_edges_soft)) / float(max(1, self.max_edges_soft))
        return float(
            np.clip(
                (0.36 * (path_low + path_high))
                + (0.42 * (node_low + node_high))
                + (0.22 * (edge_low + edge_high)),
                0.0,
                2.0,
            )
        )

    def evaluate_graph(self, graph: MissionGraph) -> Dict[str, Any]:
        """
        Evaluate one graph and return quality + feasibility diagnostics.
        """
        # Check solvability first.
        if not self._is_solvable(graph):
            return {
                "fitness": 0.0,
                "feasible": False,
                "constraint_violation": 1.0,
                "critical_edges": 0,
                "node_count": int(len(graph.nodes)),
                "descriptor_metrics": self._extract_descriptor_metrics(graph),
            }

        # Extract tension curve.
        extracted = self.extract_tension_curve(graph)
        mse = np.mean((extracted - self.target_curve) ** 2)
        curve_fitness = 1.0 - min(mse, 1.0)
        curve_trend_corr = self._curve_trend_correlation(extracted, self.target_curve)
        curve_alignment_score = float(np.clip((0.72 * curve_fitness) + (0.28 * ((curve_trend_corr + 1.0) * 0.5)), 0.0, 1.0))

        descriptor_metrics = self._extract_descriptor_metrics(graph)
        descriptor_metrics["curve_fitness"] = float(curve_fitness)
        descriptor_metrics["curve_trend_corr"] = float(curve_trend_corr)
        descriptor_metrics["curve_alignment_score"] = float(curve_alignment_score)

        # Structural backtracking proxy (non-constant; path-simple BFS does not
        # encode revisits, so we use topology descriptors instead).
        backtracking_score = self._calculate_backtracking_score(graph)

        start = graph.get_start_node()
        goal = graph.get_goal_node()
        critical_path = self._find_path(graph, start.id, goal.id) if (start and goal) else None
        critical_edges = max(0, len(critical_path) - 1) if critical_path else 0
        narrative_score = self._score_narrative_beats(graph, critical_path or [])
        descriptor_metrics["narrative_score"] = float(narrative_score)

        cognitive_score = 0.5
        cognitive_metrics: Dict[str, Any] = {}
        if not self.legacy_baseline_mode:
            try:
                # Compute cognitive metrics inline from the candidate's own
                # topology descriptors.  The previous implementation delegated
                # to ``compute_cbs_fitness`` which, when given a networkx
                # graph, used a static proxy that returned the same score for
                # identical topologies — making it impossible for the GA to
                # differentiate candidates on cognitive quality.
                #
                # The inline version below uses per-individual structural
                # features that genuinely vary between candidate topologies:
                #   - branching_factor: proportion of nodes with out-degree ≥ 2
                #   - dead_end_ratio: fraction of non-goal terminal nodes
                #   - cycle_density: cyclomatic complexity normalised by nodes
                #   - path_depth_ratio: critical path length / node count
                #
                # These are combined into a confusion_index that models how
                # cognitively taxing navigation would be, and then scored
                # against the target confusion ratio.

                n = max(1, int(len(graph.nodes)))
                e = max(0, int(len(graph.edges)))

                # Branching factor: high branching increases decision points.
                branch_nodes = 0
                dead_end_nodes = 0
                for node_id in graph.nodes.keys():
                    out_deg = int(graph.get_out_degree(node_id))
                    if out_deg >= 2:
                        branch_nodes += 1
                    if out_deg == 0:
                        node = graph.nodes[node_id]
                        if node.node_type != NodeType.GOAL:
                            dead_end_nodes += 1
                branch_pressure = float(np.clip(float(branch_nodes) / float(n), 0.0, 1.0))
                dead_end_ratio = float(np.clip(float(dead_end_nodes) / float(n), 0.0, 1.0))

                # Reuse already-computed descriptor values where available.
                cd = float(descriptor_metrics.get("cycle_density", 0.0))
                pdr = float(descriptor_metrics.get("path_depth_ratio", 0.0))

                # Confusion index: models cognitive load from topology.
                confusion_index = float(np.clip(
                    (0.35 * branch_pressure)
                    + (0.25 * dead_end_ratio)
                    + (0.20 * cd)
                    + (0.20 * (1.0 - pdr)),  # low path-depth → more wandering
                    0.0,
                    3.0,
                ))
                confusion_ratio = 1.0 + confusion_index

                # Score: how close is the topology's confusion to the target?
                target = max(1.0, float(self.target_cognitive_confusion_ratio))
                target_normalised = max(0.0, target - 1.0)
                cr_penalty = (confusion_index - target_normalised) ** 2
                cognitive_score = float(np.clip(1.0 / (1.0 + cr_penalty), 0.0, 1.0))

                path_efficiency = float(pdr)
                room_entropy = float(np.clip(
                    float(np.std([int(graph.get_out_degree(nid)) for nid in graph.nodes.keys()]))
                    / max(1.0, float(np.mean([int(graph.get_out_degree(nid)) for nid in graph.nodes.keys()])) + 1e-8),
                    0.0,
                    1.0,
                )) if n > 1 else 0.0

                cognitive_metrics = {
                    "fitness": float(cognitive_score),
                    "confusion_ratio": float(confusion_ratio),
                    "confusion_index": float(confusion_index),
                    "path_efficiency": float(path_efficiency),
                    "room_entropy": float(room_entropy),
                    "is_proxy": 0.0,
                }
            except (ImportError, RuntimeError, ValueError, TypeError, KeyError) as error:
                logger.debug("Inline cognitive scoring failed, using neutral score: %s", error)
                cognitive_metrics = {}
                cognitive_score = 0.5
        descriptor_metrics["cognitive_score"] = float(cognitive_score)
        descriptor_metrics["cognitive_confusion_ratio"] = float(cognitive_metrics.get("confusion_ratio", 0.0))
        descriptor_metrics["cognitive_path_efficiency"] = float(cognitive_metrics.get("path_efficiency", 0.0))
        descriptor_metrics["cognitive_room_entropy"] = float(cognitive_metrics.get("room_entropy", 0.0))
        descriptor_metrics["cognitive_proxy_mode"] = float(cognitive_metrics.get("is_proxy", 0.0))

        path_depth_score = 1.0 - min(
            abs(float(critical_edges - self.desired_critical_edges)) / float(max(1, self.desired_critical_edges)),
            1.0,
        )

        node_count = max(1, len(graph.nodes))
        edge_count = max(1, len(graph.edges))
        path_coverage = float(critical_edges) / float(max(1, node_count - 1))
        desired_coverage = 0.40
        coverage_score = 1.0 - min(
            abs(path_coverage - desired_coverage) / max(1e-6, desired_coverage),
            1.0,
        )
        progression_score = float(np.clip((0.65 * path_depth_score) + (0.35 * coverage_score), 0.0, 1.0))

        descriptor_score = float(np.clip(
            0.24 * self._score_target(descriptor_metrics["linearity"], self.target_linearity, tol=0.26)
            + 0.24 * self._score_target(descriptor_metrics["leniency"], self.target_leniency, tol=0.26)
            + 0.22 * self._score_target(
                descriptor_metrics["progression_complexity"],
                self.target_progression_complexity,
                tol=0.30,
            )
            + 0.30 * self._score_target(
                descriptor_metrics["topology_complexity"],
                self.target_topology_complexity,
                tol=0.32,
            ),
            0.0,
            1.0,
        ))
        feature_score = float(np.clip(
            0.17 * self._score_target(
                descriptor_metrics.get("puzzle_density", 0.0),
                self.target_puzzle_density,
                tol=0.35,
            )
            + 0.11 * self._score_target(
                descriptor_metrics.get("item_density", 0.0),
                self.target_item_density,
                tol=0.35,
            )
            + 0.16 * self._score_target(
                descriptor_metrics.get("gate_variety", 0.0),
                self.target_gate_variety,
                tol=0.40,
            )
            + 0.10 * self._score_target(
                descriptor_metrics.get("bombable_ratio", 0.0),
                self.target_bombable_ratio,
                tol=0.30,
            )
            + 0.10 * self._score_target(
                descriptor_metrics.get("soft_lock_ratio", 0.0),
                self.target_soft_lock_ratio,
                tol=0.30,
            )
            + 0.05 * self._score_target(
                descriptor_metrics.get("switch_ratio", 0.0),
                self.target_switch_ratio,
                tol=0.20,
            )
            + 0.05 * self._score_target(
                descriptor_metrics.get("stair_ratio", 0.0),
                self.target_stair_ratio,
                tol=0.20,
            )
            + 0.08 * self._score_target(
                descriptor_metrics.get("pedagogical_puzzle_variety", 0.0),
                self.target_pedagogical_puzzle_variety,
                tol=0.28,
            )
            + 0.08 * self._score_target(
                descriptor_metrics.get("skill_chain_score", 0.0),
                self.target_skill_chain_score,
                tol=0.26,
            )
            + 0.10 * self._score_target(
                descriptor_metrics.get("tutorial_climax_depth_score", 0.0),
                self.target_tutorial_climax_depth_score,
                tol=0.24,
            ),
            0.0,
            1.0,
        ))
        descriptor_score = float(np.clip((0.78 * descriptor_score) + (0.22 * feature_score), 0.0, 1.0))
        descriptor_metrics["feature_score"] = float(feature_score)
        cycle_score = self._score_target(
            descriptor_metrics.get("cycle_density", 0.0),
            self.target_cycle_density,
            tol=0.16,
        )
        shortcut_score = self._score_target(
            descriptor_metrics.get("shortcut_density", 0.0),
            self.target_shortcut_density,
            tol=0.14,
        )
        gate_depth_score = self._score_target(
            descriptor_metrics.get("gate_depth_ratio", 0.0),
            self.target_gate_depth_ratio,
            tol=0.12,
        )
        path_depth_ratio_score = self._score_target(
            descriptor_metrics.get("path_depth_ratio", 0.0),
            self.target_path_depth_ratio,
            tol=0.12,
        )
        directionality_score = self._score_target(
            descriptor_metrics.get("directionality_gap", 0.0),
            self.target_directionality_gap,
            tol=0.08,
        )
        gating_density_score = self._score_target(
            descriptor_metrics.get("gating_density", 0.0),
            self.target_gating_density,
            tol=0.10,
        )
        key_count_score = self._score_count_target(
            descriptor_metrics.get("key_count", 0.0),
            self.target_key_count,
        )
        lock_count_score = self._score_count_target(
            descriptor_metrics.get("lock_count", 0.0),
            self.target_lock_count,
        )
        key_count_gap = self._count_target_gap(
            descriptor_metrics.get("key_count", 0.0),
            self.target_key_count,
        )
        lock_count_gap = self._count_target_gap(
            descriptor_metrics.get("lock_count", 0.0),
            self.target_lock_count,
        )
        keylock_exact_score = float(np.clip(0.50 * key_count_score + 0.50 * lock_count_score, 0.0, 1.0))
        if bool(self.exact_keylock_targets_enabled):
            descriptor_score = float(np.clip((0.85 * descriptor_score) + (0.15 * keylock_exact_score), 0.0, 1.0))
        if self.target_num_nodes > 0.0:
            node_count_score = 1.0 - min(
                abs(float(node_count) - float(self.target_num_nodes)) / float(max(1.0, self.target_num_nodes)),
                1.0,
            )
        else:
            node_count_score = self._score_target(
                float(node_count),
                0.5 * float(self.min_nodes_soft + self.max_nodes_soft),
                tol=max(2.0, 0.5 * float(self.max_nodes_soft - self.min_nodes_soft)),
            )
        if self.target_num_edges > 0.0:
            edge_count_score = 1.0 - min(
                abs(float(edge_count) - float(self.target_num_edges)) / float(max(1.0, self.target_num_edges)),
                1.0,
            )
        else:
            edge_count_score = self._score_target(
                float(edge_count),
                0.5 * float(self.min_edges_soft + self.max_edges_soft),
                tol=max(3.0, 0.5 * float(self.max_edges_soft - self.min_edges_soft)),
            )
        structural_objective_score = float(np.clip(
            0.24 * cycle_score
            + 0.14 * shortcut_score
            + 0.18 * gate_depth_score
            + 0.14 * gating_density_score
            + 0.12 * path_depth_ratio_score
            + 0.06 * directionality_score
            + 0.07 * node_count_score
            + 0.05 * edge_count_score,
            0.0,
            1.0,
        ))
        if bool(self.exact_keylock_targets_enabled):
            structural_objective_score = float(
                np.clip((0.90 * structural_objective_score) + (0.10 * keylock_exact_score), 0.0, 1.0)
            )
        under_target_gap = self._under_target_gap(descriptor_metrics)
        linearity_under_gap = max(
            0.0,
            float(self.target_linearity) - float(descriptor_metrics.get("linearity", 0.0)),
        ) / max(0.08, float(self.target_linearity))
        linearity_under_gap = float(np.clip(linearity_under_gap, 0.0, 2.0))
        leniency_excess_gap = max(
            0.0,
            float(descriptor_metrics.get("leniency", 0.0)) - float(self.target_leniency),
        ) / max(0.06, float(self.target_leniency))
        leniency_excess_gap = float(np.clip(leniency_excess_gap, 0.0, 2.0))
        tutorial_climax_depth_gap = max(
            0.0,
            float(self.target_tutorial_climax_depth_score)
            - float(descriptor_metrics.get("tutorial_climax_depth_score", 0.0)),
        ) / max(0.08, float(self.target_tutorial_climax_depth_score))
        tutorial_climax_depth_gap = float(np.clip(tutorial_climax_depth_gap, 0.0, 2.0))
        shortcut_density_value = float(descriptor_metrics.get("shortcut_density", 0.0))
        shortcut_excess_gap = max(0.0, shortcut_density_value - float(self.target_shortcut_density)) / max(
            0.05,
            float(self.target_shortcut_density),
        )
        shortcut_excess_gap = float(np.clip(shortcut_excess_gap, 0.0, 2.0))
        directionality_excess_gap = max(
            0.0,
            float(descriptor_metrics.get("directionality_gap", 0.0))
            - float(self.target_directionality_gap),
        ) / max(0.05, float(self.target_directionality_gap) + 0.05)
        directionality_excess_gap = float(np.clip(directionality_excess_gap, 0.0, 2.0))
        generation_rejection_ratio = float(descriptor_metrics.get("generation_rejection_ratio", 0.0))
        rejection_excess_gap = max(
            0.0,
            generation_rejection_ratio - float(self.target_generation_rejection_ratio),
        ) / max(0.02, float(self.target_generation_rejection_ratio) + 0.02)
        rejection_excess_gap = float(np.clip(rejection_excess_gap, 0.0, 2.0))
        rejection_violation = max(
            0.0,
            generation_rejection_ratio - float(self.max_generation_rejection_ratio),
        ) / max(0.01, float(self.max_generation_rejection_ratio))
        rejection_violation = float(np.clip(rejection_violation, 0.0, 2.0))
        descriptor_metrics["structural_objective_score"] = float(structural_objective_score)
        descriptor_metrics["cycle_score"] = float(cycle_score)
        descriptor_metrics["shortcut_score"] = float(shortcut_score)
        descriptor_metrics["gate_depth_score"] = float(gate_depth_score)
        descriptor_metrics["path_depth_ratio_score"] = float(path_depth_ratio_score)
        descriptor_metrics["directionality_score"] = float(directionality_score)
        descriptor_metrics["gating_density_score"] = float(gating_density_score)
        descriptor_metrics["target_key_count"] = float(self.target_key_count)
        descriptor_metrics["target_lock_count"] = float(self.target_lock_count)
        descriptor_metrics["key_count_score"] = float(key_count_score)
        descriptor_metrics["lock_count_score"] = float(lock_count_score)
        descriptor_metrics["key_count_gap"] = float(key_count_gap)
        descriptor_metrics["lock_count_gap"] = float(lock_count_gap)
        descriptor_metrics["keylock_exact_score"] = float(keylock_exact_score)
        descriptor_metrics["exact_keylock_targets_enabled"] = float(bool(self.exact_keylock_targets_enabled))
        descriptor_metrics["node_count_score"] = float(node_count_score)
        descriptor_metrics["edge_count_score"] = float(edge_count_score)
        descriptor_metrics["under_target_gap"] = float(under_target_gap)
        descriptor_metrics["linearity_under_gap"] = float(linearity_under_gap)
        descriptor_metrics["leniency_excess_gap"] = float(leniency_excess_gap)
        descriptor_metrics["tutorial_climax_depth_gap"] = float(tutorial_climax_depth_gap)
        descriptor_metrics["shortcut_excess_gap"] = float(shortcut_excess_gap)
        descriptor_metrics["directionality_excess_gap"] = float(directionality_excess_gap)
        descriptor_metrics["rejection_excess_gap"] = float(rejection_excess_gap)
        descriptor_metrics["rejection_violation"] = float(rejection_violation)
        topology_realism_error = self._topology_realism_error(descriptor_metrics)
        descriptor_metrics["topology_realism_error"] = float(topology_realism_error)

        pareto_result = compute_pareto_objectives(
            descriptor_metrics,
            curve_alignment_score=curve_alignment_score,
            required_loops=2.0,
            required_branching=1.5,
        )
        apply_pareto_metrics(descriptor_metrics, pareto_result)

        if self.legacy_baseline_mode:
            fitness = float(np.clip(
                (0.64 * curve_fitness)
                + (0.08 * backtracking_score)
                + (0.14 * progression_score)
                + (0.14 * descriptor_score),
                0.0,
                1.0,
            ))
            narrative_weight = 0.0
            structural_weight = 0.0
            cognitive_weight = 0.0
        else:
            narrative_weight = float(self.narrative_score_weight if self.narrative_beats_enabled else 0.0)
            cognitive_weight = float(self.cognitive_score_weight)
            structural_weight = float(np.clip(0.42 - narrative_weight - cognitive_weight, 0.12, 0.42))
            fitness = (
                (0.20 * curve_fitness)
                + (0.08 * backtracking_score)
                + (0.12 * progression_score)
                + (0.18 * descriptor_score)
                + (structural_weight * structural_objective_score)
                + (narrative_weight * narrative_score)
                + (cognitive_weight * cognitive_score)
            )
        if self.legacy_baseline_mode:
            realism_multiplier = 1.0
            generation_efficiency_multiplier = 1.0
            realism_distribution_multiplier = 1.0
        else:
            realism_multiplier = float(
                np.clip(
                    1.0
                    - (0.38 * under_target_gap)
                    - (0.18 * linearity_under_gap)
                    - (0.18 * leniency_excess_gap)
                    - (0.14 * tutorial_climax_depth_gap)
                    - (0.20 * shortcut_excess_gap)
                    - (0.14 * directionality_excess_gap),
                    0.25,
                    1.0,
                )
            )
            generation_efficiency_multiplier = float(
                np.clip(
                    1.0 - (0.18 * rejection_excess_gap),
                    0.60,
                    1.0,
                )
            )
            realism_distribution_multiplier = float(np.clip(1.0 - (0.14 * topology_realism_error), 0.55, 1.0))
            fitness *= (
                realism_multiplier
                * realism_distribution_multiplier
                * generation_efficiency_multiplier
            )

        # Blend scalar fitness with Pareto objective quality.
        fitness *= float(np.clip(0.65 + (0.35 * pareto_result.pareto_score), 0.1, 1.0))
        descriptor_metrics["realism_multiplier"] = float(realism_multiplier)
        descriptor_metrics["realism_distribution_multiplier"] = float(realism_distribution_multiplier)
        descriptor_metrics["generation_efficiency_multiplier"] = float(generation_efficiency_multiplier)
        descriptor_metrics["cognitive_weight"] = float(cognitive_weight)

        structural_violation = self._structural_violation(descriptor_metrics)
        descriptor_metrics["structural_violation"] = float(structural_violation)
        violation = self._constraint_violation(
            critical_edges=critical_edges,
            node_count=node_count,
            edge_count=edge_count,
        )
        curve_alignment_violation = 0.0
        if self.min_curve_alignment_score > 0.0:
            curve_alignment_violation = max(0.0, self.min_curve_alignment_score - curve_alignment_score) / max(
                0.05,
                self.min_curve_alignment_score,
            )
        curve_alignment_violation = float(np.clip(curve_alignment_violation, 0.0, 2.0))
        descriptor_metrics["curve_alignment_violation"] = float(curve_alignment_violation)

        curve_trend_violation = 0.0
        if self.min_curve_trend_corr > -0.99:
            curve_trend_violation = max(0.0, self.min_curve_trend_corr - curve_trend_corr) / max(
                0.05,
                1.0 - self.min_curve_trend_corr,
            )
        curve_trend_violation = float(np.clip(curve_trend_violation, 0.0, 2.0))
        descriptor_metrics["curve_trend_violation"] = float(curve_trend_violation)

        narrative_violation = 0.0
        if self.min_narrative_score > 0.0:
            narrative_violation = max(0.0, self.min_narrative_score - narrative_score) / max(0.05, self.min_narrative_score)
        narrative_violation = float(np.clip(narrative_violation, 0.0, 2.0))
        descriptor_metrics["narrative_violation"] = float(narrative_violation)

        cognitive_violation = 0.0
        if self.min_cognitive_score > 0.0:
            cognitive_violation = max(0.0, self.min_cognitive_score - cognitive_score) / max(0.05, self.min_cognitive_score)
        cognitive_violation = float(np.clip(cognitive_violation, 0.0, 2.0))
        descriptor_metrics["cognitive_violation"] = float(cognitive_violation)

        directionality_violation = max(
            0.0,
            float(descriptor_metrics.get("directionality_gap", 0.0)) - float(self.max_directionality_gap),
        ) / max(1e-6, float(self.max_directionality_gap))
        directionality_violation = float(np.clip(directionality_violation, 0.0, 2.0))
        descriptor_metrics["directionality_violation"] = float(directionality_violation)
        if self.legacy_baseline_mode:
            violation = float(np.clip(violation + (0.30 * structural_violation), 0.0, 3.0))
        else:
            violation = float(
                np.clip(
                    violation
                    + (0.95 * structural_violation)
                    + (0.70 * under_target_gap)
                    + (0.22 * linearity_under_gap)
                    + (0.26 * leniency_excess_gap)
                    + (0.18 * tutorial_climax_depth_gap)
                    + (0.35 * shortcut_excess_gap)
                    + (0.22 * directionality_excess_gap)
                    + (0.15 * directionality_violation)
                    + (0.28 * rejection_excess_gap)
                    + (0.18 * rejection_violation)
                    + (0.25 * topology_realism_error)
                    + (0.34 * curve_alignment_violation)
                    + (0.18 * curve_trend_violation)
                    + (0.16 * narrative_violation)
                    + (0.20 * cognitive_violation),
                    0.0,
                    3.0,
                )
            )
            if bool(self.exact_keylock_targets_enabled):
                violation = float(
                    np.clip(
                        violation
                        + (0.30 * key_count_gap)
                        + (0.34 * lock_count_gap),
                        0.0,
                        3.0,
                    )
                )
        if not pareto_result.pareto_feasible:
            violation = float(
                np.clip(
                    violation
                    + (0.80 * pareto_result.loops_violation)
                    + (0.70 * pareto_result.branching_violation),
                    0.0,
                    3.0,
                )
            )
        feasible = bool(violation <= 1e-9)
        if not feasible:
            # Smooth penalty keeps gradient information while letting survivor
            # selection enforce feasibility-first ordering.
            fitness *= float(np.clip(1.0 - (0.25 * violation), 0.05, 1.0))

        return {
            "fitness": float(max(0.0, min(1.0, fitness))),
            "feasible": feasible,
            "constraint_violation": float(violation),
            "critical_edges": int(critical_edges),
            "node_count": int(node_count),
            "descriptor_metrics": descriptor_metrics,
        }

    @staticmethod
    def _curve_trend_correlation(extracted: np.ndarray, target: np.ndarray) -> float:
        """Robust correlation of curve trend shape in [-1, 1]."""
        x = np.asarray(extracted, dtype=np.float32).reshape(-1)
        y = np.asarray(target, dtype=np.float32).reshape(-1)
        if x.size != y.size:
            n = min(x.size, y.size)
            if n <= 1:
                return 0.0
            x = x[:n]
            y = y[:n]
        if x.size <= 1:
            return 0.0
        x_var = float(np.var(x))
        y_var = float(np.var(y))
        if x_var <= 1e-8 or y_var <= 1e-8:
            return 1.0 if float(np.mean(np.abs(x - y))) <= 1e-4 else 0.0
        corr = float(np.corrcoef(x, y)[0, 1])
        if not math.isfinite(corr):
            return 0.0
        return float(np.clip(corr, -1.0, 1.0))

    @staticmethod
    def _node_to_narrative_stage(node: Any) -> str:
        """Map a mission node type to a coarse narrative beat stage."""
        node_type = node.node_type
        if node_type == NodeType.START:
            return "START"
        if node_type == NodeType.GOAL:
            return "GOAL"
        if node_type in {NodeType.KEY, NodeType.ITEM, NodeType.SWITCH, NodeType.TUTORIAL_PUZZLE, NodeType.TREASURE}:
            return "SETUP"
        if node_type in {NodeType.LOCK, NodeType.BOSS_DOOR, NodeType.BIG_KEY}:
            return "GATE"
        if node_type in {NodeType.BOSS, NodeType.MINI_BOSS, NodeType.ARENA, NodeType.COMPLEX_PUZZLE}:
            return "CLIMAX"
        return "ESCALATION"

    def _score_narrative_beats(self, graph: MissionGraph, critical_path: Sequence[int]) -> float:
        """Score critical-path narrative pacing against a simple Zelda beat template."""
        if not self.narrative_beats_enabled:
            return 0.5
        if not critical_path:
            return 0.0

        stages = [self._node_to_narrative_stage(graph.nodes[nid]) for nid in critical_path if nid in graph.nodes]
        if not stages:
            return 0.0
        if len(stages) == 1:
            return 1.0 if stages[0] in {"START", "GOAL"} else 0.2

        expected_sequence = ["START", "SETUP", "ESCALATION", "GATE", "CLIMAX", "GOAL"]
        if len(stages) == len(expected_sequence):
            expected = expected_sequence
        else:
            x_old = np.linspace(0.0, 1.0, num=len(expected_sequence))
            x_new = np.linspace(0.0, 1.0, num=len(stages))
            idx = np.clip(np.round(np.interp(x_new, x_old, np.arange(len(expected_sequence)))), 0, len(expected_sequence) - 1)
            expected = [expected_sequence[int(i)] for i in idx]

        exact = [1.0 if a == b else 0.0 for a, b in zip(stages, expected)]
        exact_score = float(np.mean(exact)) if exact else 0.0

        stage_order = {"START": 0, "SETUP": 1, "ESCALATION": 2, "GATE": 3, "CLIMAX": 4, "GOAL": 5}
        monotonic_hits = 0
        monotonic_total = 0
        for left, right in zip(stages, stages[1:]):
            monotonic_total += 1
            if stage_order.get(right, 0) >= stage_order.get(left, 0):
                monotonic_hits += 1
        monotonic_score = float(monotonic_hits / monotonic_total) if monotonic_total > 0 else 0.0

        endpoint_bonus = 0.0
        if stages[0] == "START":
            endpoint_bonus += 0.5
        if stages[-1] == "GOAL":
            endpoint_bonus += 0.5

        return float(np.clip((0.52 * exact_score) + (0.28 * monotonic_score) + (0.20 * endpoint_bonus), 0.0, 1.0))

    def calculate_fitness(self, graph: MissionGraph) -> float:
        """
        Backward-compatible fitness accessor used by legacy callers.
        """
        result = self.evaluate_graph(graph)
        return float(result["fitness"])
    
    def _calculate_backtracking_score(self, graph: MissionGraph) -> float:
        """
        Calculate backtracking complexity metric (Thesis Upgrade #4).
        
        Measures how much the player revisits nodes during optimal traversal.
        Higher score favors dungeons with cyclic structures and shortcuts.
        
        Formula: unique_nodes_visited / total_steps_in_path
        - Linear path: score = 1.0 (each node visited once)
        - Backtracking: score < 1.0 (nodes revisited)
        - Ideal for complex dungeons: 0.6-0.85
        
        Args:
            graph: MissionGraph to evaluate
            
        Returns:
            Backtracking score (0.0-1.0)
        """
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        if not start or not goal:
            return 0.0
        
        metrics = self._extract_descriptor_metrics(graph)
        backtracking_proxy = float(metrics.get("backtracking_proxy", 0.0))
        # Aim for moderate revisit pressure, not extreme labyrinths.
        return self._score_target(backtracking_proxy, target=0.45, tol=0.35)
    
    def _is_solvable(self, graph: MissionGraph) -> bool:
        """
        Check if graph is solvable (path exists START -> GOAL).
        
        A graph is solvable if:
        1. It has both START and GOAL nodes
        2. A path exists from START to GOAL
        3. All required keys are obtainable before locks
        
        Args:
            graph: MissionGraph to check
            
        Returns:
            True if solvable
        """
        start = graph.get_start_node()
        goal = graph.get_goal_node()
        
        if not start or not goal:
            return False
        
        # Check basic connectivity
        path = self._find_path(graph, start.id, goal.id)
        if not path:
            return False
        
        # Use grammar's built-in progression validation.
        grammar = MissionGrammar()
        try:
            graph.sanitize()
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug("Graph sanitize failed during solvability check: %s", e)
        return bool(
            grammar.validate_lock_key_ordering(graph)
            and grammar.validate_progression_constraints(graph)
        )
    
    @staticmethod
    def _find_path_in_adjacency(
        adjacency: Dict[Any, List[Any]],
        start_id: Any,
        goal_id: Any,
    ) -> Optional[List[Any]]:
        """Breadth-first shortest path on an adjacency mapping."""
        if start_id == goal_id:
            return [start_id]

        visited = {start_id}
        queue = deque([(start_id, [start_id])])

        while queue:
            current, path = queue.popleft()
            for neighbor in adjacency.get(current, []):
                if neighbor in visited:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal_id:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def _find_weak_path(
        self,
        graph: MissionGraph,
        start_id: Any,
        goal_id: Any,
    ) -> Optional[List[Any]]:
        """
        Undirected shortest path over traversable mission adjacency.

        This intentionally ignores edge direction for directionality-gap
        diagnostics while still respecting traversable-edge filtering.
        """
        weak_adj: Dict[Any, List[Any]] = {nid: [] for nid in graph.nodes.keys()}
        for src, neighbors in graph.get_adjacency_map().items():
            s = src
            weak_adj.setdefault(s, [])
            for dst in neighbors:
                d = dst
                weak_adj.setdefault(d, [])
                weak_adj[s].append(d)
                weak_adj[d].append(s)

        for node_id, neighbors in list(weak_adj.items()):
            seen: Set[Any] = set()
            deduped: List[Any] = []
            for neighbor in neighbors:
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                deduped.append(neighbor)
            weak_adj[node_id] = deduped

        return self._find_path_in_adjacency(weak_adj, start_id, goal_id)

    def _find_path(
        self,
        graph: MissionGraph,
        start_id: Any,
        goal_id: Any
    ) -> Optional[List[Any]]:
        """
        Find directed path from start to goal over mission adjacency.
        """
        return self._find_path_in_adjacency(graph.get_forward_adjacency_map(), start_id, goal_id)
    
    def _interpolate(
        self,
        values: List[float],
        target_length: int
    ) -> np.ndarray:
        """
        Interpolate values to target length.
        
        Args:
            values: Original values
            target_length: Desired length
            
        Returns:
            Interpolated array
        """
        if len(values) == 0:
            return np.zeros(target_length)
        
        if len(values) == 1:
            return np.full(target_length, values[0])
        
        # Create interpolation indices
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, target_length)
        
        # Linear interpolation
        result = np.interp(x_new, x_old, values)
        
        return result
