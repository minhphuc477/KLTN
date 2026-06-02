"""Mission graph grammar orchestration and GNN conversion entrypoints."""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from .advanced_rules import (
    AddArenaRule,
    AddBossGauntlet,
    AddCollectionChallengeRule,
    AddEntangledBranchesRule,
    AddForeshadowingRule,
    AddFungibleLockRule,
    AddGatekeeperRule,
    AddHazardGateRule,
    AddItemGateRule,
    AddItemShortcutRule,
    AddMultiLockRule,
    AddPacingBreakerRule,
    AddResourceLoopRule,
    AddSecretRule,
    AddSectorRule,
    AddSkillChainRule,
    AddStairsRule,
    AddTeleportRule,
    AddValveRule,
    CreateHubRule,
    FormBigRoomRule,
    InsertSwitchRule,
    MergeRule,
    PruneDeadEndRule,
    PruneGraphRule,
    SplitRoomRule,
)
from .core_rules import BranchRule, Difficulty, InsertChallengeRule, InsertLockKeyRule, StartRule
from .graph_types import (
    LAYOUT_BASE_OFFSET,
    LAYOUT_LAYER_SPACING,
    LAYOUT_OFFSET_SPACING,
    EdgeType,
    MissionGraph,
    MissionNode,
    NodeType,
    Tensor,
    _require_torch_adapters,
)
from src.generation.grammar_validators import (
    validate_battery_reachability,
    validate_resource_loops,
    validate_skill_chains,
)

logger = logging.getLogger(__name__)

class MissionGrammar:
    """
    Grammar for generating mission graphs.
    
    Uses production rules to build a mission graph that represents
    dungeon structure, ensuring lock-key constraints are satisfied.
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Define production rules
        # Core structural rules
        self.rules = [
            StartRule(),
            
            # Basic building blocks
            InsertChallengeRule(NodeType.ENEMY),
            InsertChallengeRule(NodeType.PUZZLE),
            InsertLockKeyRule(),
            BranchRule(),
            
            # Advanced topology rules (Thesis Upgrades #1-3)
            MergeRule(),  # Creates shortcuts/cycles
            InsertSwitchRule(),  # Dynamic state changes
            AddBossGauntlet(),  # Big key hierarchy
            
            # Item-based progression
            AddItemGateRule(),  # Specific item requirements
            
            # Structural complexity
            CreateHubRule(),  # Central intersection points
            AddStairsRule(),  # Multi-floor support
            
            # Optional/hidden content
            AddSecretRule(),  # Hidden rooms
            AddTeleportRule(),  # Warp connections
            
            # Cleanup
            PruneGraphRule(),  # Simplify overly complex graphs
            
            # ========================================================
            # ADVANCED RULES (Thesis-Grade Patterns)
            # Based on Dormans "Unexplored" & Brown "Boss Keys"
            # ========================================================
            AddFungibleLockRule(),  # Economy: Fungible key inventory
            FormBigRoomRule(),  # Geometry: Merge rooms into great halls
            AddValveRule(),  # Directionality: One-way edges in cycles
            AddForeshadowingRule(),  # Design: Visual links (windows)
            AddCollectionChallengeRule(),  # Design: Multi-token gates
            AddArenaRule(),  # Pacing: Combat shutters
            AddSectorRule(),  # Structure: Thematic zones
            AddEntangledBranchesRule(),  # Design: Cross-branch dependencies
            AddHazardGateRule(),  # Design: Risky paths with protection (soft gate)
            SplitRoomRule(),  # Geometry: Virtual room layers
            
            # ========================================================
            # WAVE 3: PEDAGOGICAL & QUALITY CONTROL RULES
            # Nintendo-grade level design patterns
            # ========================================================
            AddSkillChainRule(),  # Pedagogy: Tutorial sequences
            AddPacingBreakerRule(),  # Pedagogy: Negative space/sanctuaries
            AddResourceLoopRule(),  # Safety: Farming spots prevent soft-locks
            AddGatekeeperRule(),  # Quality: Mini-boss guardians
            AddMultiLockRule(),  # Quality: Multi-switch batteries
            AddItemShortcutRule(),  # Quality: Item-gated returns
            PruneDeadEndRule(),  # Quality: Garbage collection (run late)
        ]
    
    def validate_all_constraints(self, graph: MissionGraph) -> bool:
        """
        Run all validation checks on generated graph.
        
        Performs comprehensive validation including:
        - Lock-key ordering constraints
        - Skill chain progression (Wave 3)
        - Battery reachability (Wave 3)
        - Resource loop validity (Wave 3)
        
        Args:
            graph: MissionGraph to validate
        
        Returns:
            True if all validation checks pass, False otherwise
        """
        results = {
            'anchor_nodes': self.validate_anchor_nodes(graph),
            'lock_key_ordering': self.validate_lock_key_ordering(graph),
            'progression_constraints': self.validate_progression_constraints(graph),
            'skill_chains': validate_skill_chains(graph),
            'battery_reachability': validate_battery_reachability(graph),
            'resource_loops': validate_resource_loops(graph),
        }
        
        # Log any failures
        all_passed = True
        for check, passed in results.items():
            if not passed:
                logger.warning(f"Validation failed: {check}")
                all_passed = False
            else:
                logger.debug(f"Validation passed: {check}")
        
        if all_passed:
            logger.info("All validation checks passed")
        
        return all_passed
    
    def generate(
        self,
        difficulty: Difficulty = Difficulty.MEDIUM,
        num_rooms: int = 8,
        max_keys: int = 2,
        validate_all: bool = True,
    ) -> MissionGraph:
        """
        Generate a mission graph.
        
        Args:
            difficulty: Dungeon difficulty level
            num_rooms: Approximate number of rooms
            max_keys: Maximum number of key-lock pairs
            validate_all: If True, run comprehensive validation checks after generation
            
        Returns:
            Generated MissionGraph
        """
        graph = MissionGraph()
        graph.ensure_generation_stats_defaults()
        graph.generation_stats["require_goal_gauntlet"] = True
        
        context = {
            'rng': self.rng,
            'difficulty': difficulty.value / 4.0,
            'goal_row': num_rooms // 2,
            'goal_col': num_rooms // 2,
        }
        
        # Apply start rule
        graph = self.rules[0].apply(graph, context)
        graph.sanitize()
        
        # Track how many of each have been added
        num_keys_added = 0
        num_challenges_added = 0
        
        # Apply rules until we have enough nodes
        max_iterations = num_rooms * 3
        iteration = 0
        
        while len(graph.nodes) < num_rooms and iteration < max_iterations:
            iteration += 1
            graph.sanitize()
            
            # Select a rule using adaptive weights to stage structure/gating/polish.
            applicable_rules = []
            weights = []
            
            for rule in self.rules[1:]:  # Skip start rule
                if not rule.can_apply(graph, context):
                    continue
                
                # Limit key-lock pairs
                if isinstance(rule, InsertLockKeyRule):
                    if num_keys_added >= max_keys:
                        continue

                adaptive_weight = self._compute_adaptive_rule_weight(
                    rule=rule,
                    graph=graph,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    num_rooms=num_rooms,
                    num_keys_added=num_keys_added,
                    max_keys=max_keys,
                    num_challenges_added=num_challenges_added,
                )
                if adaptive_weight <= 0.0:
                    continue

                applicable_rules.append(rule)
                weights.append(adaptive_weight)
            
            if not applicable_rules:
                break
            
            # Weighted random selection
            total_weight = sum(weights)
            r = self.rng.uniform(0, total_weight)
            cumulative = 0
            selected_rule = applicable_rules[0]
            
            for rule, weight in zip(applicable_rules, weights):
                cumulative += weight
                if r <= cumulative:
                    selected_rule = rule
                    break
            
            # Apply rule
            graph = selected_rule.apply(graph, context)
            graph.sanitize()
            
            # Track additions
            if isinstance(selected_rule, InsertLockKeyRule):
                num_keys_added += 1
            elif isinstance(selected_rule, InsertChallengeRule):
                num_challenges_added += 1

        # Repair/validation convergence loop.
        # Multiple rounds are needed because one repair can expose another issue.
        rounds = 4 if validate_all else 2
        for _ in range(rounds):
            graph = self._ensure_anchor_nodes(graph)
            graph.sanitize()
            repairs_before = int(graph.generation_stats.get("total_repairs", 0))
            round_had_repairs = False

            lock_ok = self.validate_lock_key_ordering(graph)
            if not lock_ok:
                logger.warning("Generated graph failed lock-key validation, fixing...")
                graph = self._fix_lock_key_ordering(graph)
                graph = self._ensure_anchor_nodes(graph)
                graph.sanitize()
                round_had_repairs = int(graph.generation_stats.get("total_repairs", 0)) > repairs_before

            if not validate_all:
                if round_had_repairs:
                    graph.generation_stats["repair_rounds"] = int(
                        graph.generation_stats.get("repair_rounds", 0)
                    ) + 1
                if self.validate_lock_key_ordering(graph):
                    break
                continue

            if self.validate_all_constraints(graph):
                if round_had_repairs:
                    graph.generation_stats["repair_rounds"] = int(
                        graph.generation_stats.get("repair_rounds", 0)
                    ) + 1
                break

            graph = self._repair_progression_constraints(graph)
            graph = self._repair_wave3_constraints(graph)
            graph = self._ensure_anchor_nodes(graph)
            graph.sanitize()
            round_had_repairs = int(graph.generation_stats.get("total_repairs", 0)) > repairs_before
            if round_had_repairs:
                graph.generation_stats["repair_rounds"] = int(
                    graph.generation_stats.get("repair_rounds", 0)
                ) + 1

            if self.validate_all_constraints(graph):
                break

        if validate_all and not self.validate_all_constraints(graph):
            logger.warning(
                "Graph validation failed on some checks even after repair. "
                "Graph is being returned but may have constraint violations. "
                "Consider regenerating with different seed or parameters."
            )
        elif not validate_all and not self.validate_lock_key_ordering(graph):
            logger.warning(
                "Lock-key repair could not fully satisfy constraints; "
                "invalid locks were downgraded to preserve solvability."
            )
        
        # Update positions for layout
        graph = self._layout_graph(graph)
        
        return graph

    def _compute_adaptive_rule_weight(
        self,
        rule: ProductionRule,
        graph: MissionGraph,
        iteration: int,
        max_iterations: int,
        num_rooms: int,
        num_keys_added: int,
        max_keys: int,
        num_challenges_added: int,
    ) -> float:
        """
        Compute dynamic rule weight to reduce late-stage repairs.

        The policy stages generation into:
        1) topology growth,
        2) progression/gating,
        3) cleanup/polish.
        """
        base_weight = max(0.0, float(getattr(rule, "weight", 1.0)))
        if base_weight <= 0.0:
            return 0.0

        rule_name = getattr(rule, "name", "")
        node_count = len(graph.nodes)
        progress_nodes = min(1.0, node_count / max(1, num_rooms))
        progress_iterations = min(1.0, iteration / max(1, max_iterations))
        progress = max(progress_nodes, progress_iterations * 0.8)

        if rule_name in {"PruneGraph", "PruneDeadEnd"}:
            if progress < 0.70:
                return 0.0
            base_weight *= 1.0 + (progress - 0.70) * 1.8

        if rule_name.startswith("InsertChallenge_"):
            target_challenges = max(2, int(num_rooms * 0.35))
            if num_challenges_added < target_challenges:
                base_weight *= 1.35
            if progress > 0.90:
                base_weight *= 0.75

        if rule_name == "InsertLockKey":
            if num_keys_added >= max_keys:
                return 0.0
            if node_count < 4:
                base_weight *= 0.30
            if progress < 0.25:
                base_weight *= 0.65
            elif progress <= 0.80:
                base_weight *= 1.20
            else:
                base_weight *= 0.70

        if rule_name in {"Branch", "MergeShortcut", "CreateHub", "AddStairs", "AddSector", "SplitRoom"}:
            if progress < 0.55:
                base_weight *= 1.20
            elif progress > 0.90:
                base_weight *= 0.75

        if rule_name in {"AddItemGate", "AddFungibleLock", "AddMultiLock", "AddGatekeeper", "AddHazardGate", "AddBossGauntlet"}:
            if progress < 0.40:
                base_weight *= 0.45
            elif progress > 0.90:
                base_weight *= 0.80
            else:
                base_weight *= 1.15

        if rule_name in {"AddSkillChain", "AddPacingBreaker", "AddResourceLoop", "AddItemShortcut", "AddForeshadowing"}:
            if progress < 0.55:
                base_weight *= 0.60
            else:
                base_weight *= 1.10

        return max(0.01, base_weight)

    def validate_anchor_nodes(self, graph: MissionGraph) -> bool:
        """Validate there is exactly one START and exactly one GOAL."""
        starts = graph.get_nodes_by_type(NodeType.START)
        goals = graph.get_nodes_by_type(NodeType.GOAL)
        return len(starts) == 1 and len(goals) == 1

    def _ensure_anchor_nodes(self, graph: MissionGraph) -> MissionGraph:
        """
        Enforce stable mission anchors.

        Some transformation rules can accidentally retag START/GOAL nodes.
        This pass restores a single START and GOAL while preserving topology.
        """
        graph.sanitize()
        if not graph.nodes:
            return graph

        # Ensure START exists and is unique.
        start_nodes = sorted(graph.get_nodes_by_type(NodeType.START), key=lambda n: n.id)
        if not start_nodes:
            preferred_start = graph.nodes.get(0)
            if preferred_start is None:
                first_id = min(graph.nodes.keys())
                preferred_start = graph.nodes[first_id]
            preferred_start.node_type = NodeType.START
            preferred_start.difficulty = 0.0
            preferred_start.key_id = None
            preferred_start.is_tutorial = False
            preferred_start.is_mini_boss = False
            preferred_start.is_sanctuary = False
            preferred_start.difficulty_rating = "SAFE"
            preferred_start.tension_value = 0.0
            start_nodes = [preferred_start]
        elif len(start_nodes) > 1:
            keep_start = start_nodes[0]
            for extra_start in start_nodes[1:]:
                extra_start.node_type = NodeType.EMPTY
                extra_start.is_tutorial = False
                extra_start.is_sanctuary = False
            start_nodes = [keep_start]

        # Ensure GOAL exists and is unique.
        goal_nodes = sorted(graph.get_nodes_by_type(NodeType.GOAL), key=lambda n: n.id)
        if not goal_nodes:
            preferred_goal = graph.nodes.get(1)
            if preferred_goal is None or preferred_goal.id == start_nodes[0].id:
                # Pick the highest-ID non-start node when possible.
                non_start_ids = [nid for nid in graph.nodes if nid != start_nodes[0].id]
                if non_start_ids:
                    preferred_goal = graph.nodes[max(non_start_ids)]
                else:
                    preferred_goal = start_nodes[0]
            preferred_goal.node_type = NodeType.GOAL
            preferred_goal.difficulty = 1.0
            preferred_goal.key_id = None
            preferred_goal.is_tutorial = False
            preferred_goal.is_mini_boss = False
            preferred_goal.is_sanctuary = False
            preferred_goal.difficulty_rating = "HARD"
            preferred_goal.tension_value = 1.0
            goal_nodes = [preferred_goal]
        elif len(goal_nodes) > 1:
            keep_goal = goal_nodes[0]
            for extra_goal in goal_nodes[1:]:
                extra_goal.node_type = NodeType.EMPTY
                extra_goal.is_tutorial = False
                extra_goal.is_sanctuary = False
            goal_nodes = [keep_goal]

        graph.sanitize()
        return graph

    def ensure_anchor_nodes(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for anchor-node normalization."""
        return self._ensure_anchor_nodes(graph)

    def validate_goal_gauntlet(self, graph: MissionGraph, *, log_failures: bool = True) -> bool:
        """
        Validate the final boss-goal chain used by strict VGLC topology checks.

        The canonical endgame contract is:
        approach -> BOSS_DOOR -> BOSS -> GOAL
        with GOAL as a terminal leaf node and at least one reachable BIG_KEY
        provider for the boss door.
        """
        graph.sanitize()
        goal = graph.get_goal_node()
        if goal is None:
            if log_failures:
                logger.warning("Goal gauntlet validation failed: missing GOAL node")
            return False

        boss_nodes = graph.get_nodes_by_type(NodeType.BOSS)
        if not boss_nodes:
            if log_failures:
                logger.warning("Goal gauntlet validation failed: missing BOSS node")
            return False
        if len(boss_nodes) != 1:
            if log_failures:
                logger.warning(
                    "Goal gauntlet validation failed: expected exactly one BOSS node, found %s",
                    len(boss_nodes),
                )
            return False

        goal_predecessors = list(dict.fromkeys(
            edge.source for edge in graph.edges if edge.target == goal.id
        ))
        if len(goal_predecessors) != 1:
            if log_failures:
                logger.warning(
                    "Goal gauntlet validation failed: GOAL %s has predecessors %s (expected exactly one boss predecessor)",
                    goal.id,
                    goal_predecessors,
                )
            return False

        boss_id = goal_predecessors[0]
        boss = graph.nodes.get(boss_id)
        if boss is None or boss.node_type != NodeType.BOSS:
            if log_failures:
                logger.warning(
                    "Goal gauntlet validation failed: GOAL %s is not attached to a BOSS node (neighbor=%s)",
                    goal.id,
                    boss_id,
                )
            return False

        if any(edge.source == goal.id for edge in graph.edges):
            if log_failures:
                logger.warning("Goal gauntlet validation failed: GOAL %s has outgoing edges", goal.id)
            return False

        boss_door_nodes = graph.get_nodes_by_type(NodeType.BOSS_DOOR)
        if boss_door_nodes:
            if len(boss_door_nodes) != 1:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: expected exactly one BOSS_DOOR node, found %s",
                        len(boss_door_nodes),
                    )
                return False

            boss_door = boss_door_nodes[0]
            boss_predecessors = [edge.source for edge in graph.edges if edge.target == boss_id]
            if boss_predecessors != [boss_door.id]:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: BOSS %s has predecessors %s (expected only BOSS_DOOR %s)",
                        boss_id,
                        boss_predecessors,
                        boss_door.id,
                    )
                return False

            boss_successors = [edge.target for edge in graph.edges if edge.source == boss_id]
            if boss_successors != [goal.id]:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: BOSS %s has successors %s (expected only GOAL %s)",
                        boss_id,
                        boss_successors,
                        goal.id,
                    )
                return False

            boss_door_successors = [edge.target for edge in graph.edges if edge.source == boss_door.id]
            if boss_door_successors != [boss_id]:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: BOSS_DOOR %s has successors %s (expected only BOSS %s)",
                        boss_door.id,
                        boss_door_successors,
                        boss_id,
                    )
                return False

            boss_door_predecessors = [edge.source for edge in graph.edges if edge.target == boss_door.id]
            if not boss_door_predecessors or any(
                predecessor in {boss_door.id, boss_id, goal.id}
                for predecessor in boss_door_predecessors
            ):
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: BOSS_DOOR %s has invalid predecessors %s",
                        boss_door.id,
                        boss_door_predecessors,
                    )
                return False

            if boss_door.key_id is None:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: BOSS_DOOR %s is missing key_id",
                        boss_door.id,
                    )
                return False
            has_big_key = any(
                node.node_type == NodeType.BIG_KEY and node.key_id == boss_door.key_id
                for node in graph.nodes.values()
            )
            if not has_big_key:
                if log_failures:
                    logger.warning(
                        "Goal gauntlet validation failed: no BIG_KEY provider found for BOSS_DOOR %s",
                        boss_door.id,
                    )
                return False

        return True

    def _repair_goal_gauntlet(self, graph: MissionGraph) -> MissionGraph:
        """
        Best-effort normalization of the final boss-goal chain.

        This keeps the generator aligned with the strict downstream topology
        validator even when stochastic rule sequences leave the final stretch in
        a partially constructed state.
        """
        graph.sanitize()
        goal = graph.get_goal_node()
        if goal is None:
            return graph

        start = graph.get_start_node()
        repairs = 0

        goal_incoming = [
            edge.source
            for edge in graph.edges
            if edge.target == goal.id and edge.source in graph.nodes
        ]

        boss_neighbors = [
            neighbor
            for neighbor in graph._adjacency.get(goal.id, [])
            if graph.nodes.get(neighbor) is not None
            and graph.nodes[neighbor].node_type == NodeType.BOSS
        ]
        boss_node = graph.nodes.get(boss_neighbors[0]) if boss_neighbors else None
        boss_nodes = sorted(graph.get_nodes_by_type(NodeType.BOSS), key=lambda node: node.id)
        if boss_node is None:
            boss_node = boss_nodes[0] if boss_nodes else None

        if boss_node is None:
            boss_id = max(graph.nodes.keys(), default=-1) + 1
            goal_pos = goal.position
            goal_z = goal_pos[2] if len(goal_pos) > 2 else 0
            boss_node = MissionNode(
                id=boss_id,
                node_type=NodeType.BOSS,
                position=(goal_pos[0] - 1, goal_pos[1], goal_z),
                difficulty=max(0.9, float(goal.difficulty)),
            )
            graph.add_node(boss_node)
            repairs += 1
            boss_nodes = [boss_node]

        boss_predecessors = [
            edge.source
            for edge in graph.edges
            if edge.target == boss_node.id and edge.source in graph.nodes
        ]
        primary_approach = next(
            (
                source
                for source in goal_incoming + boss_predecessors
                if source not in {goal.id, boss_node.id}
                and graph.nodes.get(source) is not None
                and graph.nodes[source].node_type != NodeType.BOSS_DOOR
            ),
            None,
        )
        if primary_approach is None:
            non_reserved_nodes = sorted(
                node_id
                for node_id, node in graph.nodes.items()
                if node_id not in {goal.id, boss_node.id}
                and node.node_type != NodeType.BOSS_DOOR
            )
            if not non_reserved_nodes:
                return graph
            primary_approach = non_reserved_nodes[0]

        boss_door_nodes = sorted(graph.get_nodes_by_type(NodeType.BOSS_DOOR), key=lambda node: node.id)
        boss_door = boss_door_nodes[0] if boss_door_nodes else None
        if boss_door is None:
            goal_pos = goal.position
            goal_z = goal_pos[2] if len(goal_pos) > 2 else 0
            boss_door_id = max(graph.nodes.keys(), default=-1) + 1
            boss_door = MissionNode(
                id=boss_door_id,
                node_type=NodeType.BOSS_DOOR,
                position=(goal_pos[0] - 2, goal_pos[1], goal_z),
                difficulty=0.9,
                key_id=boss_door_id,
            )
            graph.add_node(boss_door)
            repairs += 1
        elif boss_door.key_id is None:
            boss_door.key_id = boss_door.id
            repairs += 1

        preserved_door_approaches = [
            edge.source
            for edge in graph.edges
            if edge.target == boss_door.id
            and edge.source not in {goal.id, boss_node.id, boss_door.id}
            and graph.nodes.get(edge.source) is not None
        ]
        if preserved_door_approaches:
            primary_approach = preserved_door_approaches[0]

        for extra_boss in boss_nodes:
            if extra_boss.id == boss_node.id:
                continue
            extra_boss.node_type = NodeType.MINI_BOSS
            extra_boss.is_mini_boss = True
            repairs += 1

        for extra_boss_door in boss_door_nodes:
            if extra_boss_door.id == boss_door.id:
                continue
            extra_boss_door.node_type = NodeType.EMPTY
            extra_boss_door.key_id = None
            repairs += 1

        retained_edges: List[MissionEdge] = []
        for edge in graph.edges:
            if edge.source == goal.id:
                repairs += 1
                continue
            if edge.target == goal.id:
                repairs += 1
                continue
            if edge.source == boss_node.id or edge.target == boss_node.id:
                repairs += 1
                continue
            if edge.source == boss_door.id:
                repairs += 1
                continue
            if edge.target == boss_door.id and edge.source in {goal.id, boss_node.id, boss_door.id}:
                repairs += 1
                continue
            retained_edges.append(edge)

        graph.edges = retained_edges
        graph.rebuild_adjacency()

        def _edge_exists(source: int, target: int) -> bool:
            return any(edge.source == source and edge.target == target for edge in graph.edges)

        if not _edge_exists(primary_approach, boss_door.id):
            graph.add_edge(
                primary_approach,
                boss_door.id,
                EdgeType.BOSS_LOCKED,
                key_required=boss_door.key_id,
            )
            repairs += 1

        if not _edge_exists(boss_door.id, boss_node.id):
            graph.add_edge(boss_door.id, boss_node.id, EdgeType.PATH)
            repairs += 1

        if not _edge_exists(boss_node.id, goal.id):
            graph.add_edge(boss_node.id, goal.id, EdgeType.PATH)
            repairs += 1

        key_anchor_id = None
        if start is not None and start.id not in {goal.id, boss_node.id, boss_door.id}:
            key_anchor_id = start.id
        elif primary_approach not in {goal.id, boss_node.id, boss_door.id}:
            key_anchor_id = primary_approach
        else:
            fallback_nodes = sorted(
                node_id
                for node_id in graph.nodes
                if node_id not in {goal.id, boss_node.id, boss_door.id}
            )
            if fallback_nodes:
                key_anchor_id = fallback_nodes[0]

        if key_anchor_id is not None:
            provider_ids = [
                node.id
                for node in graph.nodes.values()
                if node.node_type == NodeType.BIG_KEY and node.key_id == boss_door.key_id
            ]
            excluded_edge = {(primary_approach, boss_door.id)}
            has_reachable_provider = bool(start) and any(
                self._is_reachable_without_edges(graph, start.id, provider_id, excluded_edge)
                for provider_id in provider_ids
            )
            if not provider_ids or not has_reachable_provider:
                anchor = graph.nodes[key_anchor_id]
                anchor_pos = anchor.position
                big_key_id = max(graph.nodes.keys(), default=-1) + 1
                big_key = MissionNode(
                    id=big_key_id,
                    node_type=NodeType.BIG_KEY,
                    position=(
                        anchor_pos[0],
                        anchor_pos[1] + 1,
                        anchor_pos[2] if len(anchor_pos) > 2 else 0,
                    ),
                    difficulty=min(0.8, max(0.4, float(anchor.difficulty) + 0.1)),
                    key_id=boss_door.key_id,
                )
                graph.add_node(big_key)
                graph.add_edge(key_anchor_id, big_key_id, EdgeType.PATH)
                repairs += 1

        graph._key_to_lock[boss_door.key_id] = boss_door.id
        graph.sanitize()
        if repairs > 0:
            graph.record_repair("goal_gauntlet_repairs", amount=int(repairs))
            logger.info(
                "Goal gauntlet repair normalized GOAL %s via BOSS %s and BOSS_DOOR %s (%d edits)",
                goal.id,
                boss_node.id,
                boss_door.id,
                repairs,
            )
        return graph
    
    def validate_lock_key_ordering(self, graph: MissionGraph, *, log_failures: bool = True) -> bool:
        """
        Validate that all keys can be reached before their locks.
        
        For each LOCK node, verifies that its required KEY is
        reachable from START without passing through the LOCK.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return False
        
        lock_types = {NodeType.LOCK, NodeType.BOSS_DOOR}
        key_types = {NodeType.KEY, NodeType.BIG_KEY}
        locks = [n for n in graph.nodes.values() if n.node_type in lock_types]
        keys = [n for n in graph.nodes.values() if n.node_type in key_types]
        
        # Build key lookup (multiple providers can map to same key_id).
        key_by_id: Dict[int, List[MissionNode]] = defaultdict(list)
        for key_node in keys:
            if key_node.key_id is not None:
                key_by_id[key_node.key_id].append(key_node)
        
        for lock in locks:
            if lock.key_id is None:
                continue
            
            providers = key_by_id.get(lock.key_id, [])
            if not providers:
                if log_failures:
                    logger.warning(f"Lock {lock.id} references non-existent key {lock.key_id}")
                return False
            
            # Check if key is reachable from start without passing through lock
            if not any(
                self._is_reachable_without(graph, start.id, key.id, exclude={lock.id})
                for key in providers
            ):
                if log_failures:
                    logger.warning(
                        f"No provider key for lock {lock.id} is reachable before lock "
                        f"(key_id={lock.key_id})"
                    )
                return False
        
        return True

    def validate_progression_constraints(self, graph: MissionGraph, *, log_failures: bool = True) -> bool:
        """
        Validate edge-level progression constraints (beyond lock-node ordering).

        Checks:
        - LOCKED/BOSS_LOCKED edges have a valid key provider reachable pre-gate.
        - ITEM_GATE edges have matching item providers reachable pre-gate.
        - MULTI_LOCK edges have enough reachable TOKEN nodes pre-gate.
        - Fungible key locks (requires_key_count) have enough reachable keys pre-gate.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return False

        # Provider indexes.
        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        item_providers: Dict[str, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type == NodeType.ITEM and node.item_type:
                item_providers[str(node.item_type)].append(node.id)

        token_nodes = [n.id for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        key_nodes = [
            n.id for n in graph.nodes.values()
            if n.node_type == NodeType.KEY
        ]

        for edge in graph.edges:
            excluded_edge = {(edge.source, edge.target)}

            if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                # Fungible small-key locks intentionally use requires_key_count
                # without a specific key_required ID.
                if edge.key_required is None:
                    if edge.edge_type == EdgeType.LOCKED and edge.requires_key_count > 0:
                        pass
                    else:
                        if log_failures:
                            logger.warning(
                                f"{edge.edge_type.name} edge {edge.source}->{edge.target} missing key_required"
                            )
                        return False
                if edge.key_required is not None:
                    providers = key_providers.get(edge.key_required, [])
                    if not providers:
                        if log_failures:
                            logger.warning(
                                f"No key provider for edge {edge.source}->{edge.target} "
                                f"(key_required={edge.key_required})"
                            )
                        return False
                    if not any(
                        self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                        for provider in providers
                    ):
                        if log_failures:
                            logger.warning(
                                f"Key for locked edge {edge.source}->{edge.target} is not reachable pre-gate"
                            )
                        return False

            if edge.requires_key_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_keys = sum(1 for key_node in key_nodes if key_node in reachable)
                if reachable_keys < edge.requires_key_count:
                    if log_failures:
                        logger.warning(
                            f"Fungible key lock {edge.source}->{edge.target} requires "
                            f"{edge.requires_key_count} but only {reachable_keys} keys are reachable pre-gate"
                        )
                    return False

            if edge.edge_type == EdgeType.ITEM_GATE and edge.item_required:
                providers = item_providers.get(str(edge.item_required), [])
                if not providers:
                    if log_failures:
                        logger.warning(
                            f"No item provider for ITEM_GATE {edge.source}->{edge.target} "
                            f"(item_required={edge.item_required})"
                        )
                    return False
                if not any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                ):
                    if log_failures:
                        logger.warning(
                            f"Item {edge.item_required} not reachable before ITEM_GATE "
                            f"{edge.source}->{edge.target}"
                        )
                    return False

            if edge.edge_type == EdgeType.MULTI_LOCK and edge.token_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_tokens = sum(1 for token_node in token_nodes if token_node in reachable)
                if reachable_tokens < edge.token_count:
                    if log_failures:
                        logger.warning(
                            f"MULTI_LOCK {edge.source}->{edge.target} requires {edge.token_count} "
                            f"tokens but only {reachable_tokens} are reachable pre-gate"
                        )
                    return False

            if edge.edge_type == EdgeType.STATE_BLOCK and edge.switches_required:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                missing_switches = [sid for sid in edge.switches_required if sid not in reachable]
                if missing_switches:
                    if log_failures:
                        logger.warning(
                            f"STATE_BLOCK {edge.source}->{edge.target} has unreachable switches before gate: "
                            f"{missing_switches}"
                        )
                    return False

        require_goal_gauntlet = bool(graph.generation_stats.get("require_goal_gauntlet", False))
        has_goal_gauntlet_artifacts = any(
            node.node_type in {NodeType.BOSS, NodeType.BOSS_DOOR, NodeType.BIG_KEY}
            for node in graph.nodes.values()
        )
        if (require_goal_gauntlet or has_goal_gauntlet_artifacts) and not self.validate_goal_gauntlet(graph, log_failures=log_failures):
            return False

        return True
    
    def _is_reachable_without(
        self,
        graph: MissionGraph,
        start: int,
        target: int,
        exclude: Set[int],
    ) -> bool:
        """BFS reachability check excluding certain nodes."""
        if start == target:
            return True
        
        visited = {start}
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in graph.get_forward_adjacency_map().get(current, []):
                if neighbor in visited or neighbor in exclude:
                    continue
                
                if neighbor == target:
                    return True
                
                visited.add(neighbor)
                queue.append(neighbor)
        
        return False

    def _is_reachable_without_edges(
        self,
        graph: MissionGraph,
        start: int,
        target: int,
        exclude_edges: Set[Tuple[int, int]],
    ) -> bool:
        """BFS reachability check excluding specific directed edges."""
        if start == target:
            return True

        visited = {start}
        queue = deque([start])

        while queue:
            current = queue.popleft()

            for neighbor in graph.get_forward_adjacency_map().get(current, []):
                if (current, neighbor) in exclude_edges:
                    continue
                if neighbor in visited:
                    continue
                if neighbor == target:
                    return True

                visited.add(neighbor)
                queue.append(neighbor)

        return False
    
    def _fix_lock_key_ordering(self, graph: MissionGraph) -> MissionGraph:
        """
        Repair invalid lock/key setups by downgrading unsatisfied gates.

        Note: Position swaps do not change graph reachability. This repair
        therefore edits progression constraints directly to restore consistency.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph

        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        demoted_lock_nodes = 0
        demoted_lock_edges = 0

        # Repair lock nodes first.
        for lock in [n for n in graph.nodes.values() if n.node_type in {NodeType.LOCK, NodeType.BOSS_DOOR}]:
            key_id = lock.key_id
            if key_id is None:
                continue

            providers = key_providers.get(key_id, [])
            reachable = any(
                self._is_reachable_without(graph, start.id, provider_id, {lock.id})
                for provider_id in providers
            )
            if providers and reachable:
                continue

            lock.node_type = NodeType.EMPTY
            lock.key_id = None
            demoted_lock_nodes += 1

            # Remove keyed requirement on outgoing edge(s) from this lock node.
            for edge in graph.edges:
                if edge.source != lock.id:
                    continue
                if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0

        graph.sanitize()

        # Repair keyed edges that still violate pre-gate reachability.
        for edge in graph.edges:
            if edge.edge_type not in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}:
                continue

            # Keep fungible key locks; they are validated via requires_key_count.
            if edge.edge_type == EdgeType.LOCKED and edge.requires_key_count > 0 and edge.key_required is None:
                continue

            key_id = edge.key_required
            providers = key_providers.get(key_id, []) if key_id is not None else []
            reachable = any(
                self._is_reachable_without_edges(graph, start.id, provider_id, {(edge.source, edge.target)})
                for provider_id in providers
            )
            if providers and reachable:
                continue

            edge.edge_type = EdgeType.PATH
            edge.key_required = None
            edge.requires_key_count = 0
            demoted_lock_edges += 1

        graph.sanitize()
        if demoted_lock_nodes > 0 or demoted_lock_edges > 0:
            graph.record_repair(
                "lock_key_repairs",
                amount=int(demoted_lock_nodes + demoted_lock_edges),
            )
            logger.info(
                "Lock-key repair: demoted %d lock nodes and %d lock edges",
                demoted_lock_nodes,
                demoted_lock_edges,
            )
        return graph

    def fix_lock_key_ordering(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for lock-key consistency repair."""
        return self._fix_lock_key_ordering(graph)

    def _repair_progression_constraints(self, graph: MissionGraph) -> MissionGraph:
        """
        Best-effort repair for progression constraints after generation.

        Strategy: preserve topology where possible; relax only unsatisfied gate
        requirements that would cause invalid progression.
        """
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph

        key_providers: Dict[int, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type in {NodeType.KEY, NodeType.BIG_KEY} and node.key_id is not None:
                key_providers[node.key_id].append(node.id)

        item_providers: Dict[str, List[int]] = defaultdict(list)
        for node in graph.nodes.values():
            if node.node_type == NodeType.ITEM and node.item_type:
                item_providers[str(node.item_type)].append(node.id)

        token_nodes = [n.id for n in graph.nodes.values() if n.node_type == NodeType.TOKEN]
        key_nodes = [
            n.id for n in graph.nodes.values()
            if n.node_type == NodeType.KEY
        ]

        relaxed_edges = 0

        for edge in graph.edges:
            excluded_edge = {(edge.source, edge.target)}

            # Malformed lock edges: neither specific key nor fungible key budget.
            if (
                edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED}
                and edge.key_required is None
                and edge.requires_key_count <= 0
            ):
                edge.edge_type = EdgeType.PATH
                edge.key_required = None
                edge.requires_key_count = 0
                relaxed_edges += 1
                continue

            # Key-specific lock handling.
            if edge.edge_type in {EdgeType.LOCKED, EdgeType.BOSS_LOCKED} and edge.key_required is not None:
                providers = key_providers.get(edge.key_required, [])
                reachable = any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                )
                if not providers or not reachable:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0
                    relaxed_edges += 1

            # Fungible locks.
            if edge.requires_key_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_keys = sum(1 for key_node in key_nodes if key_node in reachable)
                if reachable_keys <= 0:
                    edge.edge_type = EdgeType.PATH
                    edge.key_required = None
                    edge.requires_key_count = 0
                    relaxed_edges += 1
                elif reachable_keys < edge.requires_key_count:
                    edge.requires_key_count = reachable_keys
                    relaxed_edges += 1

            # Item gates.
            if edge.edge_type == EdgeType.ITEM_GATE and edge.item_required:
                providers = item_providers.get(str(edge.item_required), [])
                reachable = any(
                    self._is_reachable_without_edges(graph, start.id, provider, excluded_edge)
                    for provider in providers
                )
                if not providers or not reachable:
                    edge.edge_type = EdgeType.PATH
                    edge.item_required = None
                    relaxed_edges += 1

            # Token locks.
            if edge.edge_type == EdgeType.MULTI_LOCK and edge.token_count > 0:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                reachable_tokens = sum(1 for token_node in token_nodes if token_node in reachable)
                if reachable_tokens <= 0:
                    edge.edge_type = EdgeType.PATH
                    edge.token_count = 0
                    relaxed_edges += 1
                elif reachable_tokens < edge.token_count:
                    edge.token_count = reachable_tokens
                    relaxed_edges += 1

            # Switch gates / batteries.
            if edge.edge_type == EdgeType.STATE_BLOCK and edge.switches_required:
                reachable = graph.get_reachable_nodes(start.id, excluded_edges=excluded_edge)
                kept_switches = [sid for sid in edge.switches_required if sid in reachable]
                if not kept_switches:
                    edge.edge_type = EdgeType.PATH
                    edge.switches_required = []
                    edge.battery_id = None
                    relaxed_edges += 1
                elif len(kept_switches) != len(edge.switches_required):
                    edge.switches_required = kept_switches
                    relaxed_edges += 1

        require_goal_gauntlet = bool(graph.generation_stats.get("require_goal_gauntlet", False))
        has_goal_gauntlet_artifacts = any(
            node.node_type in {NodeType.BOSS, NodeType.BOSS_DOOR, NodeType.BIG_KEY}
            for node in graph.nodes.values()
        )
        if require_goal_gauntlet or has_goal_gauntlet_artifacts:
            graph = self._repair_goal_gauntlet(graph)
        graph.sanitize()
        if relaxed_edges > 0:
            graph.record_repair("progression_repairs", amount=int(relaxed_edges))
            logger.info("Progression repair relaxed %d edge constraints", relaxed_edges)
        return graph

    def repair_progression_constraints(self, graph: MissionGraph) -> MissionGraph:
        """Public wrapper for progression-constraint repair."""
        return self._repair_progression_constraints(graph)

    def _repair_wave3_constraints(self, graph: MissionGraph) -> MissionGraph:
        """
        Best-effort normalization for Wave 3 quality constraints.

        This pass only relaxes/normalizes metadata and does not remove critical
        structure unless required to regain consistency.
        """
        graph.sanitize()
        start = graph.get_start_node()
        changes = 0

        # If the search missed the tutorial/combat/complex arc entirely, use
        # the dedicated grammar rule as a final pedagogical repair rather than
        # leaving progression quality to chance. This keeps Block I's explicit
        # mission contract aligned with the downstream room archetype system.
        pedagogical_types = {
            NodeType.TUTORIAL_PUZZLE,
            NodeType.COMBAT_PUZZLE,
            NodeType.COMPLEX_PUZZLE,
        }
        present_pedagogical_types = {n.node_type for n in graph.nodes.values() if n.node_type in pedagogical_types}
        needs_skill_chain_repair = len(present_pedagogical_types) < len(pedagogical_types) or not validate_skill_chains(graph)
        if needs_skill_chain_repair:
            skill_chain_rule = AddSkillChainRule()
            repair_context = {"rng": self.rng, "difficulty": Difficulty.MEDIUM.value / 4.0}
            if skill_chain_rule.can_apply(graph, repair_context):
                graph = skill_chain_rule.apply(graph, repair_context)
                graph.sanitize()
                changes += 1

        # Normalize tutorial progression by nearest pedagogical successors.
        tutorial_nodes = [n for n in graph.nodes.values() if n.is_tutorial]
        pedagogical_types = {NodeType.COMBAT_PUZZLE, NodeType.COMPLEX_PUZZLE}
        for tutorial in tutorial_nodes:
            successors = [
                n for n in graph.get_forward_successors(tutorial.id, depth=3)
                if n.node_type in pedagogical_types
            ]
            if len(successors) < 2:
                continue
            successors.sort(key=lambda n: graph.get_forward_shortest_path_length(tutorial.id, n.id))
            first, second = successors[0], successors[1]
            if first.difficulty > second.difficulty:
                first.difficulty, second.difficulty = second.difficulty, first.difficulty
                first.difficulty_rating = "MODERATE"
                second.difficulty_rating = "HARD"
                changes += 1

        # Repair battery edges by keeping only reachable switches pre-gate.
        if start is not None:
            for edge in graph.edges:
                if edge.battery_id is None:
                    continue
                reachable = graph.get_reachable_nodes(
                    start.id,
                    excluded_edges={(edge.source, edge.target)},
                )
                kept_switches = [sid for sid in edge.switches_required if sid in reachable]
                if not kept_switches:
                    edge.edge_type = EdgeType.PATH
                    edge.switches_required = []
                    edge.battery_id = None
                    changes += 1
                elif len(kept_switches) != len(edge.switches_required):
                    edge.switches_required = kept_switches
                    changes += 1

            # Resource farms should only remain if they are reachable pre-gate.
            farms = [n for n in graph.nodes.values() if n.node_type == NodeType.RESOURCE_FARM]
            for farm in farms:
                if not farm.drops_resource:
                    continue
                related_gates = [e for e in graph.edges if e.item_required == farm.drops_resource]
                reachable_for_all = True
                for gate in related_gates:
                    reachable = graph.get_reachable_nodes(
                        start.id,
                        excluded_edges={(gate.source, gate.target)},
                    )
                    if farm.id not in reachable:
                        reachable_for_all = False
                        break
                if not reachable_for_all:
                    farm.node_type = NodeType.EMPTY
                    farm.drops_resource = None
                    farm.difficulty_rating = "MODERATE"
                    farm.tension_value = 0.5
                    changes += 1

        graph.sanitize()
        if changes > 0:
            graph.record_repair("wave3_repairs", amount=int(changes))
            logger.info("Wave3 repair normalized %d quality constraints", changes)
        return graph
    
    def _layout_graph(self, graph: MissionGraph) -> MissionGraph:
        """Apply a simple layout algorithm to position nodes."""
        graph.sanitize()
        start = graph.get_start_node()
        if not start:
            return graph
        
        # BFS to assign layers
        layers = {start.id: 0}
        queue = deque([start.id])
        
        while queue:
            current = queue.popleft()
            current_layer = layers[current]
            
            for neighbor in graph._adjacency.get(current, []):
                if neighbor not in layers:
                    layers[neighbor] = current_layer + 1
                    queue.append(neighbor)
        
        # Group by layer
        layer_nodes = defaultdict(list)
        for node_id, layer in layers.items():
            layer_nodes[layer].append(node_id)
        
        # Position nodes
        for layer, nodes in layer_nodes.items():
            for i, node_id in enumerate(nodes):
                # Skip nodes that were removed by rules (e.g., FormBigRoomRule, PruneGraphRule)
                if node_id not in graph.nodes:
                    continue
                
                offset = i - len(nodes) // 2
                current_pos = graph.nodes[node_id].position
                floor = current_pos[2] if len(current_pos) > 2 else 0
                # Use layout constants for consistent spacing
                graph.nodes[node_id].position = (
                    layer * LAYOUT_LAYER_SPACING,
                    offset * LAYOUT_OFFSET_SPACING + LAYOUT_BASE_OFFSET,
                    floor
                )
        
        return graph


# ============================================================================
# INTEGRATION WITH CONDITION ENCODER
# ============================================================================

def graph_to_gnn_input(
    graph: MissionGraph,
    current_node_idx: Optional[int] = None,
) -> Dict[str, Tensor]:
    """
    Convert MissionGraph to tensors for GNN conditioning.
    
    Args:
        graph: MissionGraph to convert
        current_node_idx: Current node being generated (for local context)
        
    Returns:
        Dict with:
            - edge_index: [2, E] edges
            - node_features: [N, D] features
            - tpe: [N, 8] topological encoding
            - current_node: int
    """
    adapters = _require_torch_adapters()
    return adapters.graph_to_gnn_input(graph, current_node_idx=current_node_idx)
