"""Evolutionary topology generator."""

from __future__ import annotations

import copy
import logging
import math
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from src.generation.grammar import (
    AddArenaRule,
    AddEntangledBranchesRule,
    AddGatekeeperRule,
    AddHazardGateRule,
    AddItemGateRule,
    AddMultiLockRule,
    AddSkillChainRule,
    MissionGrammar,
    MissionGraph,
    MissionNode,
    NodeType,
    EdgeType,
    PruneDeadEndRule,
)
from src.zelda_data.vglc_utils import filter_virtual_nodes, validate_topology

from ._shared import CVTEliteArchive, DEFAULT_REALISM_TUNING, DEFAULT_ZELDA_TRANSITIONS
from .converters import mission_graph_to_networkx, networkx_to_mission_graph
from .evaluator import TensionCurveEvaluator
from .executor import GraphGrammarExecutor
from .individual import Individual

logger = logging.getLogger(__name__)

class EvolutionaryTopologyGenerator:
    """
    Evolves dungeon topologies using genetic search over graph grammars.
    
    The genome is a list of grammar rule IDs. The phenotype is the MissionGraph
    produced by executing those rules sequentially.
    
    This implements a (mu+lambda) evolutionary strategy with:
    - Tournament selection
    - One-point crossover
    - Weighted mutation using Zelda transition probabilities
    """
    
    def __init__(
        self,
        target_curve: List[float],
        zelda_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        genome_length: int = 18,
        max_nodes: int = 20,
        rule_space: str = "full",
        rule_weight_overrides: Optional[Dict[str, float]] = None,
        descriptor_targets: Optional[Dict[str, float]] = None,
        transition_mix: float = 0.7,
        seed: Optional[int] = None,
        search_strategy: str = "ga",
        qd_archive_cells: int = 128,
        qd_init_random_fraction: float = 0.35,
        qd_emitter_mutation_rate: float = 0.18,
        qd_archive_path: Optional[str] = None,
        qd_load_archive: bool = False,
        qd_autosave_archive: bool = False,
        max_lock_key_rules: int = 3,
        realism_tuning: Optional[Dict[str, float]] = None,
        enable_rule_credit_assignment: bool = False,
        enforce_generation_constraints: bool = False,
        allow_candidate_repairs: bool = False,
    ):
        """
        Initialize evolutionary generator.
        
        Args:
            target_curve: Desired difficulty/tension progression (normalized 0-1)
            zelda_transition_matrix: P(RuleB | RuleA) for biased mutation
            population_size: Number of individuals per generation (mu)
            generations: Number of evolutionary iterations
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover vs. cloning
            genome_length: Length of genome (number of rules)
            max_nodes: Maximum nodes in generated graph (room count upper bound)
            rule_space: "full" (default) enables all MissionGrammar production
                rules, "core" keeps legacy 5-rule behavior.
            rule_weight_overrides: Optional map of rule_name -> weight used to
                calibrate scheduling/sampling against reference descriptors.
            descriptor_targets: Optional target descriptor means (linearity,
                leniency, progression_complexity, topology_complexity) used
                in fitness evaluation.
            transition_mix: Mixing ratio for transition-bias mutations:
                1.0 uses pure transition matrix, 0.0 uses global rule priors.
            seed: Random seed for reproducibility
            search_strategy: `ga` (default) or `cvt_emitter` for runtime QD.
            qd_archive_cells: Number of CVT archive cells for emitter search.
            qd_init_random_fraction: Bootstrap fraction sampled uniformly
                before archive emitters dominate.
            qd_emitter_mutation_rate: Mutation rate for emitter offspring.
            qd_archive_path: Optional persisted CVT archive path for warm
                starts and reproducible QD continuation.
            qd_load_archive: If True, load qd_archive_path before CVT search
                when the file exists.
            qd_autosave_archive: If True, save qd_archive_path after each
                completed generation and at the end of CVT search.
            max_lock_key_rules: Soft cap on InsertLockKey rule applications
                permitted during genome execution.
            enforce_generation_constraints: If True, reject rule outcomes that
                violate lock/progression constraints during genome execution.
                Default is False to preserve QD diversity and avoid hard-kill
                behavior in early generations.
            allow_candidate_repairs: If True, attempt local candidate repair
                when generation constraints fail.
        """
        self.target_curve = target_curve
        self.transition_matrix = zelda_transition_matrix or DEFAULT_ZELDA_TRANSITIONS
        self._has_custom_transition_matrix = zelda_transition_matrix is not None
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.genome_length = genome_length
        self.max_nodes = max_nodes
        self.rule_weight_overrides = rule_weight_overrides or {}
        self.descriptor_targets = copy.deepcopy(descriptor_targets) if descriptor_targets is not None else None
        self.transition_mix = float(np.clip(float(transition_mix), 0.0, 1.0))
        parsed_strategy = str(search_strategy).strip().lower() if search_strategy is not None else "ga"
        if parsed_strategy in {"map_elites", "cvt", "cvt_map_elites"}:
            parsed_strategy = "cvt_emitter"
        if parsed_strategy not in {"ga", "cvt_emitter"}:
            logger.warning("Unknown search_strategy='%s', defaulting to 'ga'", search_strategy)
            parsed_strategy = "ga"
        if parsed_strategy == "cvt_emitter" and CVTEliteArchive is None:
            logger.warning("CVTEliteArchive unavailable, falling back to GA strategy")
            parsed_strategy = "ga"
        self.search_strategy = parsed_strategy
        self.qd_archive_cells = int(max(32, qd_archive_cells))
        self.qd_init_random_fraction = float(np.clip(float(qd_init_random_fraction), 0.05, 0.95))
        self.qd_emitter_mutation_rate = float(np.clip(float(qd_emitter_mutation_rate), 0.01, 0.95))
        self.qd_archive_path = Path(qd_archive_path) if qd_archive_path else None
        self.qd_load_archive = bool(qd_load_archive)
        self.qd_autosave_archive = bool(qd_autosave_archive)
        self.max_lock_key_rules = int(max(0, max_lock_key_rules))
        self.realism_tuning = self._merge_realism_tuning(realism_tuning)
        self.enable_rule_credit_assignment = bool(enable_rule_credit_assignment)
        self.enforce_generation_constraints = bool(enforce_generation_constraints)
        self.allow_candidate_repairs = bool(allow_candidate_repairs)
        parsed_rule_space = str(rule_space).strip().lower() if rule_space is not None else "full"
        if parsed_rule_space not in {"core", "full"}:
            logger.warning("Unknown rule_space='%s', defaulting to 'full'", rule_space)
            parsed_rule_space = "full"
        self.rule_space = parsed_rule_space
        self.seed = seed
        self.last_best_individual: Optional[Individual] = None
        
        # Initialize RNG
        self.rng = random.Random(seed)
        
        # Initialize components
        self.executor = GraphGrammarExecutor(
            seed=seed,
            use_full_rule_space=(self.rule_space == "full"),
            max_lock_key_rules=self.max_lock_key_rules,
            rule_weight_overrides=self.rule_weight_overrides,
            enforce_generation_constraints=self.enforce_generation_constraints,
            allow_candidate_repairs=self.allow_candidate_repairs,
        )
        self.evaluator = TensionCurveEvaluator(
            target_curve,
            descriptor_targets=self.descriptor_targets,
        )

        # If descriptor targets request significantly larger topologies than
        # the caller-provided cap, softly expand max_nodes so search can
        # actually satisfy node-budget realism targets.
        target_nodes = float(max(0.0, getattr(self.evaluator, "target_num_nodes", 0.0)))
        if target_nodes > 0.0:
            floor_ratio = float(np.clip(self._rt("node_cap_floor_ratio", 0.92), 0.60, 1.05))
            expand_ratio = float(np.clip(self._rt("node_cap_expand_ratio", 1.08), 1.00, 1.60))
            hard_cap_ratio = float(np.clip(self._rt("node_cap_hard_cap_ratio", 1.25), 1.05, 2.00))
            required_floor = int(max(5, round(floor_ratio * target_nodes)))
            if int(self.max_nodes) < required_floor:
                expanded = int(max(required_floor, round(expand_ratio * target_nodes)))
                # Keep expansion bounded to avoid runaway graph growth.
                expanded = int(min(expanded, max(48, int(round(hard_cap_ratio * target_nodes)))))
                logger.info(
                    "Expanding max_nodes from %d to %d to match target_num_nodes=%.2f",
                    int(self.max_nodes),
                    expanded,
                    target_nodes,
                )
                self.max_nodes = expanded

            # Expand genome length when target topologies are much larger than
            # legacy default sequence budgets. This allows enough constructive
            # operators per individual to approach reference node/edge scale.
            gl_floor_ratio = float(np.clip(self._rt("genome_len_floor_ratio", 0.62), 0.35, 1.10))
            gl_expand_ratio = float(np.clip(self._rt("genome_len_expand_ratio", 0.78), 0.45, 1.40))
            gl_hard_cap_ratio = float(np.clip(self._rt("genome_len_hard_cap_ratio", 1.12), 0.70, 2.20))
            required_gl_floor = int(max(10, round(gl_floor_ratio * target_nodes)))
            if int(self.genome_length) < required_gl_floor:
                expanded_gl = int(max(required_gl_floor, round(gl_expand_ratio * target_nodes)))
                expanded_gl = int(min(expanded_gl, max(64, int(round(gl_hard_cap_ratio * target_nodes)))))
                logger.info(
                    "Expanding genome_length from %d to %d to match target_num_nodes=%.2f",
                    int(self.genome_length),
                    expanded_gl,
                    target_nodes,
                )
                self.genome_length = expanded_gl
        
        # Validate parameters
        if max_nodes < 5:
            logger.warning("max_nodes=%d is very low, setting to minimum of 5", max_nodes)
            self.max_nodes = 5
        
        # Statistics tracking
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.feasible_ratio_history: List[float] = []
        self.avg_violation_history: List[float] = []
        self.qd_coverage_history: List[float] = []
        self.qd_qd_score_history: List[float] = []
        self.qd_mean_fitness_history: List[float] = []
        self.qd_num_elites_history: List[float] = []
        self.qd_final_archive_stats: Dict[str, float] = {}
        
        # Rule ID bounds (skip StartRule at index 0)
        self.min_rule_id = 1
        self.max_rule_id = len(self.executor.rules) - 1
        
        # Map rule names to IDs for transition matrix
        self.rule_name_to_id = {
            name: i for i, name in enumerate(self.executor.rule_names)
        }
        self._rule_ids = list(range(self.min_rule_id, self.max_rule_id + 1))
        self._global_rule_weights: Dict[int, float] = {
            rid: max(1e-6, float(self.executor.rules[rid].weight))
            for rid in self._rule_ids
        }
        self._renormalize_global_rule_probs()
        self._topology_pressure_rule_ids = self._select_rule_ids_by_keywords(
            keywords=[
                "branch",
                "merge",
                "hub",
                "entangled",
                "split",
                "valve",
                "sector",
                "bigroom",
                "secret",
                "foreshadow",
            ]
        )
        self._gate_pressure_rule_ids = self._select_rule_ids_by_keywords(
            keywords=[
                "lock",
                "gate",
                "key",
                "gauntlet",
                "collection",
                "switch",
                "hazard",
                "multilock",
            ]
        )
        self._explicit_shortcut_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddTeleport",
                "AddItemShortcut",
                "AddResourceLoop",
            ]
        )
        self._gate_heavy_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "InsertLockKey",
                "AddFungibleLock",
                "AddItemGate",
                "AddBossGauntlet",
                "InsertSwitch",
                "AddEntangledBranches",
                "AddHazardGate",
                "AddCollectionChallenge",
                "AddMultiLock",
            ]
        )
        self._critical_path_gate_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "InsertLockKey",
                "AddFungibleLock",
                "AddBossGauntlet",
                "AddGatekeeper",
            ]
        )
        self._side_gate_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddItemGate",
                "AddEntangledBranches",
                "AddHazardGate",
                "AddMultiLock",
                "InsertSwitch",
                "AddArena",
                "AddCollectionChallenge",
            ]
        )
        self._key_inflating_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "InsertLockKey",
                "AddFungibleLock",
                "AddBossGauntlet",
                "AddCollectionChallenge",
                "CreateHub",
                "AddSecret",
                "AddForeshadowing",
                "SplitRoom",
            ]
        )
        self._non_key_gate_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddItemGate",
                "AddEntangledBranches",
                "AddHazardGate",
                "AddMultiLock",
                "InsertSwitch",
                "AddArena",
                "AddGatekeeper",
            ]
        )
        self._loop_closure_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "MergeShortcut",
                "Branch",
                "CreateHub",
                "AddEntangledBranches",
                "AddSector",
                "SplitRoom",
                "AddValve",
            ]
        )
        self._path_depth_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "InsertChallenge_ENEMY",
                "InsertChallenge_PUZZLE",
                "AddItemGate",
                "AddArena",
                "AddSkillChain",
                "AddBossGauntlet",
                "AddPacingBreaker",
            ]
        )
        self._linear_progression_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "InsertChallenge_ENEMY",
                "InsertChallenge_PUZZLE",
                "AddSkillChain",
                "AddGatekeeper",
                "AddBossGauntlet",
                "AddPacingBreaker",
            ]
        )
        self._branch_pruning_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "PruneDeadEnd",
            ]
        )
        self._pedagogical_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddSkillChain",
            ]
        )
        self._pedagogical_support_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddItemGate",
                "AddCollectionChallenge",
                "AddResourceLoop",
                "AddPacingBreaker",
                "AddGatekeeper",
            ]
        )
        self._pedagogical_depth_support_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddSkillChain",
                "AddGatekeeper",
                "AddBossGauntlet",
                "AddPacingBreaker",
                "AddItemGate",
                "AddCollectionChallenge",
            ]
        )
        self._wide_branch_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "Branch",
                "CreateHub",
                "AddEntangledBranches",
                "AddSector",
                "SplitRoom",
            ]
        )
        self._node_expansion_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "Branch",
                "CreateHub",
                "AddSector",
                "SplitRoom",
                "FormBigRoom",
                "AddEntangledBranches",
                "AddStairs",
                "AddSkillChain",
                "AddBossGauntlet",
            ]
        )
        self._edge_expansion_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "MergeShortcut",
                "AddValve",
                "AddForeshadowing",
                "AddItemShortcut",
                "AddTeleport",
                "AddResourceLoop",
                "AddHazardGate",
                "AddMultiLock",
                "AddCollectionChallenge",
            ]
        )
        self._gate_relief_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "Branch",
                "MergeShortcut",
                "CreateHub",
                "SplitRoom",
                "AddPacingBreaker",
                "AddForeshadowing",
                "AddSecret",
                "FormBigRoom",
            ]
        )
        self._directionality_heavy_rule_ids = self._select_rule_ids_by_exact_names(
            names=[
                "AddValve",
                "SplitRoom",
                "AddArena",
                "AddGatekeeper",
            ]
        )
        directionality_set = set(self._directionality_heavy_rule_ids)
        self._gate_non_directional_rule_ids = [
            rid for rid in self._gate_heavy_rule_ids if rid not in directionality_set
        ]
        self._apply_target_aware_rule_prior()
        self._apply_custom_transition_bias_to_global_prior()
        self._target_aware_rule_weights = dict(self._global_rule_weights)
        self._renormalize_global_rule_probs()
        
        logger.info(
            "Initialized EvolutionaryTopologyGenerator: target_curve_length=%d, pop_size=%d, generations=%d, genome_length=%d, rule_space=%s, max_nodes=%d, search_strategy=%s",
            len(target_curve),
            population_size,
            generations,
            genome_length,
            self.rule_space,
            self.max_nodes,
            self.search_strategy,
        )

    def evolve(self, *, directed_output: bool = False) -> nx.Graph:
        """
        Main evolutionary loop. Returns the best graph found.
        
        Process:
        1. Initialize random population
        2. For each generation:
            a. Evaluate all individuals
            b. Select parents using tournament selection
            c. Create offspring via crossover and mutation
            d. Keep best individuals (elitism)
        3. Return best graph as NetworkX
        
        Returns:
            NetworkX Graph with node attributes (type, difficulty) and valid topology
        """
        if self.search_strategy == "cvt_emitter":
            return self._evolve_cvt_emitter(directed_output=directed_output)

        logger.info("Starting evolutionary search...")
        
        # Initialize population
        population = self._generate_initial_population()
        
        # Evolutionary loop
        for gen in range(self.generations):
            # Evaluate population
            population = self._evaluate_population(population, gen)
            
            # Track statistics
            fitnesses = [ind.fitness for ind in population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            diversity = self._calculate_diversity(population)
            feasible_ratio = float(np.mean([1.0 if ind.feasible else 0.0 for ind in population])) if population else 0.0
            avg_violation = float(
                np.mean([
                    float(ind.constraint_violation if np.isfinite(ind.constraint_violation) else 1.0)
                    for ind in population
                ])
            ) if population else 0.0
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            self.feasible_ratio_history.append(feasible_ratio)
            self.avg_violation_history.append(avg_violation)
            
            # Log progress
            if gen % 10 == 0 or gen == self.generations - 1:
                logger.info(
                    "Generation %d/%d: best_fitness=%.4f, avg_fitness=%.4f, diversity=%.4f, feasible_ratio=%.2f, avg_violation=%.3f",
                    gen,
                    self.generations,
                    best_fitness,
                    avg_fitness,
                    diversity,
                    feasible_ratio,
                    avg_violation,
                )
            
            # Check for convergence
            if best_fitness >= 0.95:
                logger.info("Converged at generation %d with fitness %.4f", gen, best_fitness)
                break

            # Generation-level descriptor pressure update to reduce
            # cycle/shortcut/gate-depth realism drift.
            self._adapt_global_rule_prior_from_population(population)
            
            # Generate offspring
            offspring = self._generate_offspring(population)
            # Critical: offspring must be evaluated before survivor selection.
            # Otherwise all new individuals keep default fitness=0.0 and
            # the search degenerates into selecting only previous parents.
            offspring = self._evaluate_population(offspring, gen)
            
            # Combine and select survivors (mu+lambda)
            population = self._select_survivors(population + offspring)
        
        # Get best individual
        best = max(population, key=lambda ind: ind.fitness)
        self.last_best_individual = best
        
        logger.info(
            "Evolution complete. Best fitness: %.4f, Graph: %d nodes, %d edges",
            best.fitness,
            len(best.phenotype.nodes),
            len(best.phenotype.edges),
        )
        return self._finalize_graph_output(best.phenotype, directed_output=directed_output)

    def _finalize_graph_output(self, graph: MissionGraph, *, directed_output: bool) -> nx.Graph:
        """Convert phenotype graph to validated output graph."""
        pre_repair_graph = copy.deepcopy(graph)
        pre_repair_eval: Dict[str, Any] = {}
        try:
            pre_repair_eval = dict(self.evaluator.evaluate_graph(pre_repair_graph))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Pre-repair phenotype evaluation failed during finalization: %s", exc)

        constraint_grammar = getattr(getattr(self, "executor", None), "_constraint_grammar", None)
        if constraint_grammar is not None:
            graph = copy.deepcopy(graph)
            graph.ensure_generation_stats_defaults()
            graph.generation_stats["require_goal_gauntlet"] = True
            graph = constraint_grammar.ensure_anchor_nodes(graph)
            graph = constraint_grammar.fix_lock_key_ordering(graph)
            graph = constraint_grammar.repair_progression_constraints(graph)
            graph = self._repair_pedagogical_progression(graph, constraint_grammar=constraint_grammar)
            graph = self._repair_progression_balance(graph, constraint_grammar=constraint_grammar)
            graph = self._repair_gate_economy(graph, constraint_grammar=constraint_grammar)
            graph = constraint_grammar.ensure_anchor_nodes(graph)
            graph.sanitize()

        try:
            post_repair_eval = dict(self.evaluator.evaluate_graph(graph))
            graph.ensure_generation_stats_defaults()
            graph.generation_stats["final_repair_evaluation"] = {
                "pre_fitness": float(pre_repair_eval.get("fitness", 0.0)),
                "post_fitness": float(post_repair_eval.get("fitness", 0.0)),
                "fitness_delta": float(post_repair_eval.get("fitness", 0.0))
                - float(pre_repair_eval.get("fitness", 0.0)),
                "pre_feasible": bool(pre_repair_eval.get("feasible", False)),
                "post_feasible": bool(post_repair_eval.get("feasible", False)),
                "pre_constraint_violation": float(pre_repair_eval.get("constraint_violation", 0.0)),
                "post_constraint_violation": float(post_repair_eval.get("constraint_violation", 0.0)),
            }
            setattr(self, "last_final_repair_evaluation", dict(graph.generation_stats["final_repair_evaluation"]))
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            logger.debug("Post-repair phenotype evaluation failed during finalization: %s", exc)

        best_networkx = mission_graph_to_networkx(graph, directed=True)

        logger.debug("Applying VGLC compliance: filtering virtual nodes...")
        best_networkx_physical = filter_virtual_nodes(best_networkx)
        best_networkx_physical = self._enforce_output_node_cap(best_networkx_physical)
        best_networkx_physical = self._repair_output_connectivity(best_networkx_physical)

        topology_report = validate_topology(best_networkx_physical)
        if not topology_report.is_valid:
            logger.warning("VGLC topology validation warnings: %s", topology_report.summary())
        else:
            logger.info("VGLC topology validation: PASSED")

        if directed_output:
            return best_networkx_physical
        return best_networkx_physical.to_undirected()

    def _enforce_output_node_cap(self, graph: nx.Graph) -> nx.Graph:
        """Final safety pass for repairs that add nodes after evolutionary capping."""
        hard_cap = int(max(int(self.max_nodes) + 2, int(self.max_nodes * 1.5)))
        if graph.number_of_nodes() <= hard_cap:
            return graph

        repaired = graph.copy()
        repaired.graph["generation_stats"] = copy.deepcopy(graph.graph.get("generation_stats", {}))
        protected_types = {
            "START",
            "GOAL",
            "BOSS",
            "BOSS_DOOR",
            "BIG_KEY",
            "KEY",
            "SWITCH",
            "TOKEN",
            "MULTI_LOCK",
        }
        def _edge_type_name(data: Dict[str, Any]) -> str:
            edge_type = data.get("edge_type", data.get("label", ""))
            return str(getattr(edge_type, "name", edge_type) or "").strip().upper()

        required_items = set()
        for _src, _dst, data in repaired.edges(data=True):
            item_required = data.get("item_required")
            if item_required not in {None, ""}:
                required_items.add(str(item_required))
            elif _edge_type_name(data) == "ITEM_GATE":
                # Some imported/legacy graph edges encode the gate only by
                # edge_type. Keep a generic traversal item provider rather than
                # pruning every ITEM node under the final hard cap.
                required_items.add("ITEM")
        required_item_provider_nodes = {
            node_id
            for node_id, attrs in repaired.nodes(data=True)
            if str(attrs.get("item_type", "") or "") in required_items
            or (
                str(attrs.get("type", attrs.get("label", "")) or "").strip().upper() in {"ITEM", "PROTECTION_ITEM"}
                and "ITEM" in required_items
            )
            or (
                str(attrs.get("type", attrs.get("label", "")) or "").strip().upper() == "RESOURCE_FARM"
                and str(attrs.get("drops_resource", "") or "") in required_items
            )
        }
        removal_priority = {
            "EMPTY": 0,
            "TREASURE": 1,
            "SECRET": 2,
            "ENEMY": 3,
            "ARENA": 4,
            "ITEM": 5,
            "TUTORIAL_PUZZLE": 6,
            "COMBAT_PUZZLE": 7,
            "COMPLEX_PUZZLE": 8,
        }

        def _node_type(node_id: Any) -> str:
            attrs = repaired.nodes.get(node_id, {})
            return str(attrs.get("type", attrs.get("label", "")) or "").strip().upper()

        def _sort_key(node_id: Any) -> Tuple[int, int, str]:
            node_type = _node_type(node_id)
            degree = int(repaired.to_undirected().degree(node_id))
            return (removal_priority.get(node_type, 20), degree, str(node_id))

        progression_anchors = {
            node_id
            for node_id in repaired.nodes
            if _node_type(node_id) in protected_types
            or node_id in required_item_provider_nodes
        }

        def _progression_reachable(candidate: nx.Graph) -> bool:
            """Keep every progression anchor reachable without erasing directionality."""
            starts = [
                node_id
                for node_id in candidate.nodes
                if _node_type(node_id) == "START"
            ]
            if not starts:
                return False

            for anchor in progression_anchors:
                if anchor not in candidate:
                    return False
                if anchor in starts:
                    continue
                if not any(nx.has_path(candidate, start, anchor) for start in starts):
                    return False
            return True

        removed_nodes = 0
        while repaired.number_of_nodes() > hard_cap:
            candidates = [
                node_id
                for node_id in repaired.nodes
                if _node_type(node_id) not in protected_types
                and node_id not in required_item_provider_nodes
            ]
            if not candidates:
                break

            removed_this_round = False
            for node_id in sorted(candidates, key=_sort_key):
                candidate = repaired.copy()
                candidate.remove_node(node_id)
                if candidate.number_of_nodes() > 1 and not _progression_reachable(candidate):
                    continue
                repaired = candidate
                removed_nodes += 1
                removed_this_round = True
                break

            if not removed_this_round:
                break

        if removed_nodes:
            stats = repaired.graph.setdefault("generation_stats", {})
            stats["node_cap_pruned_nodes"] = int(stats.get("node_cap_pruned_nodes", 0)) + int(removed_nodes)
            logger.info(
                "Pruned %d non-critical output nodes to enforce hard node cap=%d",
                removed_nodes,
                hard_cap,
            )
        if repaired.number_of_nodes() > hard_cap:
            logger.warning(
                "Unable to fully enforce hard node cap=%d without removing protected progression nodes; output has %d nodes",
                hard_cap,
                repaired.number_of_nodes(),
            )
        return repaired

    def _repair_pedagogical_progression(
        self,
        graph: MissionGraph,
        *,
        constraint_grammar: Optional[MissionGrammar],
    ) -> MissionGraph:
        """Final Block I repair that backstops missing tutorial/combat/complex chains."""
        if graph is None:
            return graph

        metrics = self.evaluator._extract_descriptor_metrics(graph)
        needs_repair = (
            float(metrics.get("pedagogical_puzzle_variety", 0.0)) + 1e-6 < float(self.evaluator.target_pedagogical_puzzle_variety)
            or float(metrics.get("skill_chain_score", 0.0)) + 1e-6 < float(self.evaluator.target_skill_chain_score)
            or float(metrics.get("tutorial_climax_depth_score", 0.0)) + 1e-6 < float(self.evaluator.target_tutorial_climax_depth_score)
        )
        if not needs_repair:
            return graph

        repaired = copy.deepcopy(graph)
        rule = AddSkillChainRule()
        context = {"rng": self.rng, "difficulty": 0.5}
        repairs_applied = 0
        for _ in range(2):
            if not rule.can_apply(repaired, context):
                break
            candidate = rule.apply(repaired, context)
            if constraint_grammar is not None:
                candidate = constraint_grammar.fix_lock_key_ordering(candidate)
                candidate = constraint_grammar.repair_progression_constraints(candidate)
                candidate = constraint_grammar.ensure_anchor_nodes(candidate)
            candidate.sanitize()
            repaired = candidate
            repairs_applied += 1
            metrics = self.evaluator._extract_descriptor_metrics(repaired)
            if (
                float(metrics.get("pedagogical_puzzle_variety", 0.0)) + 1e-6 >= float(self.evaluator.target_pedagogical_puzzle_variety)
                and float(metrics.get("skill_chain_score", 0.0)) + 1e-6 >= float(self.evaluator.target_skill_chain_score)
                and float(metrics.get("tutorial_climax_depth_score", 0.0)) + 1e-6 >= float(self.evaluator.target_tutorial_climax_depth_score)
            ):
                break

        if repairs_applied > 0:
            repaired.record_repair("wave3_repairs", amount=int(repairs_applied))
            logger.info(
                "Applied %d final pedagogical progression repairs before Block I export",
                repairs_applied,
            )
        return repaired

    def _progression_balance_gap(self, metrics: Dict[str, float]) -> float:
        """Single scalar gap for final leniency/linearity balancing repairs."""
        min_gate_density = float(
            np.clip(self._rt("final_min_gate_density", 0.16), 0.0, 1.0)
        )
        max_key_surplus = float(
            max(0.0, self._rt("final_max_key_surplus", 1.0))
        )
        max_big_key_surplus = float(
            max(0.0, self._rt("final_max_big_key_surplus", 0.0))
        )
        leniency_excess = max(
            0.0,
            float(metrics.get("leniency", 0.0)) - float(self.evaluator.target_leniency),
        )
        gate_density_shortfall = max(
            0.0,
            min_gate_density - float(metrics.get("gating_density", 0.0)),
        )
        linearity_shortfall = max(
            0.0,
            float(self.evaluator.target_linearity) - float(metrics.get("linearity", 0.0)),
        )
        key_surplus_excess = max(
            0.0,
            float(metrics.get("small_key_surplus", 0.0)) - max_key_surplus,
        ) / max(1.0, max_key_surplus + 1.0)
        big_key_surplus_excess = max(
            0.0,
            float(metrics.get("boss_key_surplus", 0.0)) - max_big_key_surplus,
        )
        depth_shortfall = max(
            0.0,
            float(self.evaluator.target_tutorial_climax_depth_score)
            - float(metrics.get("tutorial_climax_depth_score", 0.0)),
        )
        variety_shortfall = max(
            0.0,
            float(self.evaluator.target_pedagogical_puzzle_variety)
            - float(metrics.get("pedagogical_puzzle_variety", 0.0)),
        )
        skill_shortfall = max(
            0.0,
            float(self.evaluator.target_skill_chain_score)
            - float(metrics.get("skill_chain_score", 0.0)),
        )
        leniency_weight = float(
            np.clip(self._rt("progression_balance_leniency_weight", 0.36), 0.0, 1.5)
        )
        linearity_weight = float(
            np.clip(self._rt("progression_balance_linearity_weight", 0.30), 0.0, 1.5)
        )
        gate_density_weight = float(
            np.clip(self._rt("progression_balance_gate_density_weight", 0.18), 0.0, 1.5)
        )
        key_surplus_weight = float(
            np.clip(self._rt("progression_balance_key_surplus_weight", 0.18), 0.0, 1.5)
        )
        big_key_surplus_weight = float(
            np.clip(self._rt("progression_balance_big_key_surplus_weight", 0.08), 0.0, 1.0)
        )
        depth_weight = float(
            np.clip(self._rt("progression_balance_depth_weight", 0.16), 0.0, 1.0)
        )
        variety_weight = float(
            np.clip(self._rt("progression_balance_variety_weight", 0.10), 0.0, 1.0)
        )
        skill_weight = float(
            np.clip(self._rt("progression_balance_skill_weight", 0.08), 0.0, 1.0)
        )
        return float(
            (leniency_weight * leniency_excess)
            + (gate_density_weight * gate_density_shortfall)
            + (linearity_weight * linearity_shortfall)
            + (key_surplus_weight * key_surplus_excess)
            + (big_key_surplus_weight * big_key_surplus_excess)
            + (depth_weight * depth_shortfall)
            + (variety_weight * variety_shortfall)
            + (skill_weight * skill_shortfall)
        )

    def _repair_progression_balance(
        self,
        graph: MissionGraph,
        *,
        constraint_grammar: Optional[MissionGrammar],
    ) -> MissionGraph:
        """
        Final Block I pass that tightens overly lenient graphs with existing gate rules.

        This uses existing grammar rules instead of bespoke topology surgery so the
        exported graph stays aligned with the rest of the grammar/search system.
        """
        if graph is None:
            return graph

        repaired = copy.deepcopy(graph)
        current_metrics = self.evaluator._extract_descriptor_metrics(repaired)
        current_gap = self._progression_balance_gap(current_metrics)
        if current_gap <= 1e-6:
            return repaired

        repair_rules = (
            PruneDeadEndRule(),
            AddGatekeeperRule(),
            AddItemGateRule(),
            AddHazardGateRule(),
            AddEntangledBranchesRule(),
            AddArenaRule(),
            AddMultiLockRule(),
        )
        context = {"rng": self.rng, "difficulty": 0.5}
        repairs_applied = 0
        max_repairs = int(
            np.clip(round(self._rt("progression_balance_repair_iterations", 3.0)), 1.0, 6.0)
        )
        for _ in range(max_repairs):
            best_candidate: Optional[MissionGraph] = None
            best_gap = current_gap
            for rule in repair_rules:
                if not rule.can_apply(copy.deepcopy(repaired), context):
                    continue
                candidate = rule.apply(copy.deepcopy(repaired), context)
                if constraint_grammar is not None:
                    candidate = constraint_grammar.fix_lock_key_ordering(candidate)
                    candidate = constraint_grammar.repair_progression_constraints(candidate)
                    candidate = self._repair_pedagogical_progression(candidate, constraint_grammar=constraint_grammar)
                    candidate = constraint_grammar.ensure_anchor_nodes(candidate)
                candidate.sanitize()
                candidate_metrics = self.evaluator._extract_descriptor_metrics(candidate)
                candidate_gap = self._progression_balance_gap(candidate_metrics)
                if candidate_gap + 1e-6 < best_gap:
                    best_candidate = candidate
                    best_gap = candidate_gap

            if best_candidate is None:
                break
            repaired = best_candidate
            current_gap = best_gap
            repairs_applied += 1
            if current_gap <= 1e-6:
                break

        if repairs_applied > 0:
            repaired.record_repair("progression_repairs", amount=int(repairs_applied))
            logger.info(
                "Applied %d final progression-balance repairs before Block I export",
                repairs_applied,
            )
        return repaired

    def _trim_surplus_reward_keys(self, graph: MissionGraph) -> Tuple[MissionGraph, int]:
        """
        Demote gratuitous side-branch KEY/BIG_KEY rewards before export.

        This specifically targets topology-expansion rules that can mint free
        reward keys without corresponding gate pressure. Critical-path keys and
        the canonical boss-gauntlet BIG_KEY are preserved.
        """
        trimmed = copy.deepcopy(graph)
        start = trimmed.get_start_node()
        goal = trimmed.get_goal_node()
        critical_path: Set[Any] = set()
        if start is not None and goal is not None:
            path = self.evaluator._find_path(trimmed, start.id, goal.id)
            if path:
                critical_path = set(path)

        max_small_key_surplus = int(max(0.0, self._rt("final_max_key_surplus", 1.0)))
        max_big_key_surplus = int(max(0.0, self._rt("final_max_big_key_surplus", 0.0)))

        small_key_demand = 0
        for edge in trimmed.edges:
            if edge.edge_type != EdgeType.LOCKED:
                continue
            small_key_demand += int(max(1, edge.requires_key_count)) if edge.requires_key_count > 0 else 1
        boss_door_key_ids = {
            node.key_id
            for node in trimmed.nodes.values()
            if node.node_type == NodeType.BOSS_DOOR and node.key_id is not None
        }

        changes = 0

        def _demote(node: MissionNode, target_type: NodeType) -> None:
            nonlocal changes
            node.node_type = target_type
            node.key_id = None
            if target_type != NodeType.ITEM:
                node.item_type = None
            changes += 1

        small_key_nodes = [
            node for node in trimmed.nodes.values()
            if node.node_type == NodeType.KEY and node.key_id is None
        ]
        small_key_nodes.sort(
            key=lambda node: (
                node.id in critical_path,
                -int(trimmed.get_node_degree(node.id)),
                node.id,
            )
        )
        keep_small_keys = int(max(0, small_key_demand + max_small_key_surplus))
        for node in small_key_nodes[keep_small_keys:]:
            if node.id in critical_path:
                continue
            replacement = NodeType.TREASURE if (node.id % 2 == 0) else NodeType.ITEM
            _demote(node, replacement)

        matching_big_keys = [
            node
            for node in trimmed.nodes.values()
            if node.node_type == NodeType.BIG_KEY and node.key_id in boss_door_key_ids
        ]
        extra_big_keys = [
            node
            for node in trimmed.nodes.values()
            if node.node_type == NodeType.BIG_KEY and node not in matching_big_keys
        ]
        if len(matching_big_keys) > len(boss_door_key_ids) + max_big_key_surplus:
            protected: Set[int] = set()
            for key_id in boss_door_key_ids:
                keeper = next((node for node in matching_big_keys if node.key_id == key_id), None)
                if keeper is not None:
                    protected.add(keeper.id)
            for node in matching_big_keys:
                if node.id in protected:
                    continue
                extra_big_keys.append(node)

        for node in extra_big_keys:
            if node.id in critical_path and node.key_id in boss_door_key_ids:
                continue
            _demote(node, NodeType.TREASURE)

        if changes > 0:
            trimmed.sanitize()
        return trimmed, int(changes)

    def _repair_gate_economy(
        self,
        graph: MissionGraph,
        *,
        constraint_grammar: Optional[MissionGrammar],
    ) -> MissionGraph:
        """
        Final export calibration for gate density and key surplus.

        The progression-balance pass improves overall shape, but this stricter
        pass focuses on two concrete Block I issues:
        - too many free reward keys relative to actual gates
        - too few meaningful gates on the exported mission graph
        """
        if graph is None:
            return graph

        repaired = copy.deepcopy(graph)
        trimmed, trim_changes = self._trim_surplus_reward_keys(repaired)
        repaired = trimmed
        if constraint_grammar is not None:
            repaired = constraint_grammar.fix_lock_key_ordering(repaired)
            repaired = constraint_grammar.repair_progression_constraints(repaired)
            repaired = self._repair_pedagogical_progression(repaired, constraint_grammar=constraint_grammar)
            repaired = constraint_grammar.ensure_anchor_nodes(repaired)
            if not constraint_grammar.validate_goal_gauntlet(repaired, log_failures=False):
                repaired = constraint_grammar.repair_progression_constraints(repaired)
        repaired.sanitize()

        current_metrics = self.evaluator._extract_descriptor_metrics(repaired)
        current_gap = self._progression_balance_gap(current_metrics)
        repair_rules = (
            PruneDeadEndRule(),
            AddGatekeeperRule(),
            AddItemGateRule(),
        )
        context = {"rng": self.rng, "difficulty": 0.55}
        repairs_applied = int(max(0, trim_changes))
        max_repairs = int(
            np.clip(round(self._rt("final_gate_calibration_iterations", 4.0)), 1.0, 8.0)
        )

        for _ in range(max_repairs):
            best_candidate: Optional[MissionGraph] = None
            best_gap = current_gap
            for rule in repair_rules:
                if not rule.can_apply(copy.deepcopy(repaired), context):
                    continue
                candidate = rule.apply(copy.deepcopy(repaired), context)
                candidate, candidate_trim_changes = self._trim_surplus_reward_keys(candidate)
                if constraint_grammar is not None:
                    candidate = constraint_grammar.fix_lock_key_ordering(candidate)
                    candidate = constraint_grammar.repair_progression_constraints(candidate)
                    candidate = self._repair_pedagogical_progression(candidate, constraint_grammar=constraint_grammar)
                    candidate = constraint_grammar.ensure_anchor_nodes(candidate)
                    if not constraint_grammar.validate_goal_gauntlet(candidate, log_failures=False):
                        continue
                candidate.sanitize()
                candidate_metrics = self.evaluator._extract_descriptor_metrics(candidate)
                candidate_gap = self._progression_balance_gap(candidate_metrics)
                if candidate_gap + 1e-6 < best_gap:
                    best_candidate = candidate
                    best_gap = candidate_gap
                    best_candidate_trim_changes = int(max(0, candidate_trim_changes))

            if best_candidate is None:
                break
            repaired = best_candidate
            current_gap = best_gap
            repairs_applied += int(1 + best_candidate_trim_changes)
            if current_gap <= 1e-6:
                break

        if repairs_applied > 0:
            repaired.record_repair("progression_repairs", amount=int(repairs_applied))
            repaired.generation_stats["gate_economy_repairs"] = int(
                repaired.generation_stats.get("gate_economy_repairs", 0)
            ) + int(repairs_applied)
            logger.info(
                "Applied %d final gate-economy calibrations before Block I export",
                repairs_applied,
            )
        if constraint_grammar is not None and not constraint_grammar.validate_goal_gauntlet(repaired, log_failures=False):
            repaired = constraint_grammar.repair_progression_constraints(repaired)
            repaired.sanitize()
        return repaired

    @staticmethod
    def _repair_output_connectivity(graph: nx.Graph) -> nx.Graph:
        """
        Connect disconnected physical components with PATH edges.

        Block I search can occasionally leave an isolated side-room after
        constraint repair. Downstream strict topology validation rejects those
        graphs, so final export stitches the components together using the
        closest non-goal anchors.
        """
        if graph.number_of_nodes() <= 1:
            return graph

        undirected = graph.to_undirected()
        if nx.is_connected(undirected):
            return graph

        repaired = graph.copy()
        repaired.graph["generation_stats"] = copy.deepcopy(graph.graph.get("generation_stats", {}))
        stats = repaired.graph.setdefault("generation_stats", {})
        protected_goal_types = {"GOAL", "BOSS", "BOSS_DOOR"}
        protected_goal_nodes = {
            node_id
            for node_id, attrs in repaired.nodes(data=True)
            if str(attrs.get("type", attrs.get("label", ""))).strip().upper() in protected_goal_types
        }

        def _node_type(node_id: Any) -> str:
            attrs = repaired.nodes.get(node_id, {})
            return str(attrs.get("type", attrs.get("label", "")) or "").strip().upper()

        def _position(node_id: Any) -> Tuple[float, float, float]:
            pos = repaired.nodes.get(node_id, {}).get("position", (0, 0, 0))
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                z = float(pos[2]) if len(pos) > 2 else 0.0
                return (float(pos[0]), float(pos[1]), z)
            return (0.0, 0.0, 0.0)

        def _candidate_nodes(component: set[Any]) -> List[Any]:
            preferred = [node_id for node_id in component if node_id not in protected_goal_nodes]
            return preferred if preferred else list(component)

        def _source_candidates(main_component_nodes: set[Any], other_component_nodes: set[Any]) -> List[Any]:
            other_types = {_node_type(node_id) for node_id in other_component_nodes}
            if other_types and other_types.issubset(protected_goal_types):
                progression_anchors = [
                    node_id
                    for node_id in main_component_nodes
                    if _node_type(node_id) in {"BOSS", "BOSS_DOOR"}
                ]
                if progression_anchors:
                    return progression_anchors
            return _candidate_nodes(main_component_nodes)

        def _distance(node_a: Any, node_b: Any) -> float:
            ax, ay, az = _position(node_a)
            bx, by, bz = _position(node_b)
            return abs(ax - bx) + abs(ay - by) + abs(az - bz)

        components = [set(component) for component in nx.connected_components(repaired.to_undirected())]
        start_candidates = [node for node, attrs in repaired.nodes(data=True) if str(attrs.get("type", "")).upper() == "START"]
        if start_candidates:
            start_node = start_candidates[0]
            main_index = next((idx for idx, component in enumerate(components) if start_node in component), 0)
        else:
            main_index = max(range(len(components)), key=lambda idx: len(components[idx]))
        main_component = set(components.pop(main_index))

        connectivity_repairs = 0
        while components:
            other_component = components.pop(0)
            best_pair: Optional[Tuple[Any, Any]] = None
            best_distance: Optional[float] = None
            for left in _source_candidates(main_component, other_component):
                for right in _candidate_nodes(other_component):
                    dist = _distance(left, right)
                    if best_distance is None or dist < best_distance:
                        best_distance = dist
                        best_pair = (left, right)

            if best_pair is None:
                logger.warning(
                    "Unable to find any node pair while repairing output connectivity; component kept disconnected: %s",
                    sorted(other_component, key=str),
                )
                main_component.update(other_component)
                continue

            source, target = best_pair
            source_type = _node_type(source)
            target_type = _node_type(target)
            protected_progression_link = target_type in protected_goal_types and source_type not in {"BOSS", "BOSS_DOOR"}
            repair_edge_type = EdgeType.BOSS_LOCKED if protected_progression_link else EdgeType.PATH
            edge_attrs = {
                "label": repair_edge_type.name.lower(),
                "edge_type": repair_edge_type.name,
                "key_required": None,
                "item_required": None,
                "switch_id": None,
                "metadata": {
                    "connectivity_repair": True,
                    "progression_gate_repair": bool(protected_progression_link),
                },
                "requires_key_count": 0,
                "token_count": 0,
                "token_id": None,
                "is_window": False,
                "hazard_damage": 0,
                "protection_item_id": None,
                "preferred_direction": None,
                "battery_id": None,
                "switches_required": [],
                "path_savings": 0,
            }
            repaired.add_edge(source, target, **edge_attrs)
            if repaired.is_directed() and _node_type(target) not in protected_goal_types and not repaired.has_edge(target, source):
                reverse_attrs = copy.deepcopy(edge_attrs)
                reverse_attrs["metadata"] = {"connectivity_repair": True, "implied_reverse": True}
                repaired.add_edge(target, source, **reverse_attrs)
            connectivity_repairs += 1
            main_component.update(other_component)
            components = [set(component) for component in nx.connected_components(repaired.to_undirected()) if not component.issubset(main_component)]

        if connectivity_repairs > 0:
            stats["connectivity_repairs"] = int(stats.get("connectivity_repairs", 0)) + int(connectivity_repairs)
            stats["total_repairs"] = int(stats.get("total_repairs", 0)) + int(connectivity_repairs)
            stats["repair_applied"] = True

        return repaired

    
    def _generate_initial_population(self) -> List[Individual]:
        """
        Create random rule sequences as starting genomes.
        
        Uses weighted sampling to prefer common rule types:
        - 40% Enemy challenges
        - 20% Puzzle challenges
        - 25% Lock-key pairs
        - 15% Branches
        
        Returns:
            List of Individual objects with random genomes
        """
        population = []
        pedagogical_seed_count = self._pedagogical_seed_genome_count()
        for _ in range(pedagogical_seed_count):
            population.append(Individual(genome=self._build_structured_seed_genome(), generation=0))

        # Weighted rule sampling.
        rule_ids = self._rule_ids
        if len(self.executor.rules) == 5 and self.rule_space != "full":
            # Legacy distribution for the core 5-rule executor.
            sampling_weights = [0.4, 0.2, 0.25, 0.15]
        else:
            # Full-rule mode: reuse grammar rule weights with a small floor
            # to preserve exploration.
            sampling_weights = [max(0.01, float(self._global_rule_weights.get(rid, 0.01))) for rid in rule_ids]

        while len(population) < self.population_size:
            # Generate random genome
            genome = []
            for _ in range(self.genome_length):
                # Weighted random choice
                rule_id = self.rng.choices(
                    rule_ids,
                    weights=sampling_weights,
                    k=1
                )[0]
                genome.append(rule_id)
            
            individual = Individual(genome=genome, generation=0)
            population.append(individual)
        
        logger.debug("Generated initial population of %d individuals", len(population))

        return population

    def _pedagogical_seed_genome_count(self) -> int:
        """How many initial genomes should be biased toward tutorial progression."""
        if self.population_size <= 0:
            return 0
        pedagogical_target = float(
            np.clip(
                (0.40 * self.evaluator.target_pedagogical_puzzle_variety)
                + (0.35 * self.evaluator.target_skill_chain_score)
                + (0.25 * self.evaluator.target_tutorial_climax_depth_score),
                0.0,
                1.0,
            )
        )
        if pedagogical_target <= 0.0:
            return 0
        seed_fraction = float(np.clip(self._rt("initial_pedagogical_seed_fraction", 0.20), 0.0, 0.80))
        min_seed = int(max(0, round(self._rt("initial_pedagogical_seed_min", 2.0))))
        max_seed_fraction = float(np.clip(self._rt("initial_pedagogical_seed_max_fraction", 0.45), 0.05, 1.0))
        proposed = int(round(float(self.population_size) * seed_fraction * pedagogical_target))
        max_allowed = int(max(1, math.ceil(float(self.population_size) * max_seed_fraction)))
        return int(max(0, min(max_allowed, max(min_seed, proposed))))

    def _choose_rule_for_seed(self, *groups: Sequence[int]) -> Optional[int]:
        """Pick one rule from the first non-empty candidate group."""
        candidates: List[int] = []
        for group in groups:
            ids = [int(rid) for rid in group if int(rid) in self._global_rule_weights]
            if ids:
                candidates.extend(ids)
                break
        if not candidates:
            return None
        weights = [max(1e-6, float(self._global_rule_weights.get(rid, 1e-6))) for rid in candidates]
        return int(self.rng.choices(candidates, weights=weights, k=1)[0])

    def _build_structured_seed_genome(self) -> List[int]:
        """Construct an initial genome that already contains a tutorial-to-climax skeleton."""
        genome = self._sample_weighted_genome()
        if not genome:
            return genome

        anchor_specs = [
            (
                0.18,
                self._choose_rule_for_seed(self._linear_progression_rule_ids, self._path_depth_rule_ids),
            ),
            (
                0.34,
                self._choose_rule_for_seed(self._pedagogical_rule_ids, self._linear_progression_rule_ids),
            ),
            (
                0.52,
                self._choose_rule_for_seed(self._non_key_gate_rule_ids, self._pedagogical_depth_support_rule_ids),
            ),
            (
                0.72,
                self._choose_rule_for_seed(self._pedagogical_depth_support_rule_ids, self._critical_path_gate_rule_ids),
            ),
            (
                0.88,
                self._choose_rule_for_seed(self._critical_path_gate_rule_ids, self._linear_progression_rule_ids),
            ),
        ]
        for ratio, rule_id in anchor_specs:
            if rule_id is None:
                continue
            idx = int(np.clip(round((len(genome) - 1) * float(ratio)), 0, len(genome) - 1))
            genome[idx] = int(rule_id)
        return genome
    
    def _evaluate_population(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Individual]:
        """
        Evaluate fitness for all individuals in population.
        
        Args:
            population: List of individuals to evaluate
            generation: Current generation number
            
        Returns:
            Population with updated fitness and phenotype
        """
        for individual in population:
            if not individual.evaluated:
                self._evaluate_individual(individual, generation=generation)
        
        return population

    def _evaluate_individual(self, individual: Individual, generation: int) -> Individual:
        """Evaluate one individual in-place."""
        individual.phenotype = self.executor.execute(
            individual.genome,
            difficulty=0.5,
            max_nodes=self.max_nodes,
        )
        eval_result = self.evaluator.evaluate_graph(individual.phenotype)
        individual.fitness = float(eval_result.get("fitness", 0.0))
        individual.feasible = bool(eval_result.get("feasible", False))
        individual.constraint_violation = float(eval_result.get("constraint_violation", 1.0))
        individual.descriptor_metrics = dict(eval_result.get("descriptor_metrics", {}))
        individual.topology_realism_error = float(
            individual.descriptor_metrics.get("topology_realism_error", float("inf"))
        )
        if self.enable_rule_credit_assignment:
            individual.rule_fitness_deltas = self._compute_rule_fitness_deltas(
                genome=individual.genome,
                base_fitness=float(individual.fitness),
            )
        individual.generation = generation
        individual.evaluated = True
        return individual

    def _compute_rule_fitness_deltas(self, genome: Sequence[int], base_fitness: float) -> Dict[int, float]:
        """
        Estimate marginal fitness credit per rule position via leave-one-out ablation.

        Positive delta means removing the rule lowered fitness (rule helped).
        Negative delta means removing the rule improved fitness (rule hurt).
        """
        g = [int(x) for x in genome]
        if not g:
            return {}
        max_samples = int(max(1, round(self._rt("rule_credit_max_samples", 8.0))))
        max_samples = min(max_samples, len(g))
        if max_samples >= len(g):
            probe_indices = list(range(len(g)))
        else:
            probe_indices = sorted(self.rng.sample(list(range(len(g))), k=max_samples))

        deltas: Dict[int, float] = {}
        for idx in probe_indices:
            pruned = g[:idx] + g[idx + 1 :]
            if not pruned:
                deltas[int(idx)] = float(base_fitness)
                continue
            probe_graph = self.executor.execute(
                pruned,
                difficulty=0.5,
                max_nodes=self.max_nodes,
            )
            probe_result = self.evaluator.evaluate_graph(probe_graph)
            probe_fitness = float(probe_result.get("fitness", 0.0))
            deltas[int(idx)] = float(base_fitness - probe_fitness)
        return deltas

    def summarize_rule_marginal_credit(
        self,
        genome: Sequence[int],
        *,
        max_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Aggregate position-level marginal deltas into per-rule average credit."""
        _, by_rule = self.compute_rule_marginal_credit(genome, max_samples=max_samples)
        return by_rule

    def compute_rule_marginal_credit(
        self,
        genome: Sequence[int],
        *,
        max_samples: Optional[int] = None,
    ) -> Tuple[Dict[int, float], Dict[str, float]]:
        """Return marginal fitness credit by genome position and by rule name."""
        genes = [int(x) for x in genome]
        if not genes:
            return {}, {}

        if max_samples is not None:
            local_max = float(max(1, int(max_samples)))
            original = self.realism_tuning.get("rule_credit_max_samples")
            self.realism_tuning["rule_credit_max_samples"] = local_max
            deltas = self._compute_rule_fitness_deltas(genes, base_fitness=self.evaluator.evaluate_graph(
                self.executor.execute(genes, difficulty=0.5, max_nodes=self.max_nodes)
            ).get("fitness", 0.0))
            if original is None:
                self.realism_tuning.pop("rule_credit_max_samples", None)
            else:
                self.realism_tuning["rule_credit_max_samples"] = original
        else:
            baseline = self.evaluator.evaluate_graph(
                self.executor.execute(genes, difficulty=0.5, max_nodes=self.max_nodes)
            )
            deltas = self._compute_rule_fitness_deltas(genes, base_fitness=float(baseline.get("fitness", 0.0)))

        by_rule: Dict[str, List[float]] = defaultdict(list)
        for idx, delta in deltas.items():
            if idx < 0 or idx >= len(genes):
                continue
            rid = genes[idx]
            if rid < 0 or rid >= len(self.executor.rule_names):
                rule_name = f"RULE_{rid}"
            else:
                rule_name = str(self.executor.rule_names[rid])
            by_rule[rule_name].append(float(delta))
        by_rule_avg = {
            str(rule_name): float(np.mean(values))
            for rule_name, values in by_rule.items()
            if values
        }
        by_index = {
            int(idx): float(delta)
            for idx, delta in sorted(deltas.items(), key=lambda item: int(item[0]))
        }
        return by_index, by_rule_avg

    @staticmethod
    def _individual_sort_key(ind: Individual) -> Tuple[int, float, float, float, float]:
        """Deb-style feasibility-first ordering key."""
        return (
            0 if bool(ind.feasible) else 1,
            0.0 if bool(ind.feasible) else float(ind.constraint_violation),
            -float(ind.fitness),
            float(ind.topology_realism_error if np.isfinite(ind.topology_realism_error) else 10.0),
            float(ind.generation_rejection_ratio if np.isfinite(ind.generation_rejection_ratio) else 1.0),
        )

    def _renormalize_global_rule_probs(self) -> None:
        """Recompute normalized rule-sampling probabilities from weights."""
        total = float(sum(float(w) for w in self._global_rule_weights.values()))
        if total <= 0.0:
            uniform = 1.0 / max(1, len(self._rule_ids))
            self._global_rule_probs = {rid: uniform for rid in self._rule_ids}
            return
        self._global_rule_probs = {
            rid: float(self._global_rule_weights[rid] / total)
            for rid in self._rule_ids
        }

    def _merge_realism_tuning(self, overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Validate and merge realism tuning overrides onto defaults."""
        merged = dict(DEFAULT_REALISM_TUNING)
        if not overrides:
            return merged
        for key, value in overrides.items():
            key_name = str(key).strip()
            if key_name not in merged:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(parsed):
                continue
            merged[key_name] = parsed
        return merged

    def _rt(self, key: str, default: float) -> float:
        """Read a realism tuning value with fallback."""
        value = self.realism_tuning.get(str(key).strip(), default)
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return parsed if math.isfinite(parsed) else float(default)

    def _select_rule_ids_by_exact_names(self, names: Sequence[str]) -> List[int]:
        """Resolve rule IDs by exact rule name match."""
        target = {str(n).strip().lower() for n in names if str(n).strip()}
        out: List[int] = []
        for rid in self._rule_ids:
            name = str(self.executor.rule_names[rid]).strip().lower()
            if name in target:
                out.append(int(rid))
        return sorted(set(out))

    def _apply_target_aware_rule_prior(self) -> None:
        """
        Shape sampling priors using descriptor targets.

        When target shortcut density is low (VGLC-like), reduce explicit
        teleport/item-shortcut pressure while preserving loop-forming rules.
        """
        target_shortcut = float(max(0.0, self.evaluator.target_shortcut_density))
        damp = 1.0
        if getattr(self, "_explicit_shortcut_rule_ids", None) and target_shortcut < 0.10:
            damp = float(np.clip(target_shortcut / 0.08, 0.20, 1.0))
            for rid in self._explicit_shortcut_rule_ids:
                self._global_rule_weights[rid] = float(
                    max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * damp)
                )

        # Calibrate gate-heavy rules:
        # - gate depth controls critical-path gate concentration,
        # - gating density controls overall gate prevalence.
        target_gate_depth = float(max(0.0, self.evaluator.target_gate_depth_ratio))
        target_gating_density = float(max(0.0, self.evaluator.target_gating_density))
        if getattr(self, "_critical_path_gate_rule_ids", None) and target_gate_depth < 0.25:
            critical_gate_damp = float(np.clip(target_gate_depth / 0.18, 0.70, 1.0))
            for rid in self._critical_path_gate_rule_ids:
                self._global_rule_weights[rid] = float(
                    max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * critical_gate_damp)
                )
        if getattr(self, "_side_gate_rule_ids", None):
            side_gate_boost = float(np.clip(0.82 + (1.35 * target_gating_density), 0.85, 1.28))
            for rid in self._side_gate_rule_ids:
                self._global_rule_weights[rid] = float(
                    max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * side_gate_boost)
                )

        # Keep directed-vs-weak path mismatch near target by damping explicit
        # one-way mechanics when directionality target is near zero.
        target_directionality = float(max(0.0, self.evaluator.target_directionality_gap))
        directionality_damp = 1.0
        if getattr(self, "_directionality_heavy_rule_ids", None) and target_directionality < 0.10:
            directionality_damp = float(np.clip((target_directionality + 0.02) / 0.10, 0.30, 1.0))
            for rid in self._directionality_heavy_rule_ids:
                self._global_rule_weights[rid] = float(
                    max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * directionality_damp)
                )
            # Preserve progression pressure by reallocating part of one-way dampening
            # budget toward gate rules that do not introduce directional asymmetry.
            if getattr(self, "_gate_non_directional_rule_ids", None):
                non_dir_gate_boost = float(np.clip(1.0 + (0.22 * (1.0 - directionality_damp)), 1.0, 1.20))
                for rid in self._gate_non_directional_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * non_dir_gate_boost)
                    )

        # If the target asks for stronger main-path structure, shift part of the
        # prior from wide branching toward critical-path/tutorial operators.
        target_linearity = float(np.clip(self.evaluator.target_linearity, 0.0, 1.0))
        if target_linearity > 0.42:
            linearity_pressure = float(np.clip((target_linearity - 0.42) / 0.24, 0.0, 1.0))
            if getattr(self, "_wide_branch_rule_ids", None):
                branch_damp = float(
                    np.clip(
                        1.0
                        - (
                            float(
                                np.clip(
                                    self._rt("prior_linearity_branch_damp_gain", 0.22),
                                    0.0,
                                    0.60,
                                )
                            )
                            * linearity_pressure
                        ),
                        0.64,
                        1.0,
                    )
                )
                for rid in self._wide_branch_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * branch_damp)
                    )
            if getattr(self, "_linear_progression_rule_ids", None):
                linear_boost = float(
                    np.clip(
                        1.0
                        + (
                            float(
                                np.clip(
                                    self._rt("prior_linearity_boost_gain", 0.34),
                                    0.0,
                                    0.80,
                                )
                            )
                            * linearity_pressure
                        ),
                        1.0,
                        1.48,
                    )
                )
                for rid in self._linear_progression_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * linear_boost)
                    )
            if getattr(self, "_branch_pruning_rule_ids", None):
                prune_boost = float(
                    np.clip(
                        1.0
                        + (
                            float(
                                np.clip(
                                    self._rt("prior_linearity_prune_boost_gain", 0.30),
                                    0.0,
                                    0.80,
                                )
                            )
                            * linearity_pressure
                        ),
                        1.0,
                        1.40,
                    )
                )
                for rid in self._branch_pruning_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * prune_boost)
                    )

        # If leniency target is moderate/strict (VGLC-like), shift part of prior
        # from key-inflating rules to non-key gate operators.
        target_leniency = float(np.clip(self.evaluator.target_leniency, 0.0, 1.0))
        if target_leniency < 0.70:
            leniency_tightness = float(np.clip((0.70 - target_leniency) / 0.40, 0.0, 1.0))
            if getattr(self, "_key_inflating_rule_ids", None):
                key_damp = float(
                    np.clip(
                        1.0
                        - (
                            float(
                                np.clip(
                                    self._rt("prior_leniency_key_damp_gain", 0.44),
                                    0.0,
                                    0.80,
                                )
                            )
                            * leniency_tightness
                        ),
                        0.40,
                        1.0,
                    )
                )
                for rid in self._key_inflating_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * key_damp)
                    )
            if getattr(self, "_non_key_gate_rule_ids", None):
                gate_boost = float(
                    np.clip(
                        1.0
                        + (
                            float(
                                np.clip(
                                    self._rt("prior_leniency_gate_boost_gain", 0.34),
                                    0.0,
                                    0.80,
                                )
                            )
                            * leniency_tightness
                        ),
                        1.0,
                        1.42,
                    )
                )
                for rid in self._non_key_gate_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * gate_boost)
                    )
            if getattr(self, "_path_depth_rule_ids", None):
                depth_boost = float(
                    np.clip(
                        1.0
                        + (
                            float(
                                np.clip(
                                    self._rt("prior_leniency_depth_boost_gain", 0.24),
                                    0.0,
                                    0.60,
                                )
                            )
                            * leniency_tightness
                        ),
                        1.0,
                        1.30,
                    )
                )
                for rid in self._path_depth_rule_ids:
                    self._global_rule_weights[rid] = float(
                        max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * depth_boost)
                    )

        # If the descriptor targets ask for pedagogical item->challenge arcs,
        # bias the prior toward the dedicated skill-chain rule instead of
        # hoping generic puzzle operators stumble into that structure.
        pedagogical_target = float(
            np.clip(
                (0.45 * self.evaluator.target_pedagogical_puzzle_variety)
                + (0.30 * self.evaluator.target_skill_chain_score)
                + (0.25 * self.evaluator.target_tutorial_climax_depth_score),
                0.0,
                1.0,
            )
        )
        if getattr(self, "_pedagogical_rule_ids", None) and pedagogical_target > 0.0:
            pedagogical_boost_gain = float(
                np.clip(self._rt("prior_pedagogical_boost_gain", 0.42), 0.0, 1.0)
            )
            pedagogical_boost_max = float(
                np.clip(self._rt("prior_pedagogical_boost_max", 1.35), 1.0, 2.0)
            )
            self._scale_rule_weight_group(
                self._pedagogical_rule_ids,
                1.0 + (pedagogical_boost_gain * pedagogical_target),
                min_factor=1.0,
                max_factor=pedagogical_boost_max,
            )
            self._scale_rule_weight_group(
                self._pedagogical_support_rule_ids,
                1.0 + (0.20 * pedagogical_target),
                min_factor=1.0,
                max_factor=1.24,
            )
            self._scale_rule_weight_group(
                self._pedagogical_depth_support_rule_ids,
                1.0 + (0.26 * pedagogical_target),
                min_factor=1.0,
                max_factor=1.30,
            )

        # Rebalance toward topology rules to avoid shrinking exploration
        # while keeping gate depth from overshooting reference.
        rebalance_ids = [
            rid
            for rid in set(self._topology_pressure_rule_ids)
            if rid not in set(self._explicit_shortcut_rule_ids)
        ]
        if rebalance_ids:
            boost = float(np.clip(1.0 + (0.40 * (1.0 - damp)), 1.0, 1.5))
            for rid in rebalance_ids:
                self._global_rule_weights[rid] = float(
                    max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * boost)
                )

        # Structural budget prior shaping:
        # nudge expansion operators when descriptor targets request larger
        # reference-like graphs (higher node/edge budgets).
        target_nodes = float(max(0.0, self.evaluator.target_num_nodes))
        target_edges = float(max(0.0, self.evaluator.target_num_edges))
        if target_nodes > 0.0:
            max_nodes_soft = float(max(1.0, float(self.max_nodes)))
            node_deficit = float(
                np.clip((target_nodes - (0.85 * max_nodes_soft)) / max(1.0, target_nodes), 0.0, 1.0)
            )
            if node_deficit > 0.0:
                node_boost_gain = float(np.clip(self._rt("prior_node_boost_gain", 0.30), 0.0, 0.80))
                node_boost_max = float(np.clip(self._rt("prior_node_boost_max", 1.25), 1.0, 2.0))
                self._scale_rule_weight_group(
                    self._node_expansion_rule_ids,
                    1.0 + (node_boost_gain * node_deficit),
                    min_factor=1.0,
                    max_factor=node_boost_max,
                )
        if target_edges > 0.0 and target_nodes > 0.0:
            target_density = float(target_edges / max(1.0, target_nodes))
            density_boost = float(np.clip((target_density - 1.35) / 1.25, 0.0, 1.0))
            if density_boost > 0.0:
                edge_boost_gain = float(np.clip(self._rt("prior_edge_boost_gain", 0.24), 0.0, 0.80))
                edge_boost_max = float(np.clip(self._rt("prior_edge_boost_max", 1.20), 1.0, 2.0))
                self._scale_rule_weight_group(
                    self._edge_expansion_rule_ids,
                    1.0 + (edge_boost_gain * density_boost),
                    min_factor=1.0,
                    max_factor=edge_boost_max,
                )

        self._renormalize_global_rule_probs()

    def _apply_custom_transition_bias_to_global_prior(self) -> None:
        """
        Preserve explicit user transition priors after target-aware shaping.

        Research-wise, the mission graph prior is the user's high-level control
        surface. If we let later realism/pedagogical priors completely swamp a
        supplied transition matrix, search stops honoring the requested style.
        This helper keeps custom transition intent alive in initial sampling and
        long-run prior relaxation, while leaving the default learned Zelda
        transition matrix unchanged.
        """
        if not getattr(self, "_has_custom_transition_matrix", False):
            return

        inbound_mass: Dict[int, float] = defaultdict(float)
        for transitions in dict(self.transition_matrix or {}).values():
            if not isinstance(transitions, dict):
                continue
            for rule_name, raw_weight in transitions.items():
                rid = self.rule_name_to_id.get(str(rule_name))
                if rid is None or rid not in self._global_rule_weights:
                    continue
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not math.isfinite(weight) or weight <= 0.0:
                    continue
                inbound_mass[int(rid)] += float(weight)

        if not inbound_mass:
            return

        max_mass = max(float(value) for value in inbound_mass.values())
        if max_mass <= 0.0:
            return

        # Keep the boost moderate: explicit user transitions should steer the
        # search, but not fully override target realism / constraint pressures.
        base_gain = float(np.clip(0.35 + (0.35 * self.transition_mix), 0.20, 0.80))
        max_boost = float(np.clip(1.0 + base_gain, 1.10, 1.80))
        for rid, mass in inbound_mass.items():
            normalized = float(np.clip(float(mass) / max_mass, 0.0, 1.0))
            boost = float(np.clip(1.0 + (base_gain * normalized), 1.0, max_boost))
            self._global_rule_weights[rid] = float(
                max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * boost)
            )

        self._renormalize_global_rule_probs()

    def _scale_rule_weight_group(
        self,
        rule_ids: Sequence[int],
        factor: float,
        *,
        min_factor: float = 0.35,
        max_factor: float = 1.70,
    ) -> None:
        """
        Scale a rule subset with clipping.
        """
        ids = [int(rid) for rid in rule_ids if int(rid) in self._global_rule_weights]
        if not ids:
            return
        local = float(np.clip(float(factor), float(min_factor), float(max_factor)))
        for rid in ids:
            self._global_rule_weights[rid] = float(
                max(1e-6, float(self._global_rule_weights.get(rid, 1e-6)) * local)
            )

    def _relax_rule_weights_to_target_prior(self, decay: float = 0.08) -> None:
        """
        Prevent long-run rule-weight drift by pulling weights toward target-aware prior.
        """
        if not getattr(self, "_target_aware_rule_weights", None):
            return
        alpha = float(np.clip(float(decay), 0.0, 1.0))
        if alpha <= 0.0:
            return
        for rid in self._rule_ids:
            base = float(self._target_aware_rule_weights.get(rid, self._global_rule_weights.get(rid, 1.0)))
            cur = float(self._global_rule_weights.get(rid, base))
            self._global_rule_weights[rid] = float(max(1e-6, ((1.0 - alpha) * cur) + (alpha * base)))

    def _adapt_global_rule_prior_from_population(self, population: Sequence[Individual]) -> None:
        """
        Generation-level pressure tuning from descriptor realism errors.
        """
        rows = [ind.descriptor_metrics for ind in population if (ind.descriptor_metrics or {})]
        if not rows:
            return

        def _mean_metric(key: str) -> float:
            vals = [float(r.get(key, 0.0)) for r in rows]
            return float(np.mean(vals)) if vals else 0.0

        mean_cycle = _mean_metric("cycle_density")
        mean_shortcut = _mean_metric("shortcut_density")
        mean_gate = _mean_metric("gate_depth_ratio")
        mean_path = _mean_metric("path_depth_ratio")
        mean_linearity = _mean_metric("linearity")
        mean_directionality = _mean_metric("directionality_gap")
        mean_gating_density = _mean_metric("gating_density")
        mean_edges = _mean_metric("edge_count")
        mean_nodes = _mean_metric("node_count")
        mean_leniency = _mean_metric("leniency")
        mean_pedagogical_variety = _mean_metric("pedagogical_puzzle_variety")
        mean_skill_chain = _mean_metric("skill_chain_score")
        mean_tutorial_climax_depth = _mean_metric("tutorial_climax_depth_score")

        cycle_target = float(max(1e-6, self.evaluator.target_cycle_density))
        shortcut_target = float(max(1e-6, self.evaluator.target_shortcut_density))
        gate_target = float(max(1e-6, self.evaluator.target_gate_depth_ratio))
        path_target = float(max(1e-6, self.evaluator.target_path_depth_ratio))
        linearity_target = float(max(1e-6, self.evaluator.target_linearity))
        directionality_target = float(max(0.0, self.evaluator.target_directionality_gap))
        gating_density_target = float(max(1e-6, self.evaluator.target_gating_density))
        leniency_target = float(max(1e-6, self.evaluator.target_leniency))
        pedagogical_variety_target = float(max(1e-6, self.evaluator.target_pedagogical_puzzle_variety))
        skill_chain_target = float(max(1e-6, self.evaluator.target_skill_chain_score))
        tutorial_climax_depth_target = float(max(1e-6, self.evaluator.target_tutorial_climax_depth_score))
        edge_target = float(max(1e-6, self.evaluator.target_num_edges)) if float(self.evaluator.target_num_edges) > 0.0 else 0.0
        node_target = float(max(1e-6, self.evaluator.target_num_nodes)) if float(self.evaluator.target_num_nodes) > 0.0 else 0.0

        cycle_error = float(np.clip((cycle_target - mean_cycle) / cycle_target, -2.0, 2.0))
        # Shortcut target can be tiny; use a softer denominator for stability.
        shortcut_error = float(
            np.clip(
                (shortcut_target - mean_shortcut) / max(0.05, shortcut_target),
                -2.0,
                2.0,
            )
        )
        gate_error = float(np.clip((gate_target - mean_gate) / gate_target, -2.0, 2.0))
        path_error = float(np.clip((path_target - mean_path) / path_target, -2.0, 2.0))
        linearity_error = float(
            np.clip(
                (linearity_target - mean_linearity) / max(0.08, linearity_target),
                -2.0,
                2.0,
            )
        )
        directionality_error = float(
            np.clip(
                (directionality_target - mean_directionality) / max(0.05, directionality_target + 0.05),
                -2.0,
                2.0,
            )
        )
        gating_density_error = float(
            np.clip(
                (gating_density_target - mean_gating_density) / max(0.05, gating_density_target),
                -2.0,
                2.0,
            )
        )
        leniency_error = float(
            np.clip(
                (leniency_target - mean_leniency) / max(0.08, leniency_target),
                -2.0,
                2.0,
            )
        )
        pedagogical_variety_error = float(
            np.clip(
                (pedagogical_variety_target - mean_pedagogical_variety) / max(0.10, pedagogical_variety_target),
                -2.0,
                2.0,
            )
        )
        skill_chain_error = float(
            np.clip(
                (skill_chain_target - mean_skill_chain) / max(0.08, skill_chain_target),
                -2.0,
                2.0,
            )
        )
        tutorial_climax_depth_error = float(
            np.clip(
                (tutorial_climax_depth_target - mean_tutorial_climax_depth) / max(0.08, tutorial_climax_depth_target),
                -2.0,
                2.0,
            )
        )
        edge_error = 0.0
        if edge_target > 0.0:
            edge_error = float(
                np.clip(
                    (edge_target - mean_edges) / max(1.0, edge_target),
                    -2.0,
                    2.0,
                )
            )
        node_error = 0.0
        if node_target > 0.0:
            node_error = float(
                np.clip(
                    (node_target - mean_nodes) / max(1.0, node_target),
                    -2.0,
                    2.0,
                )
            )
        density_error = 0.0
        if edge_target > 0.0 and node_target > 0.0 and mean_nodes > 0.0:
            target_density = float(edge_target / max(1.0, node_target))
            mean_density = float(mean_edges / max(1.0, mean_nodes))
            density_error = float(
                np.clip(
                    (target_density - mean_density) / max(0.08, target_density),
                    -2.0,
                    2.0,
                )
            )

        self._relax_rule_weights_to_target_prior(decay=0.08)

        # Increase loop closures when cycle density is under target.
        self._scale_rule_weight_group(
            self._loop_closure_rule_ids,
            1.0 + (0.28 * cycle_error) - (0.18 * max(0.0, -shortcut_error)),
            min_factor=0.65,
            max_factor=1.45,
        )

        # Explicit shortcuts get stronger negative pressure when overshooting.
        if shortcut_error >= 0.0:
            self._scale_rule_weight_group(
                self._explicit_shortcut_rule_ids,
                1.0 + (0.18 * shortcut_error),
                min_factor=0.70,
                max_factor=1.35,
            )
        else:
            self._scale_rule_weight_group(
                self._explicit_shortcut_rule_ids,
                1.0 + (0.85 * shortcut_error),
                min_factor=0.12,
                max_factor=1.0,
            )

        # Gate-depth pressure: both deficit and overshoot correction.
        self._scale_rule_weight_group(
            self._gate_heavy_rule_ids,
            1.0 + (0.40 * gate_error),
            min_factor=0.45,
            max_factor=1.45,
        )
        self._scale_rule_weight_group(
            self._side_gate_rule_ids,
            1.0 + (0.36 * gating_density_error),
            min_factor=0.55,
            max_factor=1.50,
        )

        # Directionality pressure: damp one-way-heavy operators when directed
        # path mismatch exceeds target.
        self._scale_rule_weight_group(
            self._directionality_heavy_rule_ids,
            1.0 + (0.55 * directionality_error),
            min_factor=0.30,
            max_factor=1.20,
        )
        directionality_overshoot = max(0.0, -directionality_error)
        if directionality_overshoot > 0.0:
            self._scale_rule_weight_group(
                self._gate_non_directional_rule_ids,
                1.0 + (0.20 * directionality_overshoot),
                min_factor=1.0,
                max_factor=1.30,
            )

        # If edge budget is too low, increase topology/loop operators that add
        # structure without forcing one-way progression edges.
        if edge_error > 0.0:
            self._scale_rule_weight_group(
                self._topology_pressure_rule_ids,
                1.0 + (0.18 * edge_error),
                min_factor=1.0,
                max_factor=1.28,
            )

        # Node budget pressure:
        # if room-count target is under-shot, emphasize expansion operators.
        adapt_node_gain = float(np.clip(self._rt("adapt_node_gain", 0.34), 0.0, 1.0))
        self._scale_rule_weight_group(
            self._node_expansion_rule_ids,
            1.0 + (adapt_node_gain * node_error),
            min_factor=0.72,
            max_factor=1.45,
        )

        # Edge-density pressure:
        # increase edge-forming operators when graph is too sparse for target.
        adapt_edge_density_gain = float(np.clip(self._rt("adapt_edge_density_gain", 0.30), 0.0, 1.0))
        adapt_edge_budget_gain = float(np.clip(self._rt("adapt_edge_budget_gain", 0.20), 0.0, 1.0))
        self._scale_rule_weight_group(
            self._edge_expansion_rule_ids,
            1.0 + (adapt_edge_density_gain * density_error) + (adapt_edge_budget_gain * max(0.0, edge_error)),
            min_factor=0.70,
            max_factor=1.35,
        )

        # Leniency pressure:
        # - if too lenient (too many keys per lock), boost non-key gates
        # - and damp key-inflating operators.
        leniency_overshoot = max(0.0, -leniency_error)
        if leniency_overshoot > 0.0:
            self._scale_rule_weight_group(
                self._non_key_gate_rule_ids,
                1.0
                + (
                    float(
                        np.clip(
                            self._rt("adapt_leniency_gate_boost_gain", 0.40),
                            0.0,
                            1.0,
                        )
                    )
                    * leniency_overshoot
                ),
                min_factor=1.0,
                max_factor=1.55,
            )
            self._scale_rule_weight_group(
                self._key_inflating_rule_ids,
                1.0
                - (
                    float(
                        np.clip(
                            self._rt("adapt_leniency_key_damp_gain", 0.26),
                            0.0,
                            0.80,
                        )
                    )
                    * leniency_overshoot
                ),
                min_factor=0.42,
                max_factor=1.0,
            )

        # Depth-support rules for critical-path depth realism.
        self._scale_rule_weight_group(
            self._path_depth_rule_ids,
            1.0 + (0.22 * path_error),
            min_factor=0.70,
            max_factor=1.35,
        )

        if linearity_error != 0.0:
            self._scale_rule_weight_group(
                self._linear_progression_rule_ids,
                1.0
                + (
                    float(
                        np.clip(
                            self._rt("adapt_linearity_boost_gain", 0.40),
                            0.0,
                            1.0,
                        )
                    )
                    * linearity_error
                ),
                min_factor=0.62,
                max_factor=1.55,
            )
            self._scale_rule_weight_group(
                self._wide_branch_rule_ids,
                1.0
                - (
                    float(
                        np.clip(
                            self._rt("adapt_branch_damp_gain", 0.26),
                            0.0,
                            0.80,
                        )
                    )
                    * max(0.0, linearity_error)
                ),
                min_factor=0.60,
                max_factor=1.0,
            )
            self._scale_rule_weight_group(
                getattr(self, "_branch_pruning_rule_ids", []),
                1.0
                + (
                    float(
                        np.clip(
                            self._rt("adapt_prune_boost_gain", 0.34),
                            0.0,
                            1.0,
                        )
                    )
                    * max(0.0, linearity_error)
                ),
                min_factor=1.0,
                max_factor=1.45,
            )

        pedagogical_error = float(
            np.clip(
                (0.42 * pedagogical_variety_error)
                + (0.33 * skill_chain_error)
                + (0.25 * tutorial_climax_depth_error),
                -2.0,
                2.0,
            )
        )
        adapt_pedagogical_gain = float(np.clip(self._rt("adapt_pedagogical_gain", 0.52), 0.0, 1.0))
        adapt_tutorial_climax_gain = float(np.clip(self._rt("adapt_tutorial_climax_gain", 0.40), 0.0, 1.0))
        self._scale_rule_weight_group(
            self._pedagogical_rule_ids,
            1.0 + (adapt_pedagogical_gain * pedagogical_error),
            min_factor=0.50,
            max_factor=1.55,
        )
        self._scale_rule_weight_group(
            self._pedagogical_support_rule_ids,
            1.0 + (0.24 * pedagogical_error),
            min_factor=0.72,
            max_factor=1.28,
        )
        self._scale_rule_weight_group(
            self._pedagogical_depth_support_rule_ids,
            1.0 + (adapt_tutorial_climax_gain * tutorial_climax_depth_error),
            min_factor=0.72,
            max_factor=1.32,
        )

        # If gate-depth overshoots, bias toward relief/loop rules.
        gate_overshoot = max(0.0, -gate_error)
        if gate_overshoot > 0.0:
            self._scale_rule_weight_group(
                self._gate_relief_rule_ids,
                1.0 + (0.30 * gate_overshoot),
                min_factor=1.0,
                max_factor=1.40,
            )

        self._renormalize_global_rule_probs()

    def _sample_weighted_genome(self) -> List[int]:
        """Sample one genome from global rule-weight priors."""
        rule_ids = self._rule_ids
        if len(self.executor.rules) == 5 and self.rule_space != "full":
            sampling_weights = [0.4, 0.2, 0.25, 0.15]
        else:
            sampling_weights = [max(0.01, float(self._global_rule_weights.get(rid, 0.01))) for rid in rule_ids]
        return [
            int(self.rng.choices(rule_ids, weights=sampling_weights, k=1)[0])
            for _ in range(self.genome_length)
        ]

    def _mutate_with_rate(self, genome: List[int], mutation_rate: float) -> List[int]:
        """
        Mutate genome with an explicit mutation rate (used by emitters).
        """
        mutated = genome.copy()
        local_rate = float(np.clip(float(mutation_rate), 0.0, 1.0))
        for i in range(len(mutated)):
            if self.rng.random() >= local_rate:
                continue
            current_rule_id = mutated[i]
            current_rule_name = self.executor.rule_names[
                max(0, min(current_rule_id, len(self.executor.rule_names) - 1))
            ]

            transitions = self.transition_matrix.get(current_rule_name, {})
            candidate_ids = self._rule_ids
            probs: List[float] = []
            for rid in candidate_ids:
                rule_name = self.executor.rule_names[rid]
                base_prob = float(self._global_rule_probs.get(rid, 0.0))
                trans_prob = float(max(0.0, transitions.get(rule_name, 0.0)))
                if transitions:
                    mixed_prob = (self.transition_mix * trans_prob) + ((1.0 - self.transition_mix) * base_prob)
                else:
                    mixed_prob = base_prob
                probs.append(float(max(0.0, mixed_prob)))

            total = float(sum(probs))
            if total <= 0.0:
                probs = [float(self._global_rule_probs.get(rid, 0.0)) for rid in candidate_ids]
                total = float(sum(probs))
            if total <= 0.0:
                mutated[i] = self.rng.randint(self.min_rule_id, self.max_rule_id)
                continue
            probs = [p / total for p in probs]
            mutated[i] = int(self.rng.choices(candidate_ids, weights=probs, k=1)[0])
        return mutated

    def _select_rule_ids_by_keywords(self, keywords: Sequence[str]) -> List[int]:
        """
        Build reusable rule pools for objective-pressure mutation.
        """
        keyset = [str(k).strip().lower() for k in keywords if str(k).strip()]
        out: List[int] = []
        for rid in self._rule_ids:
            name = str(self.executor.rule_names[rid]).strip().lower().replace("_", "")
            if any(k in name for k in keyset):
                out.append(int(rid))
        return sorted(set(out))

    def _estimate_structural_deficit(self, parents: Sequence[Individual]) -> Tuple[float, float, float, float, float, float]:
        """
        Estimate topology/gate deficits and shortcut over-saturation.
        """
        if not parents:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        cycle_vals = [float((p.descriptor_metrics or {}).get("cycle_density", 0.0)) for p in parents]
        shortcut_vals = [float((p.descriptor_metrics or {}).get("shortcut_density", 0.0)) for p in parents]
        gate_vals = [float((p.descriptor_metrics or {}).get("gate_depth_ratio", 0.0)) for p in parents]
        path_vals = [float((p.descriptor_metrics or {}).get("path_depth_ratio", 0.0)) for p in parents]
        gating_vals = [float((p.descriptor_metrics or {}).get("gating_density", 0.0)) for p in parents]
        edge_vals = [float((p.descriptor_metrics or {}).get("edge_count", 0.0)) for p in parents]
        directionality_vals = [float((p.descriptor_metrics or {}).get("directionality_gap", 0.0)) for p in parents]
        pedagogical_variety_vals = [
            float((p.descriptor_metrics or {}).get("pedagogical_puzzle_variety", 0.0)) for p in parents
        ]
        skill_chain_vals = [
            float((p.descriptor_metrics or {}).get("skill_chain_score", 0.0)) for p in parents
        ]
        tutorial_climax_depth_vals = [
            float((p.descriptor_metrics or {}).get("tutorial_climax_depth_score", 0.0)) for p in parents
        ]
        # Use low-quantile descriptors for deficit detection so one strong parent
        # does not hide a structural weakness.
        mean_cycle = float(np.quantile(cycle_vals, 0.25)) if cycle_vals else 0.0
        mean_shortcut = float(np.quantile(shortcut_vals, 0.25)) if shortcut_vals else 0.0
        mean_gate = float(np.quantile(gate_vals, 0.25)) if gate_vals else 0.0
        mean_path = float(np.quantile(path_vals, 0.25)) if path_vals else 0.0
        mean_gating = float(np.quantile(gating_vals, 0.25)) if gating_vals else 0.0
        mean_edges = float(np.quantile(edge_vals, 0.25)) if edge_vals else 0.0
        mean_pedagogical_variety = float(np.quantile(pedagogical_variety_vals, 0.25)) if pedagogical_variety_vals else 0.0
        mean_skill_chain = float(np.quantile(skill_chain_vals, 0.25)) if skill_chain_vals else 0.0
        mean_tutorial_climax_depth = (
            float(np.quantile(tutorial_climax_depth_vals, 0.25)) if tutorial_climax_depth_vals else 0.0
        )
        high_shortcut = float(np.quantile(shortcut_vals, 0.75)) if shortcut_vals else mean_shortcut
        high_gate = float(np.quantile(gate_vals, 0.75)) if gate_vals else mean_gate
        high_directionality = float(np.quantile(directionality_vals, 0.75)) if directionality_vals else 0.0

        cycle_def = max(0.0, self.evaluator.target_cycle_density - mean_cycle) / max(1e-6, self.evaluator.target_cycle_density)
        shortcut_def = max(0.0, self.evaluator.target_shortcut_density - mean_shortcut) / max(1e-6, self.evaluator.target_shortcut_density)
        shortcut_excess = max(0.0, high_shortcut - self.evaluator.target_shortcut_density) / max(
            0.03,
            self.evaluator.target_shortcut_density,
        )
        gate_def = max(0.0, self.evaluator.target_gate_depth_ratio - mean_gate) / max(1e-6, self.evaluator.target_gate_depth_ratio)
        gate_excess = max(0.0, high_gate - self.evaluator.target_gate_depth_ratio) / max(
            0.06,
            self.evaluator.target_gate_depth_ratio,
        )
        path_def = max(0.0, self.evaluator.target_path_depth_ratio - mean_path) / max(1e-6, self.evaluator.target_path_depth_ratio)
        gating_def = max(0.0, self.evaluator.target_gating_density - mean_gating) / max(
            1e-6,
            self.evaluator.target_gating_density,
        )
        pedagogical_variety_def = max(
            0.0,
            self.evaluator.target_pedagogical_puzzle_variety - mean_pedagogical_variety,
        ) / max(1e-6, self.evaluator.target_pedagogical_puzzle_variety)
        skill_chain_def = max(
            0.0,
            self.evaluator.target_skill_chain_score - mean_skill_chain,
        ) / max(1e-6, self.evaluator.target_skill_chain_score)
        tutorial_climax_depth_def = max(
            0.0,
            self.evaluator.target_tutorial_climax_depth_score - mean_tutorial_climax_depth,
        ) / max(1e-6, self.evaluator.target_tutorial_climax_depth_score)
        edge_def = 0.0
        if float(self.evaluator.target_num_edges) > 0.0:
            edge_def = max(0.0, float(self.evaluator.target_num_edges) - mean_edges) / max(
                1.0,
                float(self.evaluator.target_num_edges),
            )
        directionality_excess = max(
            0.0,
            high_directionality - float(self.evaluator.target_directionality_gap),
        ) / max(0.05, float(self.evaluator.target_directionality_gap) + 0.05)

        shortcut_pressure_weight = float(np.clip(self.evaluator.target_shortcut_density / 0.08, 0.08, 0.30))
        topology_deficit = float(
            np.clip((0.78 * cycle_def) + (shortcut_pressure_weight * shortcut_def) - (0.60 * shortcut_excess), 0.0, 1.5)
        )
        gate_deficit = float(
            np.clip(
                (0.38 * gate_def)
                + (0.28 * path_def)
                + (0.22 * gating_def)
                + (0.12 * edge_def)
                - (0.30 * gate_excess),
                0.0,
                1.5,
            )
        )
        pedagogical_deficit = float(
            np.clip(
                (0.40 * pedagogical_variety_def)
                + (0.35 * skill_chain_def)
                + (0.25 * tutorial_climax_depth_def),
                0.0,
                1.5,
            )
        )
        return (
            topology_deficit,
            gate_deficit,
            float(np.clip(shortcut_excess, 0.0, 2.0)),
            float(np.clip(gate_excess, 0.0, 2.0)),
            float(np.clip(directionality_excess, 0.0, 2.0)),
            pedagogical_deficit,
        )

    def _inject_rule_pressure(
        self,
        genome: List[int],
        *,
        topology_deficit: float,
        gate_deficit: float,
        shortcut_excess: float = 0.0,
        gate_excess: float = 0.0,
        directionality_excess: float = 0.0,
        pedagogical_deficit: float = 0.0,
    ) -> List[int]:
        """
        Apply targeted gene replacements toward missing topology mechanics.
        """
        pressured = list(genome)
        if not pressured:
            return pressured

        topo_multiplier = float(np.clip(1.0 - (0.55 * float(shortcut_excess)), 0.15, 1.0))
        topo_slots = int(
            np.clip(round(float(topology_deficit) * 2.0 * topo_multiplier), 0, max(0, len(pressured) // 2))
        )
        gate_slots = int(np.clip(round(float(gate_deficit) * 2.0), 0, max(0, len(pressured) // 2)))

        if self._topology_pressure_rule_ids and topo_slots > 0:
            replace_idx = self.rng.sample(range(len(pressured)), k=min(len(pressured), topo_slots))
            for idx in replace_idx:
                pressured[idx] = int(self.rng.choice(self._topology_pressure_rule_ids))

        if self._gate_pressure_rule_ids and gate_slots > 0:
            replace_idx = self.rng.sample(range(len(pressured)), k=min(len(pressured), gate_slots))
            gate_candidate_pool: List[int] = []
            if getattr(self, "_side_gate_rule_ids", None):
                gate_candidate_pool.extend(list(self._side_gate_rule_ids))
            if getattr(self, "_gate_non_directional_rule_ids", None):
                gate_candidate_pool.extend(list(self._gate_non_directional_rule_ids))
            if not gate_candidate_pool:
                gate_candidate_pool = list(self._gate_pressure_rule_ids)
            for idx in replace_idx:
                pressured[idx] = int(self.rng.choice(gate_candidate_pool))

        if self._pedagogical_rule_ids and float(pedagogical_deficit) > 0.0:
            ped_slots = int(
                np.clip(
                    round(float(pedagogical_deficit) * 1.5),
                    0,
                    max(0, len(pressured) // 3),
                )
            )
            if ped_slots > 0:
                replace_idx = self.rng.sample(range(len(pressured)), k=min(len(pressured), ped_slots))
                candidate_pool = (
                    list(self._pedagogical_rule_ids)
                    + list(self._pedagogical_rule_ids)
                    + list(self._pedagogical_rule_ids)
                    + list(self._pedagogical_support_rule_ids)
                    + list(self._pedagogical_depth_support_rule_ids)
                    + list(self._linear_progression_rule_ids)
                )
                if not candidate_pool:
                    candidate_pool = list(self._pedagogical_rule_ids)
                for idx in replace_idx:
                    pressured[idx] = int(self.rng.choice(candidate_pool))

        # If shortcut density is already above target, rewrite explicit
        # shortcut genes toward non-shortcut topology/gating operators.
        if self._explicit_shortcut_rule_ids and float(shortcut_excess) > 0.0:
            remap_slots = int(
                np.clip(
                    round(float(shortcut_excess) * 2.0),
                    0,
                    max(0, len(pressured) // 2),
                )
            )
            if remap_slots > 0:
                shortcut_positions = [
                    idx for idx, rid in enumerate(pressured)
                    if int(rid) in set(self._explicit_shortcut_rule_ids)
                ]
                replaceable = min(remap_slots, len(shortcut_positions))
                if replaceable > 0:
                    candidate_pool = [
                        rid
                        for rid in (self._topology_pressure_rule_ids + self._gate_pressure_rule_ids)
                        if rid not in set(self._explicit_shortcut_rule_ids)
                    ]
                    if candidate_pool:
                        for idx in self.rng.sample(shortcut_positions, k=replaceable):
                            pressured[idx] = int(self.rng.choice(candidate_pool))

        # If gate depth is above target, remap some gate-heavy genes to
        # non-gate topology/depth operators.
        if self._gate_heavy_rule_ids and float(gate_excess) > 0.0:
            remap_slots = int(
                np.clip(
                    round(float(gate_excess) * 1.6),
                    0,
                    max(0, len(pressured) // 2),
                )
            )
            if remap_slots > 0:
                gate_positions = [
                    idx for idx, rid in enumerate(pressured)
                    if int(rid) in set(self._gate_heavy_rule_ids)
                ]
                replaceable = min(remap_slots, len(gate_positions))
                if replaceable > 0:
                    candidate_pool = [
                        rid
                        for rid in (
                            self._loop_closure_rule_ids
                            + self._path_depth_rule_ids
                            + self._gate_relief_rule_ids
                            + self._topology_pressure_rule_ids
                        )
                        if rid not in set(self._gate_heavy_rule_ids)
                    ]
                    if candidate_pool:
                        for idx in self.rng.sample(gate_positions, k=replaceable):
                            pressured[idx] = int(self.rng.choice(candidate_pool))

        # If directionality mismatch is high, rewrite one-way-heavy genes toward
        # non-directional gate/topology operators.
        if self._directionality_heavy_rule_ids and float(directionality_excess) > 0.0:
            remap_slots = int(
                np.clip(
                    round(float(directionality_excess) * 2.2),
                    0,
                    max(0, len(pressured) // 2),
                )
            )
            if remap_slots > 0:
                directionality_positions = [
                    idx for idx, rid in enumerate(pressured)
                    if int(rid) in set(self._directionality_heavy_rule_ids)
                ]
                replaceable = min(remap_slots, len(directionality_positions))
                if replaceable > 0:
                    candidate_pool = [
                        rid
                        for rid in (
                            self._gate_non_directional_rule_ids
                            + self._gate_relief_rule_ids
                            + self._loop_closure_rule_ids
                            + self._topology_pressure_rule_ids
                        )
                        if rid not in set(self._directionality_heavy_rule_ids)
                    ]
                    if candidate_pool:
                        for idx in self.rng.sample(directionality_positions, k=replaceable):
                            pressured[idx] = int(self.rng.choice(candidate_pool))

        return pressured

    def _emit_genome_from_archive(self, archive: Any) -> List[int]:
        """Emitter-style genome proposal from CVT archive."""
        if not getattr(archive, "archive", {}):
            return self._sample_weighted_genome()

        emitter_roll = self.rng.random()
        elite = archive.get_random_elite()
        if elite is None:
            return self._sample_weighted_genome()

        # 1) Local emitter: mutate around one elite.
        if emitter_roll < 0.55:
            parent = list(int(g) for g in elite.solution)
            return self._mutate_with_rate(parent, mutation_rate=self.qd_emitter_mutation_rate)

        elites = archive.get_all_elites()
        # 2) Directional emitter: crossover two elites then mutate lightly.
        if emitter_roll < 0.85 and len(elites) >= 2:
            parent_a = list(int(g) for g in self.rng.choice(elites).solution)
            parent_b = list(int(g) for g in self.rng.choice(elites).solution)
            child, _ = self._crossover(parent_a, parent_b)
            return self._mutate_with_rate(child, mutation_rate=0.70 * self.qd_emitter_mutation_rate)

        # 3) Exploration emitter: restart from global prior.
        return self._sample_weighted_genome()

    def _new_qd_archive(self) -> Any:
        """Create a CVT archive using the generator's configured descriptor space."""
        if CVTEliteArchive is None:
            raise RuntimeError("CVTEliteArchive is unavailable")
        return CVTEliteArchive(
            num_cells=int(self.qd_archive_cells),
            feature_dims=4,
            feature_ranges=[(0.0, 1.0)] * 4,
            num_cvt_samples=max(1024, int(self.qd_archive_cells) * 24),
            seed=(None if self.seed is None else int(self.seed) + 17),
        )

    def _load_qd_archive_or_new(self) -> Any:
        """Load a persisted CVT archive when requested, otherwise create a fresh one."""
        archive = self._new_qd_archive()
        if not self.qd_load_archive:
            return archive
        if self.qd_archive_path is None:
            raise ValueError("qd_load_archive=True requires qd_archive_path.")
        if not self.qd_archive_path.exists():
            logger.info("QD archive path does not exist yet: %s", self.qd_archive_path)
            return archive

        with self.qd_archive_path.open("rb") as f:
            payload = pickle.load(f)
        loaded_archive = payload.get("archive") if isinstance(payload, dict) else payload
        if loaded_archive is None:
            raise ValueError(f"Invalid QD archive payload in {self.qd_archive_path}")
        if int(getattr(loaded_archive, "num_cells", self.qd_archive_cells)) != int(self.qd_archive_cells):
            raise ValueError(
                f"QD archive cell mismatch: file has {getattr(loaded_archive, 'num_cells', 'unknown')}, "
                f"generator uses {self.qd_archive_cells}."
            )
        if int(getattr(loaded_archive, "feature_dims", 4)) != 4:
            raise ValueError("QD archive feature dimension mismatch; expected 4D topology descriptors.")
        logger.info(
            "Loaded QD archive from %s (%d elites)",
            self.qd_archive_path,
            len(getattr(loaded_archive, "archive", {}) or {}),
        )
        return loaded_archive

    def _save_qd_archive(self, archive: Any) -> None:
        """Best-effort persistence for CVT archives used by topology QD search."""
        if self.qd_archive_path is None:
            return
        try:
            self.qd_archive_path.parent.mkdir(parents=True, exist_ok=True)
            stats = archive.get_stats()
            payload = {
                "version": 1,
                "archive": archive,
                "stats": {
                    "coverage": float(stats.coverage),
                    "qd_score": float(stats.total_fitness),
                    "mean_fitness": float(stats.mean_fitness),
                    "num_elites": int(stats.num_elites),
                    "feature_diversity": float(stats.feature_diversity),
                },
                "config": {
                    "qd_archive_cells": int(self.qd_archive_cells),
                    "feature_dims": 4,
                    "seed": self.seed,
                },
            }
            with self.qd_archive_path.open("wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (AttributeError, OSError, pickle.PickleError, TypeError, ValueError) as exc:
            logger.warning("Failed to persist QD archive to %s: %s", self.qd_archive_path, exc)

    def _evolve_cvt_emitter(self, *, directed_output: bool = False) -> nx.Graph:
        """
        Runtime QD strategy using a CVT archive + simple emitters.
        """
        if CVTEliteArchive is None:
            logger.warning("CVT archive unavailable; falling back to GA strategy")
            self.search_strategy = "ga"
            return self.evolve(directed_output=directed_output)

        logger.info("Starting CVT-emitter search...")
        total_evaluations = max(1, int(self.population_size) * max(1, int(self.generations)))
        init_random = max(8, int(round(self.qd_init_random_fraction * float(total_evaluations))))
        archive = self._load_qd_archive_or_new()

        best: Optional[Individual] = None
        batch: List[Individual] = []
        generation_counter = 0

        for eval_idx in range(total_evaluations):
            if eval_idx < init_random:
                genome = self._sample_weighted_genome()
            else:
                genome = self._emit_genome_from_archive(archive)

            ind = Individual(genome=list(int(g) for g in genome))
            ind = self._evaluate_individual(ind, generation=generation_counter)
            batch.append(ind)

            dm = ind.descriptor_metrics or {}
            features = (
                float(np.clip(dm.get("linearity", 0.0), 0.0, 1.0)),
                float(np.clip(dm.get("leniency", 0.0), 0.0, 1.0)),
                float(np.clip(dm.get("progression_complexity", 0.0), 0.0, 1.0)),
                float(np.clip(dm.get("topology_complexity", 0.0), 0.0, 1.0)),
            )
            archive.add(
                solution=list(int(g) for g in ind.genome),
                fitness=float(ind.fitness),
                features=features,
                metadata={
                    "feasible": bool(ind.feasible),
                    "constraint_violation": float(ind.constraint_violation),
                    "descriptor_metrics": dict(dm),
                },
            )

            if best is None or self._individual_sort_key(ind) < self._individual_sort_key(best):
                best = ind

            if ((eval_idx + 1) % max(1, int(self.population_size)) == 0) or (eval_idx == total_evaluations - 1):
                generation_counter += 1
                archive_stats = archive.get_stats()
                qd_stats = {
                    "coverage": float(archive_stats.coverage),
                    "qd_score": float(archive_stats.total_fitness),
                    "mean_fitness": float(archive_stats.mean_fitness),
                    "max_fitness": float(archive_stats.max_fitness),
                    "min_fitness": float(archive_stats.min_fitness),
                    "num_elites": float(archive_stats.num_elites),
                    "feature_diversity": float(archive_stats.feature_diversity),
                }
                self.qd_final_archive_stats = qd_stats
                self.qd_coverage_history.append(qd_stats["coverage"])
                self.qd_qd_score_history.append(qd_stats["qd_score"])
                self.qd_mean_fitness_history.append(qd_stats["mean_fitness"])
                self.qd_num_elites_history.append(qd_stats["num_elites"])
                if batch:
                    self._adapt_global_rule_prior_from_population(batch)
                    self.best_fitness_history.append(float(max(x.fitness for x in batch)))
                    self.avg_fitness_history.append(float(np.mean([x.fitness for x in batch])))
                    self.feasible_ratio_history.append(float(np.mean([1.0 if x.feasible else 0.0 for x in batch])))
                    self.avg_violation_history.append(
                        float(np.mean([float(x.constraint_violation if np.isfinite(x.constraint_violation) else 1.0) for x in batch]))
                    )
                else:
                    self.best_fitness_history.append(0.0)
                    self.avg_fitness_history.append(0.0)
                    self.feasible_ratio_history.append(0.0)
                    self.avg_violation_history.append(1.0)
                self.diversity_history.append(float(archive_stats.feature_diversity))
                batch = []
                if self.qd_autosave_archive:
                    self._save_qd_archive(archive)

        if best is None or best.phenotype is None:
            raise RuntimeError("CVT-emitter search produced no valid individual")

        self._save_qd_archive(archive)

        logger.info(
            "CVT-emitter complete. Best fitness: %.4f, Graph: %d nodes, %d edges",
            float(best.fitness),
            int(len(best.phenotype.nodes)),
            int(len(best.phenotype.edges)),
        )
        return self._finalize_graph_output(best.phenotype, directed_output=directed_output)
    
    def _tournament_selection(
        self,
        population: List[Individual],
        k: int = 3
    ) -> Individual:
        """
        Select best individual from k random candidates.
        
        Args:
            population: Population to select from
            k: Tournament size
            
        Returns:
            Selected Individual
        """
        k_eff = max(1, min(int(k), len(population)))
        tournament = self.rng.sample(population, k_eff)
        # Deb-style comparator:
        # 1) feasible beats infeasible
        # 2) among feasible: higher fitness wins
        # 3) among infeasible: lower violation wins (then fitness tie-break)
        winner = min(
            tournament,
            key=self._individual_sort_key,
        )
        return winner
    
    def _crossover(
        self,
        parent1: List[int],
        parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        One-point crossover: splice two rule lists.
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            Tuple of (child1, child2) genomes
        """
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy(), parent2.copy()
        
        # Select crossover point
        point = self.rng.randint(1, min(len(parent1), len(parent2)) - 1)
        
        # Create children
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, genome: List[int]) -> List[int]:
        """
        Mutate genome with weighted probabilities.
        
        If transition priors exist for the current rule, mix them with global
        full-rule priors (derived from rule weights) so mutation does not
        collapse to a tiny subset of legacy rules.
        
        Args:
            genome: Genome to mutate
            
        Returns:
            Mutated genome
        """
        return self._mutate_with_rate(genome, mutation_rate=self.mutation_rate)
    
    def _generate_offspring(self, population: List[Individual]) -> List[Individual]:
        """
        Generate offspring through selection, crossover, and mutation.
        
        Args:
            population: Current population
            
        Returns:
            List of offspring individuals
        """
        offspring = []
        
        # Generate population_size offspring
        while len(offspring) < self.population_size:
            # Select parents
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if self.rng.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(
                    parent1.genome,
                    parent2.genome
                )
            else:
                # Clone
                child1_genome = parent1.genome.copy()
                child2_genome = parent2.genome.copy()

            (
                topology_deficit,
                gate_deficit,
                shortcut_excess,
                gate_excess,
                directionality_excess,
                pedagogical_deficit,
            ) = self._estimate_structural_deficit(
                [parent1, parent2]
            )
            adaptive_mutation_rate = float(
                np.clip(
                    self.mutation_rate
                    * (
                        1.0
                        + (0.70 * topology_deficit)
                        + (0.45 * gate_deficit)
                        + (0.24 * pedagogical_deficit)
                        - (0.40 * shortcut_excess)
                        - (0.22 * gate_excess)
                        - (0.28 * directionality_excess)
                    ),
                    0.01,
                    0.95,
                )
            )

            # Mutate with deficit-adaptive rate then inject targeted rule pressure.
            child1_genome = self._mutate_with_rate(child1_genome, mutation_rate=adaptive_mutation_rate)
            child2_genome = self._mutate_with_rate(child2_genome, mutation_rate=adaptive_mutation_rate)
            child1_genome = self._inject_rule_pressure(
                child1_genome,
                topology_deficit=topology_deficit,
                gate_deficit=gate_deficit,
                shortcut_excess=shortcut_excess,
                gate_excess=gate_excess,
                directionality_excess=directionality_excess,
                pedagogical_deficit=pedagogical_deficit,
            )
            child2_genome = self._inject_rule_pressure(
                child2_genome,
                topology_deficit=topology_deficit,
                gate_deficit=gate_deficit,
                shortcut_excess=shortcut_excess,
                gate_excess=gate_excess,
                directionality_excess=directionality_excess,
                pedagogical_deficit=pedagogical_deficit,
            )
            
            # Create offspring individuals
            offspring.append(Individual(genome=child1_genome))
            if len(offspring) < self.population_size:
                offspring.append(Individual(genome=child2_genome))
        
        return offspring
    
    def _select_survivors(
        self,
        combined: List[Individual]
    ) -> List[Individual]:
        """
        Select survivors for next generation using (mu+lambda) strategy.
        
        Keeps the best population_size individuals from combined
        parent and offspring population.
        
        Args:
            combined: Combined parents and offspring
            
        Returns:
            Selected survivors
        """
        feasible = [ind for ind in combined if ind.feasible]
        infeasible = [ind for ind in combined if not ind.feasible]

        feasible.sort(
            key=lambda ind: (
                -float(ind.fitness),
                float(ind.topology_realism_error if np.isfinite(ind.topology_realism_error) else 10.0),
            )
        )
        infeasible.sort(
            key=lambda ind: (
                float(ind.constraint_violation if np.isfinite(ind.constraint_violation) else 1.0),
                -float(ind.fitness),
                float(ind.topology_realism_error if np.isfinite(ind.topology_realism_error) else 10.0),
            )
        )

        survivors = feasible[: self.population_size]
        if len(survivors) < self.population_size:
            survivors.extend(infeasible[: self.population_size - len(survivors)])

        return survivors
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """
        Calculate population diversity using genome hamming distance.
        
        Args:
            population: Population to analyze
            
        Returns:
            Average pairwise hamming distance (normalized 0-1)
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        comparisons = 0
        
        # Sample pairs to avoid O(n^2) for large populations
        sample_size = min(100, len(population))
        sample = self.rng.sample(population, sample_size)
        
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                genome1 = sample[i].genome
                genome2 = sample[j].genome
                
                # Hamming distance
                distance = sum(
                    g1 != g2 for g1, g2 in zip(genome1, genome2)
                )
                
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 0.0
        
        # Normalize by genome length
        avg_distance = total_distance / comparisons
        normalized = avg_distance / self.genome_length
        
        return normalized
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evolution statistics for analysis.
        
        Returns:
            Dictionary with fitness history, diversity, etc.
        """
        return {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'diversity_history': self.diversity_history,
            'feasible_ratio_history': self.feasible_ratio_history,
            'avg_violation_history': self.avg_violation_history,
            'qd_coverage_history': self.qd_coverage_history,
            'qd_qd_score_history': self.qd_qd_score_history,
            'qd_mean_fitness_history': self.qd_mean_fitness_history,
            'qd_num_elites_history': self.qd_num_elites_history,
            'qd_final_archive_stats': self.qd_final_archive_stats,
            'final_repair_evaluation': dict(getattr(self, "last_final_repair_evaluation", {})),
            'final_best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0.0,
            'generations_run': len(self.best_fitness_history),
            'converged': self.best_fitness_history[-1] >= 0.95 if self.best_fitness_history else False,
        }
