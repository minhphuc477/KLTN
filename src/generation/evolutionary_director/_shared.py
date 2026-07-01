"""Shared imports/constants for the evolutionary topology package."""

from __future__ import annotations

import copy
import json
import logging
import math
import pickle
import random
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
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
    PruneDeadEndRule,
    AddSkillChainRule,
    MissionGrammar,
    MissionGraph,
    MissionEdge,
    MissionNode,
    NodeType,
    EdgeType,
    StartRule,
    InsertChallengeRule,
    InsertLockKeyRule,
    BranchRule,
)
from src.core.definitions import parse_edge_type_tokens, parse_node_label_tokens
from src.zelda_data.vglc_utils import filter_virtual_nodes, validate_topology
from src.evaluation.structural_metrics import compute_branching_factor, compute_cyclomatic_complexity
from src.generation.pareto_objectives import apply_pareto_metrics, compute_pareto_objectives

from src.evaluation.map_elites import CVTEliteArchive

logger = logging.getLogger(__name__)

_SAFE_RULE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
DEFAULT_REPLAY_PAYLOAD_MAX_BYTES = 256 * 1024


def _stable_graph_node_identity(node: Any) -> Tuple[str, str]:
    """Deterministic heterogeneous node-ID identity used in topology round-trips."""
    if isinstance(node, np.generic):
        node = node.item()
    if isinstance(node, float) and float(node).is_integer():
        node = int(node)
    return (type(node).__name__, str(node))


def _stable_bidirectional_pair_key(
    src: Any,
    tgt: Any,
    edge_type_name: str,
) -> Tuple[Tuple[str, str], Tuple[str, str], str]:
    """Canonical key for mirrored bidirectional edges without salted hashing."""
    ordered = sorted(
        (_stable_graph_node_identity(src), _stable_graph_node_identity(tgt)),
        key=lambda item: (item[0], item[1]),
    )
    return (ordered[0], ordered[1], str(edge_type_name))


DEFAULT_REALISM_TUNING: Dict[str, float] = {
    "node_cap_floor_ratio": 0.92,
    "node_cap_expand_ratio": 1.08,
    "node_cap_hard_cap_ratio": 1.25,
    "genome_len_floor_ratio": 0.74,
    "genome_len_expand_ratio": 0.90,
    "genome_len_hard_cap_ratio": 1.20,
    "prior_node_boost_gain": 0.28,
    "prior_node_boost_max": 1.25,
    "prior_edge_boost_gain": 0.48,
    "prior_edge_boost_max": 1.20,
    "prior_pedagogical_boost_gain": 0.48,
    "prior_pedagogical_boost_max": 1.42,
    "prior_linearity_branch_damp_gain": 0.22,
    "prior_linearity_boost_gain": 0.34,
    "prior_linearity_prune_boost_gain": 0.30,
    "prior_leniency_key_damp_gain": 0.52,
    "prior_leniency_gate_boost_gain": 0.46,
    "prior_leniency_depth_boost_gain": 0.28,
    "adapt_node_gain": 0.36,
    "adapt_edge_density_gain": 0.62,
    "adapt_edge_budget_gain": 0.44,
    "adapt_pedagogical_gain": 0.58,
    "adapt_tutorial_climax_gain": 0.48,
    "adapt_leniency_gate_boost_gain": 0.52,
    "adapt_leniency_key_damp_gain": 0.34,
    "adapt_linearity_boost_gain": 0.40,
    "adapt_branch_damp_gain": 0.26,
    "adapt_prune_boost_gain": 0.34,
    "initial_pedagogical_seed_fraction": 0.28,
    "initial_pedagogical_seed_min": 3.0,
    "initial_pedagogical_seed_max_fraction": 0.50,
    "progression_balance_repair_iterations": 3.0,
    "progression_balance_leniency_weight": 0.36,
    "progression_balance_linearity_weight": 0.30,
    "progression_balance_depth_weight": 0.16,
    "progression_balance_variety_weight": 0.10,
    "progression_balance_skill_weight": 0.08,
    "progression_balance_gate_density_weight": 0.28,
    "progression_balance_key_surplus_weight": 0.24,
    "progression_balance_big_key_surplus_weight": 0.08,
    "final_min_gate_density": 0.20,
    "final_max_key_surplus": 1.0,
    "final_max_big_key_surplus": 0.0,
    "final_gate_calibration_iterations": 4.0,
}


# ============================================================================
# ZELDA TRANSITION MATRIX (Learned from VGLC dataset)
# ============================================================================

# Default transition probabilities P(RuleB | RuleA) learned from Zelda
# Used for biased mutation that follows typical dungeon structure patterns
DEFAULT_ZELDA_TRANSITIONS = {
    "Start": {"InsertChallenge_ENEMY": 0.4, "InsertChallenge_PUZZLE": 0.2, "Branch": 0.3, "InsertLockKey": 0.1},
    "InsertChallenge_ENEMY": {"InsertChallenge_ENEMY": 0.3, "InsertChallenge_PUZZLE": 0.2, "InsertLockKey": 0.3, "Branch": 0.2},
    "InsertChallenge_PUZZLE": {"InsertChallenge_ENEMY": 0.4, "InsertLockKey": 0.4, "Branch": 0.2},
    "InsertLockKey": {"InsertChallenge_ENEMY": 0.6, "InsertChallenge_PUZZLE": 0.2, "Branch": 0.2},
    "Branch": {"InsertChallenge_ENEMY": 0.5, "InsertChallenge_PUZZLE": 0.3, "InsertLockKey": 0.2},
}


# ============================================================================
# GENOTYPE-PHENOTYPE MAPPING
