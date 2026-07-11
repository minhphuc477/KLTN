"""Mission grammar package with import-compatible public surface."""

from .graph_types import EdgeType, MissionEdge, MissionGraph, MissionNode, NodeType
from .core_rules import BranchRule, Difficulty, InsertChallengeRule, InsertLockKeyRule, ProductionRule, StartRule
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
    SoftGateRule,
    SplitRoomRule,
)
from .mission_grammar import MissionGrammar, graph_to_gnn_input
from src.generation.grammar_validators import (
    validate_battery_reachability,
    validate_exact_progression,
    validate_resource_loops,
    validate_skill_chains,
)

__all__ = [
    'NodeType', 'EdgeType', 'MissionNode', 'MissionEdge', 'MissionGraph',
    'ProductionRule', 'StartRule', 'InsertChallengeRule', 'InsertLockKeyRule',
    'BranchRule', 'Difficulty', 'MissionGrammar', 'graph_to_gnn_input',
    'MergeRule', 'InsertSwitchRule', 'AddBossGauntlet', 'AddItemGateRule',
    'CreateHubRule', 'AddStairsRule', 'AddSecretRule', 'AddTeleportRule',
    'PruneGraphRule', 'AddFungibleLockRule', 'FormBigRoomRule', 'AddValveRule',
    'AddForeshadowingRule', 'AddCollectionChallengeRule', 'AddArenaRule',
    'AddSectorRule', 'AddEntangledBranchesRule', 'AddHazardGateRule',
    'SoftGateRule', 'SplitRoomRule', 'AddSkillChainRule', 'AddPacingBreakerRule',
    'AddResourceLoopRule', 'AddGatekeeperRule', 'AddMultiLockRule',
    'AddItemShortcutRule', 'PruneDeadEndRule',
    'validate_skill_chains', 'validate_battery_reachability', 'validate_exact_progression',
    'validate_resource_loops',
]
