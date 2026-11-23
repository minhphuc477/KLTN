"""Advanced and pedagogical mission graph production rules (facade module).

All 27 production rules are organized into focused domain modules under
`src.generation.grammar.rules` and re-exported here for complete backward
compatibility.
"""

from __future__ import annotations

from src.generation.grammar.rules import (
    _bounded_free_position,
    _layout_bounds,
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

__all__ = ['_bounded_free_position', '_layout_bounds', 'InsertSwitchRule', 'AddItemGateRule', 'AddFungibleLockRule', 'AddValveRule', 'AddHazardGateRule', 'SoftGateRule', 'AddGatekeeperRule', 'AddMultiLockRule', 'MergeRule', 'CreateHubRule', 'AddStairsRule', 'AddTeleportRule', 'FormBigRoomRule', 'AddSectorRule', 'AddEntangledBranchesRule', 'SplitRoomRule', 'AddBossGauntlet', 'AddForeshadowingRule', 'AddCollectionChallengeRule', 'AddArenaRule', 'AddSkillChainRule', 'AddPacingBreakerRule', 'AddResourceLoopRule', 'AddSecretRule', 'PruneGraphRule', 'AddItemShortcutRule', 'PruneDeadEndRule']
