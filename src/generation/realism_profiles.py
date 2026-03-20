from __future__ import annotations

from typing import Dict, List


REALISM_TUNING_PROFILES: Dict[str, Dict[str, float]] = {
    "engine_default": {},
    "gate_quality_heavy": {
        "adapt_node_gain": 0.46,
        "adapt_edge_density_gain": 0.70,
        "adapt_edge_budget_gain": 0.52,
        "prior_node_boost_gain": 0.34,
        "prior_edge_boost_gain": 0.56,
        "node_cap_floor_ratio": 0.98,
        "node_cap_expand_ratio": 1.18,
        "node_cap_hard_cap_ratio": 1.45,
    },
    "scale_heavy": {
        "adapt_node_gain": 0.50,
        "adapt_edge_density_gain": 0.78,
        "adapt_edge_budget_gain": 0.58,
        "prior_node_boost_gain": 0.40,
        "prior_edge_boost_gain": 0.64,
        "node_cap_floor_ratio": 1.00,
        "node_cap_expand_ratio": 1.22,
        "node_cap_hard_cap_ratio": 1.55,
    },
}


def get_realism_tuning_profile(name: str) -> Dict[str, float]:
    key = str(name).strip()
    if key not in REALISM_TUNING_PROFILES:
        raise KeyError(f"Unknown realism tuning profile: {name}")
    return dict(REALISM_TUNING_PROFILES[key])


def list_realism_tuning_profiles(*, include_engine_default: bool = True) -> List[str]:
    names = list(REALISM_TUNING_PROFILES.keys())
    if include_engine_default:
        return names
    return [name for name in names if name != "engine_default"]
