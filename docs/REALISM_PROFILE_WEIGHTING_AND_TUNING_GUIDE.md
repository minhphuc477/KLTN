# Realism Profile Weighting and Tuning Guide

## Purpose
This guide explains how `scripts/recommend_realism_profile.py` scores profile candidates and how to interpret realism tuning knobs in `src/generation/realism_profiles.py`.

## Weighted objective used for recommendation
The script minimizes this objective:

$$
J = w_f \cdot \text{fidelity\_js\_divergence}
+ w_o \cdot (1-\text{expressive\_overlap\_reference})
+ w_d \cdot (1-\text{descriptor\_diversity})
+ w_n \cdot \text{num\_nodes\_gap\_ratio}
+ w_e \cdot \text{num\_edges\_gap\_ratio}
+ w_c \cdot (1-\text{overall\_completeness})
+ w_v \cdot (1-\text{constraint\_valid\_rate})
$$

Where node/edge gap are relative errors against VGLC reference means.

## Default weights in script
- `fidelity=0.35`: emphasize distribution alignment with reference corpus.
- `overlap=0.20`: reward overlap in expressive-space occupancy.
- `diversity=0.20`: reward internal variation among generated samples.
- `node_gap=0.10`, `edge_gap=0.10`: keep global scale close to references.
- `completeness=6.0`, `validity=8.0`: hard penalties so invalid/incomplete outputs are strongly disfavored.

Recommended reporting preset used in current runs:
- `fidelity=0.40,overlap=0.20,diversity=0.20,node_gap=0.10,edge_gap=0.10,completeness=6.0,validity=8.0`

## Realism tuning knobs
Profiles are defined in `src/generation/realism_profiles.py`.

- `adapt_node_gain`: adaptive pressure on node-expansion operators.
- `adapt_edge_density_gain`: adaptive pressure on edge density when topology is sparse.
- `adapt_edge_budget_gain`: adaptive pressure on total edge budget.
- `prior_node_boost_gain`: prior shaping toward larger node count before adaptation.
- `prior_edge_boost_gain`: prior shaping toward larger edge count before adaptation.
- `node_cap_floor_ratio`: lower bound for effective node cap relative to target.
- `node_cap_expand_ratio`: soft expansion factor of node cap.
- `node_cap_hard_cap_ratio`: upper bound for node cap expansion.

## Current profile intent
- `engine_default`: preserves historical generator behavior (no override payload).
- `gate_quality_heavy`: balanced realism profile favoring fidelity/diversity while preserving validity.
- `scale_heavy`: stronger node/edge growth pressure; useful when room/edge budgets are under target.

## Repro commands
- Compare two profiles with default script weights:
  - `python scripts/recommend_realism_profile.py --num-generated 16 --output-dir results/profile_recommendation_16`
- Compare with explicit reporting preset:
  - `python scripts/recommend_realism_profile.py --num-generated 16 --weights "fidelity=0.40,overlap=0.20,diversity=0.20,node_gap=0.10,edge_gap=0.10,completeness=6.0,validity=8.0" --output-dir results/profile_recommendation_16_final`

## Decision rule
- Use the lowest `objective_score` as winner.
- If two profiles are close, prefer the one with:
  1) `constraint_valid_rate=1.0` and `overall_completeness=1.0`, then
  2) lower `fidelity_js_divergence`, then
  3) higher `expressive_overlap_reference`.
