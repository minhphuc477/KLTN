"""
Canonical style-token helpers for Block III conditioning.

The current neural style-token path is intentionally tied to the symbolic
sector themes that already exist in the mission-graph generator. We avoid
guessing broad visual-theme semantics from arbitrary free-form labels.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Optional


CANONICAL_STYLE_TOKENS = (
    "fire",
    "water",
    "ice",
    "forest",
    "shadow",
    "spirit",
)

STYLE_TOKEN_TO_ID = {name: idx for idx, name in enumerate(CANONICAL_STYLE_TOKENS)}
STYLE_ID_TO_TOKEN = {idx: name for name, idx in STYLE_TOKEN_TO_ID.items()}

_STYLE_ALIASES = {
    "fire": "fire",
    "lava": "fire",
    "volcano": "fire",
    "volcanic": "fire",
    "molten": "fire",
    "water": "water",
    "ocean": "water",
    "sea": "water",
    "river": "water",
    "lake": "water",
    "ice": "ice",
    "frost": "ice",
    "forest": "forest",
    "woods": "forest",
    "jungle": "forest",
    "grove": "forest",
    "shadow": "shadow",
    "dark": "shadow",
    "gloom": "shadow",
    "spirit": "spirit",
    "spectral": "spirit",
    "ethereal": "spirit",
}


def canonicalize_style_token(value: Any) -> Optional[str]:
    """Map a supported symbolic theme label to the canonical style vocabulary."""
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    direct = _STYLE_ALIASES.get(key)
    if direct is not None:
        return direct

    # Support common compound labels such as "fire-temple" or "ice_cavern"
    # without guessing arbitrary free-form style semantics.
    tokens = [token for token in re.split(r"[^a-z0-9]+", key) if token]
    for token in tokens:
        mapped = _STYLE_ALIASES.get(token)
        if mapped is not None:
            return mapped
    return None


def resolve_style_token_id(
    *values: Any,
    max_tokens: int = len(CANONICAL_STYLE_TOKENS),
) -> Optional[int]:
    """
    Resolve a supported numeric or symbolic style token ID.

    Resolution order is caller-defined via `values`.
    Numeric IDs are accepted only when they fall inside the configured token
    vocabulary. Symbolic labels are canonicalized through the repo's exact
    sector-theme vocabulary.
    """
    limit = int(max(0, max_tokens))
    for value in values:
        if value is None:
            continue
        if hasattr(value, "numel") and callable(getattr(value, "numel")):
            try:
                if int(value.numel()) != 1:
                    continue
                value = value.item()
            except Exception:
                continue
        try:
            style_id = int(value)
        except (TypeError, ValueError):
            style_name = canonicalize_style_token(value)
            if style_name is None:
                continue
            style_id = STYLE_TOKEN_TO_ID.get(style_name, -1)
        if 0 <= style_id < limit:
            return int(style_id)
    return None


def iter_style_metadata_candidates(*mappings: Optional[dict], keys: Iterable[str]) -> list[Any]:
    """Collect candidate metadata values for style-token resolution."""
    candidates: list[Any] = []
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue
        for key in keys:
            if key in mapping:
                candidates.append(mapping.get(key))
    return candidates
