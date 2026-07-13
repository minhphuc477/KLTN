"""Filesystem-backed index of canonical GUI modules.

The catalog is navigation metadata, not an import registry.  Deriving it from
the package prevents deleted or moved modules from remaining advertised as
valid APIs.
"""

from __future__ import annotations

from pathlib import Path


_GUI_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_DOMAINS = (
    "ai",
    "app",
    "common",
    "components",
    "control_panel",
    "gameplay",
    "map",
    "orchestration",
    "rendering",
    "runtime",
    "solver",
    "topology",
)


def _discover_domain_modules(domain: str) -> list[str]:
    """Return import-like module paths relative to :mod:`src.gui`."""
    root = _GUI_ROOT / domain
    if not root.is_dir():
        return []
    modules: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        modules.append(path.relative_to(_GUI_ROOT).with_suffix("").as_posix())
    return sorted(modules)


GUI_MODULE_CATEGORIES = {
    f"domain_{domain}": _discover_domain_modules(domain)
    for domain in _CANONICAL_DOMAINS
}


def list_categories() -> list[str]:
    """Return sorted category names."""
    return sorted(GUI_MODULE_CATEGORIES)


__all__ = ["GUI_MODULE_CATEGORIES", "list_categories"]
