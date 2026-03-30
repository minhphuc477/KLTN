"""
Compatibility wrapper for the canonical training entrypoint.

This module preserves historical `python -m src.train ...` workflows while
delegating all actual training behavior to `main.py train`, which owns the
validated YAML/CLI configuration system, reproducibility snapshots, and staged
training orchestration.

Examples:
    python -m src.train --config configs/zelda_hmolqd.yaml --stage diffusion
    python -m src.train --config configs/zelda_hmolqd.yaml --stage all
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import main as root_main


logger = logging.getLogger(__name__)


def main(argv: Optional[list[str]] = None) -> None:
    """
    Forward legacy training invocations to `main.py train`.

    Users should prefer `python main.py train ...`, but this wrapper keeps the
    old module path working without maintaining a second, divergent CLI surface.
    """
    forwarded = list(sys.argv[1:] if argv is None else argv)
    if forwarded and forwarded[0] == "train":
        forwarded = forwarded[1:]
    logger.debug("Forwarding legacy src.train invocation to main.py train: %s", forwarded)
    root_main.main(["train", *forwarded])


if __name__ == "__main__":
    main()
