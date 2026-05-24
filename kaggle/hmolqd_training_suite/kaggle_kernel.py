#!/usr/bin/env python
"""Kaggle script entrypoint for the H-MOLQD training suite.

This file lets a Kaggle API kernel run the current shell suite. When launched
outside a checked-out repo, it clones the repository into /kaggle/working/KLTN
first, then delegates to run_kaggle_training_suite.sh.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping


DEFAULT_REPO_URL = "https://github.com/minhphuc477/KLTN.git"
DEFAULT_KAGGLE_REPO_DIR = Path("/kaggle/working/KLTN")


def _find_repo_root_from_file() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "main.py").is_file() and (parent / "configs" / "zelda_hmolqd.yaml").is_file():
            return parent
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=os.environ.get("KLTN_REPO_URL", DEFAULT_REPO_URL))
    parser.add_argument("--repo-dir", type=Path, default=Path(os.environ.get("KLTN_REPO_DIR", DEFAULT_KAGGLE_REPO_DIR)))
    parser.add_argument("--git-depth", default=os.environ.get("KLTN_GIT_DEPTH", "1"))
    parser.add_argument("--no-clone", action="store_true", help="Fail instead of cloning when the repo is missing.")
    parser.add_argument("--profile", choices=("auto", "t4x2", "p100", "cpu"), default=None)
    parser.add_argument("--tokenizers", default=None, help='Space-separated tokenizer list, e.g. "vqvae vqvae2".')
    parser.add_argument("--branches", default=None, help='Space-separated branch list, e.g. "stage_full stage_loss010".')
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-vqvae", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--skip-fast-sampler", action="store_true")
    parser.add_argument("--skip-masked-room", action="store_true")
    return parser.parse_args(argv)


def build_suite_env(args: argparse.Namespace, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    optional = {
        "PROFILE": args.profile,
        "TOKENIZERS": args.tokenizers,
        "BRANCHES": args.branches,
        "DATA_DIR": args.data_dir,
        "OUT_ROOT": args.out_root,
    }
    for key, value in optional.items():
        if value:
            env[key] = str(value)
    if args.quick:
        env["QUICK"] = "1"
    if args.skip_vqvae:
        env["RUN_VQVAE"] = "0"
    if args.skip_diffusion:
        env["RUN_DIFFUSION"] = "0"
    if args.skip_fast_sampler:
        env["RUN_FAST_SAMPLER"] = "0"
    if args.skip_masked_room:
        env["RUN_MASKED_ROOM"] = "0"
    return env


def ensure_repo(args: argparse.Namespace) -> Path:
    discovered = _find_repo_root_from_file()
    if discovered is not None:
        return discovered

    repo_dir = args.repo_dir.expanduser().resolve()
    if (repo_dir / "main.py").is_file() and (repo_dir / "configs" / "zelda_hmolqd.yaml").is_file():
        return repo_dir
    if args.no_clone:
        raise FileNotFoundError(f"Repository not found at {repo_dir}")
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone the repository in standalone Kaggle mode.")

    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(args.git_depth), str(args.repo_url), str(repo_dir)]
    print("[kaggle] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return repo_dir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("bash") is None:
        raise RuntimeError("bash is required to run the Kaggle training suite.")
    repo_root = ensure_repo(args)
    suite_script = repo_root / "kaggle" / "hmolqd_training_suite" / "run_kaggle_training_suite.sh"
    if not suite_script.is_file():
        raise FileNotFoundError(f"Missing suite script: {suite_script}")

    env = build_suite_env(args)
    print(f"[kaggle] repo={repo_root}", flush=True)
    subprocess.run(["bash", str(suite_script)], cwd=repo_root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
