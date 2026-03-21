"""Utilities for managing GUI temp files."""

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


def list_existing_paths(paths: Iterable[str]) -> List[str]:
    """Return normalized existing file paths from an iterable."""
    out: List[str] = []
    for p in paths:
        if not p:
            continue
        try:
            s = str(p)
        except Exception:
            continue
        if os.path.exists(s):
            out.append(s)
    # de-duplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def find_temp_files(temp_dir: str, patterns: Iterable[str]) -> List[str]:
    """Find temp files matching glob patterns inside a temp directory."""
    base = Path(temp_dir)
    found: List[str] = []
    for pat in patterns:
        for m in base.glob(pat):
            found.append(str(m))
    return list_existing_paths(found)


def delete_files(paths: Iterable[str]) -> Tuple[int, List[Tuple[str, str]]]:
    """Delete files and return (deleted_count, failures[(path,error)])."""
    deleted = 0
    failures: List[Tuple[str, str]] = []
    for p in list_existing_paths(paths):
        try:
            os.remove(p)
            deleted += 1
        except Exception as e:
            failures.append((p, str(e)))
    return deleted, failures


def open_folder(path: str) -> Tuple[bool, str]:
    """Open a folder in the OS file explorer."""
    folder = str(path)
    if not os.path.isdir(folder):
        return False, f"Folder does not exist: {folder}"

    try:
        if os.name == "nt":
            os.startfile(folder)  # type: ignore[attr-defined]
        elif os.name == "posix":
            subprocess.Popen(["xdg-open", folder])
        else:
            subprocess.Popen(["open", folder])
        return True, ""
    except Exception as e:
        return False, str(e)
