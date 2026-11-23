"""Bind a run to executable source and the installed Python environment."""

from importlib import metadata
from pathlib import Path
import platform
import subprocess

from .dataset_provenance import file_sha256
from .protocol import payload_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]


def source_snapshot(root=REPO_ROOT):
    root = Path(root).resolve()
    files = set()
    for directory in ("src", "bench", "scripts", "experiments"):
        files.update(path for path in (root / directory).rglob("*.py") if "__pycache__" not in path.parts)
    files.update(path for pattern in ("*.py", "requirements*.txt", "pyproject.toml", "uv.lock", "poetry.lock")
                 for path in root.glob(pattern))
    records = [{"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
               for path in sorted(files) if path.is_file()]
    if not records:
        raise ValueError("Cannot seal an empty executable source tree.")
    return {"files": records, "sha256": payload_sha256(records)}


def environment_snapshot():
    packages = sorted((dist.metadata["Name"].lower(), dist.version)
                      for dist in metadata.distributions() if dist.metadata["Name"])
    return {"python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": [list(item) for item in packages]}


def capture_execution_provenance(root=REPO_ROOT):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            text=True, check=True).stdout.strip()
    return {"schema_version": "1.0", "git_commit": commit,
            "source": source_snapshot(root), "environment": environment_snapshot()}


def verify_execution_provenance(expected, root=REPO_ROOT):
    if not isinstance(expected, dict) or expected.get("schema_version") != "1.0":
        raise ValueError("A sealed executable source/environment snapshot is required.")
    actual = capture_execution_provenance(root)
    if actual != expected:
        changed = [key for key in ("git_commit", "source", "environment") if actual[key] != expected.get(key)]
        raise ValueError(f"Execution provenance changed after sealing: {changed}.")

