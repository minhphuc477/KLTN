from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    args = list(sys.argv[1:])
    if "--basetemp" not in args:
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        if local_appdata:
            base = Path(local_appdata) / "Temp" / "kltn_pytest_safe"
        else:
            base = Path(tempfile.gettempdir()) / "kltn_pytest_safe"
        base.mkdir(parents=True, exist_ok=True)
        args.extend(["--basetemp", str(base)])
    cmd = [sys.executable, "-m", "pytest", *args]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
