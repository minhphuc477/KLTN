import pathlib
import re

src_dir = pathlib.Path("f:/KLTN/src")
py_files = list(src_dir.rglob("*.py"))

for pf in py_files:
    rel = str(pf.relative_to("f:/KLTN")).replace("\\", "/")
    content = pf.read_text(encoding="utf-8-sig", errors="replace")
    lines = content.splitlines()
    for idx, line in enumerate(lines, 1):
        # Check explicit .cuda() calls (should be .to(device))
        if ".cuda()" in line and "if torch.cuda.is_available()" not in line and "is_available()" not in lines[max(0, idx-3):idx]:
            # Filter comments
            if not line.strip().startswith("#"):
                print(f"CUDA_CALL: {rel}:{idx} -> {line.strip()}")
