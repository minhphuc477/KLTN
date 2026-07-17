import pathlib

lines_to_check = [
    ("src/core/logic_net.py", 2740),
    ("src/core/symbolic_refiner.py", 965),
    ("src/core/vqvae.py", 587),
    ("src/evaluation/difficulty_calculator.py", 290),
    ("src/evaluation/fun_analyzers.py", 295),
    ("src/evaluation/fun_analyzers.py", 344),
    ("src/evaluation/search_benchmark_utils.py", 130),
    ("src/evaluation/structural_metrics.py", 77),
    ("src/pipeline/room_topology_conditioning.py", 525),
    ("src/simulation/cognitive_bounded_search.py", 773),
    ("src/utils/demo_recorder.py", 269),
    ("src/pipeline/generation/room_processing.py", 4109),
]

for path, lineno in lines_to_check:
    p = pathlib.Path("f:/KLTN") / path
    lines = p.read_text(encoding="utf-8-sig").splitlines()
    start = max(0, lineno - 7)
    end = min(len(lines), lineno + 2)
    print(f"=== {path}:{lineno} ===")
    for i in range(start, end):
        prefix = ">> " if i + 1 == lineno else "   "
        print(f"{prefix}{i+1:4d}: {lines[i]}")
    print()
