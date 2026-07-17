import ast, pathlib, re

src_dir = pathlib.Path("f:/KLTN/src")

def read(rel_path):
    p = src_dir / rel_path
    return p.read_text(encoding="utf-8-sig") if p.exists() else ""

print("=== 1. Checking pipeline/neural_pipeline.py ===")
np_code = read("pipeline/neural_pipeline.py")
print("  Contains .view() or .reshape():", "view(" in np_code or "reshape(" in np_code)
# Check VQTokenizer decode call inside generate_room / forward
print("  Contains decode():", "decode(" in np_code)

print("\n=== 2. Checking evaluation/evaluator.py ===")
ev_code = read("evaluation/evaluator.py")
print("  _curve_event_alignment_scores called anywhere else:", ev_code.count("_curve_event_alignment_scores"))
print("  Contains empty room handling:", "if not " in ev_code or "sum() == 0" in ev_code)

print("\n=== 3. Checking evaluation/map_elites.py ===")
me_code = read("evaluation/map_elites.py")
print("  add method check:", "def add(" in me_code)
print("  cvt boundaries check:", "cvt" in me_code.lower())

print("\n=== 4. Checking simulation/validator.py ===")
val_code = read("simulation/validator.py")
print("  opened_graph_edges populated:", "opened_graph_edges" in val_code)
# Check mutable defaults
mutable_defs = re.findall(r'def\s+[a_zA_Z0-9_]+\([^)]*=\s*([\[\{][^\]\}]*[\]\}])', val_code)
print("  Mutable default arguments in validator.py:", mutable_defs)

print("\n=== 5. Checking generation/evolutionary_director/executor.py ===")
ex_code = read("generation/evolutionary_director/executor.py")
print("  apply_rule or mutation clones graph:", "copy.deepcopy" in ex_code or ".copy()" in ex_code)
