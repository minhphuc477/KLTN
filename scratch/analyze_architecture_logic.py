import pathlib
import ast
import re

src_dir = pathlib.Path("f:/KLTN/src")

def read(rel):
    p = src_dir / rel
    return p.read_text(encoding="utf-8-sig") if p.exists() else ""

print("=== 1. Checking LogicNet <-> Diffusion Interface ===")
logic_code = read("core/logic_net.py")
diff_code = read("train_diffusion.py")
pipeline_code = read("pipeline/neural_pipeline.py")

# Check how logic_net is called during diffusion training
diff_logic_calls = [line.strip() for line in diff_code.splitlines() if "logic" in line.lower() and ("loss" in line.lower() or "net" in line.lower() or "pathfinder" in line.lower())]
print("  Diffusion training logic references (first 10):")
for c in diff_logic_calls[:10]:
    print("    -", c[:100])

# Check how LogicNet processes input tensors (what shape/dtype does transition() take?)
print("\n=== 2. Checking LogicNet input expectations ===")
if "def forward(" in logic_code:
    for cls in ["LearnableGridPathfinder", "DenseGraphPathfinder", "LogicNet"]:
        if f"class {cls}" in logic_code:
            print(f"  Class {cls} found.")
            idx = logic_code.find(f"class {cls}")
            fwd_idx = logic_code.find("def forward(", idx)
            print("    forward signature & doc:", logic_code[fwd_idx:fwd_idx+400].replace("\n", " | "))

print("\n=== 3. Checking Condition Encoder <-> Spatial Grid Alignment ===")
cond_code = read("core/condition_encoder.py")
# How are graph node positions mapped to 2D grid spatial coordinates?
print("  Spatial coordinates handling inside ConditionEncoder:")
for line in cond_code.splitlines():
    if any(k in line.lower() for k in ["spatial", "coord", "grid_pos", "bounding", "xy"]):
        if not line.strip().startswith("#"):
            print("    -", line.strip()[:100])

print("\n=== 4. Checking Evolutionary Director -> Neural Pipeline <-> Validation loop ===")
exec_code = read("generation/evolutionary_director/executor.py")
eval_code = read("generation/evolutionary_director/evaluator.py")
print("  Executor rule application and pipeline invocation:")
for line in exec_code.splitlines():
    if any(k in line.lower() for k in ["pipeline", "generate", "evaluate", "fitness", "oracle"]):
        if not line.strip().startswith("#"):
            print("    -", line.strip()[:100])
