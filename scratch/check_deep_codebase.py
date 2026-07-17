import ast
import pathlib
import re
import sys

src_dir = pathlib.Path("f:/KLTN/src")
py_files = list(src_dir.rglob("*.py"))

print(f"Scanning {len(py_files)} Python files across f:/KLTN/src ...\n")

results = {
    "mutable_defaults": [],
    "bare_except_pass": [],
    "unsafe_view": [],
    "tensor_no_device": [],
    "div_by_len": [],
    "duplicate_defs": [],
    "broken_imports": [],
}

# Collect all top-level module exports and definitions across src to check internal imports
all_exports = {}
for pf in py_files:
    mod_path = "src." + ".".join(pf.relative_to(src_dir).with_suffix("").parts)
    try:
        content = pf.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(content, filename=str(pf))
        defs = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defs.add(target.id)
        all_exports[mod_path] = defs
    except Exception as e:
        pass

for pf in py_files:
    rel_path = str(pf.relative_to("f:/KLTN")).replace("\\", "/")
    content = pf.read_text(encoding="utf-8-sig", errors="replace")
    lines = content.splitlines()
    
    try:
        tree = ast.parse(content, filename=str(pf))
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {rel_path}: {e}")
        continue

    # 1. Check duplicate definitions
    seen_defs = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in seen_defs and not node.name.startswith("_"):
                results["duplicate_defs"].append((rel_path, node.lineno, f"Duplicate definition of '{node.name}'"))
            seen_defs.add(node.name)

    # 2. Check AST for mutable defaults and bare except: pass
    for node in ast.walk(tree):
        # Mutable defaults
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg_def, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    results["mutable_defaults"].append((rel_path, node.lineno, f"Function '{node.name}' uses mutable default {ast.dump(default)} for argument '{arg_def.arg}'"))
        
        # Bare except pass
        if isinstance(node, ast.ExceptHandler):
            if node.type is None: # bare except:
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    results["bare_except_pass"].append((rel_path, node.lineno, "Bare 'except:' with only 'pass'"))

    # Line-by-line checks
    for idx, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # 3. .view() checks - check if .view is called after .transpose or .permute without .contiguous()
        if ".view(" in stripped and (".transpose(" in stripped or ".permute(" in stripped) and ".contiguous()" not in stripped:
            results["unsafe_view"].append((rel_path, idx, line.strip()))
            
        # 4. Division by len without max(1, len(...)) or guard check
        if re.search(r'/\s*len\(', stripped) and "max(" not in stripped and "if len(" not in stripped and "+ 1" not in stripped and "+ 1e-" not in stripped:
            results["div_by_len"].append((rel_path, idx, line.strip()))

# Report findings
for cat, items in results.items():
    print(f"=== {cat.upper()} ({len(items)}) ===")
    for path, line_no, msg in items[:25]:
        print(f"  {path}:{line_no} -> {msg}")
    if len(items) > 25:
        print(f"  ... and {len(items) - 25} more")
    print()
