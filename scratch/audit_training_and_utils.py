import ast, re, pathlib

src_dir = pathlib.Path("f:/KLTN/src")

def read_file(path):
    p = pathlib.Path(path)
    return p.read_text(encoding="utf-8-sig") if p.exists() else ""

print("=== 1. Checking src/train_vqvae.py ===")
vqvae_code = read_file(src_dir / "train_vqvae.py")
# Check if _run_validation has torch.no_grad and sets model.eval() and restores model.train()
val_func = ""
if "def _run_validation" in vqvae_code:
    val_func = vqvae_code.split("def _run_validation")[1].split("def ")[0]
    print("  _run_validation has torch.no_grad():", "torch.no_grad()" in val_func)
    print("  _run_validation sets model.eval():", "model.eval()" in val_func)
    print("  _run_validation restores model.train():", "model.train()" in val_func)
else:
    print("  _run_validation function not found!")

# Check if .item() calls are safe / NaN checks
print("  Contains NaN/Inf checks in loss:", "torch.isnan" in vqvae_code or "math.isnan" in vqvae_code)

print("\n=== 2. Checking src/train_lcm.py ===")
lcm_code = read_file(src_dir / "train_lcm.py")
# Check how online_model / target_model are used
print("  target_model / teacher no_grad inside loop:", "torch.no_grad()" in lcm_code)
print("  model.eval() vs model.train() in LCM:", "model.train()" in lcm_code, "model.eval()" in lcm_code)

print("\n=== 3. Checking src/utils/checkpoint.py ===")
chk_code = read_file(src_dir / "utils/checkpoint.py")
# Check what keys are saved vs loaded
print("  Keys mentioned in save:", re.findall(r'["\']([a_z_]+_state_dict|epoch|global_step|best_metric|config|scaler)[\'"]', chk_code))

print("\n=== 4. Checking src/utils/distributed.py ===")
dist_code = read_file(src_dir / "utils/distributed.py")
print("  Has barrier():", "barrier(" in dist_code)
print("  Checks is_initialized():", "is_initialized()" in dist_code)

print("\n=== 5. Checking config_system.py keys vs train_diffusion.py ===")
cfg_code = read_file(src_dir / "config_system.py")
diff_code = read_file(src_dir / "train_diffusion.py")

# Extract all getattr(config, ...) and config.get(...) calls in train_diffusion.py
attrs = set(re.findall(r'getattr\(\s*(?:self\.)?config,\s*["\']([^"\']+)["\']', diff_code))
# Extract fields defined in config_system.py (e.g. self.field_name = or field_name: in dataclasses)
defined = set(re.findall(r'([a_z_][a_z0-9_]*)\s*(?::|\=)', cfg_code))
missing_attrs = [a for a in sorted(attrs) if a not in defined and not a.startswith("_")]
print("  Attributes accessed via getattr in train_diffusion.py but not in config_system matches:", missing_attrs[:15])
