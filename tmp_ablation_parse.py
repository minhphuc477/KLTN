import re
from pathlib import Path

base = Path(r"F:/KLTN")
variants = [
    ("baseline", "outputs/_launcher_logs/vqvae_baseline.err.log", "outputs/vqvae_audit_baseline_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
    ("codebook128", "outputs/_launcher_logs/vqvae_codebook128.err.log", "outputs/vqvae_ablation_codebook128_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
    ("codebook512", "outputs/_launcher_logs/vqvae_codebook512.err.log", "outputs/vqvae_ablation_codebook512_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
    ("hidden64", "outputs/_launcher_logs/vqvae_hidden64.err.log", "outputs/vqvae_ablation_hidden64_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
    ("no_coordconv", "outputs/_launcher_logs/vqvae_no_coordconv.err.log", "outputs/vqvae_ablation_no_coordconv_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
    ("no_mrf", "outputs/_launcher_logs/vqvae_no_mrf.err.log", "outputs/vqvae_ablation_no_mrf_v2/checkpoints/vqvae/vqvae_pretrained.pth"),
]

pat_best = re.compile(r"Best val_loss:\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
pat_epoch = re.compile(r"Epoch\s*(\d+)\s*/\s*(\d+).*?val_loss\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?).*?val_accuracy\s*=\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)", re.IGNORECASE)
pat_active = re.compile(r"active_codes\s*=\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
pat_params = re.compile(r"VQ-VAE parameters:\s*([0-9][0-9,]*)", re.IGNORECASE)

rows = []
for name, log_rel, ckpt_rel in variants:
    log_path = base / log_rel
    ckpt_path = base / ckpt_rel
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    best_matches = pat_best.findall(text)
    best_val = float(best_matches[-1]) if best_matches else None

    epochs = []
    for line in text.splitlines():
        m = pat_epoch.search(line)
        if not m:
            continue
        ep, tot, vl, va = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))
        am = pat_active.search(line)
        active = None
        if am:
            a, b = int(am.group(1)), int(am.group(2))
            active = (a, b, (a / b) if b else None)
        epochs.append((ep, tot, vl, va, active))

    if best_val is None and epochs:
        best_val = min(e[2] for e in epochs)

    final_epoch = epochs[-1] if epochs else None
    final_val = final_epoch[2] if final_epoch else None
    best_acc = max((e[3] for e in epochs), default=None)
    has_300 = any((e[0] == 300 and e[1] == 300) for e in epochs)

    final_active = final_epoch[4] if final_epoch else None
    if final_active and final_active[2] is not None:
        active_str = f"{final_active[0]}/{final_active[1]} ({final_active[2]:.3f})"
        active_ratio = final_active[2]
    elif final_active:
        active_str = f"{final_active[0]}/{final_active[1]}"
        active_ratio = None
    else:
        active_str = "n/a"
        active_ratio = None

    pm = pat_params.findall(text)
    params = int(pm[-1].replace(',', '')) if pm else None

    ckpt_mb = ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else None
    stability = (final_val - best_val) if (final_val is not None and best_val is not None) else None

    rows.append({
        "variant": name,
        "best_val_loss": best_val,
        "final_epoch_val_loss": final_val,
        "stability_gap": stability,
        "best_val_accuracy": best_acc,
        "epoch_300_300": has_300,
        "final_active_codes_ratio": active_str,
        "model_params": params,
        "checkpoint_mb": ckpt_mb,
        "active_ratio": active_ratio,
    })

rows.sort(key=lambda r: (float('inf') if r['best_val_loss'] is None else r['best_val_loss']))

print("| variant | best_val_loss | final_epoch_val_loss | stability_gap | best_val_accuracy | epoch300 | final_active_codes | model_params | checkpoint_mb |")
print("|---|---:|---:|---:|---:|:---:|---|---:|---:|")
for r in rows:
    print(f"| {r['variant']} | {r['best_val_loss']:.4f} | {r['final_epoch_val_loss']:.4f} | {r['stability_gap']:.4f} | {r['best_val_accuracy']:.3f} | {'Y' if r['epoch_300_300'] else 'N'} | {r['final_active_codes_ratio']} | {r['model_params']:,} | {r['checkpoint_mb']:.2f} |")

# picks
rec_quality = min(rows, key=lambda r: r['best_val_loss'])
rec_stability = min(rows, key=lambda r: abs(r['stability_gap']))
rec_compat = next((r for r in rows if r['variant']=="codebook512"), None)
rec_eff = min(rows, key=lambda r: (r['checkpoint_mb'], r['best_val_loss']))

print("\nRECOMMENDATION")
print(f"quality={rec_quality['variant']} (best_val_loss={rec_quality['best_val_loss']:.4f})")
print(f"stability={rec_stability['variant']} (stability_gap={rec_stability['stability_gap']:.4f}, final={rec_stability['final_epoch_val_loss']:.4f}, best={rec_stability['best_val_loss']:.4f})")
if rec_compat:
    print(f"compatibility={rec_compat['variant']} (best_val_loss={rec_compat['best_val_loss']:.4f}, params={rec_compat['model_params']:,}, active={rec_compat['final_active_codes_ratio']})")
print(f"efficiency={rec_eff['variant']} (checkpoint={rec_eff['checkpoint_mb']:.2f} MB, params={rec_eff['model_params']:,}, best_val_loss={rec_eff['best_val_loss']:.4f})")
