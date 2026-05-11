"""Generate presentation-ready figures from repository artifacts.

The script intentionally uses trained checkpoints and existing benchmark
outputs where available. It writes PNG/SVG/Mermaid assets to
``results/presentation_assets_20260508`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_fast_sampler_visual_audit import _draw_stylized_tile, _tile_color
from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE
from src.core.latent_diffusion import get_noise_schedule
from src.core.symbolic_refiner import PathAnalyzer
from src.core.vqvae import create_vqvae
from src.train_vqvae import grids_to_onehot
from src.zelda_data.zelda_loader import ZeldaRoomDataset


DEFAULT_OUT_DIR = ROOT / "results" / "presentation_assets_20260508"
DEFAULT_VQVAE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "vqvae_ablation_codebook512_v2"
    / "checkpoints"
    / "vqvae"
    / "vqvae_pretrained.pth"
)

BG = "#f8fafc"
INK = "#111827"
MUTED = "#64748b"
BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#7c3aed"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _grid_rgb(grid: np.ndarray, tile_px: int = 18) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.int32)
    h, w = grid.shape
    image = Image.new("RGB", (w * tile_px, h * tile_px), _tile_color(0))
    draw = ImageDraw.Draw(image)
    for r in range(h):
        for c in range(w):
            _draw_stylized_tile(
                draw,
                tile=int(grid[r, c]),
                x0=c * tile_px,
                y0=r * tile_px,
                tile_px=tile_px,
            )
    return np.asarray(image)


def _draw_grid(ax: plt.Axes, grid: np.ndarray, title: str, *, tile_px: int = 18) -> None:
    ax.imshow(_grid_rgb(grid, tile_px=tile_px))
    ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")


def _grid_rgb_with_latent_highlight(
    grid: np.ndarray,
    latent_rc: Tuple[int, int],
    latent_shape: Tuple[int, int],
    *,
    tile_px: int = 8,
) -> np.ndarray:
    image = Image.fromarray(_grid_rgb(grid, tile_px=tile_px))
    draw = ImageDraw.Draw(image, "RGBA")
    latent_h, latent_w = latent_shape
    r, c = latent_rc
    x0 = int(round(c * ROOM_WIDTH * tile_px / latent_w))
    x1 = int(round((c + 1) * ROOM_WIDTH * tile_px / latent_w))
    y0 = int(round(r * ROOM_HEIGHT * tile_px / latent_h))
    y1 = int(round((r + 1) * ROOM_HEIGHT * tile_px / latent_h))
    draw.rectangle([x0, y0, max(x0 + 1, x1 - 1), max(y0 + 1, y1 - 1)], fill=(220, 38, 38, 48))
    line_width = max(2, tile_px // 3)
    for offset in range(line_width):
        draw.rectangle(
            [x0 + offset, y0 + offset, max(x0 + 1, x1 - 1 - offset), max(y0 + 1, y1 - 1 - offset)],
            outline=(220, 38, 38, 255),
        )
    return np.asarray(image)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_vqvae(checkpoint_path: Path) -> torch.nn.Module:
    meta_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing VQ-VAE metadata: {meta_path}")
    meta = _load_json(meta_path)
    arch = dict(meta.get("architecture", {}) or {})
    model = create_vqvae(
        num_classes=int(arch.get("num_classes", 44)),
        codebook_size=int(arch.get("codebook_size", 512)),
        latent_dim=int(arch.get("latent_dim", 64)),
        hidden_dim=int(arch.get("hidden_dim", 128)),
        commitment_cost=float(arch.get("commitment_cost", 0.25)),
        rare_tile_weight=float(arch.get("rare_tile_weight", 5.0)),
        use_ema=bool(arch.get("use_ema", True)),
        use_coordconv=bool(arch.get("use_coordconv", True)),
        mrf_penalty_weight=float(arch.get("mrf_penalty_weight", 0.05)),
        num_res_blocks=int(arch.get("num_res_blocks", 2)),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_room_tensors_and_grids() -> Tuple[List[torch.Tensor], np.ndarray]:
    dataset = ZeldaRoomDataset(str(ROOT / "Data" / "The Legend of Zelda"), normalize=True)
    if len(dataset) == 0:
        raise RuntimeError("No Zelda rooms were loaded from Data/The Legend of Zelda.")
    tensors = [dataset[int(i)] for i in range(len(dataset))]
    batch = torch.stack(tensors, dim=0)
    grids = (batch.squeeze(1).numpy() * 43.0).round().clip(0, 43).astype(np.int32)
    return tensors, grids


def _room_interest_order(grids: np.ndarray) -> List[int]:
    base_tiles = {
        int(SEMANTIC_PALETTE.get("FLOOR", 0)),
        int(SEMANTIC_PALETTE.get("WALL", 1)),
        int(SEMANTIC_PALETTE.get("VOID", 0)),
        int(SEMANTIC_PALETTE.get("DOOR_OPEN", -1)),
        int(SEMANTIC_PALETTE.get("DOOR_CLOSED", -1)),
    }
    scores: List[Tuple[int, int, int, int]] = []
    for idx, grid in enumerate(grids):
        non_base = int(np.count_nonzero(~np.isin(grid, list(base_tiles))))
        unique_tiles = int(len(np.unique(grid)))
        wall_count = int(np.count_nonzero(grid == int(SEMANTIC_PALETTE.get("WALL", 1))))
        scores.append((unique_tiles, min(non_base, 40), wall_count, -idx))
    return [idx for idx, _score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]


def _selected_room_batch(count: int = 6) -> Tuple[torch.Tensor, np.ndarray]:
    all_tensors, grids = _load_room_tensors_and_grids()
    order = _room_interest_order(grids)
    indices: List[int] = []
    seen: set[bytes] = set()
    for idx in order:
        key = grids[int(idx)].tobytes()
        if key in seen:
            continue
        seen.add(key)
        indices.append(int(idx))
        if len(indices) >= min(count, len(all_tensors)):
            break
    tensors = [all_tensors[int(i)] for i in indices]
    batch = torch.stack(tensors, dim=0)
    return batch, grids[indices]


def make_codebook_mosaic(model: torch.nn.Module, out_dir: Path) -> Dict[str, Any]:
    quantizer = getattr(model, "quantizer", None)
    if quantizer is None:
        raise RuntimeError("Loaded VQ-VAE has no quantizer.")

    tensors, grids = _load_room_tensors_and_grids()
    codebook_size = int(quantizer.embedding.weight.shape[0])
    counts = np.zeros(codebook_size, dtype=np.int64)
    occurrences: Dict[int, Tuple[int, int, int]] = {}
    latent_shape: Tuple[int, int] | None = None

    with torch.no_grad():
        for start in range(0, len(tensors), 64):
            batch = torch.stack(tensors[start : start + 64], dim=0)
            onehot = grids_to_onehot(batch, num_classes=int(model.num_classes))
            _z_q, indices = model.encode(onehot)
            arr = indices.cpu().numpy().astype(np.int64)
            if latent_shape is None:
                latent_shape = (int(arr.shape[1]), int(arr.shape[2]))
            counts += np.bincount(arr.reshape(-1), minlength=codebook_size)
            center = np.array([(arr.shape[1] - 1) / 2.0, (arr.shape[2] - 1) / 2.0])
            for local_idx in range(arr.shape[0]):
                for code_id in np.unique(arr[local_idx]):
                    code_int = int(code_id)
                    if code_int in occurrences:
                        continue
                    positions = np.argwhere(arr[local_idx] == code_int)
                    best = positions[np.argmin(np.sum((positions - center) ** 2, axis=1))]
                    occurrences[code_int] = (start + local_idx, int(best[0]), int(best[1]))

    if latent_shape is None:
        raise RuntimeError("Unable to encode any VQ-VAE room latents.")

    order = [int(code_id) for code_id in np.argsort(-counts) if counts[int(code_id)] > 0 and int(code_id) in occurrences]
    order = order[:32]

    fig, axes = plt.subplots(4, 8, figsize=(15.2, 8.7), constrained_layout=True)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, code_id in zip(axes.flatten(), order):
        room_idx, lat_r, lat_c = occurrences[int(code_id)]
        ax.imshow(
            _grid_rgb_with_latent_highlight(
                grids[int(room_idx)],
                (int(lat_r), int(lat_c)),
                latent_shape,
                tile_px=10,
            )
        )
        ax.set_title(f"code #{int(code_id)}\nuses={int(counts[int(code_id)])}", fontsize=8.5, pad=4)
        ax.axis("off")
    fig.suptitle(
        "VQ-VAE Codebook Usage Atlas (real dataset rooms; red box = latent cell assigned to code)",
        fontsize=15,
    )
    out_path = out_dir / "vqvae_codebook_prototypes.png"
    _save_fig(fig, out_path)
    return {
        "path": _rel(out_path),
        "source": _rel(DEFAULT_VQVAE_CHECKPOINT),
        "dataset_rooms": int(len(tensors)),
        "latent_shape": [int(latent_shape[0]), int(latent_shape[1])],
        "active_codes": int(np.count_nonzero(counts)),
        "selected_codes": [int(v) for v in order],
        "note": "This is a real usage atlas, not a decoded single-code prototype; each panel shows a real encoded room patch.",
    }


def make_reconstruction_pairs(model: torch.nn.Module, out_dir: Path) -> Dict[str, Any]:
    batch, real_grids = _selected_room_batch(count=6)
    with torch.no_grad():
        onehot = grids_to_onehot(batch, num_classes=int(model.num_classes))
        logits, _vq_loss, losses = model(onehot)
        recon_grids = logits.argmax(dim=1).cpu().numpy().astype(np.int32)
        acc = (recon_grids == real_grids).mean(axis=(1, 2))

    fig, axes = plt.subplots(2, 6, figsize=(14.0, 6.2))
    for idx, (real, recon) in enumerate(zip(real_grids, recon_grids)):
        row = idx // 3
        col = (idx % 3) * 2
        _draw_grid(axes[row, col], real, f"Real {idx + 1}", tile_px=9)
        _draw_grid(axes[row, col + 1], recon, f"Recon {acc[idx] * 100:.1f}%", tile_px=9)
    fig.suptitle("VQ-VAE Reconstruction: Real vs. Recon", fontsize=15, y=1.02)
    out_path = out_dir / "vqvae_reconstruction_real_vs_recon.png"
    _save_fig(fig, out_path)
    return {
        "path": _rel(out_path),
        "mean_tile_accuracy": float(np.mean(acc)),
        "source": _rel(DEFAULT_VQVAE_CHECKPOINT),
    }


def make_diffusion_latent_progression(model: torch.nn.Module, out_dir: Path) -> Dict[str, Any]:
    batch, real_grids = _selected_room_batch(count=1)
    onehot = grids_to_onehot(batch, num_classes=int(model.num_classes))
    with torch.no_grad():
        z0, _indices = model.encode(onehot)

    torch.manual_seed(20260508)
    noise = torch.randn_like(z0)
    betas = get_noise_schedule("cosine", 1000)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    steps = [999, 500, 50, 0]
    labels = ["z_T random noise", "z_t latent state", "z_50 near-clean", "z_0 clean latent"]

    latent_maps: List[np.ndarray] = []
    with torch.no_grad():
        for t in steps:
            if t == 0:
                zt = z0
            else:
                a = alpha_bar[t].view(1, 1, 1, 1)
                zt = torch.sqrt(a) * z0 + torch.sqrt(1.0 - a) * noise
            latent_maps.append(zt.pow(2).mean(dim=1).sqrt().squeeze(0).cpu().numpy())
        clean_logits = model.decode(z0, target_size=(ROOM_HEIGHT, ROOM_WIDTH))
        clean_grid = clean_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    sample_image_candidates = [
        ROOT
        / "results"
        / "thesis_ch4_evals"
        / "branch_compare_pdrop035"
        / "pdrop035"
        / "diffusion_cfg3_logic0_steps50"
        / "dungeon_grid_stylized.png",
        ROOT / "results" / "ch4_full_pipeline_real" / "diffusion_cfg3_logic0_steps50" / "dungeon_grid_stylized.png",
    ]
    sample_image = next((path for path in sample_image_candidates if path.exists()), None)

    fig, axes = plt.subplots(1, 6, figsize=(16.0, 3.9), constrained_layout=True)
    vmin = min(float(np.min(v)) for v in latent_maps)
    vmax = max(float(np.max(v)) for v in latent_maps)
    for ax, label, heat in zip(axes[:4], labels, latent_maps):
        ax.imshow(heat, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9.5, pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
    _draw_grid(axes[4], clean_grid, "VQ-VAE decode", tile_px=10)
    if sample_image is not None:
        axes[5].imshow(mpimg.imread(str(sample_image)))
        axes[5].set_title("repo sample output", fontsize=9.5, pad=5)
        axes[5].axis("off")
    else:
        _draw_grid(axes[5], real_grids[0], "reference room", tile_px=10)
    fig.suptitle("Latent Diffusion Process: Noise -> Latent Denoising Steps -> Decode", fontsize=14)
    fig.text(
        0.5,
        -0.02,
        "Heatmaps are latent energy from the repo VQ-VAE + cosine schedule; they are not decoded as rooms. Final panels show tile-space decode/output.",
        ha="center",
        va="top",
        fontsize=8.8,
        color=MUTED,
    )
    out_path = out_dir / "diffusion_latent_progression.png"
    _save_fig(fig, out_path)
    return {
        "path": _rel(out_path),
        "source": "src.core.latent_diffusion.get_noise_schedule + VQ-VAE decoder",
        "sample_output": _rel(sample_image) if sample_image is not None else None,
        "note": "Latent panels visualize state types in the diffusion schedule. They do not claim to be saved reverse-sampler intermediates.",
    }


def _box(ax: plt.Axes, xy: Tuple[float, float], w: float, h: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.4,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, color=INK)


def _arrow(ax: plt.Axes, start: Tuple[float, float], end: Tuple[float, float], color: str = MUTED) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def make_architecture_diagrams(out_dir: Path) -> Dict[str, Any]:
    fig, ax = plt.subplots(figsize=(14.0, 7.0))
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _box(ax, (0.04, 0.71), 0.23, 0.13, "Local Grid Context\nneighbor rooms + boundary doors", BLUE)
    _box(ax, (0.04, 0.39), 0.23, 0.13, "Global Mission Graph\nnodes + typed edges + TPE", PURPLE)
    _box(ax, (0.34, 0.71), 0.21, 0.13, "LocalStreamEncoder\nMLP over boundary/position", BLUE)
    _box(ax, (0.34, 0.39), 0.21, 0.13, "GlobalStreamEncoder\nGATv2/FallbackGNN", PURPLE)
    _box(ax, (0.63, 0.55), 0.21, 0.15, "CrossAttentionFusion\nlocal query over graph tokens", GREEN)
    _box(ax, (0.72, 0.29), 0.21, 0.13, "Condition Vector/Tokens\ncontext_dim=256", AMBER)
    _box(ax, (0.72, 0.07), 0.21, 0.13, "LatentDiffusionModel\nU-Net DDPM/DDIM", RED)

    _arrow(ax, (0.27, 0.775), (0.34, 0.775), BLUE)
    _arrow(ax, (0.27, 0.455), (0.34, 0.455), PURPLE)
    _arrow(ax, (0.55, 0.775), (0.63, 0.64), BLUE)
    _arrow(ax, (0.55, 0.455), (0.63, 0.62), PURPLE)
    _arrow(ax, (0.74, 0.55), (0.80, 0.42), GREEN)
    _arrow(ax, (0.825, 0.29), (0.825, 0.20), AMBER)
    _arrow(ax, (0.155, 0.52), (0.155, 0.71), MUTED)
    ax.text(0.18, 0.62, "room/node alignment", fontsize=9, color=MUTED, rotation=90, va="center")
    ax.set_title("Dual-stream Conditioning: Local Grid <-> Global Graph", fontsize=17, pad=18)
    dual_path = out_dir / "dual_stream_conditioning_flow.png"
    _save_fig(fig, dual_path)

    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        ((0.03, 0.67), "VGLC Zelda Rooms\nsemantic tile grids", BLUE),
        ((0.24, 0.67), "Semantic VQ-VAE\nencode/decode room tiles", BLUE),
        ((0.45, 0.67), "Discrete Latent Codebook\nroom vocabulary", BLUE),
        ((0.03, 0.34), "Grammar / Director\nmission DAG + descriptors", GREEN),
        ((0.24, 0.34), "DualStreamConditionEncoder\nlocal + global context", PURPLE),
        ((0.45, 0.34), "LatentDiffusionModel\nDDPM/DDIM U-Net", RED),
        ((0.64, 0.34), "VQ-VAE Decode\nlatent -> room tiles", BLUE),
        ((0.82, 0.34), "SymbolicRefiner / WFC\npath + tile cleanup", AMBER),
        ((0.45, 0.09), "LogicNet Guidance\nsoft reachability pressure", RED),
        ((0.82, 0.09), "Validators + MAP-Elites\nsolvers + QD archive", GREEN),
    ]
    for xy, text, color in boxes:
        _box(ax, xy, 0.15 if xy[0] >= 0.82 else 0.17, 0.13, text, color)
    _arrow(ax, (0.20, 0.735), (0.24, 0.735), BLUE)
    _arrow(ax, (0.41, 0.735), (0.45, 0.735), BLUE)
    _arrow(ax, (0.535, 0.67), (0.535, 0.47), BLUE)
    _arrow(ax, (0.20, 0.405), (0.24, 0.405), GREEN)
    _arrow(ax, (0.41, 0.405), (0.45, 0.405), PURPLE)
    _arrow(ax, (0.62, 0.405), (0.64, 0.405), RED)
    _arrow(ax, (0.81, 0.405), (0.82, 0.405), BLUE)
    _arrow(ax, (0.535, 0.22), (0.535, 0.34), RED)
    _arrow(ax, (0.895, 0.34), (0.895, 0.22), AMBER)
    ax.text(0.535, 0.58, "latent space", fontsize=9, color=MUTED, ha="center")
    ax.text(0.895, 0.26, "metrics", fontsize=9, color=MUTED, ha="center")
    ax.set_title("Repository Architecture: Neural-Symbolic Zelda Generator", fontsize=17, pad=18)
    repo_path = out_dir / "repo_architecture_flow.png"
    _save_fig(fig, repo_path)

    mmd = """flowchart LR
    A[Local Grid<br/>neighbor rooms + boundary doors] <--> B[Global Graph<br/>mission nodes + typed edges]
    A --> C[LocalStreamEncoder]
    B --> D[GlobalStreamEncoder<br/>GNN + TPE]
    C --> E[CrossAttentionFusion]
    D --> E
    E --> F[Condition Vector / Tokens]
    N[Random Latent Noise] --> G[LatentDiffusionModel<br/>DDPM / DDIM U-Net]
    F --> G
    L[LogicNet optional guidance] -.-> G
    G --> H[Semantic VQ-VAE Decode]
    H --> I[SymbolicRefiner / WFC Repair]
    I --> J[Solvers + MAP-Elites Metrics]
"""
    mmd_path = out_dir / "dual_stream_diffusion_flow.mmd"
    mmd_path.write_text(mmd, encoding="utf-8")
    return {"dual_stream_png": _rel(dual_path), "repo_flow_png": _rel(repo_path), "mermaid": _rel(mmd_path)}


def make_solvability_chart(out_dir: Path) -> Dict[str, Any]:
    full_path = ROOT / "results" / "old_result" / "ablation_core_quick_v3" / "ablation_summary.csv"
    no_logic_path = ROOT / "results" / "old_result" / "ablation_core_quick_part2_v1" / "ablation_summary.csv"
    room_branch_path = ROOT / "results" / "room_branch_benchmark_pdrop035_quick" / "room_branch_summary.csv"
    full_frame = pd.read_csv(full_path)
    no_logic_frame = pd.read_csv(no_logic_path)

    full_row = full_frame.loc[full_frame["config"] == "FULL"].iloc[0]
    no_logic_row = no_logic_frame.loc[no_logic_frame["config"] == "NO_LOGIC"].iloc[0]
    labels = ["Full pipeline\nLogicNet included", "NO_LOGIC\nquick ablation"]
    values = [float(full_row["solvability_rate"]), float(no_logic_row["solvability_rate"])]
    ns = [int(full_row["n"]), int(no_logic_row["n"])]
    colors = [GREEN, "#94a3b8"]

    fig, (ax, note_ax) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        gridspec_kw={"width_ratios": [1.05, 1.0]},
        constrained_layout=True,
    )
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.56, edgecolor=INK, linewidth=0.6, alpha=0.55)
    ax.scatter(x, values, s=160, c=colors, edgecolors=INK, linewidths=0.8, zorder=3)
    ax.axhline(0.0, color=INK, linewidth=1.0)
    ax.set_ylim(0, 1.05 if max(values) > 0.12 else 0.12)
    ax.set_ylabel("Solved-room rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Available LogicNet Ablation Evidence")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    for bar, value, n in zip(bars, values, ns):
        solved = int(round(value * n))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{solved}/{n} solved\n{value * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    note_ax.axis("off")
    note_ax.set_title("Evidence Status", loc="left", fontsize=12, pad=8)
    note_lines = [
        "Matched ON/OFF LogicNet benchmark: not found",
        "This plot uses the only local quick ablation rows:",
        f"- FULL from {_rel(full_path)}",
        f"- NO_LOGIC from {_rel(no_logic_path)}",
        "",
        "Do not present this as a strong causal claim.",
        "The room-branch files compare reference-room maps,",
        "not LogicNet, so they are intentionally excluded.",
    ]
    y = 0.95
    for line in note_lines:
        color = RED if line.startswith("Matched") or line.startswith("Do not") else INK
        weight = "bold" if color == RED else "normal"
        note_ax.text(0.0, y, line, ha="left", va="top", fontsize=9.5, color=color, weight=weight)
        y -= 0.105 if line else 0.065
    out_path = out_dir / "solvability_logicnet_comparison.png"
    _save_fig(fig, out_path)
    return {
        "path": _rel(out_path),
        "logic_on_source": _rel(full_path),
        "logic_off_source": _rel(no_logic_path),
        "excluded_reference_branch_source": _rel(room_branch_path),
        "note": "No matched LogicNet ON/OFF benchmark was found; the figure is an evidence audit using quick ablation rows.",
    }


def make_repair_before_after(out_dir: Path) -> Dict[str, Any]:
    wall = int(SEMANTIC_PALETTE["WALL"])
    floor = int(SEMANTIC_PALETTE["FLOOR"])
    start_tile = int(SEMANTIC_PALETTE["START"])
    goal_tile = int(SEMANTIC_PALETTE["TRIFORCE"])
    door = int(SEMANTIC_PALETTE["DOOR_OPEN"])

    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH), floor, dtype=np.int32)
    grid[0, :] = wall
    grid[-1, :] = wall
    grid[:, 0] = wall
    grid[:, -1] = wall
    grid[8, 1:-1] = wall
    # Deliberately place a door marker on the outer shell, not in the blocking
    # wall. The room therefore has both a geometric gap and a misaligned door.
    grid[4, -1] = door
    start = (2, ROOM_WIDTH // 2)
    goal = (13, ROOM_WIDTH // 2)
    grid[start] = start_tile
    grid[goal] = goal_tile

    analyzer = PathAnalyzer()
    before_failures = analyzer.analyze_grid(grid, start=start, goal=goal)

    repaired = grid.copy()
    # Deterministic path-guided repair: open the blocking wall on the critical
    # path and move the side door to a reachable boundary cell.
    repair_row = 8
    repair_col = ROOM_WIDTH // 2
    repaired[repair_row, repair_col] = floor
    repaired[4, -1] = wall
    repaired[repair_row, -2] = floor
    repaired[repair_row, -1] = door
    repaired[start] = start_tile
    repaired[goal] = goal_tile
    after_failures = analyzer.analyze_grid(repaired, start=start, goal=goal)
    success = len(after_failures) == 0
    changed = grid != repaired

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    _draw_grid(axes[0], grid, f"Before: disconnected={bool(before_failures)}", tile_px=13)
    _draw_grid(axes[1], repaired, f"After repair: success={bool(success)}", tile_px=13)
    axes[2].imshow(_grid_rgb(repaired, tile_px=13))
    overlay = np.zeros((*changed.shape, 4), dtype=np.float32)
    overlay[changed] = [1.0, 0.15, 0.15, 0.55]
    axes[2].imshow(overlay, interpolation="nearest", extent=(0, ROOM_WIDTH * 13, ROOM_HEIGHT * 13, 0))
    axes[2].set_title(f"Changed tiles: {int(changed.sum())}", fontsize=10, pad=6)
    axes[2].axis("off")
    fig.suptitle("Symbolic/WFC-style Repair: Geometry Gap / Misaligned Door -> Valid Room", fontsize=14, y=1.02)
    out_path = out_dir / "symbolic_repair_before_after.png"
    _save_fig(fig, out_path)
    return {
        "path": _rel(out_path),
        "before_failure_count": int(len(before_failures)),
        "after_failure_count": int(len(after_failures)),
        "success": bool(success),
    }


def make_map_elites_behavior_space(out_dir: Path) -> Dict[str, Any]:
    candidates = [
        ROOT / "results" / "map_elites_full_n64" / "matched_budget_raw.csv",
        ROOT / "results" / "map_elites_default_20260507" / "matched_budget_raw.csv",
        ROOT / "results" / "map_elites_quick_20260507" / "matched_budget_raw.csv",
    ]
    raw_path = next((path for path in candidates if path.exists()), candidates[0])
    frame = pd.read_csv(raw_path)
    x_desc = "linearity"
    y_desc = "leniency"
    quality = "fitness"
    bins = 10 if len(frame) >= 50 else 8
    heat = np.full((bins, bins), np.nan, dtype=float)
    counts = np.zeros((bins, bins), dtype=int)
    xvals = frame[x_desc].to_numpy(dtype=float)
    yvals = frame[y_desc].to_numpy(dtype=float)
    qvals = frame[quality].to_numpy(dtype=float)
    xi = np.clip((xvals * bins).astype(int), 0, bins - 1)
    yi = np.clip((yvals * bins).astype(int), 0, bins - 1)
    for xbin, ybin, q in zip(xi, yi, qvals):
        current = heat[ybin, xbin]
        heat[ybin, xbin] = q if math.isnan(current) else max(current, float(q))
        counts[ybin, xbin] += 1

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), constrained_layout=True)
    im = axes[0].imshow(heat, origin="lower", cmap="viridis", vmin=np.nanmin(qvals), vmax=np.nanmax(qvals))
    axes[0].set_title(f"MAP-Elites Descriptor Grid (n={len(frame)})")
    axes[0].set_xlabel("Linearity bin")
    axes[0].set_ylabel("Leniency bin")
    for y in range(bins):
        for x in range(bins):
            if counts[y, x] > 0:
                axes[0].text(x, y, str(int(counts[y, x])), ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.045, label="Best fitness")

    sc = axes[1].scatter(xvals, yvals, c=qvals, cmap="viridis", s=90, edgecolors=INK, linewidths=0.6)
    axes[1].set_title("Descriptors vs. Quality")
    axes[1].set_xlabel("Linearity")
    axes[1].set_ylabel("Leniency")
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(alpha=0.25, linestyle="--")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.045, label="Fitness")
    out_path = out_dir / "map_elites_behavior_space.png"
    _save_fig(fig, out_path)
    return {"path": _rel(out_path), "source": _rel(raw_path), "rows": int(len(frame))}


def make_case_gallery(out_dir: Path) -> Dict[str, Any]:
    rows_path = ROOT / "results" / "thesis_ch4_evals" / "chapter4_eval_rows.csv"
    frame = pd.read_csv(rows_path)
    selected = frame.loc[
        (frame["group"] == "branch_compare_pdrop035")
        & (frame["variant"].isin(["diffusion_cfg3_logic0_steps50", "fast_cfg3_logic0_steps4", "masked_room_full"]))
    ].copy()
    if selected.empty:
        return {}

    fig, axes = plt.subplots(1, len(selected), figsize=(14.0, 4.3))
    if len(selected) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, selected.iterrows()):
        image_path = ROOT / str(row["image_stylized"])
        ax.imshow(mpimg.imread(str(image_path)))
        title = str(row["variant"]).replace("_", " ")
        metrics = (
            f"astar={bool(row['astar_solvable'])}, cbs={bool(row['cbs_success'])}, "
            f"repair={float(row['repair_rate']) * 100:.0f}%"
        )
        ax.set_title(f"{title}\n{metrics}", fontsize=9.2)
        ax.axis("off")
    fig.suptitle("Generated Dungeon Branch Examples", fontsize=14, y=1.02)
    out_path = out_dir / "generated_branch_examples.png"
    _save_fig(fig, out_path)
    return {"path": _rel(out_path), "source": _rel(rows_path)}


def make_all(out_dir: Path, checkpoint_path: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    global DEFAULT_VQVAE_CHECKPOINT
    DEFAULT_VQVAE_CHECKPOINT = checkpoint_path

    manifest: Dict[str, Any] = {
        "output_dir": _rel(out_dir),
        "vqvae_checkpoint": _rel(checkpoint_path),
        "figures": {},
    }

    model = _load_vqvae(checkpoint_path)
    manifest["figures"]["codebook_mosaic"] = make_codebook_mosaic(model, out_dir)
    manifest["figures"]["reconstruction_pairs"] = make_reconstruction_pairs(model, out_dir)
    manifest["figures"]["diffusion_progression"] = make_diffusion_latent_progression(model, out_dir)
    manifest["figures"]["architecture"] = make_architecture_diagrams(out_dir)
    manifest["figures"]["solvability_logicnet"] = make_solvability_chart(out_dir)
    manifest["figures"]["symbolic_repair"] = make_repair_before_after(out_dir)
    manifest["figures"]["map_elites_behavior_space"] = make_map_elites_behavior_space(out_dir)
    manifest["figures"]["generated_branch_examples"] = make_case_gallery(out_dir)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation figures from KLTN artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--vqvae-checkpoint", type=Path, default=DEFAULT_VQVAE_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_all(args.out_dir, args.vqvae_checkpoint)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
