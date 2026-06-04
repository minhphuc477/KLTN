"""Autoregressive GPT-style baseline over flattened Zelda room tokens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.baselines.common import (
    BaselineEvalConfig,
    evaluate_generated_grids,
    flatten_grids_to_tokens,
    load_room_grids,
    set_reproducible_seed,
    write_json_report,
)


def _require_transformers():
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise RuntimeError(
            "The LLM baseline requires transformers. Install the optional training "
            "dependency before running without --dry-run."
        ) from exc
    return GPT2Config, GPT2LMHeadModel


def build_tiny_gpt2(*, seq_len: int, vocab_size: int, n_embd: int, n_layer: int, n_head: int):
    GPT2Config, GPT2LMHeadModel = _require_transformers()
    config = GPT2Config(
        vocab_size=int(vocab_size),
        n_positions=int(seq_len),
        n_ctx=int(seq_len),
        n_embd=int(n_embd),
        n_layer=int(n_layer),
        n_head=int(n_head),
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config)


def generate_tokens(
    model,
    *,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Autoregressively sample `[N, seq_len]` tokens under no grad."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    tokens = torch.zeros((int(num_samples), 1), dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        while tokens.shape[1] < int(seq_len):
            logits = model(tokens).logits[:, -1, :] / max(1e-6, float(temperature))
            probs = torch.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all():
                probs = torch.full_like(probs, 1.0 / float(vocab_size))
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)
            tokens = torch.cat([tokens, next_token.clamp(0, int(vocab_size) - 1)], dim=1)
    return tokens[:, : int(seq_len)].detach().cpu()


def train_llm(
    tokens: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_embd: int,
    n_layer: int,
    n_head: int,
    dry_run: bool,
) -> tuple[torch.nn.Module, List[float]]:
    """Train next-token CE without using argmax in the gradient path."""
    x = torch.as_tensor(tokens[:, :-1], dtype=torch.long)
    y = torch.as_tensor(tokens[:, 1:], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=int(batch_size), shuffle=True, drop_last=False)
    model = build_tiny_gpt2(
        seq_len=int(tokens.shape[1]),
        vocab_size=int(tokens.max()) + 1 if int(tokens.max()) >= 43 else 44,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    losses: List[float] = []
    max_epochs = 1 if dry_run else int(epochs)
    for _epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            out = model(batch_x).logits
            loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), batch_y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite LLM baseline loss.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if dry_run:
                return model, losses
    return model, losses


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate flattened-token GPT baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("Data/The Legend of Zelda"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines/llm"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-generate", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-pcbs", action="store_true")
    args = parser.parse_args(argv)

    set_reproducible_seed(args.seed)
    reference = load_room_grids(args.data_dir, max_samples=(8 if args.dry_run else args.max_train_samples))
    tokens = flatten_grids_to_tokens(reference)
    device = torch.device(args.device)
    model, losses = train_llm(
        tokens,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dry_run=args.dry_run,
    )
    num_generate = 4 if args.dry_run else int(args.num_generate)
    sampled = generate_tokens(
        model,
        num_samples=num_generate,
        seq_len=int(tokens.shape[1]),
        vocab_size=44,
        device=device,
        seed=args.seed,
        temperature=args.temperature,
    ).numpy()
    generated = sampled.reshape(num_generate, *reference[0].shape).astype(np.int32)
    report = evaluate_generated_grids(
        list(generated),
        reference,
        BaselineEvalConfig(name="autoregressive_gpt2_flattened", seed=args.seed, run_pcbs=bool(args.run_pcbs and not args.dry_run)),
    )
    report["llm"] = {
        "dry_run": bool(args.dry_run),
        "seq_len": int(tokens.shape[1]),
        "vocab_size": 44,
        "losses": losses,
        "final_loss": float(losses[-1]) if losses else None,
        "note": "Training uses next-token cross entropy; sampling argmax/multinomial is under torch.no_grad().",
    }
    output_path = write_json_report(args.output_dir / "llm_baseline_report.json", report)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
