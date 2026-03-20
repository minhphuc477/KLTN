"""
Thesis-grade ablation runner with fixed-seed paired comparisons.

Implements:
1) Core ablations: FULL vs NO_EVOLUTION / NO_WFC / NO_LOGIC
2) Requested sweeps:
   - VQ codebook size (128/512/2048) via categorical codebook cap
   - latent diffusion vs categorical
   - conditioning with/without TPE
   - logic guidance strength sweep
   - WFC on/off
3) Significance reporting (paired bootstrap CI + random-sign permutation p-value)
4) Multiple-comparison control (Benjamini-Hochberg FDR-adjusted p-values)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

# Ensure repository root is importable when script is executed directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark_suite import (
    extract_graph_descriptor,
    load_vglc_reference_graphs,
    load_vglc_reference_rooms,
)
from src.generation.evolutionary_director import mission_graph_to_networkx
from src.generation.evolutionary_director import networkx_to_mission_graph
from src.generation.grammar import Difficulty, MissionGrammar
from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
from src.simulation.cognitive_bounded_search import solve_with_cbs
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    name: str
    use_evolution: bool = True
    use_wfc: bool = True
    logic_guidance_scale: float = 1.0
    latent_sampler: str = "diffusion"  # diffusion | categorical
    categorical_codebook_size: Optional[int] = None
    use_tpe: bool = True


def _tile_distribution(grids: Sequence[np.ndarray]) -> Dict[int, float]:
    counts: Dict[int, int] = {}
    total = 0
    for grid in grids:
        arr = np.asarray(grid, dtype=np.int32)
        unique, freq = np.unique(arr, return_counts=True)
        for k, v in zip(unique.tolist(), freq.tolist()):
            counts[int(k)] = counts.get(int(k), 0) + int(v)
            total += int(v)
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def _kl_divergence(reference: Dict[int, float], generated: Dict[int, float], eps: float = 1e-9) -> float:
    if not reference:
        return 0.0
    keys = sorted(set(reference.keys()) | set(generated.keys()))
    p = np.array([float(reference.get(k, 0.0)) + eps for k in keys], dtype=np.float64)
    q = np.array([float(generated.get(k, 0.0)) + eps for k in keys], dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    return float(np.sum(p * np.log(p / q)))


def _descriptor_vector(G: nx.Graph) -> np.ndarray:
    d = extract_graph_descriptor(G, grammar=None)
    return np.array(
        [d.linearity, d.leniency, d.progression_complexity, d.topology_complexity],
        dtype=np.float64,
    )


def _graph_edit_distance_proxy(Ga: nx.Graph, Gb: nx.Graph) -> float:
    na = float(Ga.number_of_nodes())
    nb = float(Gb.number_of_nodes())
    ea = float(Ga.number_of_edges())
    eb = float(Gb.number_of_edges())
    node_term = abs(na - nb) / max(1.0, max(na, nb))
    edge_term = abs(ea - eb) / max(1.0, max(ea, eb))

    def _type_hist(G: nx.Graph) -> Dict[str, int]:
        h: Dict[str, int] = {}
        for _, attrs in G.nodes(data=True):
            t = str(attrs.get("type", attrs.get("label", "unknown"))).lower()
            h[t] = h.get(t, 0) + 1
        return h

    ha = _type_hist(Ga)
    hb = _type_hist(Gb)
    keys = sorted(set(ha.keys()) | set(hb.keys()))
    if keys:
        va = np.array([ha.get(k, 0) for k in keys], dtype=np.float64)
        vb = np.array([hb.get(k, 0) for k in keys], dtype=np.float64)
        va /= max(np.sum(va), 1.0)
        vb /= max(np.sum(vb), 1.0)
        type_term = float(np.mean(np.abs(va - vb)))
    else:
        type_term = 0.0

    return float(0.4 * node_term + 0.35 * edge_term + 0.25 * type_term)


def _nearest_graph_edit_distance(G: nx.Graph, refs: Sequence[nx.Graph], max_refs: int = 20) -> float:
    if not refs:
        return 0.0
    candidates = list(refs[: max(1, int(max_refs))])
    dists = [_graph_edit_distance_proxy(G, R) for R in candidates]
    return float(min(dists)) if dists else 0.0


def _pairwise_diversity(vectors: Sequence[np.ndarray]) -> float:
    if len(vectors) < 2:
        return 0.0
    arr = np.stack(vectors, axis=0)
    total = 0.0
    count = 0
    for i in range(arr.shape[0]):
        for j in range(i + 1, arr.shape[0]):
            total += float(np.linalg.norm(arr[i] - arr[j]) / np.sqrt(arr.shape[1]))
            count += 1
    return float(total / max(1, count))


def _paired_bootstrap_ci(
    deltas: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if deltas.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = np.mean(deltas[idx], axis=1)
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return low, high


def _paired_sign_permutation_pvalue(
    deltas: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> float:
    if deltas.size == 0:
        return 1.0
    observed = abs(float(np.mean(deltas)))
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    abs_d = np.abs(deltas)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, deltas.size))
    perm_means = np.mean(signs * abs_d[None, :], axis=1)
    p = (1.0 + float(np.sum(np.abs(perm_means) >= observed))) / float(n_perm + 1)
    return float(p)


def _benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR-adjusted p-values (q-values).
    """
    arr = np.asarray([float(p) for p in p_values], dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        return []
    order = np.argsort(arr)
    ranked = arr[order]
    q = np.zeros(n, dtype=np.float64)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = float(i + 1)
        raw = float(ranked[i]) * float(n) / rank
        prev = min(prev, raw)
        q[i] = prev
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return [float(v) for v in out.tolist()]


class AblationStudy:
    def __init__(
        self,
        *,
        output_dir: Path,
        data_root: Path,
        num_rooms: int,
        target_curve: Sequence[float],
        diffusion_steps: int,
        cbs_timeout: int,
        evolution_population: int,
        evolution_generations: int,
        max_runtime_sec: Optional[float] = None,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_root = data_root
        self.num_rooms = int(num_rooms)
        self.target_curve = list(float(v) for v in target_curve)
        self.diffusion_steps = int(diffusion_steps)
        self.cbs_timeout = int(cbs_timeout)
        self.evolution_population = int(evolution_population)
        self.evolution_generations = int(evolution_generations)
        self.max_runtime_sec = float(max_runtime_sec) if max_runtime_sec is not None else None

        self.reference_graphs = load_vglc_reference_graphs(self.data_root, limit=64)
        ref_rooms = load_vglc_reference_rooms(self.data_root, max_rooms=256)
        self.reference_tile_dist = _tile_distribution(ref_rooms)
        self.reference_vectors = (
            np.stack([_descriptor_vector(g) for g in self.reference_graphs], axis=0)
            if self.reference_graphs
            else np.zeros((0, 4), dtype=np.float64)
        )

        self._pipeline: Optional[NeuralSymbolicDungeonPipeline] = None
        self._constraint_grammar = MissionGrammar(seed=2026)

    def _get_pipeline(self) -> NeuralSymbolicDungeonPipeline:
        if self._pipeline is None:
            self._pipeline = NeuralSymbolicDungeonPipeline(
                vqvae_checkpoint=None,
                diffusion_checkpoint=None,
                logic_net_checkpoint=None,
                device="auto",
                use_learned_refiner_rules=True,
                enable_logging=False,
            )
        return self._pipeline

    def _build_non_evolution_graph(self, seed: int) -> nx.Graph:
        grammar = MissionGrammar(seed=seed)
        graph = grammar.generate(
            difficulty=Difficulty.MEDIUM,
            num_rooms=self.num_rooms,
            max_keys=max(1, self.num_rooms // 4),
            validate_all=True,
        )
        return mission_graph_to_networkx(graph)

    def _optimal_and_cbs_metrics(self, grid: np.ndarray, seed: int) -> Tuple[bool, float, float, float]:
        optimal_success = False
        optimal_len = 0
        cbs_success = False
        cbs_len = 0
        confusion_index = float("nan")
        try:
            env = ZeldaLogicEnv(semantic_grid=grid)
            astar = StateSpaceAStar(env, timeout=200000, search_mode="astar")
            optimal_success, optimal_path, _ = astar.solve()
            optimal_len = len(optimal_path or [])
        except Exception:
            optimal_success = False
            optimal_len = 0

        try:
            cbs_success, cbs_path, _, cbs_metrics = solve_with_cbs(
                grid,
                persona="balanced",
                timeout=self.cbs_timeout,
                seed=seed,
            )
            cbs_len = len(cbs_path or [])
            confusion_index = float(cbs_metrics.confusion_index)
            if not cbs_success:
                cbs_len = 0
        except Exception:
            cbs_len = 0

        confusion_ratio = float("nan")
        if optimal_success and optimal_len > 0 and cbs_success and cbs_len > 0:
            confusion_ratio = float(cbs_len / max(1, optimal_len))

        if optimal_success and optimal_len > 0 and cbs_len > 0:
            path_optimal = float(optimal_len / max(1, cbs_len))
        else:
            path_optimal = 0.0
        return bool(optimal_success), float(confusion_ratio), float(path_optimal), float(confusion_index)

    def _vq_reconstruction_error(self, pipeline: NeuralSymbolicDungeonPipeline, room_grid: np.ndarray) -> float:
        try:
            num_classes = int(getattr(pipeline.vqvae, "num_classes", 44))
            h, w = room_grid.shape
            x = np.zeros((1, num_classes, h, w), dtype=np.float32)
            clipped = np.clip(room_grid.astype(np.int64), 0, num_classes - 1)
            for r in range(h):
                for c in range(w):
                    x[0, int(clipped[r, c]), r, c] = 1.0
            xt = torch.from_numpy(x).to(pipeline.device)
            with torch.no_grad():
                z_q, _ = pipeline.vqvae.encode(xt)
                logits = pipeline.vqvae.decode(z_q, target_size=(h, w))
                recon = logits.argmax(dim=1).detach().cpu().numpy()[0]
            return float(np.mean(recon != clipped))
        except Exception:
            return float("nan")

    def _run_single(self, cfg: ExperimentConfig, seed: int) -> Dict[str, Any]:
        started = time.time()
        row: Dict[str, Any] = {
            "config": cfg.name,
            "seed": int(seed),
            "success": False,
            "solvable": False,
            "confusion_ratio": np.nan,
            "confusion_index": np.nan,
            "path_optimal": 0.0,
            "tile_prior_kl": np.nan,
            "graph_edit_distance": np.nan,
            "generation_time_sec": np.nan,
            "novelty": np.nan,
            "reconstruction_error": np.nan,
            "constraint_valid": np.nan,
            "room_repair_rate": np.nan,
            "tiles_repaired": np.nan,
            "error": "",
        }

        try:
            pipeline = self._get_pipeline()
            mission_graph = None
            generate_topology = bool(cfg.use_evolution)
            if not cfg.use_evolution:
                mission_graph = self._build_non_evolution_graph(seed=seed)

            result = pipeline.generate_dungeon(
                mission_graph=mission_graph,
                generate_topology=generate_topology,
                target_curve=self.target_curve,
                num_rooms=self.num_rooms,
                population_size=self.evolution_population,
                generations=self.evolution_generations,
                guidance_scale=7.5,
                logic_guidance_scale=float(cfg.logic_guidance_scale),
                num_diffusion_steps=self.diffusion_steps,
                latent_sampler=cfg.latent_sampler,
                categorical_codebook_size=cfg.categorical_codebook_size,
                use_topological_positional_encoding=bool(cfg.use_tpe),
                apply_repair=bool(cfg.use_wfc),
                enable_map_elites=False,
                seed=seed,
            )

            grid = np.asarray(result.dungeon_grid, dtype=np.int32)
            graph = result.mission_graph
            desc_vec = _descriptor_vector(graph)

            tile_kl = _kl_divergence(self.reference_tile_dist, _tile_distribution([grid]))
            graph_ged = _nearest_graph_edit_distance(graph, self.reference_graphs, max_refs=24)

            novelty = 0.0
            if self.reference_vectors.size > 0:
                nearest = np.min(np.linalg.norm(self.reference_vectors - desc_vec[None, :], axis=1))
                novelty = float(nearest / np.sqrt(desc_vec.shape[0]))

            solvable, confusion_ratio, path_optimal, confusion_index = self._optimal_and_cbs_metrics(grid, seed=seed)

            first_room = next(iter(result.rooms.values())).room_grid if result.rooms else grid
            recon_error = self._vq_reconstruction_error(pipeline, np.asarray(first_room, dtype=np.int32))
            constraint_valid = float("nan")
            try:
                mission = networkx_to_mission_graph(graph)
                mission.sanitize()
                constraint_valid = float(
                    self._constraint_grammar.validate_lock_key_ordering(mission)
                    and self._constraint_grammar.validate_progression_constraints(mission)
                )
            except Exception:
                constraint_valid = float("nan")

            room_repair_rate = float(result.metrics.get("repair_rate", float("nan")))
            tiles_repaired = float(result.metrics.get("total_tiles_repaired", float("nan")))

            row.update(
                {
                    "success": True,
                    "solvable": bool(solvable),
                    "confusion_ratio": float(confusion_ratio),
                    "confusion_index": float(confusion_index),
                    "path_optimal": float(path_optimal),
                    "tile_prior_kl": float(tile_kl),
                    "graph_edit_distance": float(graph_ged),
                    "generation_time_sec": float(time.time() - started),
                    "novelty": float(novelty),
                    "reconstruction_error": float(recon_error),
                    "constraint_valid": float(constraint_valid),
                    "room_repair_rate": float(room_repair_rate),
                    "tiles_repaired": float(tiles_repaired),
                    "_descriptor_vec": desc_vec.tolist(),
                }
            )
        except Exception as e:
            row["generation_time_sec"] = float(time.time() - started)
            row["error"] = f"{type(e).__name__}: {e}"
        return row

    def run(self, configs: Sequence[ExperimentConfig], seeds: Sequence[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        records: List[Dict[str, Any]] = []
        descriptor_store: Dict[str, List[np.ndarray]] = {cfg.name: [] for cfg in configs}
        started = time.time()
        stop_early = False

        for cfg in configs:
            if stop_early:
                break
            logger.info("Running config=%s (%d seeds)", cfg.name, len(seeds))
            for seed in seeds:
                if self.max_runtime_sec is not None:
                    elapsed = float(time.time() - started)
                    if elapsed >= self.max_runtime_sec:
                        logger.warning(
                            "Stopping ablation early due to runtime budget (elapsed=%.1fs, budget=%.1fs)",
                            elapsed,
                            self.max_runtime_sec,
                        )
                        stop_early = True
                        break
                row = self._run_single(cfg, int(seed))
                vec = row.pop("_descriptor_vec", None)
                if vec is not None:
                    descriptor_store[cfg.name].append(np.asarray(vec, dtype=np.float64))
                records.append(row)

        df = pd.DataFrame(records)
        summary_rows: List[Dict[str, Any]] = []
        for cfg in configs:
            sub = df[df["config"] == cfg.name]
            summary_rows.append(
                {
                    "config": cfg.name,
                    "n": int(len(sub)),
                    "success_rate": float(sub["success"].mean()) if len(sub) > 0 else 0.0,
                    "solvability_rate": float(sub["solvable"].mean()) if len(sub) > 0 else 0.0,
                    "confusion_ratio": float(sub["confusion_ratio"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "confusion_index": float(sub["confusion_index"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "path_optimal": float(sub["path_optimal"].mean(skipna=True)) if len(sub) > 0 else 0.0,
                    "tile_prior_kl": float(sub["tile_prior_kl"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "graph_edit_distance": float(sub["graph_edit_distance"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "generation_time_sec": float(sub["generation_time_sec"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "novelty": float(sub["novelty"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "reconstruction_error": float(sub["reconstruction_error"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "constraint_valid_rate": float(sub["constraint_valid"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "room_repair_rate": float(sub["room_repair_rate"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "tiles_repaired": float(sub["tiles_repaired"].mean(skipna=True)) if len(sub) > 0 else float("nan"),
                    "diversity": float(_pairwise_diversity(descriptor_store.get(cfg.name, []))),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        return df, summary_df

    @staticmethod
    def significance_report(
        df: pd.DataFrame,
        *,
        baseline: str = "FULL",
        metrics: Optional[Sequence[str]] = None,
        seed: int = 0,
    ) -> pd.DataFrame:
        if metrics is None:
            metrics = [
                "solvable",
                "confusion_ratio",
                "confusion_index",
                "path_optimal",
                "tile_prior_kl",
                "graph_edit_distance",
                "generation_time_sec",
                "novelty",
                "reconstruction_error",
                "constraint_valid",
                "room_repair_rate",
                "tiles_repaired",
            ]

        rows: List[Dict[str, Any]] = []
        base = df[df["config"] == baseline]
        other_configs = [c for c in sorted(df["config"].unique()) if c != baseline]

        for cfg in other_configs:
            other = df[df["config"] == cfg]
            merged = base.merge(other, on="seed", suffixes=("_base", "_cfg"))
            if merged.empty:
                continue
            for i, metric in enumerate(metrics):
                bcol = f"{metric}_base"
                ccol = f"{metric}_cfg"
                if bcol not in merged.columns or ccol not in merged.columns:
                    continue
                left = merged[ccol].astype(np.float64)
                right = merged[bcol].astype(np.float64)
                deltas = (left - right).to_numpy(dtype=np.float64)
                deltas = deltas[np.isfinite(deltas)]
                if deltas.size == 0:
                    continue
                mean_delta = float(np.mean(deltas))
                ci_low, ci_high = _paired_bootstrap_ci(
                    deltas,
                    n_boot=2000,
                    alpha=0.05,
                    seed=seed + 17 * (i + 1),
                )
                p_value = _paired_sign_permutation_pvalue(
                    deltas,
                    n_perm=4000,
                    seed=seed + 31 * (i + 1),
                )
                std = float(np.std(deltas))
                effect = float(mean_delta / std) if std > 1e-9 else 0.0
                rows.append(
                    {
                        "config": cfg,
                        "metric": metric,
                        "n_pairs": int(deltas.size),
                        "delta_mean_cfg_minus_full": mean_delta,
                        "delta_ci_low": ci_low,
                        "delta_ci_high": ci_high,
                        "p_value": p_value,
                        "effect_size_d": effect,
                    }
                )
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        q_values = _benjamini_hochberg(out["p_value"].astype(float).tolist())
        out["p_value_bh_fdr"] = q_values
        out["significant_fdr_0_05"] = out["p_value_bh_fdr"] < 0.05
        return out

    def export(
        self,
        *,
        configs: Sequence[ExperimentConfig],
        seeds: Sequence[int],
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        sig_df: pd.DataFrame,
    ) -> None:
        def _fmt_table(df: pd.DataFrame) -> str:
            try:
                return df.to_markdown(index=False)
            except Exception:
                return df.to_string(index=False)

        raw_path = self.output_dir / "ablation_raw.csv"
        summary_path = self.output_dir / "ablation_summary.csv"
        sig_path = self.output_dir / "ablation_significance.csv"
        json_path = self.output_dir / "ablation_report.json"
        md_path = self.output_dir / "ablation_report.md"

        raw_df.to_csv(raw_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        sig_df.to_csv(sig_path, index=False)

        payload = {
            "configs": [asdict(c) for c in configs],
            "seeds": list(int(s) for s in seeds),
            "summary": summary_df.to_dict(orient="records"),
            "significance": sig_df.to_dict(orient="records"),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines = [
            "# Ablation Study Report",
            "",
            "## Configurations",
        ]
        for cfg in configs:
            lines.append(f"- `{cfg.name}`: {asdict(cfg)}")
        lines.extend(
            [
                "",
                "## Summary Metrics",
                "",
                _fmt_table(summary_df),
                "",
                "## Paired Significance (vs FULL)",
                "",
                _fmt_table(sig_df) if not sig_df.empty else "_No paired comparisons available_",
            ]
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info("Saved ablation outputs to %s", self.output_dir)


def build_experiment_set(include_extended: bool = True) -> List[ExperimentConfig]:
    core = [
        ExperimentConfig(name="FULL"),
        ExperimentConfig(name="NO_EVOLUTION", use_evolution=False),
        ExperimentConfig(name="NO_WFC", use_wfc=False),
        ExperimentConfig(name="NO_LOGIC", logic_guidance_scale=0.0),
    ]
    if not include_extended:
        return core

    extended = [
        ExperimentConfig(name="VQ_CODEBOOK_128", latent_sampler="categorical", categorical_codebook_size=128),
        ExperimentConfig(name="VQ_CODEBOOK_512", latent_sampler="categorical", categorical_codebook_size=512),
        ExperimentConfig(name="VQ_CODEBOOK_2048", latent_sampler="categorical", categorical_codebook_size=2048),
        ExperimentConfig(name="LATENT_DIFFUSION", latent_sampler="diffusion"),
        ExperimentConfig(name="LATENT_CATEGORICAL", latent_sampler="categorical"),
        ExperimentConfig(name="COND_NO_TPE", use_tpe=False),
        ExperimentConfig(name="LOGIC_G_0.25", logic_guidance_scale=0.25),
        ExperimentConfig(name="LOGIC_G_0.50", logic_guidance_scale=0.50),
        ExperimentConfig(name="LOGIC_G_1.50", logic_guidance_scale=1.50),
        ExperimentConfig(name="LOGIC_G_2.00", logic_guidance_scale=2.00),
    ]
    return core + extended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-seed thesis ablation protocol.")
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    parser.add_argument("--data-root", type=Path, default=Path("Data") / "The Legend of Zelda")
    parser.add_argument("--num-samples", type=int, default=8, help="Seeds per configuration")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for fixed-seed protocol")
    parser.add_argument("--num-rooms", type=int, default=8)
    parser.add_argument("--target-curve", type=str, default="0.2,0.4,0.6,0.8,0.7,0.5,0.3,0.2")
    parser.add_argument("--diffusion-steps", type=int, default=25)
    parser.add_argument("--cbs-timeout", type=int, default=120000)
    parser.add_argument("--evolution-population", type=int, default=24)
    parser.add_argument("--evolution-generations", type=int, default=30)
    parser.add_argument(
        "--max-runtime-sec",
        type=float,
        default=None,
        help="Optional wall-clock budget. If exceeded, stop and export partial results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tractable quick profile for iterative thesis experiments.",
    )
    parser.add_argument(
        "--kaggle-t4x2",
        action="store_true",
        help="Apply a Kaggle T4 x2 preset for larger fixed-seed ablation runs.",
    )
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--configs", type=str, default="", help="Comma-separated subset of config names")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    target_curve = [float(v) for v in str(args.target_curve).split(",") if str(v).strip()]
    if args.quick:
        args.num_samples = min(int(args.num_samples), 2)
        args.diffusion_steps = min(int(args.diffusion_steps), 10)
        args.evolution_population = min(int(args.evolution_population), 12)
        args.evolution_generations = min(int(args.evolution_generations), 8)
        args.cbs_timeout = min(int(args.cbs_timeout), 30000)
        if args.max_runtime_sec is None:
            args.max_runtime_sec = 420.0
        logger.info(
            "Quick profile active: samples=%d, diffusion_steps=%d, pop=%d, gens=%d, cbs_timeout=%d, max_runtime_sec=%s",
            args.num_samples,
            args.diffusion_steps,
            args.evolution_population,
            args.evolution_generations,
            args.cbs_timeout,
            str(args.max_runtime_sec),
        )
    if args.kaggle_t4x2:
        args.num_samples = max(int(args.num_samples), 12)
        args.diffusion_steps = max(int(args.diffusion_steps), 25)
        args.evolution_population = max(int(args.evolution_population), 32)
        args.evolution_generations = max(int(args.evolution_generations), 40)
        args.cbs_timeout = max(int(args.cbs_timeout), 60000)
        if args.max_runtime_sec is None:
            args.max_runtime_sec = 10800.0
        logger.info(
            "Kaggle T4 x2 profile active: samples=%d, diffusion_steps=%d, pop=%d, gens=%d, cbs_timeout=%d, max_runtime_sec=%s",
            args.num_samples,
            args.diffusion_steps,
            args.evolution_population,
            args.evolution_generations,
            args.cbs_timeout,
            str(args.max_runtime_sec),
        )
    configs = build_experiment_set(include_extended=not args.core_only)
    if args.configs.strip():
        selected = {c.strip() for c in args.configs.split(",") if c.strip()}
        configs = [cfg for cfg in configs if cfg.name in selected]
        if not configs:
            raise ValueError("No matching configs after --configs filtering.")

    seeds = [int(args.seed) + i for i in range(int(args.num_samples))]
    study = AblationStudy(
        output_dir=args.output,
        data_root=args.data_root,
        num_rooms=args.num_rooms,
        target_curve=target_curve,
        diffusion_steps=args.diffusion_steps,
        cbs_timeout=args.cbs_timeout,
        evolution_population=args.evolution_population,
        evolution_generations=args.evolution_generations,
        max_runtime_sec=args.max_runtime_sec,
    )

    raw_df, summary_df = study.run(configs=configs, seeds=seeds)
    sig_df = study.significance_report(raw_df, baseline="FULL", seed=args.seed + 999)
    study.export(
        configs=configs,
        seeds=seeds,
        raw_df=raw_df,
        summary_df=summary_df,
        sig_df=sig_df,
    )

    logger.info("Ablation complete. Output: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
