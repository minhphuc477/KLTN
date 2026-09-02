"""
Distributed training helpers for single-node multi-GPU runs.

The project training code uses multiple custom module methods instead of a
single monolithic `nn.Module.forward(...)`. Because of that, the lowest-risk
parallelization path is:

1. launch one process per GPU with `torchrun`
2. shard the dataset with `DistributedSampler`
3. average gradients and scalar metrics across ranks with collectives

This is compatible with Kaggle dual-T4 notebook sessions while keeping the
existing trainer structure largely intact.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0


def get_env_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_env_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", str(get_env_rank())))


def get_env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_torchrun_environment() -> bool:
    return get_env_world_size() > 1 or "LOCAL_RANK" in os.environ


def build_torchrun_command(
    *,
    script_path: str,
    script_args: Sequence[str],
    nproc_per_node: int,
    master_port: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(int(nproc_per_node)),
        "--master-port",
        str(int(master_port)),
        str(script_path),
        *[str(arg) for arg in script_args],
    ]


def maybe_launch_with_torchrun(
    *,
    enabled: bool,
    nproc_per_node: int,
    master_port: int,
    script_path: str,
    script_args: Sequence[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> bool:
    if not enabled:
        return False
    if is_torchrun_environment():
        return False
    requested = int(max(1, nproc_per_node))
    if requested <= 1:
        return False
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed GPU launch requested, but CUDA is not available.")
    available = int(torch.cuda.device_count())
    if requested > available:
        raise RuntimeError(
            f"Requested nproc_per_node={requested}, but only {available} CUDA device(s) are visible."
        )

    cmd = build_torchrun_command(
        script_path=script_path,
        script_args=script_args,
        nproc_per_node=requested,
        master_port=int(master_port),
    )
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    subprocess.run(cmd, check=True, env=env)
    return True


def initialize_distributed(
    *,
    enabled: bool,
    backend: str = "nccl",
) -> DistributedContext:
    requested = bool(enabled) or is_torchrun_environment()
    if not requested:
        return DistributedContext(enabled=False, backend=str(backend), world_size=1, rank=0, local_rank=0)

    world_size = get_env_world_size()
    rank = get_env_rank()
    local_rank = get_env_local_rank()

    resolved_backend = str(backend).strip().lower()
    if resolved_backend == "nccl" and not torch.cuda.is_available():
        resolved_backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        init_kwargs = {
            "backend": resolved_backend,
            "init_method": "env://",
        }
        if torch.cuda.is_available():
            init_kwargs["device_id"] = local_rank
        try:
            dist.init_process_group(**init_kwargs)
        except TypeError:
            init_kwargs.pop("device_id", None)
            dist.init_process_group(**init_kwargs)

    return DistributedContext(
        enabled=True,
        backend=resolved_backend,
        world_size=int(world_size),
        rank=int(rank),
        local_rank=int(local_rank),
    )


def destroy_distributed(context: Optional[DistributedContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_barrier(context: Optional[DistributedContext]) -> None:
    if context is None or not context.enabled:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_ranks_true(
    value: bool,
    *,
    device: torch.device,
    context: Optional[DistributedContext],
) -> bool:
    """Return true only when every distributed worker reports true."""
    if context is None or not context.enabled or int(context.world_size) <= 1:
        return bool(value)
    flag = torch.tensor(
        1 if bool(value) else 0,
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def resolve_device(device: str, context: Optional[DistributedContext]) -> torch.device:
    requested = str(device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available() and requested in {"auto", "cuda"}:
        if context is not None and context.enabled:
            return torch.device("cuda", int(context.local_rank))
        return torch.device("cuda")
    return torch.device("cpu")


def make_distributed_sampler(
    dataset,
    *,
    context: Optional[DistributedContext],
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> Optional[DistributedSampler]:
    if context is None or not context.enabled or int(context.world_size) <= 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=int(context.world_size),
        rank=int(context.rank),
        shuffle=bool(shuffle),
        drop_last=bool(drop_last),
        seed=int(seed),
    )


def average_gradients(
    modules: Iterable[torch.nn.Module],
    *,
    context: Optional[DistributedContext],
) -> None:
    if context is None or not context.enabled or int(context.world_size) <= 1:
        return

    seen = set()
    for module in modules:
        for param in module.parameters():
            key = int(param.data_ptr())
            if key in seen:
                continue
            seen.add(key)

            has_grad = torch.tensor(
                1.0 if param.grad is not None else 0.0,
                device=param.device,
                dtype=torch.float32,
            )
            dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
            if float(has_grad.item()) <= 0.0:
                continue

            if param.grad is None:
                reduced_grad = torch.zeros_like(param.data)
                dist.all_reduce(reduced_grad, op=dist.ReduceOp.SUM)
                reduced_grad.div_(float(context.world_size))
                param.grad = reduced_grad
            else:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data.div_(float(context.world_size))


def average_module_parameters(
    module: torch.nn.Module,
    *,
    context: Optional[DistributedContext],
) -> None:
    """Average module parameters and buffers across distributed ranks."""
    if context is None or not context.enabled or int(context.world_size) <= 1:
        return

    with torch.no_grad():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if tensor.data.is_floating_point() or tensor.data.is_complex():
                dist.all_reduce(tensor.data, op=dist.ReduceOp.SUM)
                tensor.data.div_(float(context.world_size))
            else:
                dist.broadcast(tensor.data, src=0)


def reduce_scalar_metrics(
    metrics: Dict[str, float],
    *,
    device: torch.device,
    context: Optional[DistributedContext],
    average: bool = True,
) -> Dict[str, float]:
    """All-reduce scalar metrics across distributed ranks.

    **Key-agreement safety**: ``dist.all_reduce`` requires all ranks to submit
    identically-shaped tensors.  If one rank computes a different set of metric
    keys (e.g. because a loss term was skipped on that rank), the tensor sizes
    will disagree, causing NCCL to crash or Gloo to hang indefinitely.

    This function negotiates a canonical key order by broadcasting rank-0's
    sorted key list to all ranks before the collective.  Ranks that observe a
    key mismatch raise an informative ``RuntimeError`` immediately.
    """
    if context is None or not context.enabled or int(context.world_size) <= 1:
        return {str(k): float(v) for k, v in metrics.items()}

    if not metrics:
        return {}

    import pickle as _pickle

    # --- Step 1: Broadcast the canonical key list from rank 0 ---
    # Encode the key list as bytes so we can broadcast it as a fixed-size tensor.
    local_keys = sorted(metrics.keys())

    if context.rank == 0:
        key_bytes = _pickle.dumps(local_keys)
        size_tensor = torch.tensor([len(key_bytes)], device=device, dtype=torch.int64)
    else:
        size_tensor = torch.zeros(1, device=device, dtype=torch.int64)

    dist.broadcast(size_tensor, src=0)
    key_buf_size = int(size_tensor.item())

    if context.rank == 0:
        key_buf = torch.frombuffer(bytearray(key_bytes), dtype=torch.uint8).to(device)
    else:
        key_buf = torch.zeros(key_buf_size, device=device, dtype=torch.uint8)

    dist.broadcast(key_buf, src=0)
    canonical_keys: List[str] = _pickle.loads(bytes(key_buf.cpu().numpy().tobytes()))

    # --- Step 2: Validate local keys against the canonical set ---
    if local_keys != canonical_keys:
        raise RuntimeError(
            f"[Rank {context.rank}] reduce_scalar_metrics: local metric keys "
            f"{local_keys} do not match rank-0 keys {canonical_keys}.  "
            f"Ensure all ranks compute the same set of metrics each step, or "
            f"fill missing keys with 0.0 before calling reduce_scalar_metrics."
        )

    # --- Step 3: All-reduce the values tensor (shapes guaranteed to match) ---
    values = torch.tensor(
        [float(metrics[key]) for key in canonical_keys],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    if average:
        values.div_(float(context.world_size))
    return {key: float(values[idx].item()) for idx, key in enumerate(canonical_keys)}
