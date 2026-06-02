from __future__ import annotations

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from src.utils.distributed import (
    DistributedContext,
    average_module_parameters,
    build_torchrun_command,
    make_distributed_sampler,
)


def test_build_torchrun_command_uses_current_python_and_script_args():
    cmd = build_torchrun_command(
        script_path="main.py",
        script_args=["train", "--config", "configs/zelda_hmolqd.yaml"],
        nproc_per_node=2,
        master_port=29577,
    )

    assert cmd[1:7] == [
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        "2",
        "--master-port",
        "29577",
    ]
    assert cmd[7:] == ["main.py", "train", "--config", "configs/zelda_hmolqd.yaml"]


def test_make_distributed_sampler_returns_sampler_when_world_size_exceeds_one():
    dataset = TensorDataset(torch.arange(8))
    context = DistributedContext(enabled=True, backend="nccl", world_size=2, rank=1, local_rank=1)

    sampler = make_distributed_sampler(
        dataset,
        context=context,
        shuffle=True,
        drop_last=True,
        seed=123,
    )

    assert isinstance(sampler, DistributedSampler)
    assert sampler.num_replicas == 2
    assert sampler.rank == 1


def test_average_module_parameters_noops_without_distributed_context():
    module = torch.nn.Linear(2, 2)
    before = {key: value.detach().clone() for key, value in module.state_dict().items()}

    average_module_parameters(module, context=DistributedContext(enabled=False))

    for key, value in module.state_dict().items():
        assert torch.allclose(value, before[key])
