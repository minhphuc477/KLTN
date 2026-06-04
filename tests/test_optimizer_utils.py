import torch
from torch import nn

from src.utils.optimization import adamw_decay_param_groups, adamw_decay_param_groups_for_modules


def test_adamw_decay_param_groups_excludes_bias_and_norm_scales():
    module = nn.Sequential(
        nn.Linear(4, 8),
        nn.LayerNorm(8),
        nn.Conv2d(1, 2, kernel_size=3),
    )

    groups = adamw_decay_param_groups(
        module.named_parameters(),
        weight_decay=0.1,
        base_name="test",
    )
    by_decay = {float(group["weight_decay"]): set(map(id, group["params"])) for group in groups}

    assert 0.1 in by_decay
    assert 0.0 in by_decay
    assert id(module[0].weight) in by_decay[0.1]
    assert id(module[2].weight) in by_decay[0.1]
    assert id(module[0].bias) in by_decay[0.0]
    assert id(module[1].weight) in by_decay[0.0]
    assert id(module[1].bias) in by_decay[0.0]


def test_adamw_decay_param_groups_for_modules_deduplicates_tied_parameters():
    shared = nn.Linear(4, 4)
    wrapper_a = nn.Sequential(shared)
    wrapper_b = nn.Sequential(shared)

    groups = adamw_decay_param_groups_for_modules(
        (("a", wrapper_a), ("b", wrapper_b)),
        weight_decay=0.2,
    )

    all_params = [param for group in groups for param in group["params"]]
    assert len(all_params) == len({id(param) for param in all_params})
    assert sum(param is shared.weight for param in all_params) == 1
    assert sum(param is shared.bias for param in all_params) == 1
