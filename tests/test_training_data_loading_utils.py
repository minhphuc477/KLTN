import random

import numpy as np
import pytest
import torch

from src.utils.data_loading import dataloader_runtime_kwargs, seed_dataloader_worker
from src.zelda_data.zelda_loader import RandomD4Symmetry, create_dataloader


def test_dataloader_runtime_kwargs_omits_worker_only_options_without_workers():
    kwargs = dataloader_runtime_kwargs(num_workers=0, pin_memory=True, prefetch_factor=4)

    assert kwargs == {"num_workers": 0, "pin_memory": True}


def test_dataloader_runtime_kwargs_enables_persistent_workers_when_workers_exist():
    kwargs = dataloader_runtime_kwargs(num_workers=2, pin_memory=False, prefetch_factor=4)

    assert kwargs["num_workers"] == 2
    assert kwargs["pin_memory"] is False
    assert kwargs["worker_init_fn"] is seed_dataloader_worker
    assert kwargs["persistent_workers"] is True
    assert kwargs["prefetch_factor"] == 4


def test_seed_dataloader_worker_seeds_numpy_and_python_random(monkeypatch):
    monkeypatch.setattr(torch, "initial_seed", lambda: 123456789)

    seed_dataloader_worker(0)
    numpy_value_a = float(np.random.random())
    python_value_a = float(random.random())
    seed_dataloader_worker(0)
    numpy_value_b = float(np.random.random())
    python_value_b = float(random.random())

    assert numpy_value_a == pytest.approx(numpy_value_b)
    assert python_value_a == pytest.approx(python_value_b)


def test_random_d4_symmetry_preserves_rectangular_room_shape():
    transform = RandomD4Symmetry()
    grid = torch.arange(11 * 16, dtype=torch.float32).view(1, 11, 16)

    for _ in range(64):
        augmented = transform(grid)
        assert augmented.shape == grid.shape


def test_create_dataloader_forwards_augmentation_to_dataset():
    loader = create_dataloader(
        "Data/The Legend of Zelda",
        batch_size=4,
        shuffle=False,
        drop_last=False,
        use_vglc=True,
        room_level=True,
        load_graphs=False,
        dungeon_ids=[9],
        variants=[1],
        augment=True,
    )

    assert loader.dataset.augment is True


def test_create_dataloader_rejects_graph_augmentation_without_metadata_transform():
    with pytest.raises(ValueError, match="augment=True cannot be combined with load_graphs=True"):
        create_dataloader(
            "Data/The Legend of Zelda",
            batch_size=2,
            shuffle=False,
            drop_last=False,
            use_vglc=True,
            room_level=True,
            load_graphs=True,
            dungeon_ids=[9],
            variants=[1],
            augment=True,
        )
