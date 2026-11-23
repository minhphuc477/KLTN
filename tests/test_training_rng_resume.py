"""Fresh stochastic training must agree across an epoch-boundary restart."""

import io
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from src.utils.training_rng import capture_training_rng, restore_training_rng


def test_saved_rng_reproduces_optimizer_trajectory_and_sampler_draws():
    prior = capture_training_rng()
    try:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        dataset = TensorDataset(torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24)
        def setup():
            model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.4), torch.nn.Linear(4, 1))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            sampler = RandomSampler(dataset, replacement=True, num_samples=8, generator=torch.Generator().manual_seed(4))
            return model, optimizer, sampler
        def epoch(model, optimizer, sampler):
            for (batch,) in DataLoader(dataset, batch_size=2, sampler=sampler):
                optimizer.zero_grad()
                target = torch.full((2, 1), random.random() + float(np.random.random()))
                (model(batch) - target).square().mean().backward()
                optimizer.step()
        model, optimizer, sampler = setup()
        epoch(model, optimizer, sampler)
        stream = io.BytesIO()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "rng": capture_training_rng(sampler=sampler)}, stream)
        epoch(model, optimizer, sampler)
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        resumed, resumed_optimizer, resumed_sampler = setup()
        stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        resumed.load_state_dict(payload["model"])
        resumed_optimizer.load_state_dict(payload["optimizer"])
        restore_training_rng(payload["rng"], sampler=resumed_sampler)
        epoch(resumed, resumed_optimizer, resumed_sampler)
        assert all(torch.equal(expected[name], value) for name, value in resumed.state_dict().items())
    finally:
        restore_training_rng(prior)


def test_resume_rejects_missing_rng_and_sampler_mismatch():
    with pytest.raises(ValueError, match="complete supported"):
        restore_training_rng({})
    state = capture_training_rng(sampler=SimpleNamespace(generator=torch.Generator()))
    with pytest.raises(ValueError, match="sampler generator"):
        restore_training_rng(state)
