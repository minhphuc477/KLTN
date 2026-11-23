"""Epoch-boundary RNG snapshots compatible with restricted torch loading."""

import random

import numpy as np
import torch


def capture_training_rng(*, sampler=None):
    numpy_state = np.random.get_state()
    generator = getattr(sampler, "generator", None)
    return {
        "version": 1,
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_initialized() else [],
        "sampler_generator": generator.get_state().clone() if generator is not None else None,
    }


def restore_training_rng(state, *, sampler=None):
    """Restore after model construction, before creating the next loader iterator.

    Workers must be recreated each epoch; persistent worker RNG state is not
    accessible from the training process. Mid-epoch resume is not supported.
    """
    required = {"version", "python", "numpy", "torch_cpu", "torch_cuda", "sampler_generator"}
    if not isinstance(state, dict) or not required.issubset(state) or state["version"] != 1:
        raise ValueError("Resume checkpoint lacks a complete supported training RNG snapshot.")
    generator = getattr(sampler, "generator", None)
    saved_generator = state["sampler_generator"]
    if (generator is None) != (saved_generator is None):
        raise ValueError("Resume sampler generator differs from the saved training sampler.")
    cuda_states = state["torch_cuda"]
    if len(cuda_states) != (torch.cuda.device_count() if torch.cuda.is_initialized() else 0):
        raise ValueError("Resume CUDA RNG device count differs from the saved process.")
    # Validate using isolated generators before changing the live RNG streams.
    python_rng = random.Random()
    python_rng.setstate(state["python"])
    numpy_state = state["numpy"]
    numpy_rng = np.random.RandomState()
    numpy_rng.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    cpu_rng = torch.Generator()
    cpu_rng.set_state(state["torch_cpu"].cpu())
    if saved_generator is not None:
        torch.Generator().set_state(saved_generator.cpu())
    random.setstate(python_rng.getstate())
    np.random.set_state(numpy_rng.get_state())
    torch.set_rng_state(cpu_rng.get_state())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    if generator is not None:
        generator.set_state(saved_generator.cpu())
