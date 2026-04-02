from types import SimpleNamespace

import torch

from src.optimization.lcm_lora import DEFAULT_LORA_TARGETS, freeze_non_lora_parameters, inject_lora_into_model
from src.train_lcm import ConsistencyLoRATrainer


class _TinyCrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4, bias=False)
        self.k = torch.nn.Linear(4, 4, bias=False)
        self.v = torch.nn.Linear(4, 4, bias=False)


class _TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn = _TinyCrossAttention()


class _TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.denoiser = _TinyDenoiser()


class _FastSamplerTestConfig(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def test_fast_sampler_resume_checkpoint_round_trip(tmp_path):
    student = _TinyStudent()
    inject_lora_into_model(
        student.denoiser,
        rank=2,
        alpha=4.0,
        target_modules=DEFAULT_LORA_TARGETS,
    )
    freeze_non_lora_parameters(student)

    trainer = ConsistencyLoRATrainer.__new__(ConsistencyLoRATrainer)
    trainer.student = student
    trainer.device = torch.device("cpu")
    trainer.config = _FastSamplerTestConfig(
        base_diffusion_checkpoint="base_diffusion.pth",
        num_inference_steps=4,
        lora_rank=2,
        lora_alpha=4.0,
    )
    trainer.optimizer = torch.optim.AdamW(
        [p for p in trainer.student.parameters() if p.requires_grad],
        lr=1e-3,
    )
    trainer.epoch = 5
    trainer.global_step = 21

    tracked_param = next(p for name, p in trainer.student.named_parameters() if ".lora." in name)
    original = tracked_param.detach().clone()

    resume_path = tmp_path / "fast_sampler_resume.pth"
    trainer.save_resume_checkpoint(str(resume_path), {"val_loss": 0.75})

    payload = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert "optimizer_state_dict" in payload
    assert payload["epoch"] == 5
    assert payload["global_step"] == 21
    assert payload["metadata"]["num_inference_steps"] == 4

    with torch.no_grad():
        tracked_param.zero_()
    trainer.epoch = 0
    trainer.global_step = 0

    trainer.load_checkpoint(str(resume_path))

    assert trainer.epoch == 5
    assert trainer.global_step == 21
    assert torch.allclose(tracked_param, original)
