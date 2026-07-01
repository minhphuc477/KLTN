from types import SimpleNamespace
import copy
import json

import torch

from src.core.latent_diffusion import TimestepEmbedding
from src.optimization.lcm_lora import (
    DEFAULT_LORA_TARGETS,
    LoRALinear,
    freeze_non_lora_parameters,
    inject_lora_into_model,
)
from src.train_lcm import ConsistencyLoRATrainer, reevaluate_fast_sampler_checkpoint_candidates


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


class _EvalModeStub:
    def eval(self):
        return self

    def train(self):
        return self


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
    trainer.teacher = copy.deepcopy(student)
    for param in trainer.teacher.parameters():
        param.requires_grad = False
    trainer.device = torch.device("cpu")
    trainer.config = _FastSamplerTestConfig(
        base_diffusion_checkpoint="base_diffusion.pth",
        num_inference_steps=4,
        ema_decay=0.95,
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
    assert "ema_target_lora_state_dict" in payload
    assert payload["epoch"] == 5
    assert payload["global_step"] == 21
    assert payload["metadata"]["num_inference_steps"] == 4
    assert payload["metadata"]["target_update"] == "ema"
    assert payload["metadata"]["distillation_objective"] == "trajectory_consistency"
    assert payload["metadata"]["topology_anchor_policy"]["version"]

    with torch.no_grad():
        tracked_param.zero_()
    trainer.epoch = 0
    trainer.global_step = 0

    trainer.load_checkpoint(str(resume_path))

    assert trainer.epoch == 5
    assert trainer.global_step == 21
    assert torch.allclose(tracked_param, original)


def test_fast_sampler_ema_target_updates_from_online_student():
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
    trainer.teacher = copy.deepcopy(student)
    trainer.config = _FastSamplerTestConfig(ema_decay=0.75)

    student_param = next(p for name, p in trainer.student.named_parameters() if ".lora." in name)
    teacher_param = next(p for name, p in trainer.teacher.named_parameters() if ".lora." in name)
    with torch.no_grad():
        student_param.fill_(4.0)
        teacher_param.zero_()

    trainer._update_teacher_ema()

    assert torch.allclose(teacher_param, torch.ones_like(teacher_param))


def test_fast_sampler_timestep_pairs_always_advance_to_lower_noise():
    trainer = ConsistencyLoRATrainer.__new__(ConsistencyLoRATrainer)
    trainer.device = torch.device("cpu")
    trainer.target_timesteps = torch.tensor([999, 749, 500, 250, 0])

    current, previous = trainer._sample_batch_timestep_pairs(128)

    assert current.shape == previous.shape == (128,)
    assert torch.all(current > previous)


def test_teacher_ddim_step_anchors_zero_timestep_to_clean_prediction():
    trainer = ConsistencyLoRATrainer.__new__(ConsistencyLoRATrainer)
    trainer.ode_teacher = SimpleNamespace(alphas_cumprod=torch.tensor([0.25, 0.5], dtype=torch.float32))
    pred_x0 = torch.tensor([[[[0.75]]]], dtype=torch.float32)
    pred_noise = torch.tensor([[[[10.0]]]], dtype=torch.float32)
    t_previous = torch.tensor([0], dtype=torch.long)

    x_previous = ConsistencyLoRATrainer._teacher_ddim_step(trainer, pred_x0, pred_noise, t_previous)

    assert torch.allclose(x_previous, pred_x0)


def test_fast_sampler_deployable_adapter_exports_ema_target(monkeypatch):
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
    trainer.teacher = copy.deepcopy(student)
    trainer.config = _FastSamplerTestConfig(
        base_diffusion_checkpoint="base_diffusion.pth",
        num_inference_steps=4,
        ema_decay=0.95,
        lora_rank=2,
        lora_alpha=4.0,
    )
    with torch.no_grad():
        for name, target_param in trainer.teacher.named_parameters():
            if ".lora." in name:
                target_param.fill_(3.0)

    captured = {}
    monkeypatch.setattr("src.train_lcm.save_fast_sampler_checkpoint", lambda path, **kwargs: captured.update(kwargs))

    trainer.save_checkpoint("unused.pth")

    assert captured["lora_state_dict"]
    assert all(
        torch.allclose(param, torch.full_like(param, 3.0))
        for param in captured["lora_state_dict"].values()
    )
    assert captured["adapter_export"] == "ema_target"


def test_injected_lora_matches_wrapped_linear_device_and_dtype():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = _TinyStudent().to(device=device, dtype=torch.float64)

    inject_lora_into_model(
        student.denoiser,
        rank=2,
        alpha=4.0,
        target_modules=DEFAULT_LORA_TARGETS,
    )

    lora_modules = [module for module in student.modules() if isinstance(module, LoRALinear)]
    assert lora_modules

    for module in lora_modules:
        assert module.lora.lora_A.device == module.base_layer.weight.device
        assert module.lora.lora_B.device == module.base_layer.weight.device
        assert module.lora.lora_A.dtype == module.base_layer.weight.dtype
        assert module.lora.lora_B.dtype == module.base_layer.weight.dtype


def test_timestep_embedding_matches_mlp_dtype():
    embed = TimestepEmbedding(dim=8).to(dtype=torch.float64)
    t = torch.tensor([1, 7], dtype=torch.long)

    out = embed(t)

    assert out.dtype == torch.float64


def test_fast_sampler_validate_respects_max_batches_and_eval_seed(monkeypatch):
    trainer = ConsistencyLoRATrainer.__new__(ConsistencyLoRATrainer)
    trainer.student = _EvalModeStub()
    trainer.device = torch.device("cpu")

    calls = []

    def _fake_distill_step_eval(real_maps, graph_list=None, *, batch_index=0, eval_seed=None):
        calls.append((int(batch_index), None if eval_seed is None else int(eval_seed)))
        return {
            "val_loss": float(batch_index + 1),
            "val_x0_loss": 0.0,
            "val_prediction_loss": 0.0,
            "val_decode_ce_loss": float(batch_index + 3),
        }

    monkeypatch.setattr(trainer, "distill_step_eval", _fake_distill_step_eval)

    batches = [torch.zeros(1, 1, 1, 1) for _ in range(3)]
    metrics = trainer.validate(batches, max_batches=2, eval_seed=99)

    assert calls == [(0, 99), (1, 99)]
    assert metrics["val_loss"] == 1.5
    assert metrics["val_decode_ce_loss"] == 3.5


def test_reselect_fast_sampler_checkpoint_prefers_lowest_deterministic_val_loss(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "fast_sampler"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "fast_sampler_best.pth").write_bytes(b"best")
    (checkpoint_dir / "fast_sampler_final.pth").write_bytes(b"final")

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.base_bundle = SimpleNamespace(config=SimpleNamespace(data_dir="unused"))
            self.loaded = None

        def load_checkpoint(self, path):
            self.loaded = path
            return {}

        def validate(self, dataloader, *, max_batches=None, eval_seed=None):
            filename = str(self.loaded)
            score = 0.2 if filename.endswith("fast_sampler_best.pth") else 0.1
            return {
                "val_loss": score,
                "val_x0_loss": score,
                "val_prediction_loss": score,
                "val_decode_ce_loss": score + 0.05,
            }

        def save_checkpoint(self, path, metrics=None):
            torch.save({"metrics": dict(metrics or {}), "selected_from": self.loaded}, path)

    monkeypatch.setattr("src.train_lcm.ConsistencyLoRATrainer", _FakeTrainer)
    monkeypatch.setattr(
        "src.train_lcm._create_fast_sampler_dataloaders",
        lambda config, data_dir: ([], [], "val", 8, 2),
    )

    config = _FastSamplerTestConfig(
        checkpoint_dir=str(checkpoint_dir),
        data_dir="unused",
        seed=123,
        validation_max_batches=4,
        best_checkpoint_metric="val_loss",
    )

    ranking = reevaluate_fast_sampler_checkpoint_candidates(config, checkpoint_dir=str(checkpoint_dir))

    assert ranking["selected_checkpoint"] == "fast_sampler_final.pth"
    assert (checkpoint_dir / "fast_sampler_best_reselected.pth").exists()
    saved = torch.load(checkpoint_dir / "fast_sampler_best_reselected.pth", map_location="cpu", weights_only=False)
    assert str(saved["selected_from"]).endswith("fast_sampler_final.pth")

    ranking_json = json.loads((checkpoint_dir / "fast_sampler_checkpoint_ranking.json").read_text(encoding="utf-8"))
    assert ranking_json["selected_checkpoint"] == "fast_sampler_final.pth"


def test_reselect_fast_sampler_checkpoint_can_rank_by_decode_ce(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "fast_sampler"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "fast_sampler_best.pth").write_bytes(b"best")
    (checkpoint_dir / "fast_sampler_final.pth").write_bytes(b"final")

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.base_bundle = SimpleNamespace(config=SimpleNamespace(data_dir="unused"))
            self.loaded = None

        def load_checkpoint(self, path):
            self.loaded = path
            return {}

        def validate(self, dataloader, *, max_batches=None, eval_seed=None):
            filename = str(self.loaded)
            if filename.endswith("fast_sampler_best.pth"):
                return {
                    "val_loss": 0.20,
                    "val_x0_loss": 0.20,
                    "val_prediction_loss": 0.20,
                    "val_decode_ce_loss": 0.30,
                }
            return {
                "val_loss": 0.24,
                "val_x0_loss": 0.24,
                "val_prediction_loss": 0.24,
                "val_decode_ce_loss": 0.10,
            }

        def save_checkpoint(self, path, metrics=None):
            torch.save({"metrics": dict(metrics or {}), "selected_from": self.loaded}, path)

    monkeypatch.setattr("src.train_lcm.ConsistencyLoRATrainer", _FakeTrainer)
    monkeypatch.setattr(
        "src.train_lcm._create_fast_sampler_dataloaders",
        lambda config, data_dir: ([], [], "val", 8, 2),
    )

    config = _FastSamplerTestConfig(
        checkpoint_dir=str(checkpoint_dir),
        data_dir="unused",
        seed=123,
        validation_max_batches=4,
        best_checkpoint_metric="val_decode_ce_loss",
    )

    ranking = reevaluate_fast_sampler_checkpoint_candidates(config, checkpoint_dir=str(checkpoint_dir))

    assert ranking["metric_name"] == "val_decode_ce_loss"


def test_reselect_fast_sampler_checkpoint_can_rank_by_topology_decode_ce(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "fast_sampler"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "fast_sampler_best.pth").write_bytes(b"best")
    (checkpoint_dir / "fast_sampler_final.pth").write_bytes(b"final")

    class _FakeTrainer:
        def __init__(self, config):
            self.config = config
            self.base_bundle = SimpleNamespace(config=SimpleNamespace(data_dir="unused"))
            self.loaded = None

        def load_checkpoint(self, path):
            self.loaded = path
            return {}

        def validate(self, dataloader, *, max_batches=None, eval_seed=None):
            filename = str(self.loaded)
            if filename.endswith("fast_sampler_best.pth"):
                return {
                    "val_loss": 0.20,
                    "val_x0_loss": 0.20,
                    "val_prediction_loss": 0.20,
                    "val_decode_ce_loss": 0.16,
                    "val_topology_decode_ce_loss": 0.28,
                }
            return {
                "val_loss": 0.24,
                "val_x0_loss": 0.24,
                "val_prediction_loss": 0.24,
                "val_decode_ce_loss": 0.18,
                "val_topology_decode_ce_loss": 0.08,
            }

        def save_checkpoint(self, path, metrics=None):
            torch.save({"metrics": dict(metrics or {}), "selected_from": self.loaded}, path)

    monkeypatch.setattr("src.train_lcm.ConsistencyLoRATrainer", _FakeTrainer)
    monkeypatch.setattr(
        "src.train_lcm._create_fast_sampler_dataloaders",
        lambda config, data_dir: ([], [], "val", 8, 2),
    )

    config = _FastSamplerTestConfig(
        checkpoint_dir=str(checkpoint_dir),
        data_dir="unused",
        seed=123,
        validation_max_batches=4,
        best_checkpoint_metric="val_topology_decode_ce_loss",
    )

    ranking = reevaluate_fast_sampler_checkpoint_candidates(config, checkpoint_dir=str(checkpoint_dir))

    assert ranking["metric_name"] == "val_topology_decode_ce_loss"
    assert ranking["selected_checkpoint"] == "fast_sampler_final.pth"
    assert ranking["selected_checkpoint"] == "fast_sampler_final.pth"
