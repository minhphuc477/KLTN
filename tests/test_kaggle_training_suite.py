import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(relative_path: str, module_name: str):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_kaggle_config_presets_wire_vqvae2_and_stage_branches():
    mod = _load_script_module("kaggle/hmolqd_training_suite/make_kaggle_config.py", "make_kaggle_config_test")
    config = {}

    mod._apply_profile(config, "t4x2")
    assert config["dataset"]["batch_size"] == 4
    assert config["distributed"]["backend"] == "nccl"
    assert config["distributed"]["nproc_per_node"] == 2
    assert config["distributed"]["cuda_visible_devices"] == "0,1"

    mod._apply_tokenizer(config, "vqvae2")
    assert config["vqvae"]["architecture"] == "vqvae2"
    assert config["vqvae"]["codebook_size"] == 256
    assert config["vqvae"]["top_codebook_size"] == 128
    assert config["vqvae"]["top_latent_dim"] == 32

    mod._apply_stage_branch(config, "stage_tokens_only")
    for section in mod.STAGE_SECTIONS:
        assert config[section]["puzzle_stage_conditioning_enabled"] is True
        assert config[section]["puzzle_stage_topology_enabled"] is False
        assert config[section]["puzzle_stage_semantics_loss_weight"] == 0.25


def test_kaggle_artifact_manifest_packages_summaries_and_final_checkpoints(tmp_path):
    mod = _load_script_module(
        "kaggle/hmolqd_training_suite/collect_training_artifacts.py",
        "collect_training_artifacts_test",
    )
    summary_path = tmp_path / "tokenizers" / "vqvae2" / "checkpoints" / "vqvae" / "vqvae_run_summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps({"epoch_to_best": 1, "codebook_utilization": 0.5}), encoding="utf-8")

    best_checkpoint = tmp_path / "downstream" / "run" / "checkpoints" / "diffusion" / "best_model.pth"
    retained_checkpoint = tmp_path / "downstream" / "run" / "checkpoints" / "diffusion" / "checkpoint_epoch_1.pth"
    best_checkpoint.parent.mkdir(parents=True)
    best_checkpoint.write_bytes(b"best")
    retained_checkpoint.write_bytes(b"retained")

    manifest = mod.build_manifest(tmp_path, include_checkpoints=True)
    packaged = {record["path"].replace("\\", "/") for record in manifest["packaged_files"]}
    checkpoints = {record["path"].replace("\\", "/") for record in manifest["checkpoints"]}

    assert "tokenizers/vqvae2/checkpoints/vqvae/vqvae_run_summary.json" in packaged
    assert "downstream/run/checkpoints/diffusion/best_model.pth" in packaged
    assert "downstream/run/checkpoints/diffusion/checkpoint_epoch_1.pth" not in packaged
    assert "downstream/run/checkpoints/diffusion/checkpoint_epoch_1.pth" in checkpoints
    assert manifest["summary_files"][0]["summary"]["epoch_to_best"] == 1
