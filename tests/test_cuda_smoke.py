import pytest
import torch

from src.core.latent_diffusion import create_latent_diffusion
from src.core.logic_net import soft_min


pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the manual GPU smoke suite.")


def test_cuda_soft_min_fp16_is_finite():
    _require_cuda()
    values = torch.tensor(
        [[0.0, 1.0, 100.0], [1.0, 2.0, 3.0]],
        device="cuda",
        dtype=torch.float16,
    )

    result = soft_min(values, dim=-1, temperature=0.1)

    assert result.device.type == "cuda"
    assert torch.isfinite(result).all()


def test_cuda_autocast_latent_diffusion_denoiser_is_finite():
    _require_cuda()
    model = create_latent_diffusion(
        latent_dim=4,
        model_channels=8,
        context_dim=8,
        num_timesteps=10,
        unet_channel_mult=(1, 2),
        unet_num_res_blocks=1,
        unet_attention_resolutions=(1,),
        unet_num_heads=2,
        unet_dropout=0.0,
    ).cuda().eval()
    latent = torch.randn(1, 4, 4, 4, device="cuda")
    timestep = torch.tensor([2], device="cuda")
    context = torch.randn(1, 8, device="cuda")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.denoiser(latent, timestep, context)

    assert output.shape == latent.shape
    assert torch.isfinite(output).all()
