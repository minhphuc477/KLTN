from src.optimization.lcm_lora import LCMLoRAFastSampler
import torch
import torch.nn as nn

class DummyDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
    def forward(self, latent, t, condition=None):
        return torch.zeros_like(latent)

model = DummyDiffusion()
sampler = LCMLoRAFastSampler(diffusion_model=model, use_fp16=False, compile_model=False, num_inference_steps=2)
cond = torch.zeros(1, 256)
out = sampler.sample_fast(condition=cond, latent_shape=(1,4), guidance_scale=1.0, seed=42)
print('sampled_shape=', out.shape)
