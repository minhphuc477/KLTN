# Consistency Models

- PDF: [consistency_models.pdf](../../consistency_models.pdf)
- Topic: diffusion_latent
- Reference IDs: arxiv:2303.01469

## Abstract / Core Idea
Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose consistency models, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet 64x64 and LSUN 256x256.

## Method Signals
- Keywords phat hien: diffusion

## Conclusion / Findings
s. We demonstrate the efficacy of consistency models on sev- eral image datasets, including CIFAR-10 (Krizhevsky et al., 2009), ImageNet 64 ˆ 64 (Deng et al., 2009), and LSUN 256 ˆ 256 (Yu et al., 2015). Empirically, we observe that as a distillation approach, consistency models outperform existing diffusion distillation methods like progressive dis- tillation (Salimans & Ho, 2022) across a variety of datasets in few-step generation: On CIFAR-10, consistency models reach new state-of-the-art FIDs of 3.55 and 2.93 for one-step and two-step generation; on ImageNet 64 ˆ 64, it achieves record-breaking FIDs of 6.20 and 4.70 with one and two net- work evaluations respectively. When trained as standalone generative models, consistency models can match or surpass the quality of one-step samples from progressive distillation, despite having no access to pre-trained diffusion models. They are also able to outperform many GANs, and exist- ing non-adversarial, single-step generative models across multiple datasets. Furthermore, we show that consistency models can be used to perform a wide range of zero-shot data editing tasks, including image denoising, interpolation, inpainting, colorization, super-resolution, and stroke-guided image editing (SDEdit, Consistency Models (a) EDM (FID=6.69) (b) CT with single-step generation (FID=20.70) (c) CT with two-step generation (FID=11.76) Figure 21: Uncurated samples from LSUN Cat 256 ˆ 256. All corresponding samples use the same initial noise. 42

## Relevance To KLTN
- Bai nay duoc xep vao nhom diffusion_latent trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
