# Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference

- PDF: [latent_consistency_models_synthesizing_high_resolution_images_with_few_step_infe.pdf](../../latent_consistency_models_synthesizing_high_resolution_images_with_few_step_infe.pdf)
- Topic: diffusion_latent
- Reference IDs: arxiv:2310.04378

## Abstract / Core Idea
Latent Diffusion models (LDMs) have achieved remarkable results in synthesizing high-resolution images. However, the iterative sampling process is computationally intensive and leads to slow generation. Inspired by Consistency Models (song et al.), we propose Latent Consistency Models (LCMs), enabling swift inference with minimal steps on any pre-trained LDMs, including Stable Diffusion (rombach et al). Viewing the guided reverse diffusion process as solving an augmented probability flow ODE (PF-ODE), LCMs are designed to directly predict the solution of such ODE in latent space, mitigating the need for numerous iterations and allowing rapid, high-fidelity sampling. Efficiently distilled from pre-trained classifier-free guided diffusion models, a high-quality 768 x 768 2~4-step LCM takes only 32 A100 GPU hours for training. Furthermore, we introduce Latent Consistency Fine-tuning (LCF), a novel method that is tailored for fine-tuning LCMs on customized image datasets. Evaluation on the LAION-5B-Aesthetics dataset demonstrates that LCMs achieve state-of-the-art text-to-image generation performance with few-step inference. Project Page: https://latent-consistency-models.github.io/

## Method Signals
- Keywords phat hien: diffusion

## Conclusion / Findings
in 2 ∼4 steps or even one step, significantly accelerating text-to-image generation. We employ LCM to distill the Dreamer-V7 version of SD in just 4,000 training iterations. In this paper, we introduce Latent Consistency Models (LCMs) for fast, high-resolution image generation. Mirroring LDMs, we employ consistency models in the image latent space of a pre- trained auto-encoder from Stable Diffusion (Rombach et al., 2022). We propose a one-stage guided distillation method to efficiently convert a pre-trained guided diffusion model into a latent consis- tency model by solving an augmented PF-ODE. Additionally, we propose Latent Consistency Fine- tuning, which allows fine-tuning a pre-trained LCM to support few-step inference on customized image datasets. Our main contributions are summarized as follows: • We propose Latent Consistency Models (LCMs) for fast, high-resolution image generation. LCMs employ consistency models in the image latent space, enabling fast few-step or even one-step high-fidelity sampling on pre-trained latent diffusion models (e.g., Stable Diffusion (SD)). • We provide a simple and efficient one-stage guided consistency distillation method to distill SD for few-step (2∼4) or even 1-step sampling. We propose the SKIPPING -STEP technique to further 2 Preprint 2-Steps Inference Figure 8: More generated images results with LCM 2-steps inference (768 ×768 Resolution). We employ LCM to distill the Dreamer-V7 version of SD in just 4,000 training iterations. 18

## Relevance To KLTN
- Bai nay duoc xep vao nhom diffusion_latent trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
