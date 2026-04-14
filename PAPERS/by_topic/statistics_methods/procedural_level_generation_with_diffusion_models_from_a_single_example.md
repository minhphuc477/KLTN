# Procedural Level Generation with Diffusion Models from a Single Example

- PDF: [procedural_level_generation_with_diffusion_models_from_a_single_example.pdf](../../procedural_level_generation_with_diffusion_models_from_a_single_example.pdf)
- Topic: statistics_methods
- Reference IDs: doi:10.1609/aaai.v38i9.28865

## Abstract / Core Idea
Level generation is a central focus of Procedural Content Generation (PCG), yet deep learning-based approaches are limited by scarce training data, i.e., human-designed levels. Despite being a dominant framework, Generative Adversarial Networks (GANs) exhibit a substantial quality gap between generated and human-authored levels, alongside rising training costs, particularly with increasing token complexity. In this paper, we introduce a diffusion-based generative model that learns from just one example. Our approach involves two core components: 1) an efficient yet expressive level representation, and 2) a latent denoising network with constrained receptive fields. To start with, our method utilizes token semantic labels, similar to word embeddings, to provide dense representations. This strategy not only surpasses one-hot encoding in representing larger game levels but also improves stability and accelerates convergence in latent diffusion. In addition, we adapt the denoising network architecture to confine the receptive field to localized patches of the data, aiming to facilitate single-example learning. Extensive experiments demonstrate that our model is capable of generating stylistically congruent samples of arbitrary sizes compared to manually designed levels. It suits a wide range of level structures with fewer artifacts than GAN-based approaches. The source code is available at https://github.com/shiqi-dai/diffusioncraft.

## Method Signals
- Keywords phat hien: diffusion, graph, pcg

## Conclusion / Findings
f overview of the main achievem Sudhakaran, S.; Grbic, D.; Li, S.; Katona, A.; Najarro, E.; Glanois, C.; and Risi, S. 2021. Growing 3d artefacts and functional machines with neural cellular automata. arXiv preprint arXiv:2103.08737. Summerville, A.; Snodgrass, S.; Guzdial, M.; Holmg˚ard, C.; Hoover, A. K.; Isaksen, A.; Nealen, A.; and Togelius, J. 2018. Procedural content generation via machine learning (PCGML). IEEE Transactions on Games, 10(3): 257–270. Summerville, A. J.; Snodgrass, S.; Mateas, M.; and n’on Vil- lar, S. O. 2016. The VGLC: The Video Game Level Cor- pus. Proceedings of the 7th Workshop on Procedural Con- tent Generation. Torrado, R. R.; Khalifa, A.; Green, M. C.; Justesen, N.; Risi, S.; and Togelius, J. 2020. Bootstrapping conditional gans for video game level generation. In 2020 IEEE Conference on Games (CoG), 41–48. IEEE. V olz, V .; Schrum, J.; Liu, J.; Lucas, S. M.; Smith, A.; and Risi, S. 2018. Evolving mario levels in the latent space of a deep convolutional generative adversarial network. In Pro- ceedings of the genetic and evolutionary computation con- ference, 221–228. Wang, W.; Bao, J.; Zhou, W.; Chen, D.; Chen, D.; Yuan, L.; and Li, H. 2022. SinDiffusion: Learning a Diffu- sion Model from a Single Natural Image. arXiv preprint arXiv:2211.12445. Wu, R.; and Zheng, C. 2022. Learning to Generate 3D Shapes from a Single Example. ACM Transactions on Graphics (TOG), 41(6): 1–19. The Thirty-Eighth AAAI Conference on Artiﬁcial Intelligence (AAAI-24) 10029

## Relevance To KLTN
- Bai nay duoc xep vao nhom statistics_methods trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
