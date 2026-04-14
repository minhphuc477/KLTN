# Discovering Representations for Black-box Optimization

- PDF: [discovering_representations_for_black_box_optimization.pdf](../../discovering_representations_for_black_box_optimization.pdf)
- Topic: quality_diversity_evolution
- Reference IDs: arxiv:2003.04389

## Abstract / Core Idea
The encoding of solutions in black-box optimization is a delicate, handcrafted balance between expressiveness and domain knowledge -- between exploring a wide variety of solutions, and ensuring that those solutions are useful. Our main insight is that this process can be automated by generating a dataset of high-performing solutions with a quality diversity algorithm (here, MAP-Elites), then learning a representation with a generative model (here, a Variational Autoencoder) from that dataset. Our second insight is that this representation can be used to scale quality diversity optimization to higher dimensions -- but only if we carefully mix solutions generated with the learned representation and those generated with traditional variation operators. We demonstrate these capabilities by learning an low-dimensional encoding for the inverse kinematics of a thousand joint planar arm. The results show that learned representations make it possible to solve high-dimensional problems with orders of magnitude fewer evaluations than the standard MAP-Elites, and that, once solved, the produced encoding can be used for rapid optimization of novel, but similar, tasks. The presented techniques not only scale up quality diversity algorithms to high dimensions, but show that black-box optimization encodings can be automatically learned, rather than hand designed.

## Method Signals
- Keywords phat hien: map-elites, quality diversity

## Conclusion / Findings
sian mutation, to create candidates which could not be captured by the current DDE. At the same time we leverage the DDE to generalize common patterns across the map and create new solutions that are likely to be high-performing. To avoid introducing new hyper-parameters, we tune this exploration/exploitation trade- off optimally using a multi-armed bandit algorithm [23]. This new algorithm, DDE-Elites, reframes optimization as a search for representations (Figure 1). Integrating MAP-Elites with a VAE makes it possible to apply quality diversity to high-dimensional search spaces, and to find effective representations for future uses. We envision application to domains that have straightforward but expansive low-level representations, for instance: joints positions at 20Hz for a walking robot (12 × 100 = 1200 joint positions for a 5-second gait of a robot Discovering Representations for Black-box Optimization GECCO ’20, July 8–12, 2020, Cancún, Mexico B. Hyperparameters of DDE Experiments Hyperparameter Value Isometric Mutation Strength 0.003 Line Mutation Strength 0.1 Batch Size 100 Bandit Options, [0.00:0.00:1.00], [0.25:0.00:0.75], [0.50:0.00:0.50], [0.75:0.00:0.25], [1.00:0.00:0.00], [0.00:0.25:0.75], [0.00:0.50:0.50], [0.00:0.75:0.25], [0.00:1.00:0.00] Bandit Window Length 1000 Generations per VAE Training 1 Epochs per VAE Training 5 Mutation Strength when Searching DDE 0.15 Latent Vector Length [Arm20] 10 Latent Vector Length [Arm200] 32 Latent Vector Length [Arm1000] 32

## Relevance To KLTN
- Bai nay duoc xep vao nhom quality_diversity_evolution trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
