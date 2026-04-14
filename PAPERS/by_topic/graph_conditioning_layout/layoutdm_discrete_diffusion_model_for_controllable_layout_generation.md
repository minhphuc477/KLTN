# LayoutDM: Discrete Diffusion Model for Controllable Layout Generation

- PDF: [layoutdm_discrete_diffusion_model_for_controllable_layout_generation.pdf](../../layoutdm_discrete_diffusion_model_for_controllable_layout_generation.pdf)
- Topic: graph_conditioning_layout
- Reference IDs: arxiv:2303.08137

## Abstract / Core Idea
Controllable layout generation aims at synthesizing plausible arrangement of element bounding boxes with optional constraints, such as type or position of a specific element. In this work, we try to solve a broad range of layout generation tasks in a single model that is based on discrete state-space diffusion models. Our model, named LayoutDM, naturally handles the structured layout data in the discrete representation and learns to progressively infer a noiseless layout from the initial input, where we model the layout corruption process by modality-wise discrete diffusion. For conditional generation, we propose to inject layout constraints in the form of masking or logit adjustment during inference. We show in the experiments that our LayoutDM successfully generates high-quality layouts and outperforms both task-specific and task-agnostic baselines on several layout tasks.

## Method Signals
- Keywords phat hien: diffusion, vq, layout, constraint

## Conclusion / Findings
relational constraints [21, 23], el- ement completion [12], and refinement [40]. Some attempt at solving multiple tasks in a single model [22, 37]. BLT [22] points out that the recent autoregressive de- coders [2, 12] are not fully capable of considering partial inputs, i.e. known elements or attributes, during generation because they have a fixed generation order. BLT addresses the conditional generation by fill-in-the-blank task formu- lation using a bidirectional Transformer encoder similar to masked language models [6]. However, BLT cannot solve layout completion demonstrated in the decoder-based mod- els because of the requirement of the known number of el- ements. Our LayoutDM enjoys the best of both worlds and supports a broader range of conditional generation tasks in a single model. Another layout-specific consideration is the complex user-specified constraints, such as the positional require- ments between two boxes (e.g., a header box should be Partial 1.0 1.5 2.0 2.5 Overlap 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 0.11Alignment Rico 0.00 0.05 0.10 0.15 Overlap 0 2 4 6 8Alignment PubLayNet LayoutTrans. MaskGIT* BLT BART VQDiffusion* LayoutDM Real Data Unconditional 0.8 1.0 1.2 1.4 Overlap 0.0 0.2 0.4 0.6 0.8 1.0Alignment Rico 0.0 0.2 0.4 0.6 0.8 Overlap 0.025 0.050 0.075 0.100 0.125 0.150 0.175 0.200Alignment PubLayNet BART LayoutTrans. VQDiffusion* LayoutDM LayoutTrans.-Ordered MaskGIT* BLT Real Data Figure 18. (cont.) Alignment and overlap of different models. 26

## Relevance To KLTN
- Bai nay duoc xep vao nhom graph_conditioning_layout trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
