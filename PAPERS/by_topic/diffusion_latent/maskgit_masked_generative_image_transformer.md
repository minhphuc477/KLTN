# MaskGIT: Masked Generative Image Transformer

- PDF: [maskgit_masked_generative_image_transformer.pdf](../../maskgit_masked_generative_image_transformer.pdf)
- Topic: diffusion_latent
- Reference IDs: arxiv:2202.04200, doi:10.1109/CVPR52688.2022.01103

## Abstract / Core Idea
Generative transformers have experienced rapid popularity growth in the computer vision community in synthesizing high-fidelity and high-resolution images. The best generative transformer models so far, however, still treat an image naively as a sequence of tokens, and decode an image sequentially following the raster scan ordering (i.e. line-by-line). We find this strategy neither optimal nor efficient. This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation. Our experiments demonstrate that MaskGIT significantly outperforms the state-of-the-art transformer model on the ImageNet dataset, and accelerates autoregressive decoding by up to 64x. Besides, we illustrate that MaskGIT can be easily extended to various image editing tasks, such as inpainting, extrapolation, and image manipulation.

## Method Signals
- Keywords phat hien: see_pdf_text

## Conclusion / Findings
sketch and then progressively reﬁnes it by ﬁlling or tweaking the de- tails, which is in clear contrast to the line-by-line printing used in previous work [7, 15]. Additionally, treating image as a ﬂat sequence means that the autoregressive sequence length grows quadratically, easily forming an extremely long sequence–longer than any natural language sentence. This poses challenges for not only modeling long-term cor- relation but also renders the decoding intractable. For exam- ple, it takes a considerable 30 seconds to generate a single image on a GPU autoregressively with 32x32 tokens. This paper introduces a new bidirectional transformer for image synthesis called Masked Generative Image Trans- former (MaskGIT). During training, MaskGIT is trained on a similar proxy task to the mask prediction in BERT [11]. At inference time, MaskGIT adopts a novel non-autoregressive decoding method to synthesize an image in constant number of steps. Speciﬁcally, at each iteration, the model predicts all tokens simultaneously in parallel but only keeps the most conﬁdent ones. The remaining tokens are masked out and will be re-predicted in the next iteration. The mask ratio is decreased until all tokens are generated with a few itera- tions of reﬁnement. As illustrated Input Our Outpainting Samples (A) (B) Input —— Our Outpainting Samples —— Groundtruth (C) Input ——Our Inpainting Samples —— Groundtruth (D) ——Our Class-conditional Samples —— (E) Figure 21. Limitations and Failure Cases. 23

## Relevance To KLTN
- Bai nay duoc xep vao nhom diffusion_latent trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
