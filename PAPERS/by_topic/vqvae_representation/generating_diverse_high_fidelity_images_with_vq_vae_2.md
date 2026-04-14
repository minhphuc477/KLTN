# Generating Diverse High-Fidelity Images with VQ-VAE-2

- PDF: [generating_diverse_high_fidelity_images_with_vq_vae_2.pdf](../../generating_diverse_high_fidelity_images_with_vq_vae_2.pdf)
- Topic: vqvae_representation
- Reference IDs: arxiv:1906.00446

## Abstract / Core Idea
We explore the use of Vector Quantized Variational AutoEncoder (VQ-VAE) models for large scale image generation. To this end, we scale and enhance the autoregressive priors used in VQ-VAE to generate synthetic samples of much higher coherence and fidelity than possible before. We use simple feed-forward encoder and decoder networks, making our model an attractive candidate for applications where the encoding and/or decoding speed is critical. Additionally, VQ-VAE requires sampling an autoregressive model only in the compressed latent space, which is an order of magnitude faster than sampling in the pixel space, especially for large images. We demonstrate that a multi-scale hierarchical organization of VQ-VAE, augmented with powerful priors over the latent codes, is able to generate samples with quality that rivals that of state of the art Generative Adversarial Networks on multifaceted datasets such as ImageNet, while not suffering from GAN's known shortcomings such as mode collapse and lack of diversity.

## Method Signals
- Keywords phat hien: vq

## Conclusion / Findings
ucing inductive biases such as multi-scale [34, 35, 26, 21] or by modeling the dominant bit planes in an image [17, 16]. In this paper we use ideas from lossy compression to relieve the generative model from modeling negligible information. Indeed, techniques such as JPEG [39] have shown that it is often possible to remove more than 80% of the data without noticeably changing the perceived image quality. As proposed by [37], we compress images into a discrete latent space by vector-quantizing intermediate representations of an autoencoder. These representations are over 30x smaller than the original image, but still allow the decoder to reconstruct the images with little distortion. The prior over these discrete representations can be modeled with a state of the art PixelCNN [ 35, 36] with self-attention [38], called PixelSnail [6]. When sampling from this prior, the decoded images also exhibit the same high quality and coherence of the reconstructions (see Fig. 1). Furthermore, the training and sampling of this generative model over the discrete latent space is also 30x faster than when directly applied to the pixels, allowing us to train on much higher resolution images. Finally, the encoder and decoder used in this work reta B Additional Samples Please follow the following link to access the full version of our paper, rendered without lossy compression, which includes additional samples. https://drive.google.com/file/d/1H2nr_Cu7OK18tRemsWn_6o5DGMNYentM/ view?usp=sharing 15

## Relevance To KLTN
- Bai nay duoc xep vao nhom vqvae_representation trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
