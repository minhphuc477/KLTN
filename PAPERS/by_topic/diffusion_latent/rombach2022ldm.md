# By decomposing the image formation process into a se- quential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image dat

- PDF: [rombach2022ldm.pdf](../../rombach2022ldm.pdf)
- Topic: diffusion_latent

## Abstract / Core Idea
By decomposing the image formation process into a se- quential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation al- lows for a guiding mechanism to control the image gen- eration process without retraining. However, since these models typically operate directly in pixel space, optimiza- tion of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evalu- ations. To enable DM training on limited computational resources while retaining their quality and ﬂexibility, we apply them in the latent space of powerful pretrained au- toencoders. In contrast to previous work, training diffusion models on such a representation allows for the ﬁrst time to reach a near-optimal point between complexity reduc- tion and detail preservation, greatly boosting visual ﬁdelity. By introducing cross-attention layers into the model archi- tecture, we turn diffusion models into powerful and ﬂexi- ble generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state-of-the-art scores for im- age inpainting and class-conditional image synthesis and highly competitive performance on various tasks, includ- ing text-to-image

## Method Signals
- Keywords phat hien: diffusion

## Conclusion / Findings
Nearest Neighbors on the CelebA-HQ dataset Figure 32. Nearest neighbors of our best CelebA-HQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 43 Nearest Neighbors on the FFHQ dataset Figure 33. Nearest neighbors of our best FFHQ model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 44 Nearest Neighbors on the LSUN-Churches dataset Figure 34. Nearest neighbors of our best LSUN-Churches model, computed in the feature space of a VGG-16 [79]. The leftmost sample is from our model. The remaining samples in each row are its 10 nearest neighbors. 45

## Relevance To KLTN
- Bai nay duoc xep vao nhom diffusion_latent trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
