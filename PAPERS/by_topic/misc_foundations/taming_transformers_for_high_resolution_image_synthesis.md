# Taming Transformers for High-Resolution Image Synthesis

- PDF: [taming_transformers_for_high_resolution_image_synthesis.pdf](../../taming_transformers_for_high_resolution_image_synthesis.pdf)
- Topic: misc_foundations
- Reference IDs: arxiv:2012.09841

## Abstract / Core Idea
Designed to learn long-range interactions on sequential data, transformers continue to show state-of-the-art results on a wide variety of tasks. In contrast to CNNs, they contain no inductive bias that prioritizes local interactions. This makes them expressive, but also computationally infeasible for long sequences, such as high-resolution images. We demonstrate how combining the effectiveness of the inductive bias of CNNs with the expressivity of transformers enables them to model and thereby synthesize high-resolution images. We show how to (i) use CNNs to learn a context-rich vocabulary of image constituents, and in turn (ii) utilize transformers to efficiently model their composition within high-resolution images. Our approach is readily applied to conditional synthesis tasks, where both non-spatial information, such as object classes, and spatial information, such as segmentations, can control the generated image. In particular, we present the first results on semantically-guided synthesis of megapixel images with transformers and obtain the state of the art among autoregressive models on class-conditional ImageNet. Code and pretrained models can be found at https://github.com/CompVis/taming-transformers .

## Method Signals
- Keywords phat hien: see_pdf_text

## Conclusion / Findings
ason transform [78] Zhisheng Xiao, Qing Yan, Yi-an Chen, and Yali Amit. Generative latent ﬂow: A framework for non-adversarial image generation. CoRR, abs/1905.10485, 2019. 2 [79] Fisher Yu, Yinda Zhang, Shuran Song, Ari Seff, and Jianxiong Xiao. Lsun: Construction of a large-scale image dataset using deep learning with humans in the loop. arXiv preprint arXiv:1506.03365, 2015. 5 [80] Pan Zhang, Bo Zhang, Dong Chen, Lu Yuan, and Fang Wen. Cross-Domain Correspondence Learning for Exemplar-Based Image Translation. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2020. 2 [81] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. In CVPR, 2018. 4, 11, 13 [82] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018. 45, 46 [83] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through the ade20k dataset. arXiv preprint arXiv:1608.05442, 2016. 6 [84] Tinghui Zhou, Shubham Tulsiani, Weilun Sun, Jitendra Malik, and Alexei A. Efros. View synthesis by appearance ﬂow, 2017. 2 [85] Peihao Zhu, Rameen Abdal, Yipeng Qin, and Peter Wonka. Sean: Image synthesis with semantic region-adaptive normalization, 2019. 2 52

## Relevance To KLTN
- Bai nay duoc xep vao nhom misc_foundations trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
