# Generative Adversarial Networks (GANs) have shown im- pressive results for image generation

- PDF: [rodriguez2019bootstrapping.pdf](../../rodriguez2019bootstrapping.pdf)
- Topic: statistics_methods

## Abstract / Core Idea
Generative Adversarial Networks (GANs) have shown im- pressive results for image generation. However, GANs face challenges in generating contents with certain types of con- straints, such as game levels. Speciﬁcally, it is difﬁcult to generate levels that have aesthetic appeal and are playable at the same time. Additionally, because training data usually is limited, it is challenging to generate unique levels with cur- rent GANs. In this paper, we propose a new GAN architec- ture named Conditional Embedding Self-Attention Genera- tive Adversarial Network (CESAGAN) and a new bootstrap- ping training procedure. The CESAGAN is a modiﬁcation of the self-attention GAN that incorporates an embedding fea- ture vector input to condition the training of the discriminator and generator. This allows the network to model non-local dependency between game objects, and to count objects. Ad- ditionally, to reduce the number of levels necessary to train the GAN, we propose a bootstrapping mechanism in which playable generated levels are added to the training set. The results demonstrate that the new approach does not only gen- erate a larger number of levels that are playable but also gen- erates fewer duplicate levels compared to a standard GAN.

## Method Signals
- Keywords phat hien: pcg

## Conclusion / Findings
We introduce a new GAN architecture – Conditional Em- bedding Self-Attention Generative Adversarial Network (CESAGAN) with bootstrapping mechanism – for video game level generation. The results of the experiments con- ﬁrm the original concern that the state-of-art in GAN has limitations when applied to procedural content generation (PCG). In particular, GANs have difﬁculty in generating playable and unique levels when few training samples are available. To address this challenge, we introduce Con- ditional Embedding Self-Attention Generative Adversarial Network (CESAGAN) with bootstrapping. This new archi- tecture is a modiﬁcation of SAGAN, with an additional fea- ture conditional vector to train the discriminator and gener- ator. The results show a considerable improvement in playa- bility and diversity for 15,000 generated levels with respect to the state-of-art. One of the next challenges for CESAGAN with bootstrapping is to train on more complex video games such as Boulderdash or train with more complex architec- tures in place of the conditional feature. In addition, using a deep neural network to select the most relevant levels for bootstrapping could decrease the number of duplicate levels even further. Acknowledgements Ahmed Khalifa acknowledges the ﬁnancial support from NSF grant (Award number 1717324 - “RI: Small: General Intelligence through Algorithm Invention and S

## Relevance To KLTN
- Bai nay duoc xep vao nhom statistics_methods trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
