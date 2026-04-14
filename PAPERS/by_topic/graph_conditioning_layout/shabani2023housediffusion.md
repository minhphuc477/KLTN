# The paper presents a novel approach for vector- floorplan generation via a diffusion model, which denoises 2D coordinates of room/door corners with two inference ob- jectives: 1) a

- PDF: [shabani2023housediffusion.pdf](../../shabani2023housediffusion.pdf)
- Topic: graph_conditioning_layout

## Abstract / Core Idea
The paper presents a novel approach for vector- floorplan generation via a diffusion model, which denoises 2D coordinates of room/door corners with two inference ob- jectives: 1) a single-step noise as the continuous quantity to precisely invert the continuous forward process; and 2) the final 2D coordinate as the discrete quantity to establish ge- ometric incident relationships such as parallelism, orthog- onality, and corner-sharing. Our task is graph-conditioned floorplan generation, a common workflow in floorplan de- sign. We represent a floorplan as 1D polygonal loops, each of which corresponds to a room or a door. Our dif- fusion model employs a Transformer architecture at the core, which controls the attention masks based on the in- put graph-constraint and directly generates vector-graphics floorplans via a discrete and continuous denoising pro- cess. We have evaluated our approach on RPLAN dataset. The proposed approach makes significant improvements in all the metrics against the state-of-the-art with significant margins, while being capable of generating non-Manhattan structures and controlling the exact number of corners per room. A project website with supplementary video and doc- ument is here https://aminshabani.github.io/housediffusion.

## Method Signals
- Keywords phat hien: diffusion, graph, constraint

## Conclusion / Findings
This paper presents a novel floorplan generative model that directly generates vector-graphics floorplans. The ap- proach uses a Diffusion Model with a Transformer network module at the core, which denoises 2D pixel coordinates both in discrete and continuous numeric representations. The discrete representation ensures precise geometric inci- dent relationships among rooms and doors. The transformer module has three types of attentions that exploit the struc- tural relationships of architectural components. Qualitative and quantitative evaluations demonstrate that the proposed system makes significant improvements over the current state-of-the-art with large margins in all the metrics, while boasting new capabilities such as the generation of non- Manhattan structures or the exact specification of the num- 8 ber of corners. This paper is the first compelling method to directly generate vector-graphics structured geometry. Our future work is the handling of large-scale buildings. We will share all our code and models. Acknowledgement: This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement.

## Relevance To KLTN
- Bai nay duoc xep vao nhom graph_conditioning_layout trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
