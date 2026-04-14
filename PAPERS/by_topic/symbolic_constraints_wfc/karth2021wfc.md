# WaveFunctionCollapse is Constraint Solving in the Wild

- PDF: [karth2021wfc.pdf](../../karth2021wfc.pdf)
- Topic: symbolic_constraints_wfc
- Reference IDs: doi:10.1145/3102071.3110566

## Abstract / Core Idea
WaveFunctionCollapse is Constraint Solving in the Wild Isaac Karth University of California Santa Cruz Department of Computational Media Santa Cruz, CA ikarth@ucsc.edu Adam M. Smith University of California Santa Cruz Department of Computational Media Santa Cruz, CA amsmith@ucsc.edu ABSTRACT Maxim Gumin’s WaveFunctionCollapse (WFC) algorithm is an example-driven image generation algorithm emerging from the craft practice of procedural content generation. In WFC, new im- ages are generated in the style of given examples by ensuring every local window of the output occurs somewhere in the input. Op- erationally, WFC implements a non-backtracking, greedy search method. This paper examines WFC as an instance of constraint solving methods. We trace WFC’s explosive influence on the tech- nical artist community, explain its operation in terms of ideas from the constraint solving literature, and probe its strengths by means of a surrogate implementation using answer set programming. CCS CONCEPTS • Theory of computation → Constraint and logic program- ming; Random walks and Markov chains; • Applied comput- ing → Media arts; Fine arts; • Mathematics of computing → Solvers; KEYWORDS constraint solving, procedural content generation, texture synthesis ACM Reference format: Isaac Karth and Adam M. Smith. 2017. WaveFunctionCollapse is Constraint Solving in the Wild. In Proceedings of FDG’17, Hyannis, MA, USA, August 14-17, 2017, 10 pages. https://doi.org/10.1145/3102071.3110566 1 INTRODUCT

## Method Signals
- Keywords phat hien: constraint

## Conclusion / Findings
nt generation is more accessible. Even though many users treat the algorithm as a black box, they are able to effectively use it to create interesting content. ACKNOWLEDGMENTS The authors would like to thank generative artists Maxim Gumin, Joseph Parker, Brian Bucklew, Oskar Stålberg, and Martin O’Leary for their correspondence about their respective projects, and to Free- hold Games and Joseph Parker for permission to use their images. Additionally we would like to thank Ruben Fitch for discussions towards producing a pseudocode version of Gumin’s original code. REFERENCES [1] Alexei A Efros and Thomas K Leung. 1999. Texture synthesis by non-parametric sampling. In Computer Vision, 1999. The Proceedings of the Seventh IEEE Interna- tional Conference on, Vol. 2. IEEE, IEEE Computer Society, 1999, 1033–1038. [2] Leif Foged and Ian D Horswill. 2015. Rolling Your Own Finite-Domain Constraint Solver. A K Peters/CRC Press, 283–302. [3] Freehold Games. 2017. Caves of Qud. (2017). [4] Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. 2015. A Neural Algo- rithm of Artistic Style. CoRR abs/1508.06576 (2015). http://arxiv.org/abs/1508. 06576 [5] Martin Gebser, Roland Kaminski, Benjamin Kaufmann, and Torsten Schaub. 2012. Answer Set Solving in Practice . Morgan and Claypool Publishers. [6] Martin Gebser, Benjamin Kaufmann, and Torsten Schaub. 2012. Conflict-driven answer set solving: From theory to practice. Artificial Intelligence 187 (2012), 52–89. [7] Carla P Gomes, Ashish Sabh

## Relevance To KLTN
- Bai nay duoc xep vao nhom symbolic_constraints_wfc trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
