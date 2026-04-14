# The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games

- PDF: [the_procedural_content_generation_benchmark_an_open_source_testbed_for_generativ.pdf](../../the_procedural_content_generation_benchmark_an_open_source_testbed_for_generativ.pdf)
- Topic: quality_diversity_evolution
- Reference IDs: arxiv:2503.21474

## Abstract / Core Idea
This paper introduces the Procedural Content Generation Benchmark for evaluating generative algorithms on different game content creation tasks. The benchmark comes with 12 game-related problems with multiple variants on each problem. Problems vary from creating levels of different kinds to creating rule sets for simple arcade games. Each problem has its own content representation, control parameters, and evaluation metrics for quality, diversity, and controllability. This benchmark is intended as a first step towards a standardized way of comparing generative algorithms. We use the benchmark to score three baseline algorithms: a random generator, an evolution strategy, and a genetic algorithm. Results show that some problems are easier to solve than others, as well as the impact the chosen objective has on quality, diversity, and controllability of the generated artifacts.

## Method Signals
- Keywords phat hien: zelda

## Conclusion / Findings
uni00000050/uni00000044/uni00000003/uni00000016/uni00000011/uni00000015 /uni00000027/uni00000048/uni00000048/uni00000053/uni00000036/uni00000048/uni00000048/uni0000004e/uni00000010/uni00000055/uni00000014 Figure 4: The number of feasible and unique solutions over 100 separate runs on Binary, Sokoban, and Zelda using six different methods (three search-based generators, one con- structive generator, and two few-shot LLM generators). generator starts by building a 2D maze using Prim’s algorithm [8]. This generated maze is used as-is for Binary. For Sokoban and Zelda, the script erases more than 50% of the walls to allow for open areas, then adds the missing objects in the level at random locations (for Sokoban, these locations are restricted such that crates have no more than one side blocked). LLM generators use a simple prompt that explains the goal of the game and how to play it, followed by the goal of the generator and five example levels. Figure 4 shows the comparison between these algorithms from the perspective of quality (number of feasible solutions) and diver- sity (number of unique solutions) over 100 runs, extending the find- ings from Fig. 3. GA has more feasible solutions overall, although the constructive algorithm surpasses it in Sokoban. In Sokoban, the script is fairly thorough (e.g. constraining where crates can be placed) and thus it is not surprising that it can generate many feasible solutions. It is worth noting, however, that most of these solutions are

## Relevance To KLTN
- Bai nay duoc xep vao nhom quality_diversity_evolution trong pipeline neural-symbolic topology-first.
- Dung de doi chieu voi khoi tuong ung trong docs va Chapter 3/4.
