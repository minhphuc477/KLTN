## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, invoke the `skill` tool with `skill: "graphify"` before doing anything else.

Rules:
- For codebase questions, first run `python -m graphify query "<question>"` when graphify-out/graph.json exists. Use `python -m graphify path "<A>" "<B>"` for relationships and `python -m graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `python -m graphify update .` to keep the graph current (AST-only, no API cost).

---

## Anti-Cargo-Cult & Minimalist Engineering Rules (Chống Lắp Ráp Vô Nghĩa & Feature Bloat)

This project strictly adheres to minimalist, mathematically sound, neuro-symbolic engineering. Any agent working on this repository MUST abide by the following non-negotiable rules:

### 1. Zero Cargo-Cult Architecture (Cấm Lắp Ráp Vô Nghĩa & Đẻ Module)
- **Do NOT add new modules, attention variants, neural layers, or auxiliary loss functions to patch generative symptoms.**
  *Historical anti-patterns explicitly banned:*
  - Adding handcrafted maze overrides (`puzzle_room_scaffold`) because the neural model failed to generate enough blocks.
  - Adding extreme logit biases (`+10,000`) or hard token overwrites after decoding.
  - Bolting on linear attention kernels (`Hedgehog`), distillation hacks (`LCM-LoRA`), or multiple redundant pathfinders (`VIN`, `Perturb-and-MAP`, `CNN`, `Bellman-Ford`).
  - Proposing multi-room chained loss gradients across dungeons when the training set only has 8 dungeons.
- **Before adding ANY new code or module, prove that the existing minimalist stack cannot solve the problem.**

### 2. Neuro-Symbolic Division of Responsibility (Phân Định Rạch Ròi Nơ-ron vs Ký Hiệu)
- **Symbolic Layer (Block I Grammar, Block VI Assembler, Block VII StateSpaceAStar):**
  - Owns global quest correctness, key-lock dependency scheduling, room coordinate placement, and formal solvability proofs.
  - Do NOT force the neural network to learn NP-complete global combinatorial quest planning from scratch.
- **Neural Layer (Block III GraphGPS, Block IV G-LMDM Masked Transformer):**
  - Owns conditional micro room geometry ($16 \times 11$) synthesis from data.
  - Conditioned on global context via GraphGPS + RRWP and exact boundary sockets at $t=0$.
- **Differentiable Regularization Layer (Block V LogicNet):**
  - Owns local intra-room traversability continuity via a single soft Bellman-Ford objective ($\alpha_{\text{logic}} = 0.1$).
  - Do NOT multiply loss terms ($\alpha_1, \alpha_2, \alpha_3, \dots$) or introduce complex multi-objective balancing on small data.

### 3. Small-Data Budget & Hyperparameter Restraint
- Ground truth reality: VGLC training set has **349 rooms across 8 dungeons (only 43 unique interior layouts)**.
- The generative denoiser MUST remain parameter-efficient ($\le 100\text{K}$ denoiser parameters; $\le 4\text{M}$ total system parameters).
- Keep hyperparameter surfaces minimal. If a solution requires tuning $>1$ sensitive loss weight or alters batch statistics, it is rejected.

### 4. Information-First Debugging (Sửa Dữ Liệu Đầu Vào Trước Khi Thêm Thuật Toán)
- If the neural generator fails to satisfy a mission constraint (e.g., misplacing doors or ignoring locked status):
  1. Check if the network was provided sufficient conditioning features (node attributes, boundary socket labels, relative positional encodings).
  2. Fix data feeding and feature conditioning first.
  3. Never add post-hoc patches or complex auxiliary losses to compensate for missing input features.

### 5. Truth in Attribution (Minh Bạch Phân Bổ 4 Giai Đoạn)
- Always report Schema 2.0 attribution: `raw` $\to$ `constrained` $\to$ `repaired` $\to$ `compiled`.
- Never claim that the neural network "learned full game logic" if symbolic repair was required.
- Measure and report `repair_rate` and `tiles_repaired_mean` honestly as metrics of neural prior quality.
