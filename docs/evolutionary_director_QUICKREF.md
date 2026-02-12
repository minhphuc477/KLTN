# Evolutionary Topology Director - Quick Reference

## 🎯 What It Does

Generates Zelda-like dungeon topologies by **evolving sequences of graph grammar rules** to match target difficulty curves.

**NOT random generation** — this is **evolutionary search** over rule sequences.

---

## 🚀 Quick Start (30 seconds)

```python
from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

# 1. Define target curve
target = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# 2. Create generator
gen = EvolutionaryTopologyGenerator(
    target_curve=target,
    population_size=50,
    generations=100,
    seed=42,
)

# 3. Evolve!
graph = gen.evolve()

# 4. Analyze
print(f"Fitness: {gen.get_statistics()['final_best_fitness']:.4f}")
print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
```

---

## 📊 Target Curve Patterns

| Pattern | Curve Example | Use Case |
|---------|---------------|----------|
| **Linear** | `[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]` | Tutorial dungeon |
| **Wave** | `[0.2, 0.6, 0.3, 0.7, 0.4, 0.9]` | Multi-boss arena |
| **Plateau** | `[0.1, 0.4, 0.4, 0.6, 0.6, 1.0]` | Exploration zones |
| **Spike** | `[0.2, 0.3, 0.9, 0.4, 0.5, 1.0]` | Surprise encounter |

---

## ⚙️ Key Parameters

```python
EvolutionaryTopologyGenerator(
    target_curve=[0.2, 0.5, 0.8, 1.0],  # Your desired curve
    
    # Evolution settings
    population_size=50,      # ↑ = better quality, ↓ = faster
    generations=100,         # ↑ = more time to converge
    
    # Genetic operators
    mutation_rate=0.15,      # ↑ = more exploration
    crossover_rate=0.7,      # ↑ = more exploitation
    
    # Genome settings
    genome_length=18,        # ↑ = more complex dungeons
    
    # Optional
    zelda_transition_matrix=None,  # Custom rule probabilities
    seed=42,                        # Reproducibility
)
```

---

## 🎮 Common Use Cases

### Tutorial Dungeon (Gradual Difficulty)
```python
target = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
gen = EvolutionaryTopologyGenerator(target, population_size=30, generations=50)
graph = gen.evolve()
```

### Arena Challenge (Wave Pattern)
```python
target = [0.3, 0.7, 0.4, 0.8, 0.5, 1.0]
gen = EvolutionaryTopologyGenerator(target, mutation_rate=0.2, genome_length=20)
graph = gen.evolve()
```

### Metroidvania (Branching Exploration)
```python
target = [0.2, 0.4, 0.4, 0.6, 0.6, 0.9]
custom_matrix = {
    "Start": {"Branch": 0.5, "InsertChallenge_ENEMY": 0.3, ...},
    # ... favor branching
}
gen = EvolutionaryTopologyGenerator(
    target, 
    zelda_transition_matrix=custom_matrix,
    genome_length=30,
)
graph = gen.evolve()
```

### Quick Prototype (Fast Iteration)
```python
target = [0.3, 0.7, 1.0]
gen = EvolutionaryTopologyGenerator(
    target, 
    population_size=20, 
    generations=30,
    genome_length=10,
)
graph = gen.evolve()
```

---

## 📈 Understanding Output

### Graph Structure
```python
graph = gen.evolve()  # Returns networkx.Graph

# Node attributes
for node in graph.nodes():
    print(graph.nodes[node]['type'])        # START, GOAL, ENEMY, KEY, etc.
    print(graph.nodes[node]['difficulty'])  # 0.0 - 1.0
    print(graph.nodes[node]['position'])    # (row, col)

# Check path
import networkx as nx
start = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'START'][0]
goal = [n for n in graph.nodes() if graph.nodes[n]['type'] == 'GOAL'][0]
path = nx.shortest_path(graph, start, goal)
print(f"Critical path length: {len(path)} nodes")
```

### Statistics
```python
stats = gen.get_statistics()

print(f"Best Fitness: {stats['final_best_fitness']:.4f}")  # 0.0-1.0
print(f"Converged: {stats['converged']}")                  # True/False
print(f"Generations: {stats['generations_run']}")

# Fitness over time
import matplotlib.pyplot as plt
plt.plot(stats['best_fitness_history'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Low fitness (< 0.5)** | ↑ `genome_length`, ↑ `population_size`, ↑ `generations` |
| **Premature convergence** | ↑ `mutation_rate`, ↓ `crossover_rate`, ↑ `population_size` |
| **Too many nodes** | ↓ `genome_length` (default: 18 → try 12) |
| **Takes too long** | ↓ `population_size`, ↓ `generations`, ↓ `genome_length` |
| **Not branching enough** | Use custom `zelda_transition_matrix` favoring `"Branch"` |
| **Too random** | ↓ `mutation_rate`, ↑ `crossover_rate` |

---

## 🧬 How It Works (30-second version)

1. **Genome** = `[1, 4, 2, 5, 1, 3, ...]` (rule IDs)
2. **Execute** genome → apply rules sequentially → build graph
3. **Evaluate** fitness = curve matching + solvability
4. **Evolve** via selection + crossover + mutation
5. **Return** best graph after N generations

---

## 📚 Grammar Rules

| ID | Rule | Effect |
|----|------|--------|
| 0 | StartRule | Creates START + GOAL (automatic) |
| 1 | InsertChallenge_ENEMY | Adds enemy encounter |
| 2 | InsertChallenge_PUZZLE | Adds puzzle room |
| 3 | InsertLockKey | Adds key → lock pair |
| 4 | Branch | Creates alternate path |

---

## ✅ Success Criteria

**Good Result:**
- Fitness > 0.7 (ideally > 0.9)
- Graph has 8-20 nodes
- Connected (START → GOAL path exists)
- Convergence in < 50 generations

**Adjust if:**
- Fitness < 0.5 after 100 generations
- Graphs too simple (< 5 nodes) or too complex (> 30 nodes)
- Not converging (diversity stays high)

---

## 🔗 Integration

### With Block II (2D Layout)
```python
# Block I: Generate topology
topology = gen.evolve()

# Block II: Generate 2D layout (future)
from src.generation.layout_generator import Layout2DGenerator
layout_gen = Layout2DGenerator(topology)
grid = layout_gen.generate()  # numpy array
```

### Export for Game Engine
```python
import pickle

# Save graph
with open('dungeon_topology.pkl', 'wb') as f:
    pickle.dump(graph, f)

# Load later
with open('dungeon_topology.pkl', 'rb') as f:
    graph = pickle.load(f)
```

### Batch Generation
```python
variants = []
for i in range(10):
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.5, 0.8, 1.0],
        seed=1000 + i,  # Different seed each time
    )
    graph = gen.evolve()
    variants.append(graph)

# Pick best
best = max(variants, key=lambda g: g.number_of_nodes())
```

---

## 📖 Full Documentation

- **Full README**: `docs/evolutionary_director_README.md`
- **Examples**: `examples/evolutionary_generation_demo.py`
- **Tests**: `tests/test_evolutionary_director.py`
- **Source**: `src/generation/evolutionary_director.py`

---

## 🎓 Research Context

**This implements:**
- Search-Based PCG (Togelius et al., 2011)
- Graph Grammar Generation (Dormans & Bakkes, 2011)
- Tension Curve Matching (Smith et al., 2010)

**Key Innovation:** Evolves **rule sequences** (genotype), not graphs (phenotype)

---

## 🆘 Quick Help

```bash
# Run test suite
cd KLTN
export PYTHONPATH="${PWD}"  # or $env:PYTHONPATH on Windows
python src/generation/evolutionary_director.py

# Run examples
python examples/evolutionary_generation_demo.py

# Run unit tests
pytest tests/test_evolutionary_director.py -v
```

---

## 🎯 Remember

1. **Target curve** = your design intent
2. **Fitness** = how well it matches
3. **Generation count** = trade-off: speed vs. quality
4. **Seed** = reproducibility is your friend
5. **Genome length** = complexity ceiling

---

**Version**: 1.0.0  
**Last Updated**: 2026-02-13  
**Status**: ✅ Production-Ready
