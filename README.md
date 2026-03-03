# quarimo (クアリモ)

**Quartet-based entropy analysis for detecting parallel evolution across phylogenetic tree ensembles.**

*Quarimo* uses cross-entropy products of quartet frequencies across bootstrap ensembles to estimate the likelihood of parallel evolution.

Quarimo is a made up word formed by smashing "quartet" and "mori" (森, forest) together.

## Features

- **Fast quartet topology analysis** : Bulk queries across large tree collections
- **Per-group topology counts** : Results broken out by labeled tree group with shape `(n_quartets, n_groups, 3)`
- **Deterministic and sampled quartets** : The `Quartets` class provides explicit lists, random sampling, and on-GPU generation from a shared infinite sequence
- **Multiple backends** : Python, CPU-parallel (Numba), and CUDA GPU acceleration
- **Memory efficient** : CSR-like flat-packed layout for large datasets
- **Clean API** : Context managers for logging, warnings, and backend selection
- **Well tested** : Comprehensive test suite with 450+ tests

## Installation

### Basic installation (CPU only)

```bash
pip install quarimo
```

### With CPU parallelization

```bash
pip install quarimo[parallel]
```

### With GPU acceleration (requires CUDA)

```bash
pip install quarimo[gpu]
```

### Install all optional features

```bash
pip install quarimo[all]
```

## Quick Start

```python
from quarimo import Forest, Quartets

# Load trees from NEWICK strings
trees = [
    '((A:1,B:1):1,(C:1,D:1):1);',
    '((A:1,C:1):1,(B:1,D:1):1);',
    '((A:1,D:1):1,(B:1,C:1):1);',
]

# Create collection (single unnamed group)
forest = Forest(trees)

# Build a quartet query
q = Quartets.from_list(forest, [('A', 'B', 'C', 'D')])

# Query quartet topology - returns (n_quartets, n_groups, 3)
counts = forest.quartet_topology(q)

print(counts.shape)   # (1, 1, 3)
print(counts[0, 0])   # [1 1 1]  - one tree supports each topology
```

## The Quartets Class

`Quartets` represents a window into an infinite deterministic sequence of four-taxon sets.
It is the required input to `Forest.quartet_topology()`.

```python
from quarimo import Quartets

# Explicit quartet list (by taxon name or by global ID)
q = Quartets.from_list(forest, [('A', 'B', 'C', 'D'), ('A', 'B', 'C', 'E')])

# Random sampling - reproducible with seed
q = Quartets.random(forest, count=100_000, seed=42)

# Full constructor: explicit seed + random tail starting at offset
q = Quartets(forest, seed=[('A', 'B', 'C', 'D')], offset=0, count=50_000)
```

**Type rule:** all elements in every quartet must be the same type - either all `str`
(taxon names) or all `int` (global IDs).  Mixing types raises `TypeError`.

## Advanced Usage

### Per-Group Topology Counts

When a `Forest` is constructed from a `dict`, trees are organized into named groups.
`quartet_topology()` always returns counts with shape `(n_quartets, n_groups, 3)`,
where axis 1 corresponds to `forest.unique_groups` (sorted alphabetically).

```python
# Organize trees by group
forest = Forest({
    'gene_A': ['((A:1,B:1):1,(C:1,D:1):1);', '((A:1,B:1):1,(C:1,D:1):1);'],
    'gene_B': ['((A:1,C:1):1,(B:1,D:1):1);', '((A:1,D:1):1,(B:1,C:1):1);'],
})

q = Quartets.from_list(forest, [('A', 'B', 'C', 'D')])
counts = forest.quartet_topology(q)

print(counts.shape)              # (1, 2, 3)
print(forest.unique_groups)      # ['gene_A', 'gene_B']

gi_A = forest.unique_groups.index('gene_A')
gi_B = forest.unique_groups.index('gene_B')
print(counts[0, gi_A])           # topology distribution for gene_A trees
print(counts[0, gi_B])           # topology distribution for gene_B trees
```

For `Forest(list)`, there is one auto-labeled group, so the shape is `(n_quartets, 1, 3)`.

### Steiner Distances

```python
counts, steiner = forest.quartet_topology(q, steiner=True)

# Both arrays have shape (n_quartets, n_groups, 3)
# steiner[qi, gi, k] = sum of Steiner spanning lengths for group gi, topology k
# Mean Steiner per group per topology:
mean_steiner = steiner / counts.clip(min=1)
```

### Context Managers

```python
from quarimo import quiet, use_backend, silent_benchmark

# Suppress logging during construction
with quiet():
    forest = Forest(large_tree_list)

# Force specific backend
with use_backend('cpu-parallel'):
    counts = forest.quartet_topology(q)

# Silent benchmarking
with silent_benchmark('cuda'):
    counts = forest.quartet_topology(q)
```

### Backend Selection

```python
from quarimo import get_available_backends

# Check available backends
print(get_available_backends())

# Per-call backend selection
counts = forest.quartet_topology(q)                         # 'best' (default)
counts = forest.quartet_topology(q, backend='python')       # pure Python
counts = forest.quartet_topology(q, backend='cpu-parallel') # Numba JIT
counts = forest.quartet_topology(q, backend='cuda')         # GPU
```

## Documentation

<!-- TODO: Add documentation link when available -->
Full documentation coming soon.

For now, see docstrings in the code:

```python
from quarimo import Forest, Quartets
help(Forest)
help(Forest.quartet_topology)
help(Quartets)
```

## Development

### Setup development environment

```bash
git clone https://github.com/yourusername/quarimo.git
cd quarimo
pip install -e .[dev]
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ -m "not large_scale"   # skip slow benchmarks
```

### Run linters

```bash
black . --line-length=100
isort . --profile=black
flake8 .
mypy quarimo/
```

## Performance

The package supports three computational backends:

- **Python**: Pure Python fallback (always available)
- **CPU-parallel**: Numba JIT compilation with parallel execution across quartets
- **CUDA**: GPU acceleration; also supports on-GPU quartet generation via `Quartets.random()`,
  eliminating host-to-device quartet transfer for large random samples

Typical speedups relative to Python:

- CPU-parallel: 10–100×
- CUDA: 100–1000× (depending on dataset size)

## Citation

<!-- TODO: Add citation information when paper is published -->
If you use this software in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

<!-- TODO: Add support links when available -->
- Issues: [GitHub Issues](https://github.com/yourusername/quarimo/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/quarimo/discussions)
