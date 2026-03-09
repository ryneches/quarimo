# quarimo (クアリモ)

**Quartet-based entropy analysis for detecting parallel evolution across phylogenetic tree ensembles.**

*Quarimo* uses cross-entropy products of quartet frequencies across bootstrap ensembles to estimate the likelihood of parallel evolution.

Quarimo is a made up word formed by smashing "quartet" and "mori" (森, forest) together.

## Features

- **Fast quartet topology analysis** : Bulk queries across large tree collections
- **Per-group topology counts** : Results broken out by labeled tree group with shape `(n_quartets, n_groups, 4)`
- **Polytomy-aware** : Multifurcating trees are automatically binarized; unresolvable quartets are tracked separately (topology k=3)
- **Result dataclasses** : Structured return types with `.to_frame()` for Polars DataFrame output in long or wide form
- **Deterministic and sampled quartets** : The `Quartets` class provides explicit lists, random sampling, and on-GPU generation from a shared infinite sequence
- **Multiple backends** : Python, CPU-parallel (Numba), and CUDA GPU acceleration
- **Memory efficient** : CSR-like flat-packed layout; GPU arrays uploaded once at construction
- **Clean API** : Context managers for logging and backend selection
- **Well tested** : Comprehensive test suite with 488+ tests

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

# Query quartet topology — returns a QuartetTopologyResult
result = forest.quartet_topology(q)

print(result.counts.shape)   # (1, 1, 4)
print(result.counts[0, 0])   # [1 1 1 0]  — one tree for each of the three resolved topologies, zero unresolved
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

**Type rule:** all elements in every quartet must be the same type — either all `str`
(taxon names) or all `int` (global IDs).  Mixing types raises `TypeError`.

## Advanced Usage

### Per-Group Topology Counts

When a `Forest` is constructed from a `dict`, trees are organized into named groups.
`quartet_topology()` always returns counts with shape `(n_quartets, n_groups, 4)`,
where axis 1 corresponds to `forest.unique_groups` (sorted alphabetically).

```python
forest = Forest({
    'gene_A': ['((A:1,B:1):1,(C:1,D:1):1);', '((A:1,B:1):1,(C:1,D:1):1);'],
    'gene_B': ['((A:1,C:1):1,(B:1,D:1):1);', '((A:1,D:1):1,(B:1,C:1):1);'],
})

q = Quartets.from_list(forest, [('A', 'B', 'C', 'D')])
result = forest.quartet_topology(q)

print(result.counts.shape)       # (1, 2, 4)
print(forest.unique_groups)      # ['gene_A', 'gene_B']

gi_A = forest.unique_groups.index('gene_A')
gi_B = forest.unique_groups.index('gene_B')
print(result.counts[0, gi_A])    # [2 0 0 0]  — gene_A trees both vote topology 0
print(result.counts[0, gi_B])    # [0 1 1 0]  — gene_B trees split between topologies 1 and 2
```

For `Forest(list)`, there is one auto-labeled group, so the shape is `(n_quartets, 1, 4)`.

### Topology Encoding

For a quartet with taxa sorted by global ID as (a, b, c, d):

| k | Split | Meaning |
|---|-------|---------|
| 0 | (a,b) \| (c,d) | topology 0 |
| 1 | (a,c) \| (b,d) | topology 1 |
| 2 | (a,d) \| (b,c) | topology 2 |
| 3 | unresolved | quartet spans a polytomy-inserted node and all three pair-sums tie |

### Polytomy Handling

Multifurcating trees are automatically binarized by inserting zero-length internal branches.
Quarimo tracks these inserted nodes using a CSR sparse list and uses them to detect unresolvable quartets.

A quartet is classified as **unresolved** (k=3) only when:
1. At least one of its six pairwise LCA nodes is a polytomy-inserted node, **and**
2. All three four-point pair-sums are exactly equal (an unambiguous signal from the zero-length sentinel branch)

Quartets that span a polytomy-inserted node but still have a clear winner are assigned to the winning resolved topology normally — partial polytomies do not automatically make quartets unresolvable.

```python
# Trifurcating tree — quarimo binarizes silently and logs stats at INFO level
forest = Forest(['(A:1,B:1,C:1,D:1);'])  # quadrifurcating root

result = forest.quartet_topology(Quartets.from_list(forest, [('A', 'B', 'C', 'D')]))
# result.counts[0, 0, 3] > 0 means unresolvable for this quartet
```

**Zero-length user branches:** if your input trees contain explicit zero-length branches
(`:0` or `:0.0`), quarimo emits a WARNING — these are treated as real branches contributing
0 to root distances.  If you intend a polytomy at that position, collapse the zero-length
branch into an explicit multifurcation in the NEWICK string before loading.

### Result Dataclasses

`quartet_topology()` returns a `QuartetTopologyResult` dataclass with direct access to the
underlying arrays and a `to_frame()` method for labelled DataFrame output.

```python
result = forest.quartet_topology(q, steiner=True)

result.counts          # int32  (n_quartets, n_groups, 4)
result.steiner         # float64 (n_quartets, n_groups, 4), or None if steiner=False
result.groups          # ['gene_A', 'gene_B', ...]  — axis-1 labels
result.quartets        # the Quartets object used to produce this result
result.global_names    # taxon name lookup: global_names[gid] = name

# Mean Steiner length per topology (avoid division by zero):
mean_steiner = result.steiner / result.counts.clip(min=1)
```

#### DataFrame output

```python
# Long form — one row per (quartet, group, topology)
df = result.to_frame('long')
# Columns: quartet_idx, a, b, c, d, group, topology, count[, steiner_sum]

# Wide form — one row per quartet
df = result.to_frame('wide')
# Columns: quartet_idx, a, b, c, d, {group}_t{k}[, {group}_steiner_t{k}]
```

`quartet_idx` is a combinadic integer that uniquely identifies the quartet by its
four taxon global IDs — use it as the join key between result frames.

By default, `to_frame()` calls `.unique()` before returning (`deduplicate=True`).
This handles the case where random sampling produces the same quartet more than once,
which would otherwise cause many-to-many join explosions.  Pass `deduplicate=False`
to keep all rows and preserve the correspondence between row position and `qi` index.

```python
# Join topology counts and QED scores on quartet_idx (wide, 1-to-1)
topo_df = counts.to_frame('wide')
qed_df  = forest.qed(counts).to_frame('wide')
joined  = qed_df.join(topo_df, on='quartet_idx', how='left')

# Or in long form (1-to-n_groups×4); filter by group/topology after joining
topo_long = counts.to_frame('long')
qed_long  = forest.qed(counts).to_frame('long')
```

### QED (Quartet Ensemble Discordance)

`forest.qed(counts)` compares topology distributions between pairs of groups,
returning a `QEDResult` with scores in [-1, +1].

```python
result  = forest.quartet_topology(q)
qed     = forest.qed(result)              # all group pairs by default
qed.scores.shape                          # (n_quartets, n_pairs)

# Restrict to specific pairs
import numpy as np
pairs = np.array([[0, 1]], dtype=np.int32)   # compare group 0 vs group 1 only
qed   = forest.qed(result, group_pairs=pairs)
```

+1 means both groups have the same dominant topology; −1 means they disagree.

### Logging

Quarimo uses Python's standard `logging` module under the `quarimo` parent logger.
All child loggers (`quarimo._forest`, `quarimo._logging`, etc.) inherit from it.

By default the root logger controls whether quarimo messages appear.
To configure quarimo's verbosity independently:

```python
import logging

# Show only warnings and above from quarimo (suppress INFO construction messages)
logging.getLogger('quarimo').setLevel(logging.WARNING)

# Silence quarimo completely
logging.getLogger('quarimo').setLevel(logging.CRITICAL)

# Restore default behaviour
logging.getLogger('quarimo').setLevel(logging.NOTSET)
```

To format quarimo messages alongside your own application logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(name)s  %(levelname)s  %(message)s',
)
# quarimo messages now appear through your handler with timestamps
```

#### Logging context managers

```python
from quarimo import quiet, suppress_logger

# Temporarily silence all quarimo output
with quiet():
    forest = Forest(large_tree_list)

# Show only warnings during construction, then restore
with quiet(logging.WARNING):
    forest = Forest(large_tree_list)

# Fine-grained: suppress a single child logger
with suppress_logger('quarimo._forest'):
    forest = Forest(large_tree_list)
```

### Context Managers

```python
from quarimo import quiet, use_backend, silent_benchmark

# Force a specific backend for one block
with use_backend('cpu-parallel'):
    counts = forest.quartet_topology(q)

# Silent benchmarking — suppresses quarimo logging and forces backend
with silent_benchmark('cuda'):
    counts = forest.quartet_topology(q)
```

### Backend Selection

```python
from quarimo import get_available_backends

print(get_available_backends())   # e.g. ['python', 'cpu-parallel', 'cuda']

# Per-call backend selection
counts = forest.quartet_topology(q)                         # 'best' (default)
counts = forest.quartet_topology(q, backend='python')       # pure Python
counts = forest.quartet_topology(q, backend='cpu-parallel') # Numba JIT
counts = forest.quartet_topology(q, backend='cuda')         # GPU
```

## Performance

The package supports three computational backends:

- **Python**: Pure Python fallback, always available, useful for debugging
- **CPU-parallel**: Numba JIT compilation with `prange` parallelism across quartets; compiles on first call
- **CUDA**: GPU acceleration; quartet generation runs on-device via `Quartets.random()`, eliminating host-to-device quartet transfer for large random samples

Typical speedups relative to pure Python for large quartet counts (≥10k):

- CPU-parallel: 10–100× depending on core count
- CUDA: 100–1000× depending on dataset size and GPU memory bandwidth

Scaling is dominated by the number of (quartet, tree) pairs evaluated — effectively `n_quartets × n_trees`.  The CSR flat-packed layout keeps all tree arrays contiguous in memory, and GPU arrays are uploaded once at `Forest` construction rather than per query.

## Documentation

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

## Citation

<!-- TODO: Add citation information when paper is published -->
If you use this software in your research, please cite:

```
[Citation information to be added]
```

## License

MIT License — see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

<!-- TODO: Add support links when available -->
- Issues: [GitHub Issues](https://github.com/yourusername/quarimo/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/quarimo/discussions)
