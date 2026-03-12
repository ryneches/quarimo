# quarimo (クアリモ)

[![Tests](https://github.com/ryneches/quarimo/actions/workflows/tests.yml/badge.svg)](https://github.com/ryneches/quarimo/actions/workflows/tests.yml)
[![Build](https://github.com/ryneches/quarimo/actions/workflows/build.yml/badge.svg)](https://github.com/ryneches/quarimo/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/quarimo)](https://pypi.org/project/quarimo/)
[![Python](https://img.shields.io/pypi/pyversions/quarimo)](https://pypi.org/project/quarimo/)
[![Downloads](https://img.shields.io/pypi/dm/quarimo)](https://pypi.org/project/quarimo/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](LICENSE)
[![Coverage](https://codecov.io/gh/ryneches/quarimo/branch/main/graph/badge.svg)](https://codecov.io/gh/ryneches/quarimo)

**Quartet-based entropy analysis for detecting parallel evolution across phylogenetic tree ensembles.**

*Quarimo* uses cross-entropy products of quartet frequencies across bootstrap ensembles to estimate the likelihood of parallel evolution.

Quarimo is a made up word formed by smashing "quartet" and "mori" (森, forest) together.

## Features

- **Fast quartet topology analysis** : Bulk queries across large tree collections
- **Per-group topology counts** : Results broken out by labeled tree group with shape `(n_quartets, n_groups, 4)`
- **Polytomy-aware** : Multifurcating trees are automatically binarized; unresolvable quartets are tracked separately (topology k=3)
- **Result dataclasses** : Structured return types with `.to_frame()` for Polars DataFrame output in long or wide form
- **Deterministic and sampled quartets** : The `Quartets` class provides explicit lists, random sampling, and on-GPU generation from a shared infinite sequence
- **Multiple backends** : Python, CPU-parallel (Numba), CUDA GPU, and Apple Silicon Metal GPU
- **Memory efficient** : CSR-like flat-packed layout; GPU arrays uploaded once at construction
- **Clean API** : Context managers for logging and backend selection
- **Well tested** : Comprehensive test suite with 520+ tests

## Installation

### Basic installation (CPU only)

```bash
pip install quarimo
```

### With CPU parallelization

```bash
pip install quarimo[parallel]
```

### With GPU acceleration (NVIDIA CUDA)

```bash
pip install quarimo[gpu]
```

### With Apple Silicon Metal GPU

```bash
pip install quarimo[apple]
```

### Install all optional features

```bash
pip install quarimo[all]          # numba (CPU + CUDA)
pip install quarimo[all,apple]    # add Metal GPU for Apple Silicon
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
print(result.counts[0, 0])   # [1 1 1 0]  — one tree for each resolved topology, zero unresolved
```

## The Quartets Class

`Quartets` represents a window into an infinite deterministic sequence of four-taxon sets.
It is the required input to `Forest.quartet_topology()`.

```python
from quarimo import Quartets

# Explicit quartet list (by taxon name or by global ID)
q = Quartets.from_list(forest, [('A', 'B', 'C', 'D'), ('A', 'B', 'C', 'E')])

# Random sampling — reproducible with seed
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

result.counts          # int32   (n_quartets, n_groups, 4)
result.steiner         # float64 (n_quartets, n_groups, 4), or None if steiner=False
result.steiner_min     # float64 (n_quartets, n_groups, 4), NaN where count == 0
result.steiner_max     # float64 (n_quartets, n_groups, 4), NaN where count == 0
result.steiner_var     # float64 (n_quartets, n_groups, 4), NaN where count == 0
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
# Columns: quartet_idx, a, b, c, d, group, topology, count[, steiner_sum, steiner_min, steiner_max, steiner_var]

# Wide form — one row per quartet
df = result.to_frame('wide')
# Columns: quartet_idx, a, b, c, d, {group}_t{k}[, {group}_steiner_t{k}, ...]
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

### Backend Selection

Quarimo selects the fastest available backend automatically.  The priority order is:

```
python  <  cpu-parallel  <  mlx  <  cuda
```

On any given machine, `cuda` and `mlx` are mutually exclusive in practice (Apple Silicon
cannot run CUDA).

Use `use_backend()` to force a specific backend for a block of code:

```python
from quarimo import use_backend

with use_backend('cpu-parallel'):
    result = forest.quartet_topology(q)

with use_backend('cuda'):          # raises ValueError if CUDA is unavailable
    result = forest.quartet_topology(q)

with use_backend('mlx'):           # raises ValueError if MLX is unavailable
    result = forest.quartet_topology(q)
```

`use_backend()` yields the resolved backend name, which is useful for logging:

```python
with use_backend('best') as b:
    print(f"using {b}")            # e.g. "using cuda"
    result = forest.quartet_topology(q)
```

You can also pass `backend=` directly to `quartet_topology()`:

```python
result = forest.quartet_topology(q)                         # 'best' (default)
result = forest.quartet_topology(q, backend='python')       # pure Python
result = forest.quartet_topology(q, backend='cpu-parallel') # Numba JIT
result = forest.quartet_topology(q, backend='cuda')         # NVIDIA GPU
result = forest.quartet_topology(q, backend='mlx')          # Apple Silicon GPU
```

To inspect what is available on the current machine:

```python
from quarimo import get_available_backends, get_backend_info

print(get_available_backends())   # e.g. ['python', 'cpu-parallel', 'mlx']
print(get_backend_info())
```

#### Silent benchmarking

```python
from quarimo import silent_benchmark

# Suppresses quarimo logging and forces backend
with silent_benchmark('cuda'):
    result = forest.quartet_topology(q)
```

## Performance

Quarimo supports four computational backends:

| Backend | Hardware | Relative speed |
|---------|----------|----------------|
| `python` | Any CPU | 1× (baseline) |
| `cpu-parallel` | Any CPU (Numba JIT + prange) | 10–100× |
| `mlx` | Apple Silicon M-series (Metal GPU) | 50–500× |
| `cuda` | NVIDIA GPU (Numba CUDA) | 100–1000× |

Speedup estimates are for large quartet counts (≥10k quartets) relative to pure Python.
Actual performance depends on core/GPU count, memory bandwidth, and dataset size.

Scaling is dominated by the number of (quartet, tree) pairs evaluated —
effectively `n_quartets × n_trees`.  The CSR flat-packed layout keeps all tree arrays
contiguous in memory.

**Apple Silicon note:** the `mlx` backend benefits from Apple's Unified Memory Architecture
(UMA) — CPU and GPU share the same physical memory, so there is no host-to-device copy
cost when uploading forest arrays.

**CUDA note:** when using `Quartets.random()`, quartet generation runs on-device,
eliminating host-to-device quartet transfer for large random samples.

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
git clone https://github.com/ryneches/quarimo.git
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
