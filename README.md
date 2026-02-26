# quarimo (クアリモ)

**Quartet-based entropy analysis for detecting parallel evolution across phylogenetic tree ensembles.**

*Quarimo* uses cross-entropy products of quartet frequencies across bootstrap ensembles to estimate the likelihood of parallel evolution.

Quarimo is a made up word formed by smashing "quartet" and "mori" (森, forest) together.

## Features

- **Fast quartet topology analysis** - Bulk queries across large tree collections
- **Multiple backends** - Python, CPU-parallel (numba), and CUDA GPU acceleration
- **Memory efficient** - CSR-like flat-packed layout for large datasets
- **Clean API** - Context managers for logging, warnings, and backend selection
- **Well tested** - Comprehensive test suite with 83+ tests
- **Type hints** - Full type annotations for IDE support

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
from quarimo import Forest

# Load trees from NEWICK strings
trees = [
    '((A:1,B:1):1,(C:1,D:1):1);',
    '((A:1,C:1):1,(B:1,D:1):1);',
    '((A:1,D:1):1,(B:1,C:1):1);',
]

# Create collection
c = Forest(trees)

# Query quartet topology
quartets = [('A', 'B', 'C', 'D')]
counts = c.quartet_topology(quartets)

print(f"Topology counts: {counts}")
# Output: [[1 1 1]]  (one tree supports each topology)
```

## Advanced Usage

### Context Managers

```python
from quarimo import quiet, use_backend, silent_benchmark

# Suppress logging during construction
with quiet():
    c = Forest(large_tree_list)

# Force specific backend
with use_backend('cpu-parallel'):
    counts = c.quartet_topology(quartets)

# Silent benchmarking
with silent_benchmark('cuda'):
    counts = c.quartet_topology(quartets)
```

### Grouped Trees

```python
# Organize trees by group
grouped_trees = {
    'species_A': ['((A:1,B:1):1,(C:1,D:1):1);', ...],
    'species_B': ['((A:1,C:1):1,(B:1,D:1):1);', ...],
}

c = Forest(grouped_trees)

# Query and split results by group
counts, dists = c.quartet_topology(quartets, steiner=True)
by_group = c.split_quartet_results_by_group(counts, dists)
```

### Backend Selection

```python
from quarimo import get_available_backends

# Check available backends
backends = get_available_backends()
print(f"Available: {backends}")

# Force specific backend (parameter)
counts = c.quartet_topology(quartets, backend='cpu-parallel')

# Force specific backend (context manager)
with use_backend('python'):
    counts = c.quartet_topology(quartets)
```

## Documentation

<!-- TODO: Add documentation link when available -->
Full documentation coming soon.

For now, see docstrings in the code:

```python
from quarimo import Forest
help(Forest)
help(Forest.quartet_topology)
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
```

### Run linters

```bash
black .
isort .
flake8 .
mypy *.py
```

## Performance

The package supports three computational backends:

- **Python**: Pure Python fallback (always available)
- **CPU-parallel**: Numba JIT compilation with parallel execution
- **CUDA**: GPU acceleration for large-scale analyses

Typical speedups with numba:

- CPU-parallel: 10-100x faster than Python
- CUDA: 100-1000x faster than Python (depending on dataset size)

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
