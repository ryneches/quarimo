"""
tests/bench_forest.py
=====================
Scaling benchmarks for quarimo's quartet_topology() and qed() across all
available backends.

Design
------
Each backend has its own test class, sized for that hardware tier:

  Backend        | n_quartets range  | n_trees  | Notes
  ---------------|-------------------|----------|------------------------
  python         | 10 – 5 000        | 100      | Pure Python baseline
  cpu-parallel   | 1 K – 100 K       | 1 000    | Numba JIT (with warmup)
  mlx            | 10 K – 1 M        | 10 000   | Apple Silicon Metal GPU
  cuda           | 10 K – 1 M        | 10 000   | NVIDIA GPU

Because MLX and CUDA require dedicated hardware that is mutually exclusive
in practice, GPU benchmarks are hardware-gated via pytest markers
(@pytest.mark.requires_mlx, @pytest.mark.requires_cuda).

Forest fixtures (scope='module') are built once per benchmark session, so
fixture construction cost is not included in timing.  JIT-compiled kernels
(cpu-parallel, cuda) perform one warmup call before each benchmark loop so
that compilation time is excluded from steady-state measurements.

Cross-platform integration
--------------------------
Every benchmark records uniform keys in extra_info so that JSON results
from different machines can be merged in post-processing:

  backend, n_quartets, n_trees, n_leaves, n_groups, tree_type, mode, pairs

Throughput in pairs/s = pairs / benchmark.stats['mean'] can be computed
from the saved JSON produced by:

    pytest tests/bench_forest.py --benchmark-json=results.json --benchmark-only

Running
-------
    # Standard benchmarks only (no large_scale mark)
    pytest tests/bench_forest.py -m "not large_scale" --benchmark-only

    # Include GPU-scale large benchmarks
    pytest tests/bench_forest.py --benchmark-only

    # Save for cross-run comparison
    pytest tests/bench_forest.py --benchmark-json=results.json --benchmark-only
    pytest tests/bench_forest.py --benchmark-compare=results.json
"""

from __future__ import annotations

import random

import pytest

from quarimo._context import quiet, use_backend
from quarimo._forest import Forest
from quarimo._quartets import Quartets


# ======================================================================== #
# Tunable constants                                                         #
# ======================================================================== #

BRANCH_LENGTH: float = 1.0
RANDOM_SEED: int = 42    # seed for NEWICK leaf-order shuffling
QUARTET_SEED: int = 7    # seed for Quartets.random()

# ── Construction / branch-distance grid ─────────────────────────────────
TREE_COUNTS_SMALL: list[tuple[str, int]] = [
    ("trees10",   10),
    ("trees100",  100),
    ("trees1000", 1_000),
]
TREE_COUNTS_LARGE: list[tuple[str, int]] = [
    ("trees5000",  5_000),
    ("trees10000", 10_000),
]
LEAF_COUNTS: list[tuple[str, int]] = [
    ("leaves10",  10),
    ("leaves50",  50),
    ("leaves200", 200),
]

# ── Per-backend quartet counts ───────────────────────────────────────────
# Chosen so each non-large_scale benchmark finishes within a few seconds on
# the target hardware.  Large-scale variants are gated with @large_scale.
# Throughput = pairs / benchmark timing (pairs = n_quartets × n_trees).
PYTHON_QUARTET_COUNTS: list[tuple[str, int]] = [
    ("q10",   10),
    ("q100",  100),
    ("q1k",   1_000),
    ("q5k",   5_000),
]
CPU_QUARTET_COUNTS: list[tuple[str, int]] = [
    ("q1k",   1_000),
    ("q10k",  10_000),
    ("q100k", 100_000),
]
GPU_QUARTET_COUNTS: list[tuple[str, int]] = [
    ("q10k",  10_000),
    ("q100k", 100_000),
    ("q1m",   1_000_000),
]
GPU_QUARTET_COUNTS_LARGE: list[tuple[str, int]] = [
    ("q10m", 10_000_000),
]

# ── Fixed forest sizes for per-backend quartet benchmarks ────────────────
# n_leaves is kept constant so that scaling is driven by n_quartets × n_trees
# rather than tree size.  n_leaves=50 → C(50,4)=230 300 unique quartets, which
# comfortably covers all quartet count tiers (including 1M with repetition).
PYTHON_N_TREES:  int = 100
PYTHON_N_LEAVES: int = 50
CPU_N_TREES:     int = 1_000
CPU_N_LEAVES:    int = 50
GPU_N_TREES:     int = 10_000
GPU_N_LEAVES:    int = 50

# ── QED benchmark parameters ─────────────────────────────────────────────
QED_QUARTET_COUNTS: list[tuple[str, int]] = [
    ("q1k",   1_000),
    ("q10k",  10_000),
    ("q100k", 100_000),
]
QED_N_GROUPS:          int = 4
QED_N_TREES_PER_GROUP: int = 250   # total trees = 4 × 250 = 1 000
QED_N_LEAVES:          int = 50


# ======================================================================== #
# Tree generators                                                           #
# ======================================================================== #


def _leaf_name(i: int) -> str:
    """Zero-based index → unique short taxon name (A, B, …, Z, AA, AB, …)."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if i < len(alphabet):
        return alphabet[i]
    name = []
    i += 1
    while i > 0:
        i, remainder = divmod(i - 1, len(alphabet))
        name.append(alphabet[remainder])
    return "".join(reversed(name))


def balanced_newick(
    n_leaves: int,
    branch_length: float = BRANCH_LENGTH,
    leaf_names: list | None = None,
) -> str:
    """Balanced binary tree NEWICK string for n_leaves taxa."""
    if n_leaves < 2:
        raise ValueError(f"n_leaves must be >= 2, got {n_leaves}")
    names = leaf_names if leaf_names is not None else [_leaf_name(i) for i in range(n_leaves)]

    def _build(ns):
        if len(ns) == 1:
            return f"{ns[0]}:{branch_length}"
        mid = len(ns) // 2
        return f"({_build(ns[:mid])},{_build(ns[mid:])}):{branch_length}"

    return _build(names) + ";"


def caterpillar_newick(
    n_leaves: int,
    branch_length: float = BRANCH_LENGTH,
    leaf_names: list | None = None,
) -> str:
    """Caterpillar (maximally unbalanced) binary tree NEWICK string."""
    if n_leaves < 2:
        raise ValueError(f"n_leaves must be >= 2, got {n_leaves}")
    names = leaf_names if leaf_names is not None else [_leaf_name(i) for i in range(n_leaves)]
    tree = f"({names[-2]}:{branch_length},{names[-1]}:{branch_length})"
    for name in reversed(names[:-2]):
        tree = f"({name}:{branch_length},{tree}:{branch_length})"
    return tree + ";"


def generate_collection_newicks(
    n_trees: int,
    n_leaves: int,
    *,
    tree_type: str = "balanced",
    branch_length: float = BRANCH_LENGTH,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """
    Return n_trees NEWICK strings, each with n_leaves leaves.

    Each tree uses the same taxon set but with a shuffled leaf order so that
    the global_to_local mapping is exercised.  tree_type controls topology:
    'balanced', 'caterpillar', or 'mixed' (alternating).
    """
    rng = random.Random(seed)
    canonical = [_leaf_name(i) for i in range(n_leaves)]
    newicks = []
    for i in range(n_trees):
        shuffled = canonical[:]
        rng.shuffle(shuffled)
        if tree_type == "mixed":
            gen = balanced_newick if i % 2 == 0 else caterpillar_newick
        elif tree_type == "balanced":
            gen = balanced_newick
        elif tree_type == "caterpillar":
            gen = caterpillar_newick
        else:
            raise ValueError(f"Unknown tree_type {tree_type!r}")
        newicks.append(gen(n_leaves, branch_length=branch_length, leaf_names=shuffled))
    return newicks


def _info(
    benchmark,
    *,
    backend: str,
    n_quartets: int,
    n_trees: int,
    n_leaves: int,
    n_groups: int = 1,
    tree_type: str = "balanced",
    mode: str = "counts",
) -> None:
    """
    Populate benchmark.extra_info with uniform keys.

    These keys are preserved in --benchmark-json output, enabling
    cross-platform throughput analysis:
      pairs_per_second ≈ pairs / benchmark.stats['mean']
    """
    benchmark.extra_info.update(
        {
            "backend":    backend,
            "n_quartets": n_quartets,
            "n_trees":    n_trees,
            "n_leaves":   n_leaves,
            "n_groups":   n_groups,
            "tree_type":  tree_type,
            "mode":       mode,
            "pairs":      n_quartets * n_trees,
        }
    )


# ======================================================================== #
# Module-scoped forest fixtures                                             #
# ======================================================================== #


@pytest.fixture(scope="module")
def python_forest():
    """100-tree, 50-leaf balanced forest for Python baseline benchmarks."""
    with quiet():
        return Forest(generate_collection_newicks(PYTHON_N_TREES, PYTHON_N_LEAVES))


@pytest.fixture(scope="module")
def cpu_forest():
    """1 000-tree, 50-leaf balanced forest for cpu-parallel benchmarks."""
    with quiet():
        return Forest(generate_collection_newicks(CPU_N_TREES, CPU_N_LEAVES))


@pytest.fixture(scope="module")
def gpu_forest():
    """10 000-tree, 50-leaf balanced forest for GPU benchmarks (mlx / cuda)."""
    with quiet():
        return Forest(generate_collection_newicks(GPU_N_TREES, GPU_N_LEAVES))


@pytest.fixture(scope="module")
def qed_forest():
    """
    Grouped forest for QED benchmarks.

    4 groups × 250 trees = 1 000 trees total, all with 50 leaves.
    Each group has a distinct random seed so leaf orderings vary across groups.
    """
    groups = {
        f"group_{g}": generate_collection_newicks(
            QED_N_TREES_PER_GROUP, QED_N_LEAVES, seed=RANDOM_SEED + g * 100
        )
        for g in range(QED_N_GROUPS)
    }
    with quiet():
        return Forest(groups)


# ======================================================================== #
# 1. Construction benchmarks                                                #
# ======================================================================== #


class TestBenchConstruction:
    """
    Forest construction: NEWICK parsing + Euler tour + sparse table + CSR packing.

    Scaling axes: n_trees × n_leaves.  Separate balanced and caterpillar runs
    confirm that topology shape does not materially affect construction cost
    (both are O(n_nodes) per tree).
    """

    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_SMALL], ids=[k for k, _ in TREE_COUNTS_SMALL])
    def test_balanced(self, benchmark, n_trees, n_leaves):
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="balanced")
        benchmark.extra_info.update(
            {"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "balanced", "n_nodes": 2 * n_leaves - 1}
        )
        benchmark(Forest, newicks)

    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_SMALL], ids=[k for k, _ in TREE_COUNTS_SMALL])
    def test_caterpillar(self, benchmark, n_trees, n_leaves):
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="caterpillar")
        benchmark.extra_info.update(
            {"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "caterpillar", "n_nodes": 2 * n_leaves - 1}
        )
        benchmark(Forest, newicks)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_LARGE], ids=[k for k, _ in TREE_COUNTS_LARGE])
    def test_large(self, benchmark, n_trees, n_leaves):
        """Large-scale construction — confirms O(n_trees) scaling slope."""
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="mixed")
        benchmark.extra_info.update(
            {"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "mixed", "n_nodes": 2 * n_leaves - 1}
        )
        benchmark(Forest, newicks)


# ======================================================================== #
# 2. Python baseline benchmarks                                             #
# ======================================================================== #


class TestBenchPython:
    """
    quartet_topology() with backend='python' (pure Python, no JIT).

    This class establishes the 1× baseline.  All other backend classes can be
    compared against it by dividing pairs_per_second values.

    Fixed forest: 100 trees × 50 leaves.
    """

    @pytest.mark.parametrize("n_q", [v for _, v in PYTHON_QUARTET_COUNTS], ids=[k for k, _ in PYTHON_QUARTET_COUNTS])
    def test_counts(self, benchmark, python_forest, n_q):
        """Counts-only mode — 6 RMQ calls per (quartet, tree) pair."""
        q = Quartets.random(python_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="python", n_quartets=n_q, n_trees=PYTHON_N_TREES,
              n_leaves=PYTHON_N_LEAVES, mode="counts")
        with use_backend("python"):
            benchmark(python_forest.quartet_topology, q)

    @pytest.mark.parametrize("n_q", [v for _, v in PYTHON_QUARTET_COUNTS], ids=[k for k, _ in PYTHON_QUARTET_COUNTS])
    def test_steiner(self, benchmark, python_forest, n_q):
        """Steiner mode — quantifies overhead relative to counts-only."""
        q = Quartets.random(python_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="python", n_quartets=n_q, n_trees=PYTHON_N_TREES,
              n_leaves=PYTHON_N_LEAVES, mode="steiner")
        with use_backend("python"):
            benchmark(python_forest.quartet_topology, q, steiner=True)


# ======================================================================== #
# 3. CPU-parallel benchmarks                                                #
# ======================================================================== #


@pytest.mark.requires_cpu_parallel
class TestBenchCPUParallel:
    """
    quartet_topology() with backend='cpu-parallel' (Numba JIT + prange).

    A warmup call precedes each benchmark loop so that Numba JIT compilation
    (which happens only on first invocation with a given argument signature)
    is excluded from the timed rounds.

    Fixed forest: 1 000 trees × 50 leaves.
    """

    @pytest.mark.parametrize("n_q", [v for _, v in CPU_QUARTET_COUNTS], ids=[k for k, _ in CPU_QUARTET_COUNTS])
    def test_counts(self, benchmark, cpu_forest, n_q):
        q = Quartets.random(cpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cpu-parallel", n_quartets=n_q, n_trees=CPU_N_TREES,
              n_leaves=CPU_N_LEAVES, mode="counts")
        with use_backend("cpu-parallel"):
            cpu_forest.quartet_topology(q)          # JIT warmup
            benchmark(cpu_forest.quartet_topology, q)

    @pytest.mark.parametrize("n_q", [v for _, v in CPU_QUARTET_COUNTS], ids=[k for k, _ in CPU_QUARTET_COUNTS])
    def test_steiner(self, benchmark, cpu_forest, n_q):
        q = Quartets.random(cpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cpu-parallel", n_quartets=n_q, n_trees=CPU_N_TREES,
              n_leaves=CPU_N_LEAVES, mode="steiner")
        with use_backend("cpu-parallel"):
            cpu_forest.quartet_topology(q, steiner=True)   # JIT warmup
            benchmark(cpu_forest.quartet_topology, q, steiner=True)


# ======================================================================== #
# 4. MLX benchmarks (Apple Silicon Metal GPU)                              #
# ======================================================================== #


@pytest.mark.requires_mlx
class TestBenchMLX:
    """
    quartet_topology() with backend='mlx' (Apple Silicon Metal GPU).

    Apple's Unified Memory Architecture means there is no host-to-device copy
    cost, so array sizes large enough to saturate the GPU are appropriate.
    Quartets.random() generates quartets lazily; materialization cost is
    included in the benchmark timing (part of the real workload).

    A warmup call triggers Metal JIT compilation before the timed loop.

    Fixed forest: 10 000 trees × 50 leaves.
    """

    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS], ids=[k for k, _ in GPU_QUARTET_COUNTS])
    def test_counts(self, benchmark, gpu_forest, n_q):
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="mlx", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="counts")
        with use_backend("mlx"):
            gpu_forest.quartet_topology(q)          # Metal JIT warmup
            benchmark(gpu_forest.quartet_topology, q)

    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS], ids=[k for k, _ in GPU_QUARTET_COUNTS])
    def test_steiner(self, benchmark, gpu_forest, n_q):
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="mlx", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="steiner")
        with use_backend("mlx"):
            gpu_forest.quartet_topology(q, steiner=True)   # warmup
            benchmark(gpu_forest.quartet_topology, q, steiner=True)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS_LARGE], ids=[k for k, _ in GPU_QUARTET_COUNTS_LARGE])
    def test_counts_large(self, benchmark, gpu_forest, n_q):
        """Extreme-scale counts — shows peak GPU throughput."""
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="mlx", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="counts")
        with use_backend("mlx"):
            gpu_forest.quartet_topology(q)
            benchmark(gpu_forest.quartet_topology, q)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS_LARGE], ids=[k for k, _ in GPU_QUARTET_COUNTS_LARGE])
    def test_steiner_large(self, benchmark, gpu_forest, n_q):
        """Extreme-scale Steiner — shows memory bandwidth at peak GPU throughput."""
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="mlx", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="steiner")
        with use_backend("mlx"):
            gpu_forest.quartet_topology(q, steiner=True)
            benchmark(gpu_forest.quartet_topology, q, steiner=True)


# ======================================================================== #
# 5. CUDA benchmarks (NVIDIA GPU)                                           #
# ======================================================================== #


@pytest.mark.requires_cuda
class TestBenchCUDA:
    """
    quartet_topology() with backend='cuda' (NVIDIA GPU via Numba CUDA).

    When Quartets.random() is used, quartet generation runs on-device,
    eliminating the host-to-device quartet transfer for large random samples.
    A warmup call is performed to trigger Numba CUDA JIT compilation.

    Fixed forest: 10 000 trees × 50 leaves.
    """

    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS], ids=[k for k, _ in GPU_QUARTET_COUNTS])
    def test_counts(self, benchmark, gpu_forest, n_q):
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cuda", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="counts")
        with use_backend("cuda"):
            gpu_forest.quartet_topology(q)          # CUDA JIT warmup
            benchmark(gpu_forest.quartet_topology, q)

    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS], ids=[k for k, _ in GPU_QUARTET_COUNTS])
    def test_steiner(self, benchmark, gpu_forest, n_q):
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cuda", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="steiner")
        with use_backend("cuda"):
            gpu_forest.quartet_topology(q, steiner=True)   # warmup
            benchmark(gpu_forest.quartet_topology, q, steiner=True)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS_LARGE], ids=[k for k, _ in GPU_QUARTET_COUNTS_LARGE])
    def test_counts_large(self, benchmark, gpu_forest, n_q):
        """Extreme-scale counts — shows peak GPU throughput."""
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cuda", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="counts")
        with use_backend("cuda"):
            gpu_forest.quartet_topology(q)
            benchmark(gpu_forest.quartet_topology, q)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_q", [v for _, v in GPU_QUARTET_COUNTS_LARGE], ids=[k for k, _ in GPU_QUARTET_COUNTS_LARGE])
    def test_steiner_large(self, benchmark, gpu_forest, n_q):
        """Extreme-scale Steiner — shows memory bandwidth at peak GPU throughput."""
        q = Quartets.random(gpu_forest, count=n_q, seed=QUARTET_SEED)
        _info(benchmark, backend="cuda", n_quartets=n_q, n_trees=GPU_N_TREES,
              n_leaves=GPU_N_LEAVES, mode="steiner")
        with use_backend("cuda"):
            gpu_forest.quartet_topology(q, steiner=True)
            benchmark(gpu_forest.quartet_topology, q, steiner=True)


# ======================================================================== #
# 6. QED benchmarks                                                         #
# ======================================================================== #


class TestBenchQED:
    """
    forest.qed() benchmark on a grouped forest (4 groups × 250 trees = 1 000 trees).

    Two phases are timed separately because in real workflows the topology
    result is typically computed once and qed() is called repeatedly
    (e.g. with different group-pair selections):

      test_quartet_topology_grouped — quartet_topology() on grouped forest
          Includes per-group accumulation overhead not present in single-group runs.

      test_qed_from_precomputed — qed() on a pre-computed topology result
          Isolates the discordance-score computation cost.

    The best available backend is used for quartet_topology().  qed() always
    runs on CPU (it is a pure NumPy kernel).
    """

    @pytest.mark.parametrize("n_q", [v for _, v in QED_QUARTET_COUNTS], ids=[k for k, _ in QED_QUARTET_COUNTS])
    def test_quartet_topology_grouped(self, benchmark, qed_forest, n_q):
        """Topology counting with per-group accumulation (n_groups=4)."""
        q = Quartets.random(qed_forest, count=n_q, seed=QUARTET_SEED)
        n_trees_total = QED_N_GROUPS * QED_N_TREES_PER_GROUP
        benchmark.extra_info.update(
            {
                "n_quartets": n_q,
                "n_trees":    n_trees_total,
                "n_leaves":   QED_N_LEAVES,
                "n_groups":   QED_N_GROUPS,
                "tree_type":  "balanced",
                "mode":       "counts",
                "pairs":      n_q * n_trees_total,
            }
        )
        benchmark(qed_forest.quartet_topology, q)

    @pytest.mark.parametrize("n_q", [v for _, v in QED_QUARTET_COUNTS], ids=[k for k, _ in QED_QUARTET_COUNTS])
    def test_qed_from_precomputed(self, benchmark, qed_forest, n_q):
        """qed() called on a pre-computed QuartetTopologyResult."""
        q = Quartets.random(qed_forest, count=n_q, seed=QUARTET_SEED)
        with quiet():
            result = qed_forest.quartet_topology(q)
        n_trees_total = QED_N_GROUPS * QED_N_TREES_PER_GROUP
        n_pairs = QED_N_GROUPS * (QED_N_GROUPS - 1) // 2
        benchmark.extra_info.update(
            {
                "n_quartets": n_q,
                "n_trees":    n_trees_total,
                "n_leaves":   QED_N_LEAVES,
                "n_groups":   QED_N_GROUPS,
                "n_pairs":    n_pairs,
                "mode":       "qed",
            }
        )
        benchmark(qed_forest.qed, result)


# ======================================================================== #
# 7. Branch distance benchmarks                                             #
# ======================================================================== #


class TestBenchBranchDistance:
    """
    branch_distance() for a single taxon pair across all trees.

    One RMQ call per tree where both taxa are present — this establishes the
    per-RMQ baseline cost and confirms O(n_trees) scaling, which is the same
    inner loop that drives quartet_topology (6 RMQ calls per pair instead of 1).
    """

    TAXON_A: str = _leaf_name(0)   # "A"
    TAXON_B: str = _leaf_name(1)   # "B"

    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_SMALL], ids=[k for k, _ in TREE_COUNTS_SMALL])
    def test_balanced(self, benchmark, n_trees, n_leaves):
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="balanced")
        c = Forest(newicks)
        benchmark.extra_info.update({"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "balanced"})
        benchmark(c.branch_distance, self.TAXON_A, self.TAXON_B)

    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_SMALL], ids=[k for k, _ in TREE_COUNTS_SMALL])
    def test_caterpillar(self, benchmark, n_trees, n_leaves):
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="caterpillar")
        c = Forest(newicks)
        benchmark.extra_info.update({"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "caterpillar"})
        benchmark(c.branch_distance, self.TAXON_A, self.TAXON_B)

    @pytest.mark.large_scale
    @pytest.mark.parametrize("n_leaves", [v for _, v in LEAF_COUNTS], ids=[k for k, _ in LEAF_COUNTS])
    @pytest.mark.parametrize("n_trees",  [v for _, v in TREE_COUNTS_LARGE], ids=[k for k, _ in TREE_COUNTS_LARGE])
    def test_large(self, benchmark, n_trees, n_leaves):
        """Large-scale branch distance — confirms O(n_trees) scaling slope."""
        newicks = generate_collection_newicks(n_trees, n_leaves, tree_type="mixed")
        c = Forest(newicks)
        benchmark.extra_info.update({"n_trees": n_trees, "n_leaves": n_leaves, "tree_type": "mixed"})
        benchmark(c.branch_distance, self.TAXON_A, self.TAXON_B)
