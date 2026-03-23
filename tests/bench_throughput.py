"""
tests/bench_throughput.py
=========================
Quartet topology throughput benchmarks across a matrix of forest sizes,
group counts, leaf counts, Steiner modes, and backends.

Three scaling sweeps
--------------------
fixed_forest  (TestThroughputFixedForest)
    Total forest size is held constant (FIXED_FOREST_N_TREES trees).
    The number of tree groups varies across FIXED_FOREST_N_GROUPS.
    Each group receives floor(n_trees / n_groups) trees (last group takes
    the remainder).  Measures how per-group accumulation overhead scales
    relative to raw kernel throughput as group count increases.

fixed_groups  (TestThroughputFixedGroups)
    Group count is held constant (FIXED_GROUPS_N_GROUPS groups).
    Total forest size varies across FIXED_GROUPS_N_TREES.
    All sweeps start above the GPU L2-cache escape threshold so every
    point is bandwidth-bound rather than cache-bound.  Measures how
    kernel throughput scales with forest depth (trees per group).

fixed_trees  (TestThroughputFixedTrees)
    Tree count is held constant (FIXED_TREES_N_TREES trees, one group).
    Leaf count per tree varies across FIXED_TREES_N_LEAVES.
    Tests the O(1) LCA hypothesis: calculation-phase throughput should be
    independent of tree size because each quartet query requires exactly
    four sparse-table lookups regardless of n_leaves.  A throughput drop
    at larger leaf counts indicates sparse-table cache spill (O(n log n)
    per tree).

Timing phases
-------------
Every call to quartet_topology() emits a structured INFO log line:

    ⏱ t_query_load=<s> t_calc=<s> t_retrieve=<s>

These are captured by a _PhaseCapture log handler attached for the
duration of the benchmark loop.  The median across BENCH_ROUNDS rounds
is stored in benchmark.extra_info alongside t_device_load (measured
directly around Forest construction, which is excluded from all plots).

Primary metric: quartets_per_second = n_quartets / t_calc

JSON output
-----------
Run with --benchmark-json to save results for the throughput_benchmarks.py
Marimo notebook.  Run each sweep separately or all at once:

    # All three sweeps
    pytest tests/bench_throughput.py \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json

    # Individual sweeps
    pytest tests/bench_throughput.py::TestThroughputFixedForest \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    pytest tests/bench_throughput.py::TestThroughputFixedGroups \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json
    pytest tests/bench_throughput.py::TestThroughputFixedTrees \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json

Tuning
------
Per-backend quartet counts (N_QUARTETS) are sized for ~2 s of kernel
time per trial at the maximum forest size, giving stable estimates without
long runs.  Reduce cuda/mlx if individual trials exceed ~5 s; reduce
python if the fixed_trees sweep at high leaf counts is too slow.

Adjust FIXED_FOREST_N_GROUPS, FIXED_GROUPS_N_TREES, and FIXED_TREES_N_LEAVES
to change the sweep ranges.
"""

from __future__ import annotations

import logging
import random
import re
from time import perf_counter

import numpy as np
import pytest

from quarimo import Forest, Quartets
from quarimo._context import quiet, use_backend


# ======================================================================== #
# Configuration — tune these lists to adjust problem sizes                 #
# ======================================================================== #

N_LEAVES: int = 20      # taxa per tree; C(20,4) = 4 845 unique quartets
RANDOM_SEED:  int = 42
QUARTET_SEED: int = 7

# Number of timed rounds per trial (excluding one warmup round)
BENCH_ROUNDS: int = 3

# ── Scaling axis 1: fixed total forest size, varying number of groups ───
# 20_000 trees × ~3.6 KB/tree ≈ 72 MB — exceeds L2 cache on A100 (40 MB),
# H100 (50 MB) and GB10, so throughput reflects HBM bandwidth rather than
# cache capacity.
FIXED_FOREST_N_TREES:  int       = 20_000
FIXED_FOREST_N_GROUPS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ── Scaling axis 2: fixed group count, varying total forest size ─────────
# Sweep starts well above the L2-escape threshold so every point is
# bandwidth-bound rather than cache-bound.
FIXED_GROUPS_N_GROUPS: int       = 5
FIXED_GROUPS_N_TREES:  list[int] = [
    2_000, 4_000, 6_000, 8_000, 10_000,
    12_000, 14_000, 16_000, 18_000, 20_000,
]

# ── Scaling axis 3: fixed tree count, varying number of leaves ───────────
# Tests the O(1) LCA hypothesis: throughput should be flat across leaf counts
# because each quartet query requires exactly 4 sparse-table lookups.
# A drop at large leaf counts indicates sparse-table cache spill (O(n log n)
# per tree).  Forest construction (excluded from timing) is slow at high
# leaf counts; only the calculation phase is measured.
FIXED_TREES_N_TREES:  int       = 1_000
FIXED_TREES_N_LEAVES: list[int] = [
    1_000, 2_000, 3_000, 4_000, 5_000,
    6_000, 7_000, 8_000, 9_000, 10_000,
]

# ── Per-backend quartet counts ────────────────────────────────────────────
# Sized for ~2 s of kernel time per trial at the maximum forest size above,
# which gives stable throughput estimates without long runs.
# Reduce cuda/mlx if individual trials exceed ~5 s on your hardware.
N_QUARTETS: dict[str, int] = {
    "python":       10,
    "cpu-parallel": 10_000,
    "cuda":         500_000,
    "mlx":          500_000,
}


# ======================================================================== #
# Tree / forest generators                                                  #
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


def _balanced_newick(
    leaf_names: list[str],
    branch_length: float = 1.0,
) -> str:
    """Balanced binary tree NEWICK from an ordered leaf list."""
    def _build(ns: list[str]) -> str:
        if len(ns) == 1:
            return f"{ns[0]}:{branch_length}"
        mid = len(ns) // 2
        return f"({_build(ns[:mid])},{_build(ns[mid:])}):{branch_length}"
    return _build(leaf_names) + ";"


def _detect_gpu(backend: str) -> str | None:
    """Return a short GPU name string for the active backend, or None."""
    if backend == "cuda":
        try:
            from numba import cuda as numba_cuda
            device = numba_cuda.get_current_device()
            name = device.name
            return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:
            return None
    if backend == "mlx":
        try:
            import mlx.core as mx
            info = mx.metal.device_info()
            return info.get("device_name") or "Apple Metal"
        except Exception:
            return "Apple Metal"
    return None


def _make_grouped_forest(
    n_total_trees: int,
    n_groups: int,
    n_leaves: int = N_LEAVES,
    seed: int = RANDOM_SEED,
) -> Forest:
    """
    Build a grouped Forest with n_total_trees split across n_groups groups.

    Trees per group = floor(n_total_trees / n_groups); the first
    (n_total_trees % n_groups) groups each get one extra tree.  All trees
    share the same taxon set with independently shuffled leaf orders.
    """
    rng = random.Random(seed)
    canonical = [_leaf_name(i) for i in range(n_leaves)]
    groups: dict[str, list[str]] = {}
    base, remainder = divmod(n_total_trees, n_groups)
    for g in range(n_groups):
        count = base + (1 if g < remainder else 0)
        trees = []
        for _ in range(count):
            names = canonical[:]
            rng.shuffle(names)
            trees.append(_balanced_newick(names))
        groups[f"g{g}"] = trees
    with quiet():
        return Forest(groups)


# ======================================================================== #
# Log capture for phase timings                                             #
# ======================================================================== #

_TIMING_RE = re.compile(
    r"⏱ t_query_load=(?P<tql>\S+) t_calc=(?P<tc>\S+) t_retrieve=(?P<tr>\S+)"
)


class _PhaseCapture(logging.Handler):
    """
    Captures ⏱ timing lines emitted by quartet_topology() at INFO level.

    Each call to quartet_topology() writes exactly one matching line.
    Attach before the benchmark loop; detach after; call median_phases()
    to get stable per-phase estimates across BENCH_ROUNDS rounds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rounds: list[dict[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = _TIMING_RE.search(record.getMessage())
        if m:
            self.rounds.append(
                {
                    "t_query_load": float(m.group("tql")),
                    "t_calc":       float(m.group("tc")),
                    "t_retrieve":   float(m.group("tr")),
                }
            )

    def median_phases(self) -> dict[str, float]:
        if not self.rounds:
            return {"t_query_load": 0.0, "t_calc": 0.0, "t_retrieve": 0.0}
        return {
            k: float(np.median([r[k] for r in self.rounds]))
            for k in ("t_query_load", "t_calc", "t_retrieve")
        }


# ======================================================================== #
# Shared trial runner                                                       #
# ======================================================================== #


def _run_trial(
    benchmark,
    *,
    n_total_trees: int,
    n_groups: int,
    backend: str,
    steiner: bool,
    sweep: str,         # 'fixed_forest', 'fixed_groups', or 'fixed_trees'
    n_leaves: int = N_LEAVES,
    morton_order: bool = False,  # [MORTON_SCHED]
) -> None:
    """
    Build a forest, warm up the kernel, then run BENCH_ROUNDS timed calls
    capturing all four phase timings via the ⏱ log protocol.

    benchmark.extra_info is populated with uniform keys so that JSON files
    from different machines can be merged in the Marimo notebook.
    """
    n_quartets = N_QUARTETS[backend]

    # ── Phase 1: device load (Forest construction + optional GPU upload) ──
    t0 = perf_counter()
    forest = _make_grouped_forest(n_total_trees, n_groups, n_leaves=n_leaves)
    t_device_load = perf_counter() - t0

    q = Quartets.random(forest, count=n_quartets, seed=QUARTET_SEED)

    # ── Warmup: triggers JIT/Metal compilation; excluded from timing ──────
    with quiet(), use_backend(backend):
        forest.quartet_topology(q, steiner=steiner, morton_order=morton_order)

    # ── Phases 2–4: attach log capture, then run the benchmark loop ───────
    # Temporarily enable INFO-level logging on the forest logger so that the
    # ⏱ timing lines emitted by quartet_topology() reach the capture handler.
    # The original level is restored in the finally block.
    capture = _PhaseCapture()
    _forest_logger = logging.getLogger("quarimo._forest")
    _orig_level = _forest_logger.level
    _forest_logger.setLevel(logging.INFO)
    _forest_logger.addHandler(capture)
    try:
        with use_backend(backend):
            benchmark.pedantic(
                forest.quartet_topology,
                args=(q,),
                kwargs={"steiner": steiner, "morton_order": morton_order},
                rounds=BENCH_ROUNDS,
                iterations=1,
            )
    finally:
        _forest_logger.removeHandler(capture)
        _forest_logger.setLevel(_orig_level)

    phases = capture.median_phases()
    t_calc = phases["t_calc"]

    benchmark.extra_info.update(
        {
            "sweep":               sweep,
            "backend":             backend,
            "gpu_name":            _detect_gpu(backend),
            "n_trees":             n_total_trees,
            "n_groups":            n_groups,
            "n_leaves":            n_leaves,
            "n_quartets":          n_quartets,
            "steiner":             steiner,
            "morton_order":        morton_order,         # [MORTON_SCHED]
            "t_device_load":       t_device_load,
            "t_query_load":        phases["t_query_load"],
            "t_calc":              t_calc,
            "t_retrieve":          phases["t_retrieve"],
            "quartets_per_second": n_quartets / t_calc if t_calc > 0 else 0.0,
        }
    )


# ======================================================================== #
# 1. Fixed forest size, varying number of tree groups                      #
# ======================================================================== #


class TestThroughputFixedForest:
    """
    Sweep 1: fixed total forest size, varying number of tree groups.

    Holds the total tree count constant at FIXED_FOREST_N_TREES and sweeps
    n_groups across FIXED_FOREST_N_GROUPS.  Each group receives
    floor(n_trees / n_groups) trees; any remainder goes to the first groups.

    The total kernel work (n_quartets × n_trees) is fixed, so throughput
    should be roughly constant.  Any decline with increasing group count
    reflects per-group accumulation overhead in the kernel's inner loop.

    Both steiner=False and steiner=True are tested; comparing them isolates
    the cost of accumulating Steiner-length statistics alongside topology counts.
    """

    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_python(self, benchmark, n_groups, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="python",
            steiner=steiner,
            sweep="fixed_forest",
            morton_order=morton_order,
        )

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_cpu_parallel(self, benchmark, n_groups, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="cpu-parallel",
            steiner=steiner,
            sweep="fixed_forest",
            morton_order=morton_order,
        )

    @pytest.mark.requires_cuda
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_cuda(self, benchmark, n_groups, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="cuda",
            steiner=steiner,
            sweep="fixed_forest",
            morton_order=morton_order,
        )

    @pytest.mark.requires_mlx
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_mlx(self, benchmark, n_groups, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="mlx",
            steiner=steiner,
            sweep="fixed_forest",
            morton_order=morton_order,
        )


# ======================================================================== #
# 2. Fixed group count, varying total forest size                          #
# ======================================================================== #


class TestThroughputFixedGroups:
    """
    Sweep 2: fixed group count, varying total forest size.

    Holds the group count constant at FIXED_GROUPS_N_GROUPS and sweeps
    n_trees across FIXED_GROUPS_N_TREES.  All sizes start above the GPU
    L2-cache escape threshold (~7 200 trees at 20 leaves/tree ≈ 26 MB for
    A100), so every data point is bandwidth-bound rather than cache-bound.

    Measures how kernel throughput scales with forest depth (trees per group).
    GPU backends should show near-linear growth up to peak device occupancy;
    CPU-parallel backends may saturate at available core count.

    Both steiner=False and steiner=True are tested.
    """

    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_python(self, benchmark, n_trees, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="python",
            steiner=steiner,
            sweep="fixed_groups",
            morton_order=morton_order,
        )

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_cpu_parallel(self, benchmark, n_trees, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="cpu-parallel",
            steiner=steiner,
            sweep="fixed_groups",
            morton_order=morton_order,
        )

    @pytest.mark.requires_cuda
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_cuda(self, benchmark, n_trees, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="cuda",
            steiner=steiner,
            sweep="fixed_groups",
            morton_order=morton_order,
        )

    @pytest.mark.requires_mlx
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_mlx(self, benchmark, n_trees, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="mlx",
            steiner=steiner,
            sweep="fixed_groups",
            morton_order=morton_order,
        )


# ======================================================================== #
# 3. Fixed tree count, varying number of leaves                            #
# ======================================================================== #


class TestThroughputFixedTrees:
    """
    Fixed tree count (FIXED_TREES_N_TREES trees), varying leaf count.

    Sweeps n_leaves ∈ FIXED_TREES_N_LEAVES with a single group.
    Tests whether calculation-phase throughput is independent of tree size,
    as predicted by the O(1) LCA algorithm.  Cache effects will appear as a
    throughput drop at larger leaf counts where the sparse table outgrows the
    GPU's L2 cache.  Per-tree sparse table size grows as O(n log n) in the
    number of leaves, so cache spill happens well before the tree count limit.

    Note: forest construction is slow at large leaf counts (excluded from
    timing via t_device_load), but the calculation phase should remain flat.
    """

    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_leaves", FIXED_TREES_N_LEAVES)
    def test_python(self, benchmark, n_leaves, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_TREES_N_TREES,
            n_groups=1,
            backend="python",
            steiner=steiner,
            sweep="fixed_trees",
            n_leaves=n_leaves,
            morton_order=morton_order,
        )

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_leaves", FIXED_TREES_N_LEAVES)
    def test_cpu_parallel(self, benchmark, n_leaves, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_TREES_N_TREES,
            n_groups=1,
            backend="cpu-parallel",
            steiner=steiner,
            sweep="fixed_trees",
            n_leaves=n_leaves,
            morton_order=morton_order,
        )

    @pytest.mark.requires_cuda
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_leaves", FIXED_TREES_N_LEAVES)
    def test_cuda(self, benchmark, n_leaves, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_TREES_N_TREES,
            n_groups=1,
            backend="cuda",
            steiner=steiner,
            sweep="fixed_trees",
            n_leaves=n_leaves,
            morton_order=morton_order,
        )

    @pytest.mark.requires_mlx
    @pytest.mark.parametrize("morton_order", [False, True], ids=["standard", "morton"])  # [MORTON_SCHED]
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_leaves", FIXED_TREES_N_LEAVES)
    def test_mlx(self, benchmark, n_leaves, steiner, morton_order):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_TREES_N_TREES,
            n_groups=1,
            backend="mlx",
            steiner=steiner,
            sweep="fixed_trees",
            n_leaves=n_leaves,
            morton_order=morton_order,
        )
