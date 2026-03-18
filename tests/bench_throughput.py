"""
tests/bench_throughput.py
=========================
Quartet topology throughput benchmarks across a matrix of forest sizes,
group counts, Steiner modes, and backends.

Two scaling sweeps
------------------
fixed_forest
    Total forest size is held constant (FIXED_FOREST_N_TREES trees).
    The number of tree groups varies from 1 to len(FIXED_FOREST_N_GROUPS).
    Each group receives floor(n_trees / n_groups) trees (last group takes
    the remainder).  Measures how per-group accumulation overhead scales.

fixed_groups
    Group count is held constant (FIXED_GROUPS_N_GROUPS groups).
    Total forest size varies across FIXED_GROUPS_N_TREES.
    Measures how kernel throughput scales with forest depth.

Four timing phases
------------------
Every call to quartet_topology() emits a structured INFO log line:

    ⏱ t_query_load=<s> t_calc=<s> t_retrieve=<s>

These are captured by a _PhaseCapture log handler attached for the
duration of the benchmark loop.  The median across BENCH_ROUNDS rounds
is stored in benchmark.extra_info alongside t_device_load (measured
directly around Forest construction).

Primary metric: quartets_per_second = n_quartets / t_calc

JSON output
-----------
Run with --benchmark-json to save results for the throughput_benchmarks.py
Marimo notebook:

    pytest tests/bench_throughput.py -m "not large_scale" \\
        --benchmark-json=docs/benchmark_results/throughput_$(hostname)_$(date +%Y%m%d).json

Tuning
------
Adjust N_QUARTETS (per backend), FIXED_FOREST_N_GROUPS, and
FIXED_GROUPS_N_TREES at the top of this file.  The defaults are small
for fast plot development; replace with realistic values once satisfied.
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
FIXED_FOREST_N_TREES:  int       = 1_000
FIXED_FOREST_N_GROUPS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ── Scaling axis 2: fixed group count, varying total forest size ─────────
FIXED_GROUPS_N_GROUPS: int       = 5
FIXED_GROUPS_N_TREES:  list[int] = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000]

# ── Per-backend quartet counts ────────────────────────────────────────────
# Scale up once satisfied with the plot shapes; current values are sized
# for fast development on CPU hardware (< 2 s per non-large_scale trial).
N_QUARTETS: dict[str, int] = {
    "python":       100,
    "cpu-parallel": 1_000,
    "cuda":         10_000,
    "mlx":          10_000,
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
    sweep: str,         # 'fixed_forest' or 'fixed_groups'
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
    forest = _make_grouped_forest(n_total_trees, n_groups)
    t_device_load = perf_counter() - t0

    q = Quartets.random(forest, count=n_quartets, seed=QUARTET_SEED)

    # ── Warmup: triggers JIT/Metal compilation; excluded from timing ──────
    with quiet(), use_backend(backend):
        forest.quartet_topology(q, steiner=steiner)

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
                kwargs={"steiner": steiner},
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
            "n_trees":             n_total_trees,
            "n_groups":            n_groups,
            "n_leaves":            N_LEAVES,
            "n_quartets":          n_quartets,
            "steiner":             steiner,
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
    Fixed total forest size (FIXED_FOREST_N_TREES trees), varying n_groups.

    Sweeps n_groups ∈ FIXED_FOREST_N_GROUPS while holding the total tree
    count constant.  Measures how per-group accumulation overhead grows
    relative to computation throughput as the group count increases.

    Steiner=False and Steiner=True are both tested so the notebook can
    show overhead from Steiner statistics accumulation.
    """

    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_python(self, benchmark, n_groups, steiner):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="python",
            steiner=steiner,
            sweep="fixed_forest",
        )

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_cpu_parallel(self, benchmark, n_groups, steiner):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="cpu-parallel",
            steiner=steiner,
            sweep="fixed_forest",
        )

    @pytest.mark.requires_cuda
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_cuda(self, benchmark, n_groups, steiner):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="cuda",
            steiner=steiner,
            sweep="fixed_forest",
        )

    @pytest.mark.requires_mlx
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_groups", FIXED_FOREST_N_GROUPS)
    def test_mlx(self, benchmark, n_groups, steiner):
        _run_trial(
            benchmark,
            n_total_trees=FIXED_FOREST_N_TREES,
            n_groups=n_groups,
            backend="mlx",
            steiner=steiner,
            sweep="fixed_forest",
        )


# ======================================================================== #
# 2. Fixed group count, varying total forest size                          #
# ======================================================================== #


class TestThroughputFixedGroups:
    """
    Fixed number of groups (FIXED_GROUPS_N_GROUPS), varying total forest size.

    Sweeps n_trees ∈ FIXED_GROUPS_N_TREES with FIXED_GROUPS_N_GROUPS groups.
    Measures how kernel throughput scales with forest depth (more trees per
    group).  GPU backends should show near-linear throughput growth up to
    occupancy saturation; CPU backends may saturate at available core count.
    """

    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_python(self, benchmark, n_trees, steiner):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="python",
            steiner=steiner,
            sweep="fixed_groups",
        )

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_cpu_parallel(self, benchmark, n_trees, steiner):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="cpu-parallel",
            steiner=steiner,
            sweep="fixed_groups",
        )

    @pytest.mark.requires_cuda
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_cuda(self, benchmark, n_trees, steiner):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="cuda",
            steiner=steiner,
            sweep="fixed_groups",
        )

    @pytest.mark.requires_mlx
    @pytest.mark.parametrize("steiner", [False, True], ids=["counts", "steiner"])
    @pytest.mark.parametrize("n_trees", FIXED_GROUPS_N_TREES)
    def test_mlx(self, benchmark, n_trees, steiner):
        _run_trial(
            benchmark,
            n_total_trees=n_trees,
            n_groups=FIXED_GROUPS_N_GROUPS,
            backend="mlx",
            steiner=steiner,
            sweep="fixed_groups",
        )
