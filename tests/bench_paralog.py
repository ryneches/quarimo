"""
tests/bench_paralog.py
======================
Benchmarks and convergence-data tests for Forest.resolve_paralogs().

Two test classes with distinct purposes:

  TestBenchResolveParalogs
  ────────────────────────
  Cross-kernel fidelity benchmarks across a three-axis condition matrix:

    paralog_freq    — fraction of genomes with > 1 copy
                      LOW  (1 paralog genome / 16 total ≈  6 %)
                      MED  (3 paralog genomes / 18 total ≈ 17 %)
                      HIGH (5 paralog genomes / 20 total ≈ 25 %)

    paralog_intensity — copies per paralog genome
                      k2 (2 copies)  k3 (3 copies)

    background_discordance — level of ILS-like noise in non-paralog quartets
                      concordant  : all trees within a group share the same
                                    leaf order (identical topology → zero
                                    within-group quartet noise; optimiser sees
                                    a pure between-group signal)
                      mixed       : each tree is an independent random balanced
                                    tree (realistic ILS-like background noise)
                      discordant  : alternating balanced / caterpillar trees
                                    with independent leaf shuffles (maximum
                                    within-group quartet conflict)

  Benchmarks are split by backend (python, cpu-parallel) so that timing data
  from different machines can be combined.  A separate TestParalogKernelAgreement
  class asserts that every backend returns bit-identical counts for every cell
  in the condition matrix — fidelity is thus verified independently of timing.

  Problem size is chosen so that:
    • python backend completes each benchmark cell in < 2 s (feasible in CI)
    • cpu-parallel backend has enough work for low relative timing noise
  Specifically: 4 groups × 30 trees × 2 000 quartets = 240 K (qi, tree) pairs
  per full pass, plus optimizer sweeps.


  TestStressResolveParalogs
  ─────────────────────────
  Data-collection tests for plotting optimizer convergence (qed_history).
  The complete qed_history list is stored in benchmark.extra_info so that
  results saved with --benchmark-json can be parsed and plotted directly.

  Two-axis matrix: paralog_freq × paralog_intensity.

  A conservative non-large_scale tier (≈ feasible with the pure-Python
  kernel) and a larger @large_scale tier are provided; both tiers use
  backend="best" so the fastest available kernel is selected automatically.

  Non-large_scale:   3 groups × 30 trees × 2 000 quartets (≈ 180 K pairs)
  @large_scale:      3 groups × 300 trees × 50 000 quartets (≈ 45 M pairs)
                     (requires_cpu_parallel — Python would time out)

Running
───────
  # Fidelity + bench + stress (no large_scale), benchmarks timed
  pytest tests/bench_paralog.py -m "not large_scale" --benchmark-only

  # Stress-only: see qed_history in console output
  pytest tests/bench_paralog.py -k "Stress" -m "not large_scale" -s

  # Full stress matrix including large sizes (needs Numba)
  pytest tests/bench_paralog.py -k "Stress" --benchmark-json=paralog_stress.json

  # Cross-kernel fidelity only (needs Numba)
  pytest tests/bench_paralog.py -k "Agreement"
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from quarimo import Forest, Quartets
from quarimo._backend import backends
from quarimo._context import quiet


# ======================================================================== #
# Configuration                                                             #
# ======================================================================== #

BRANCH_LENGTH: float = 1.0
RANDOM_SEED: int = 42
QUARTET_SEED: int = 17

# ── Benchmark forest shape ────────────────────────────────────────────────
BENCH_N_GROUPS:          int = 4
BENCH_N_TREES_PER_GROUP: int = 30
BENCH_N_SINGLETONS:      int = 15   # non-paralog genomes (leaves)
BENCH_N_QUARTETS:        int = 2_000

# ── Benchmark parametrize axes ───────────────────────────────────────────
#
# paralog_freq: (id, n_paralog_genomes)
#   total genomes = n_singletons + n_paralog_genomes; fraction = npg / total
PARALOG_FREQ: list[tuple[str, int]] = [
    ("pf_low", 1),   #  1 / 16 ≈  6 %
    ("pf_med", 3),   #  3 / 18 ≈ 17 %
    ("pf_hi",  5),   #  5 / 20 ≈ 25 %
]

# paralog_intensity: (id, copies_per_genome)
PARALOG_INTENSITY: list[tuple[str, int]] = [
    ("k2", 2),
    ("k3", 3),
]

# background_discordance: (id, discordance_mode)
DISCORDANCE: list[tuple[str, str]] = [
    ("concordant",  "concordant"),
    ("mixed",       "mixed"),
    ("discordant",  "discordant"),
]

# ── Stress test forest shape ──────────────────────────────────────────────
STRESS_N_GROUPS:          int = 3
STRESS_N_TREES_PER_GROUP: int = 30    # non-large_scale tier
STRESS_N_SINGLETONS:      int = 15
STRESS_N_QUARTETS:        int = 2_000

STRESS_N_TREES_PER_GROUP_LARGE: int = 300
STRESS_N_QUARTETS_LARGE:        int = 50_000

# ── Stress parametrize axes (conservative 3 × 2 matrix) ──────────────────
STRESS_PARALOG_FREQ: list[tuple[str, int]] = [
    ("pf_low", 1),
    ("pf_med", 3),
    ("pf_hi",  5),
]
STRESS_PARALOG_INTENSITY: list[tuple[str, int]] = [
    ("k2", 2),
    ("k3", 3),
]


# ======================================================================== #
# Tree generators                                                           #
# ======================================================================== #


def _singleton_name(i: int) -> str:
    return f"S{i}"


def _paralog_copy_name(genome_idx: int, copy_idx: int) -> str:
    return f"P{genome_idx}c{copy_idx}"


def _balanced_newick(leaf_names: list[str], branch_length: float = BRANCH_LENGTH) -> str:
    """Balanced binary tree NEWICK from an ordered leaf list."""
    def _build(ns: list[str]) -> str:
        if len(ns) == 1:
            return f"{ns[0]}:{branch_length}"
        mid = len(ns) // 2
        return f"({_build(ns[:mid])},{_build(ns[mid:])}):{branch_length}"
    return _build(leaf_names) + ";"


def _caterpillar_newick(leaf_names: list[str], branch_length: float = BRANCH_LENGTH) -> str:
    """Caterpillar (maximally unbalanced) binary tree NEWICK."""
    t = f"({leaf_names[-2]}:{branch_length},{leaf_names[-1]}:{branch_length})"
    for name in reversed(leaf_names[:-2]):
        t = f"({name}:{branch_length},{t}:{branch_length})"
    return t + ";"


def make_taxon_map(n_paralog_genomes: int, copies_per_genome: int) -> dict[str, str]:
    """Map each paralog copy leaf name → genome name."""
    return {
        _paralog_copy_name(g, c): f"P{g}"
        for g in range(n_paralog_genomes)
        for c in range(copies_per_genome)
    }


def generate_paralog_ensemble(
    n_groups: int,
    n_trees_per_group: int,
    n_singletons: int,
    n_paralog_genomes: int,
    copies_per_genome: int = 2,
    discordance: str = "mixed",
    branch_length: float = BRANCH_LENGTH,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Build a MUL-tree forest dict and taxon_map for benchmarking.

    Every tree contains all singletons plus all paralog copies.  Three
    discordance modes control how much within-group quartet noise the
    non-paralog taxa introduce:

    concordant
        Each group has one fixed leaf order used for every tree in the group.
        Different groups have independently shuffled orderings, so between-group
        copy-slot disagreement is the sole source of discordance.  The
        optimiser sees the cleanest possible signal.

    mixed
        Each (group, tree) pair has an independently shuffled balanced topology.
        Models realistic ILS-like background discordance.

    discordant
        Each (group, tree) pair has an independently shuffled topology chosen
        alternately from balanced and caterpillar families, maximising
        within-group quartet conflict.

    Parameters
    ----------
    n_groups, n_trees_per_group : ensemble shape
    n_singletons   : non-paralog leaves per tree
    n_paralog_genomes : number of genomes contributing > 1 copy
    copies_per_genome : k (copies per paralog genome)
    discordance    : 'concordant' | 'mixed' | 'discordant'
    seed           : RNG seed for leaf-order shuffling

    Returns
    -------
    (trees_dict, taxon_map)
    """
    if discordance not in ("concordant", "mixed", "discordant"):
        raise ValueError(f"Unknown discordance mode {discordance!r}")

    rng = random.Random(seed)
    all_leaves: list[str] = (
        [_singleton_name(i) for i in range(n_singletons)]
        + [
            _paralog_copy_name(g, c)
            for g in range(n_paralog_genomes)
            for c in range(copies_per_genome)
        ]
    )
    tmap = make_taxon_map(n_paralog_genomes, copies_per_genome)

    trees: dict[str, list[str]] = {}
    for g in range(n_groups):
        group_trees: list[str] = []

        if discordance == "concordant":
            # One fixed ordering per group — all trees within the group are
            # identical (pure across-group copy-slot disagreement signal).
            base = all_leaves[:]
            rng.shuffle(base)
            group_trees = [_balanced_newick(base, branch_length)] * n_trees_per_group

        elif discordance == "mixed":
            for _ in range(n_trees_per_group):
                s = all_leaves[:]
                rng.shuffle(s)
                group_trees.append(_balanced_newick(s, branch_length))

        else:  # discordant
            for t in range(n_trees_per_group):
                s = all_leaves[:]
                rng.shuffle(s)
                if (g * n_trees_per_group + t) % 2 == 0:
                    group_trees.append(_balanced_newick(s, branch_length))
                else:
                    group_trees.append(_caterpillar_newick(s, branch_length))

        trees[f"group{g}"] = group_trees

    return trees, tmap


def _build(
    n_groups: int,
    n_trees_per_group: int,
    n_singletons: int,
    n_paralog_genomes: int,
    copies_per_genome: int,
    n_quartets: int,
    discordance: str = "mixed",
    seed: int = RANDOM_SEED,
) -> tuple[Forest, Quartets]:
    """Construct a Forest + Quartets pair (outside any timed region)."""
    trees, tmap = generate_paralog_ensemble(
        n_groups, n_trees_per_group, n_singletons,
        n_paralog_genomes, copies_per_genome, discordance, seed=seed,
    )
    with quiet():
        f = Forest(trees, taxon_map=tmap)
    q = Quartets.random(f, count=n_quartets, seed=QUARTET_SEED)
    return f, q


def _extra_info(
    benchmark,
    *,
    backend: str,
    n_trees_per_group: int,
    n_groups: int,
    n_paralog_genomes: int,
    copies_per_genome: int,
    n_quartets: int,
    n_leaves: int,
    n_global_taxa: int,
    discordance: str,
    qed_history: list[float] | None = None,
    n_optimizer_iters: int | None = None,
) -> None:
    """Populate benchmark.extra_info with uniform keys for all cells."""
    n_trees = n_trees_per_group * n_groups
    d = {
        "backend":           backend,
        "n_trees_per_group": n_trees_per_group,
        "n_groups":          n_groups,
        "n_trees":           n_trees,
        "n_leaves":          n_leaves,
        "n_global_taxa":     n_global_taxa,
        "n_paralog_genomes": n_paralog_genomes,
        "copies_per_genome": copies_per_genome,
        "n_quartets":        n_quartets,
        "discordance":       discordance,
        "cells":             n_quartets * n_trees,
    }
    if qed_history is not None:
        d["qed_initial"] = qed_history[0]
        d["qed_final"]   = qed_history[-1]
        d["qed_history"] = qed_history        # full list for plotting
    if n_optimizer_iters is not None:
        d["n_optimizer_iters"] = n_optimizer_iters
    benchmark.extra_info.update(d)


# ======================================================================== #
# 1. Benchmarks (timing + fidelity via separate class)                     #
# ======================================================================== #


class TestBenchResolveParalogs:
    """
    Timed benchmarks for resolve_paralogs() — one test per backend per cell.

    The condition matrix has three axes:
      paralog_freq × paralog_intensity × background_discordance

    Forest and quartet objects are built once outside the timed loop.
    forest.global_to_local is reset to its initial state before every
    timed iteration so each round starts from identical conditions.

    A pre-run outside the benchmark loop captures optimizer metadata
    (n_iterations, qed_history) which is stored in extra_info.

    To compute throughput from saved JSON:
        cells_per_second = extra_info["cells"] / stats["mean"]
    """

    @pytest.mark.parametrize("disc",  [v for _, v in DISCORDANCE],         ids=[k for k, _ in DISCORDANCE])
    @pytest.mark.parametrize("cpg",   [v for _, v in PARALOG_INTENSITY],   ids=[k for k, _ in PARALOG_INTENSITY])
    @pytest.mark.parametrize("npg",   [v for _, v in PARALOG_FREQ],        ids=[k for k, _ in PARALOG_FREQ])
    def test_python(self, benchmark, npg, cpg, disc):
        f, q = _build(BENCH_N_GROUPS, BENCH_N_TREES_PER_GROUP,
                      BENCH_N_SINGLETONS, npg, cpg, BENCH_N_QUARTETS,
                      discordance=disc)
        init_g2l = f.global_to_local.copy()
        _, opt0 = f.resolve_paralogs(q, backend="python")   # metadata + warmup
        f.global_to_local[:] = init_g2l

        _extra_info(benchmark, backend="python",
                    n_trees_per_group=BENCH_N_TREES_PER_GROUP,
                    n_groups=BENCH_N_GROUPS,
                    n_paralog_genomes=npg, copies_per_genome=cpg,
                    n_quartets=BENCH_N_QUARTETS,
                    n_leaves=f._trees[0].n_leaves,
                    n_global_taxa=f.n_global_taxa,
                    discordance=disc,
                    qed_history=opt0.qed_history,
                    n_optimizer_iters=opt0.n_iterations)

        def _run():
            f.global_to_local[:] = init_g2l
            return f.resolve_paralogs(q, backend="python")

        benchmark(_run)

    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("disc",  [v for _, v in DISCORDANCE],         ids=[k for k, _ in DISCORDANCE])
    @pytest.mark.parametrize("cpg",   [v for _, v in PARALOG_INTENSITY],   ids=[k for k, _ in PARALOG_INTENSITY])
    @pytest.mark.parametrize("npg",   [v for _, v in PARALOG_FREQ],        ids=[k for k, _ in PARALOG_FREQ])
    def test_cpu_parallel(self, benchmark, npg, cpg, disc):
        """Numba delta kernel — JIT warmup excluded from timing."""
        f, q = _build(BENCH_N_GROUPS, BENCH_N_TREES_PER_GROUP,
                      BENCH_N_SINGLETONS, npg, cpg, BENCH_N_QUARTETS,
                      discordance=disc)
        init_g2l = f.global_to_local.copy()
        _, opt0 = f.resolve_paralogs(q, backend="cpu-parallel")   # JIT warmup
        f.global_to_local[:] = init_g2l

        _extra_info(benchmark, backend="cpu-parallel",
                    n_trees_per_group=BENCH_N_TREES_PER_GROUP,
                    n_groups=BENCH_N_GROUPS,
                    n_paralog_genomes=npg, copies_per_genome=cpg,
                    n_quartets=BENCH_N_QUARTETS,
                    n_leaves=f._trees[0].n_leaves,
                    n_global_taxa=f.n_global_taxa,
                    discordance=disc,
                    qed_history=opt0.qed_history,
                    n_optimizer_iters=opt0.n_iterations)

        def _run():
            f.global_to_local[:] = init_g2l
            return f.resolve_paralogs(q, backend="cpu-parallel")

        benchmark(_run)


# ======================================================================== #
# 2. Cross-kernel fidelity                                                  #
# ======================================================================== #


@pytest.mark.requires_cpu_parallel
class TestParalogKernelAgreement:
    """
    Assert that python and cpu-parallel backends produce bit-identical
    results for every cell in the full condition matrix.

    Checked per cell:
      • counts arrays are equal (np.testing.assert_array_equal)
      • qed_history has the same length and values (within 1e-10)
      • n_iterations agree

    This class is separate from the timing benchmarks so that fidelity
    failures are reported as distinct test failures with clear error messages.
    """

    @pytest.mark.parametrize("disc",  [v for _, v in DISCORDANCE],         ids=[k for k, _ in DISCORDANCE])
    @pytest.mark.parametrize("cpg",   [v for _, v in PARALOG_INTENSITY],   ids=[k for k, _ in PARALOG_INTENSITY])
    @pytest.mark.parametrize("npg",   [v for _, v in PARALOG_FREQ],        ids=[k for k, _ in PARALOG_FREQ])
    def test_python_vs_cpu_parallel(self, npg, cpg, disc):
        f, q = _build(BENCH_N_GROUPS, BENCH_N_TREES_PER_GROUP,
                      BENCH_N_SINGLETONS, npg, cpg, BENCH_N_QUARTETS,
                      discordance=disc)
        init_g2l = f.global_to_local.copy()

        result_py, opt_py = f.resolve_paralogs(q, backend="python")
        py_counts = result_py.counts.copy()
        py_hist   = list(opt_py.qed_history)

        f.global_to_local[:] = init_g2l
        result_nb, opt_nb = f.resolve_paralogs(q, backend="cpu-parallel")

        ctx = f"(npg={npg}, k={cpg}, disc={disc!r})"

        np.testing.assert_array_equal(
            py_counts, result_nb.counts,
            err_msg=f"counts mismatch python vs cpu-parallel {ctx}",
        )
        assert len(py_hist) == len(opt_nb.qed_history), (
            f"qed_history length differs python={len(py_hist)} "
            f"vs nb={len(opt_nb.qed_history)} {ctx}"
        )
        for i, (qpy, qnb) in enumerate(zip(py_hist, opt_nb.qed_history)):
            assert abs(qpy - qnb) < 1e-10, (
                f"qed_history[{i}] python={qpy:.10f} vs cpu-parallel={qnb:.10f} {ctx}"
            )
        assert opt_py.n_iterations == opt_nb.n_iterations, (
            f"n_iterations differ python={opt_py.n_iterations} "
            f"vs nb={opt_nb.n_iterations} {ctx}"
        )


# ======================================================================== #
# 3. Convergence stress tests (qed_history data collection)                #
# ======================================================================== #


class TestStressResolveParalogs:
    """
    Data-collection benchmarks for optimizer convergence analysis.

    Each cell runs resolve_paralogs() with the best available backend and
    records the full qed_history in benchmark.extra_info.  Saving results
    with --benchmark-json produces a file that can be parsed and plotted:

        import json, matplotlib.pyplot as plt
        data = json.load(open("paralog_stress.json"))
        for b in data["benchmarks"]:
            ei = b["extra_info"]
            plt.plot(ei["qed_history"],
                     label=f"pf={ei['n_paralog_genomes']} k={ei['copies_per_genome']}")

    Three invariants are checked after each run:
      (1) Count conservation — sum over topology slots unchanged before / after
      (2) Non-negativity     — no count bin < 0
      (3) QED monotonicity   — qed_history is non-decreasing

    Two tiers:
      non-large_scale : 3 groups × 30 trees × 2 000 quartets  (Python-feasible)
      @large_scale    : 3 groups × 300 trees × 50 000 quartets (requires Numba)
    """

    # ── non-large_scale tier ─────────────────────────────────────────────

    @pytest.mark.parametrize("cpg",  [v for _, v in STRESS_PARALOG_INTENSITY], ids=[k for k, _ in STRESS_PARALOG_INTENSITY])
    @pytest.mark.parametrize("npg",  [v for _, v in STRESS_PARALOG_FREQ],      ids=[k for k, _ in STRESS_PARALOG_FREQ])
    def test_convergence(self, benchmark, npg, cpg):
        """
        Conservative matrix — qed_history for each paralog_freq × paralog_intensity
        cell.  Runs with the best available backend.  Non-large_scale (Python-feasible).
        """
        f, q = _build(STRESS_N_GROUPS, STRESS_N_TREES_PER_GROUP,
                      STRESS_N_SINGLETONS, npg, cpg, STRESS_N_QUARTETS,
                      discordance="mixed")
        init_g2l = f.global_to_local.copy()
        counts_before = f.quartet_topology(q, backend="python").counts.copy()

        # Resolve once so the reported backend name is concrete, not "best"
        resolved_backend = backends.resolve("best")

        # One outer run for invariant checks and metadata
        result0, opt0 = f.resolve_paralogs(q, backend=resolved_backend)
        _check_invariants(counts_before, result0, opt0,
                          label=f"npg={npg} k={cpg}")

        _summarise(npg, cpg, opt0)          # console output for -s mode

        f.global_to_local[:] = init_g2l    # reset for timed loop

        _extra_info(benchmark, backend=resolved_backend,
                    n_trees_per_group=STRESS_N_TREES_PER_GROUP,
                    n_groups=STRESS_N_GROUPS,
                    n_paralog_genomes=npg, copies_per_genome=cpg,
                    n_quartets=STRESS_N_QUARTETS,
                    n_leaves=f._trees[0].n_leaves,
                    n_global_taxa=f.n_global_taxa,
                    discordance="mixed",
                    qed_history=opt0.qed_history,
                    n_optimizer_iters=opt0.n_iterations)

        def _run():
            f.global_to_local[:] = init_g2l
            return f.resolve_paralogs(q, backend=resolved_backend)

        benchmark(_run)

    # ── large_scale tier ─────────────────────────────────────────────────

    @pytest.mark.large_scale
    @pytest.mark.requires_cpu_parallel
    @pytest.mark.parametrize("cpg",  [v for _, v in STRESS_PARALOG_INTENSITY], ids=[k for k, _ in STRESS_PARALOG_INTENSITY])
    @pytest.mark.parametrize("npg",  [v for _, v in STRESS_PARALOG_FREQ],      ids=[k for k, _ in STRESS_PARALOG_FREQ])
    def test_convergence_large(self, benchmark, npg, cpg):
        """
        Large-scale tier — 3 groups × 300 trees × 50 000 quartets.
        Requires Numba (cpu-parallel).  Produces detailed qed_history curves
        suitable for convergence plots across the full 3 × 2 matrix.
        """
        f, q = _build(STRESS_N_GROUPS, STRESS_N_TREES_PER_GROUP_LARGE,
                      STRESS_N_SINGLETONS, npg, cpg, STRESS_N_QUARTETS_LARGE,
                      discordance="mixed")
        init_g2l = f.global_to_local.copy()
        counts_before = f.quartet_topology(q, backend="cpu-parallel").counts.copy()

        result0, opt0 = f.resolve_paralogs(q, backend="cpu-parallel")
        _check_invariants(counts_before, result0, opt0,
                          label=f"npg={npg} k={cpg} LARGE")

        _summarise(npg, cpg, opt0)

        f.global_to_local[:] = init_g2l

        _extra_info(benchmark, backend="cpu-parallel",
                    n_trees_per_group=STRESS_N_TREES_PER_GROUP_LARGE,
                    n_groups=STRESS_N_GROUPS,
                    n_paralog_genomes=npg, copies_per_genome=cpg,
                    n_quartets=STRESS_N_QUARTETS_LARGE,
                    n_leaves=f._trees[0].n_leaves,
                    n_global_taxa=f.n_global_taxa,
                    discordance="mixed",
                    qed_history=opt0.qed_history,
                    n_optimizer_iters=opt0.n_iterations)

        def _run():
            f.global_to_local[:] = init_g2l
            return f.resolve_paralogs(q, backend="cpu-parallel")

        benchmark(_run)


# ======================================================================== #
# Shared helpers                                                            #
# ======================================================================== #


def _check_invariants(counts_before, result, opt, *, label: str) -> None:
    """Assert the three invariants that constitute a functional spec."""
    # (1) Count conservation
    np.testing.assert_array_equal(
        counts_before.sum(axis=2),
        result.counts.sum(axis=2),
        err_msg=f"[{label}] count totals changed after optimization",
    )
    # (2) Non-negativity
    assert np.all(result.counts >= 0), (
        f"[{label}] negative counts after optimization "
        f"(min={result.counts.min()})"
    )
    # (3) QED monotonicity
    hist = opt.qed_history
    for i in range(len(hist) - 1):
        assert hist[i + 1] >= hist[i] - 1e-9, (
            f"[{label}] qed_history decreased at sweep {i}: "
            f"{hist[i]:.6f} → {hist[i + 1]:.6f}"
        )


def _summarise(npg: int, cpg: int, opt) -> None:
    """Print a compact qed_history summary (visible with pytest -s)."""
    hist = opt.qed_history
    delta = hist[-1] - hist[0]
    arrow = "↑" if delta > 1e-6 else ("=" if abs(delta) < 1e-6 else "↓")
    print(
        f"\n  [stress npg={npg} k={cpg}] "
        f"sweeps={opt.n_iterations}  converged={opt.converged}  "
        f"QED: {hist[0]:+.4f} → {hist[-1]:+.4f} ({arrow}{abs(delta):.4f})  "
        f"history={[f'{v:+.3f}' for v in hist]}"
    )
