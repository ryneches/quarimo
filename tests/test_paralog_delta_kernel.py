"""
tests/test_paralog_delta_kernel.py
===================================
Tests for Step 4: the quartet-counts delta kernel.

Covers:
- Identity permutation via delta → counts unchanged (python + cpu-parallel)
- Swap permutation via delta agrees with a fresh full pass
- Count conservation: total count per quartet is unchanged for non-paralog taxa
- apply_quartet_counts_delta returns correct trial_global_to_local
- Empty affected set → counts unchanged
"""

import numpy as np
import pytest

from quarimo import Forest, Quartets, use_backend
from quarimo._paralog import build_paralog_data
from quarimo._backend import backends


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Two trees: tree0 has 1 G copy, tree1 has 2 G copies + singletons X, Y
FOREST_TREES = [
    "(G1:1,(X:1,Y:1):1);",
    "((G1:1,G2:1):1,(X:1,Y:1):1);",
]
TAXON_MAP = {"G1": "G", "G2": "G"}


def make_forest():
    return Forest(FOREST_TREES, taxon_map=TAXON_MAP)


def make_quartets(f):
    # The single possible quartet over 4 global taxa
    return Quartets(f, seed=[tuple(range(f.n_global_taxa))], offset=0, count=1)


def full_pass_counts(f, q, g2l=None):
    """Run a full-pass topology count.  Optionally swap global_to_local."""
    if g2l is not None:
        old = f.global_to_local.copy()
        f.global_to_local[:] = g2l
        # _kernel_data holds a reference, so the in-place update propagates
        # automatically (same underlying ndarray).
        result = f.quartet_topology(q, backend="python")
        f.global_to_local[:] = old
    else:
        result = f.quartet_topology(q, backend="python")
    return result.counts.copy()


# ── Helpers ───────────────────────────────────────────────────────────────────

BACKENDS = ["python"]
if backends.numba:
    BACKENDS.append("cpu-parallel")


# ── Identity permutation ─────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_identity_permutation_counts_unchanged(backend):
    """Identity permutation via delta must leave counts_out unchanged."""
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    counts_ref = full_pass_counts(f, q)
    counts_delta = counts_ref.copy()

    li = pd.genome_names.index("G")
    k = int(pd.copy_offsets[li + 1]) - int(pd.copy_offsets[li])
    perm = np.arange(k, dtype=np.int32)

    with use_backend(backend):
        f.apply_quartet_counts_delta(pd, li, perm, counts_delta, backend=backend)

    np.testing.assert_array_equal(
        counts_delta, counts_ref,
        err_msg=f"Identity permutation changed counts ({backend})",
    )


# ── Swap agrees with full re-pass ────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_swap_agrees_with_full_pass(backend):
    """
    Delta-updated counts after a swap must equal a fresh full pass run
    with the swapped global_to_local.
    """
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    li = pd.genome_names.index("G")
    perm = np.array([1, 0], dtype=np.int32)

    # Full pass with original assignment
    counts_before = full_pass_counts(f, q)
    counts_delta = counts_before.copy()

    # Apply delta
    trial_g2l = f.apply_quartet_counts_delta(pd, li, perm, counts_delta, backend=backend)

    # Full pass with swapped assignment
    counts_after_full = full_pass_counts(f, q, g2l=trial_g2l)

    np.testing.assert_array_equal(
        counts_delta, counts_after_full,
        err_msg=f"Delta counts don't match full re-pass ({backend})",
    )


# ── Count conservation ────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_total_count_conserved(backend):
    """
    The total count summed over all topology slots for each quartet must
    be the same before and after any permutation (permutation only
    reshuffles topology attribution, not tree membership).
    """
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    li = pd.genome_names.index("G")
    perm = np.array([1, 0], dtype=np.int32)

    counts_before = full_pass_counts(f, q)
    total_before = counts_before.sum(axis=(1, 2))  # per quartet

    counts_delta = counts_before.copy()
    f.apply_quartet_counts_delta(pd, li, perm, counts_delta, backend=backend)
    total_after = counts_delta.sum(axis=(1, 2))

    np.testing.assert_array_equal(
        total_after, total_before,
        err_msg="Total count per quartet changed after delta",
    )


# ── Return value ──────────────────────────────────────────────────────────────


def test_returns_trial_global_to_local():
    """apply_quartet_counts_delta returns the trial global_to_local."""
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    li = pd.genome_names.index("G")
    perm = np.array([1, 0], dtype=np.int32)
    counts = full_pass_counts(f, q)

    trial = f.apply_quartet_counts_delta(pd, li, perm, counts)
    assert trial.shape == f.global_to_local.shape
    assert trial.dtype == np.int32

    # The trial mapping should differ from the original in the permuted slots
    _, _, _, affected_qi = pd.apply_permutation(li, perm, f.global_to_local)
    assert not np.array_equal(trial, f.global_to_local), \
        "trial_global_to_local should differ from original after a swap"


# ── Empty affected set ────────────────────────────────────────────────────────


def test_empty_affected_quartets_no_change():
    """
    When no quartets are affected (impossible genome index for a no-op),
    counts must remain unchanged.
    """
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    counts_ref = full_pass_counts(f, q)
    counts_delta = counts_ref.copy()

    # Build a paralog_data where the paralog copy global IDs are not in
    # any quartet — force this by using a fake copy with gid = n_global_taxa-99
    # The simplest approach: identity permutation always produces no delta
    li = pd.genome_names.index("G")
    k = int(pd.copy_offsets[li + 1]) - int(pd.copy_offsets[li])
    perm = np.arange(k, dtype=np.int32)  # identity

    f.apply_quartet_counts_delta(pd, li, perm, counts_delta)
    np.testing.assert_array_equal(counts_delta, counts_ref)


# ── Double-swap recovers original ─────────────────────────────────────────────


@pytest.mark.parametrize("backend", BACKENDS)
def test_double_swap_recovers_original(backend):
    """Applying the swap permutation twice returns to the original counts."""
    f = make_forest()
    q = make_quartets(f)
    pd = build_paralog_data(f, q)

    li = pd.genome_names.index("G")
    perm = np.array([1, 0], dtype=np.int32)

    counts_orig = full_pass_counts(f, q)
    counts_work = counts_orig.copy()

    # First swap
    trial1 = f.apply_quartet_counts_delta(pd, li, perm, counts_work, backend=backend)
    # Second swap (apply_permutation on trial1 back to original g2l)
    # We need to build a temporary pd with trial1 as the "current" g2l
    # The simplest way: call apply_permutation manually, then call the kernel
    trial2, affected_trees, affected_taxa, affected_qi = pd.apply_permutation(
        li, perm, trial1
    )
    if len(affected_qi) > 0 and len(affected_trees) > 0:
        # Apply delta from trial1 back to trial2
        from quarimo._forest import Forest as F
        kd = f._kernel_data
        F._python_quartet_counts_delta(
            affected_taxa,
            affected_qi,
            trial1,   # old
            trial2,   # new
            affected_trees,
            kd.all_first_occ,
            kd.all_root_distance,
            kd.all_euler_tour,
            kd.all_euler_depth,
            kd.all_sparse_table,
            kd.all_log2_table,
            kd.node_offsets,
            kd.tour_offsets,
            kd.sp_offsets,
            kd.lg_offsets,
            kd.sp_tour_widths,
            kd.tree_to_group_idx,
            kd.polytomy_offsets,
            kd.polytomy_nodes,
            kd.tree_multiplicities,
            counts_work,
        )

    np.testing.assert_array_equal(
        counts_work, counts_orig,
        err_msg="Double-swap did not recover original counts",
    )
