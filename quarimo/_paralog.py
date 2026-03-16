"""
_paralog.py
===========
ParalogData dataclass and supporting functions for MUL-tree (multi-labeled
tree) paralog analysis.

Public API
----------
  ParalogData
      Dataclass bundling all paralog-specific arrays for a Forest + Quartets
      pair.  Provides apply_permutation() for delta-kernel dispatch.

  build_paralog_data(forest, quartets) -> ParalogData
      Factory function.  Iterates over all quartets once to build the inverted
      taxon → quartet index, then packages it with the Forest's paralog arrays.

  ParalogOptimizer
      Coordinate-descent optimiser over paralog copy-slot assignments.
      Maximises mean QED by trying all k! permutations per genome per sweep.
"""

from dataclasses import dataclass
from itertools import permutations as _iperms
from typing import List, Tuple

import numpy as np


@dataclass
class ParalogData:
    """
    All paralog-specific data needed for copy-slot optimisation.

    Fields
    ------
    genome_names : list[str]
        Names of genomes with > 1 copy in any tree (same ordering as the
        first axis of *assignments*).
    n_paralog_genomes : int
        ``len(genome_names)``.
    copy_offsets : int32 [n_paralog_genomes + 1]
        CSR into *copy_global_ids*.
        ``copy_global_ids[copy_offsets[li] : copy_offsets[li+1]]`` gives the
        global taxon IDs assigned to copy slots 0, 1, … for genome *li*.
    copy_global_ids : int32 [total_copies]
        Flat array of global IDs for each copy slot.
    leaf_offsets : int32 [n_paralog_genomes * (n_trees + 1)]
        Packed per-genome CSR into *leaf_nodes* (global offsets).
        For genome *li*, the sub-array
        ``leaf_offsets[li*(n_trees+1) : (li+1)*(n_trees+1)]`` maps tree
        index → range in *leaf_nodes*.
    leaf_nodes : int32 [total_paralog_leaves]
        Local leaf node IDs in CSR order.
    assignments : int32 [n_paralog_genomes, max_copies, n_trees]
        ``assignments[li, ci, ti]`` = local leaf node ID for copy slot *ci*
        of genome *li* in tree *ti*, or ``-1`` if absent.
    taxon_quartet_offsets : int32 [n_global_taxa + 1]
        CSR into *taxon_quartet_ids*.
        ``taxon_quartet_ids[taxon_quartet_offsets[gid] :
        taxon_quartet_offsets[gid+1]]`` lists every quartet index *qi* where
        global taxon *gid* appears (in ascending order).
    taxon_quartet_ids : int32 [total]
        Flat array of quartet indices per taxon.
    quartet_taxa : int32 [n_quartets, 4]
        ``quartet_taxa[qi]`` = (t0, t1, t2, t3) global taxon IDs for the
        quartet at position *qi* in the quartet sequence.
    """

    genome_names: List[str]
    n_paralog_genomes: int
    copy_offsets: np.ndarray
    copy_global_ids: np.ndarray
    leaf_offsets: np.ndarray
    leaf_nodes: np.ndarray
    assignments: np.ndarray
    taxon_quartet_offsets: np.ndarray
    taxon_quartet_ids: np.ndarray
    quartet_taxa: np.ndarray

    def apply_permutation(
        self,
        li: int,
        perm: np.ndarray,
        global_to_local: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply copy-slot permutation *perm* for genome *li* and identify
        the quartets and trees that are affected.

        The permutation reassigns leaves to copy slots: copy slot *ci* in
        the new mapping receives the leaf that was previously assigned to
        copy slot ``perm[ci]``.  The identity permutation ``[0, 1, …, k-1]``
        leaves everything unchanged.

        Parameters
        ----------
        li : int
            Paralog genome index (into *genome_names*).
        perm : int array [k]
            New ordering of copy slots.  ``perm[ci]`` = old copy slot whose
            leaf is placed into position *ci*.  A swap of copy0 and copy1 is
            expressed as ``[1, 0]``.
        global_to_local : int32 ndarray [n_trees, n_global_taxa]
            Current copy-slot → local-leaf mapping (not modified in place).

        Returns
        -------
        trial_global_to_local : int32 ndarray [n_trees, n_global_taxa]
            Modified mapping with copy slots permuted for genome *li*.
        affected_tree_ids : int32 ndarray
            Indices of trees where ≥ 2 copies of genome *li* are present
            (the only trees where the permutation can change topology).
        affected_quartet_taxa : int32 ndarray [n_affected_quartets, 4]
            (t0, t1, t2, t3) for each quartet that involves any global ID
            belonging to genome *li*.
        affected_quartet_qi : int32 ndarray [n_affected_quartets]
            Row index into the counts array for each affected quartet.
        """
        # --- Global IDs for this genome's copy slots --------------------- #
        cstart = int(self.copy_offsets[li])
        cend = int(self.copy_offsets[li + 1])
        gids = self.copy_global_ids[cstart:cend]  # int32[k]
        k = len(gids)

        # --- Build trial global_to_local ---------------------------------- #
        trial = global_to_local.copy()
        old_vals = global_to_local[:, gids]  # int32[n_trees, k]

        # Trees where ≥ 2 copies are present (permutation is non-trivial)
        n_present = (old_vals >= 0).sum(axis=1)
        affected_tree_ids = np.where(n_present >= 2)[0].astype(np.int32)

        # Apply permutation: copy slot ci gets the leaf from old slot perm[ci]
        for ci in range(k):
            trial[:, int(gids[ci])] = old_vals[:, int(perm[ci])]

        # --- Collect affected quartet indices ----------------------------- #
        qi_set: set = set()
        for gid in gids.tolist():
            qstart = int(self.taxon_quartet_offsets[gid])
            qend = int(self.taxon_quartet_offsets[gid + 1])
            qi_set.update(self.taxon_quartet_ids[qstart:qend].tolist())

        if not qi_set:
            return (
                trial,
                affected_tree_ids,
                np.empty((0, 4), dtype=np.int32),
                np.empty(0, dtype=np.int32),
            )

        affected_qi = np.array(sorted(qi_set), dtype=np.int32)
        affected_taxa = self.quartet_taxa[affected_qi]  # int32[n_affected, 4]
        return trial, affected_tree_ids, affected_taxa, affected_qi


def build_paralog_data(forest, quartets) -> ParalogData:
    """
    Build a :class:`ParalogData` from a forest and a :class:`Quartets` object.

    Iterates over all quartets once to build the inverted taxon → quartet
    index (``taxon_quartet_offsets`` / ``taxon_quartet_ids``) and stores
    the quartet taxa array (``quartet_taxa``).  Then packages these with
    the Forest's pre-built paralog arrays.

    Parameters
    ----------
    forest : Forest
        Forest with ``taxon_map`` applied.  ``forest.paralog_genome_names``
        must be non-empty.
    quartets : Quartets
        Quartet sequence for which the inverted index is built.

    Returns
    -------
    ParalogData
    """
    n_quartets = quartets.count
    n_global_taxa = forest.n_global_taxa

    # --- Build quartet_taxa: int32[n_quartets, 4] ------------------------- #
    quartet_taxa = np.empty((n_quartets, 4), dtype=np.int32)
    for qi, (t0, t1, t2, t3) in enumerate(quartets):
        quartet_taxa[qi, 0] = t0
        quartet_taxa[qi, 1] = t1
        quartet_taxa[qi, 2] = t2
        quartet_taxa[qi, 3] = t3

    # --- Build inverted index: taxon -> sorted list of quartet indices ----- #
    taxon_lists: List[List[int]] = [[] for _ in range(n_global_taxa)]
    for qi in range(n_quartets):
        for col in range(4):
            taxon_lists[int(quartet_taxa[qi, col])].append(qi)

    counts = np.array([len(lst) for lst in taxon_lists], dtype=np.int32)
    taxon_quartet_offsets = np.zeros(n_global_taxa + 1, dtype=np.int32)
    taxon_quartet_offsets[1:] = np.cumsum(counts)
    total = int(taxon_quartet_offsets[-1])

    taxon_quartet_ids = np.empty(total, dtype=np.int32)
    for gid, lst in enumerate(taxon_lists):
        s = int(taxon_quartet_offsets[gid])
        taxon_quartet_ids[s : s + len(lst)] = lst

    return ParalogData(
        genome_names=forest.paralog_genome_names,
        n_paralog_genomes=len(forest.paralog_genome_names),
        copy_offsets=forest.paralog_copy_offsets,
        copy_global_ids=forest.paralog_copy_global_ids,
        leaf_offsets=forest.paralog_leaf_offsets,
        leaf_nodes=forest.paralog_leaf_nodes,
        assignments=forest.paralog_assignments.copy(),
        taxon_quartet_offsets=taxon_quartet_offsets,
        taxon_quartet_ids=taxon_quartet_ids,
        quartet_taxa=quartet_taxa,
    )


class ParalogOptimizer:
    """
    Coordinate-descent optimiser over paralog copy-slot assignments.

    For each paralog genome, exhaustively tries all *k!* permutations of
    its *k* copy slots and accepts the one that maximises mean QED.  Sweeps
    continue until no genome can be improved (convergence) or *max_iter*
    sweeps are exhausted.

    Parameters
    ----------
    forest : Forest
        The forest whose ``global_to_local`` will be updated in-place as
        swaps are accepted.
    quartets : Quartets
        The quartet sequence used for the full pass and QED computation.
    counts : np.ndarray, int32, shape (n_quartets, n_groups, 4)
        Initial topology counts from the full pass.  A copy is taken; the
        original array is not modified.
    paralog_data : ParalogData
        Built from *forest* and *quartets* via :func:`build_paralog_data`.
    backend : str
        Computational backend to use for delta kernel calls.  When
        ``'cuda'`` is resolved, ``counts`` is uploaded to the GPU once and
        kept resident for the duration of the optimisation loop, eliminating
        repeated H→D PCIe transfers.  When ``'mlx'`` is resolved, no
        special device-resident state is needed because Apple Silicon's
        Unified Memory Architecture makes wrapping numpy arrays as
        ``mx.array`` essentially free.  Default is ``'best'`` (auto-select).
    """

    def __init__(
        self,
        forest,
        quartets,
        counts: np.ndarray,
        paralog_data: ParalogData,
        backend: str = "best",
    ):
        from quarimo._backend import backends

        self.forest = forest
        self.quartets = quartets
        self.counts = counts.copy()
        self.paralog_data = paralog_data
        self._resolved_backend: str = backends.resolve(backend)

        # Device-resident counts (CUDA path only) ─────────────────────── #
        self._d_counts = None  # cuda DeviceNDArray or None
        self._ckd = None       # ForestKernelData or None

        if self._resolved_backend == "cuda" and backends.cuda:
            from numba import cuda as _cuda
            from quarimo._cuda_kernels import (
                _quartet_counts_delta_cuda as _delta_cuda,
                _counts_d2d_copy_cuda as _d2d_cuda,
            )
            self._cuda = _cuda
            self._delta_cuda = _delta_cuda
            self._d2d_cuda = _d2d_cuda
            self._ckd = forest._cuda_kernel_data
            self._d_counts = _cuda.to_device(self.counts)

        self._current_qed: float = self._compute_qed(self.counts)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _compute_qed(self, counts: np.ndarray) -> float:
        """Mean QED over all quartet × group-pair combinations."""
        from quarimo._results import QuartetTopologyResult

        result = QuartetTopologyResult(
            counts=counts,
            steiner=None,
            steiner_min=None,
            steiner_max=None,
            steiner_var=None,
            groups=self.forest.unique_groups,
            quartets=self.quartets,
            global_names=self.forest.global_names,
        )
        qed = self.forest.qed(result)
        scores = np.asarray(qed.scores)
        if scores.size == 0:
            return 0.0
        return float(np.nanmean(scores))

    def _update_assignments(self, li: int, new_g2l: np.ndarray) -> None:
        """Recompute paralog_data.assignments[li] from the updated mapping."""
        start = int(self.paralog_data.copy_offsets[li])
        end = int(self.paralog_data.copy_offsets[li + 1])
        for ci in range(end - start):
            gid = int(self.paralog_data.copy_global_ids[start + ci])
            for ti in range(self.forest.n_trees):
                self.paralog_data.assignments[li, ci, ti] = int(new_g2l[ti, gid])

    def _delta_on_device(
        self,
        li: int,
        perm: np.ndarray,
        d_counts_target,
    ) -> np.ndarray:
        """
        Run ``_quartet_counts_delta_cuda`` directly against a device array.

        Uploads only the small per-call arrays (trial_g2l, affected_taxa,
        affected_qi, affected_tree_ids); the structural forest arrays stay
        on the device via ``_ckd``.

        Returns the trial ``global_to_local`` (host array) so the caller
        can apply it if the swap is accepted.
        """
        trial_g2l, affected_tree_ids, affected_taxa, affected_qi = (
            self.paralog_data.apply_permutation(li, perm, self.forest.global_to_local)
        )
        n_aq = len(affected_qi)
        n_at = len(affected_tree_ids)
        if n_aq == 0 or n_at == 0:
            return trial_g2l

        ckd = self._ckd
        d_delta_taxa = self._cuda.to_device(affected_taxa)
        d_delta_qi   = self._cuda.to_device(affected_qi)
        d_old_g2l    = self._cuda.to_device(self.forest.global_to_local)
        d_new_g2l    = self._cuda.to_device(trial_g2l)
        d_tree_ids   = self._cuda.to_device(affected_tree_ids)

        tpb = (16, 16)
        bpg = (
            (n_aq + tpb[0] - 1) // tpb[0],
            (n_at + tpb[1] - 1) // tpb[1],
        )
        self._delta_cuda[bpg, tpb](
            d_delta_taxa, d_delta_qi, np.int32(n_aq),
            d_old_g2l, d_new_g2l, d_tree_ids, np.int32(n_at),
            ckd.all_first_occ, ckd.all_root_distance, ckd.all_euler_tour,
            ckd.all_euler_depth, ckd.all_sparse_table, ckd.all_log2_table,
            ckd.node_offsets, ckd.tour_offsets, ckd.sp_offsets, ckd.lg_offsets,
            ckd.sp_tour_widths, ckd.tree_to_group_idx,
            ckd.polytomy_offsets, ckd.polytomy_nodes,
            d_counts_target,
        )
        return trial_g2l

    def _d2d_copy(self, src, dst) -> None:
        """Device-to-device copy of a 3D counts array via ``_counts_d2d_copy_cuda``."""
        shape = dst.shape
        tpb = (8, 4, 4)
        bpg = (
            (shape[0] + tpb[0] - 1) // tpb[0],
            (shape[1] + tpb[1] - 1) // tpb[1],
            (shape[2] + tpb[2] - 1) // tpb[2],
        )
        self._d2d_cuda[bpg, tpb](src, dst)

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def evaluate_swap(self, li: int, perm: np.ndarray) -> float:
        """
        Return the QED delta (new QED − current QED) from permuting genome
        *li* by *perm*, without mutating any state.

        On the CUDA path, creates a device-side trial copy (D→D, no H→D
        transfer) and downloads only the result for QED computation.

        Parameters
        ----------
        li : int
            Paralog genome index.
        perm : int array [k]
            Copy-slot permutation to evaluate.

        Returns
        -------
        float
            Positive means the permutation improves QED.
        """
        if self._d_counts is not None:
            # CUDA path: D→D copy, delta on device, D→H download
            d_trial = self._cuda.device_array_like(self._d_counts)
            self._d2d_copy(self._d_counts, d_trial)
            self._delta_on_device(li, perm, d_trial)
            counts_trial = d_trial.copy_to_host()
            return self._compute_qed(counts_trial) - self._current_qed

        counts_trial = self.counts.copy()
        self.forest.apply_quartet_counts_delta(self.paralog_data, li, perm, counts_trial)
        return self._compute_qed(counts_trial) - self._current_qed

    def apply_swap(self, li: int, perm: np.ndarray) -> None:
        """
        Apply permutation *perm* for genome *li* in-place: update counts,
        ``forest.global_to_local``, and ``paralog_data.assignments``.

        On the CUDA path, the delta is applied directly to the device-resident
        ``_d_counts`` (no H→D upload) and the host copy is synced once via
        D→H download.

        Parameters
        ----------
        li : int
            Paralog genome index.
        perm : int array [k]
            Copy-slot permutation to apply.
        """
        if self._d_counts is not None:
            # CUDA path: delta directly on resident device array
            trial_g2l = self._delta_on_device(li, perm, self._d_counts)
            self._d_counts.copy_to_host(self.counts)
            self.forest.global_to_local[:] = trial_g2l
            self._update_assignments(li, trial_g2l)
            self._current_qed = self._compute_qed(self.counts)
            return

        trial_g2l = self.forest.apply_quartet_counts_delta(
            self.paralog_data, li, perm, self.counts
        )
        self.forest.global_to_local[:] = trial_g2l
        self._update_assignments(li, trial_g2l)
        self._current_qed = self._compute_qed(self.counts)

    def optimize(self, max_iter: int = 100, rng_seed=None):
        """
        Run coordinate descent until convergence or *max_iter* sweeps.

        For each paralog genome in each sweep, all *k!* non-identity
        permutations are evaluated; the one with the largest positive QED
        delta is applied.  After every full sweep the mean QED is appended
        to ``qed_history``.

        Parameters
        ----------
        max_iter : int
            Maximum number of complete sweeps.
        rng_seed : ignored
            Reserved for future stochastic tie-breaking; currently unused.

        Returns
        -------
        OptimizationResult
        """
        from quarimo._results import OptimizationResult

        qed_history: List[float] = [self._current_qed]
        converged = False
        n_iterations = 0

        for _ in range(max_iter):
            n_iterations += 1
            improved = False

            for li in range(self.paralog_data.n_paralog_genomes):
                k = (
                    int(self.paralog_data.copy_offsets[li + 1])
                    - int(self.paralog_data.copy_offsets[li])
                )
                identity = tuple(range(k))
                best_perm: np.ndarray | None = None
                best_delta = 0.0

                for perm_tuple in _iperms(range(k)):
                    if perm_tuple == identity:
                        continue
                    perm = np.array(perm_tuple, dtype=np.int32)
                    delta = self.evaluate_swap(li, perm)
                    if delta > best_delta:
                        best_delta = delta
                        best_perm = perm

                if best_perm is not None:
                    self.apply_swap(li, best_perm)
                    improved = True

            qed_history.append(self._current_qed)

            if not improved:
                converged = True
                break

        return OptimizationResult(
            assignments=self.paralog_data.assignments.copy(),
            qed_history=qed_history,
            converged=converged,
            n_iterations=n_iterations,
            genome_names=list(self.paralog_data.genome_names),
        )
