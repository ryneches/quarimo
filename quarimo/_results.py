"""
_results.py
===========
Result dataclasses returned by Forest public methods.

Each class holds raw numpy arrays as direct references (zero copy overhead)
and exposes a ``to_frame()`` method for conversion to a labelled Polars
DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    import polars as pl
    from quarimo._quartets import Quartets


@dataclass
class BranchDistanceResult:
    """
    Return value of ``Forest.branch_distance()``.

    Attributes
    ----------
    distances : np.ndarray, float64, shape (n_trees,)
        Patristic distance between the two queried taxa for every tree in the
        collection.  NaN where either taxon is absent from a tree.
    """

    distances: np.ndarray  # float64, shape (n_trees,)

    # ------------------------------------------------------------------
    # Array protocol — lets existing code treat the result as an ndarray.
    # Indexing, iteration, and numpy functions all delegate to .distances.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.distances)

    def __getitem__(self, key):
        return self.distances[key]

    def __iter__(self):
        return iter(self.distances)

    def __array__(self, dtype=None, copy=None):
        return self.distances if dtype is None else self.distances.astype(dtype)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_frame(self) -> "pl.DataFrame":
        """
        Convert to a Polars DataFrame.

        Not yet implemented — placeholder for future long/wide form output
        once ``Forest.branch_distance()`` is redesigned to carry group and
        tree-label metadata.
        """
        raise NotImplementedError(
            "BranchDistanceResult.to_frame() is not yet implemented."
        )


@dataclass
class QuartetTopologyResult:
    """
    Return value of ``Forest.quartet_topology()``.

    Attributes
    ----------
    counts : np.ndarray, int32, shape (n_quartets, n_groups, 4)
        counts[qi, gi, k] = number of trees in group gi where quartet qi
        has topology k.  Trees where any of the four taxa are absent do not
        contribute.  k=3 accumulates unresolved (polytomy) counts.

    steiner : np.ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner[qi, gi, k] = summed Steiner spanning length for group gi,
        topology k.  None when ``steiner=False`` was passed to
        ``quartet_topology()``.  Mean Steiner per tree:
        ``steiner[qi, gi, k] / max(counts[qi, gi, k], 1)``.
        k=3 accumulates Steiner sums for unresolved quartets.

    steiner_min : np.ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner_min[qi, gi, k] = minimum Steiner spanning length across all
        trees in group gi that voted for topology k.  ``np.nan`` where
        ``counts[qi, gi, k] == 0`` (no tree contributed).  None when
        ``steiner=False`` was passed to ``quartet_topology()``.

    steiner_max : np.ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner_max[qi, gi, k] = maximum Steiner spanning length across all
        trees in group gi that voted for topology k.  ``np.nan`` where
        ``counts[qi, gi, k] == 0`` (no tree contributed).  None when
        ``steiner=False`` was passed to ``quartet_topology()``.

    steiner_var : np.ndarray or None, float64, shape (n_quartets, n_groups, 4)
        steiner_var[qi, gi, k] = population variance of Steiner spanning length
        across all trees in group gi that voted for topology k.
        Computed as ``sum_sq/n - (sum/n)^2``.  ``np.nan`` where
        ``counts[qi, gi, k] == 0``.  None when ``steiner=False`` was passed to
        ``quartet_topology()``.  For n=1 trees the variance is exactly 0.

    groups : list of str
        Group axis labels, sorted alphabetically.  Axis 1 of ``counts`` and
        ``steiner`` corresponds to this list.

    quartets : Quartets
        The query object used to produce this result.  Iterated in
        ``to_frame()`` to materialise taxon labels.

    global_names : list of str
        global_names[gid] = taxon name.  Used to look up taxon labels from
        the sorted global IDs stored in each quartet.
    """

    counts: np.ndarray              # int32, (n_quartets, n_groups, 4)
    steiner: Optional[np.ndarray]   # float64, (n_quartets, n_groups, 4) or None
    steiner_min: Optional[np.ndarray]  # float64, (n_quartets, n_groups, 4) or None
    steiner_max: Optional[np.ndarray]  # float64, (n_quartets, n_groups, 4) or None
    steiner_var: Optional[np.ndarray]  # float64, (n_quartets, n_groups, 4) or None
    groups: List[str]
    quartets: "Quartets"
    global_names: List[str]

    def __post_init__(self) -> None:
        # Replace kernel sentinels (+inf / -inf) with NaN in empty cells.
        # Cells where counts == 0 never received a Steiner observation;
        # NaN is numpy's canonical missing float and keeps nanmin/nanmax clean.
        if self.steiner_min is not None:
            empty = self.counts == 0
            self.steiner_min[empty] = np.nan
            self.steiner_max[empty] = np.nan  # type: ignore[index]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_frame(self, form: str = "long", deduplicate: bool = True) -> "pl.DataFrame":
        """
        Convert to a Polars DataFrame.

        Parameters
        ----------
        form : {'long', 'wide'}, default 'long'
            Shape of the output.

            ``'long'``
                One row per (quartet, group, topology) triple.  Columns:
                ``quartet_idx``, ``a``, ``b``, ``c``, ``d`` (taxon names,
                sorted by global ID), ``group``, ``topology`` (int 0–3),
                ``count``, and optionally ``steiner_sum``, ``steiner_min``,
                ``steiner_max``, ``steiner_var``.
                topology=3 means unresolved (polytomy).
                Total rows: n_quartets × n_groups × 4.

            ``'wide'``
                One row per quartet.  Columns ``quartet_idx``, ``a``–``d``
                are followed by one count column per (group, topology)
                combination named ``{group}_t{k}`` (k=0–3), and optionally
                Steiner columns ``{group}_steiner_t{k}``,
                ``{group}_steiner_min_t{k}``, ``{group}_steiner_max_t{k}``,
                ``{group}_steiner_var_t{k}``.
                Total rows: n_quartets.

        deduplicate : { True, False }, default True
            Remove duplicated rows before returning.

        Join key
        --------
        ``quartet_idx`` is a combinadic integer that uniquely and stably
        identifies each quartet by its four global taxon IDs (see
        :meth:`~quarimo._quartets.Quartets.quartet_index`).  Use it as the
        join key when combining this DataFrame with a :class:`QEDResult`
        DataFrame — both forms produce a 1-to-1 join on ``quartet_idx``
        for wide output; for long output each QED row matches n_groups × 3
        topology rows (filter by ``group`` after joining). Note that if
        the same quartet is queried more than once (an important
        consideration when sampling), duplicated indicies will cause the
        join to fail. By default, .to_frame() calls .unique() before
        returning, which may cause the row indecies not to match the
        query. If this behavior is not wanted, use dedupliacte=False.

        Topology encoding
        -----------------
        For a quartet with taxa sorted by global ID as (a, b, c, d):

          topology 0 → (a, b) | (c, d)
          topology 1 → (a, c) | (b, d)
          topology 2 → (a, d) | (b, c)
          topology 3 → unresolved (polytomy)

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl

        n_q, n_g, _ = self.counts.shape

        # Quartet identity columns — cached in the Quartets object, zero extra cost
        # on repeated calls.  Row order matches the kernel output exactly.
        q_df = self.quartets.to_frame()  # n_q rows: quartet_idx, a, b, c, d

        if form == "long":
            # Expand each quartet row n_g * 4 times (one per group × topology).
            # repeat_by + explode is a native polars operation — no numpy round-trip.
            base = q_df.select([
                pl.col(c).repeat_by(n_g * 4).explode() for c in q_df.columns
            ])

            gi = np.tile(np.repeat(np.arange(n_g, dtype=np.int32), 4), n_q)
            ti = np.tile(np.arange(4, dtype=np.int32), n_q * n_g)

            data: dict = {
                **{c: base[c] for c in base.columns},
                "group":    pl.Series(self.groups).gather(pl.Series(gi)),
                "topology": pl.Series(ti),
                "count":    pl.Series(self.counts.ravel()),
            }
            if self.steiner is not None:
                data["steiner_sum"] = pl.Series(self.steiner.ravel())
                data["steiner_min"] = pl.Series(self.steiner_min.ravel()).fill_nan(None)
                data["steiner_max"] = pl.Series(self.steiner_max.ravel()).fill_nan(None)
                data["steiner_var"] = pl.Series(self.steiner_var.ravel()).fill_nan(None)

            df = pl.DataFrame(data)
            return df.unique() if deduplicate else df

        elif form == "wide":
            # Wide: one row per quartet — q_df is already the right shape.
            data = {c: q_df[c] for c in q_df.columns}
            for gi, group in enumerate(self.groups):
                for k in range(4):
                    data[f"{group}_t{k}"] = pl.Series(self.counts[:, gi, k])
                    if self.steiner is not None:
                        data[f"{group}_steiner_t{k}"]     = pl.Series(self.steiner[:, gi, k])
                        data[f"{group}_steiner_min_t{k}"] = pl.Series(self.steiner_min[:, gi, k]).fill_nan(None)
                        data[f"{group}_steiner_max_t{k}"] = pl.Series(self.steiner_max[:, gi, k]).fill_nan(None)
                        data[f"{group}_steiner_var_t{k}"] = pl.Series(self.steiner_var[:, gi, k]).fill_nan(None)

            df = pl.DataFrame(data)
            return df.unique() if deduplicate else df

        else:
            raise ValueError(f"form must be 'long' or 'wide', got {form!r}")


@dataclass
class OptimizationResult:
    """
    Return value of ``Forest.resolve_paralogs()``.

    Attributes
    ----------
    assignments : np.ndarray, int32, shape (n_paralog_genomes, max_copies, n_trees)
        Final copy-slot → local-leaf assignment.
        ``assignments[li, ci, ti]`` = local leaf node ID for copy slot *ci*
        of genome *li* in tree *ti*, or ``-1`` if absent.
    qed_history : list[float]
        Mean QED recorded once before optimisation begins and then after each
        complete sweep across all paralog genomes.  Monotonically
        non-decreasing (the optimiser only accepts improvements).
    converged : bool
        True when a complete sweep produced no improvement.
    n_iterations : int
        Number of complete sweeps executed.
    genome_names : list[str]
        Genome labels indexed by the first axis of *assignments*.
    """

    assignments: np.ndarray   # int32 [n_paralog_genomes, max_copies, n_trees]
    qed_history: List[float]
    converged: bool
    n_iterations: int
    genome_names: List[str]


@dataclass
class QEDResult:
    """
    Return value of ``Forest.qed()``.

    Attributes
    ----------
    scores : np.ndarray, float64, shape (n_quartets, n_pairs)
        ``scores[qi, pi]`` is the Quartet Ensemble Discordance comparing the
        two groups in ``group_pairs[pi]`` for quartet ``qi``.
        Values lie in [-1, +1].

    groups : list of str
        All group labels in the forest, in axis-1 order of the source
        ``counts`` array.  Used to map ``group_pairs`` indices to names.

    group_pairs : np.ndarray, int32, shape (n_pairs, 2)
        Each row ``[g1, g2]`` is the ordered pair of group indices that was
        compared.

    quartets : Quartets or None
        The query object used to produce this result.  Required by
        ``to_frame()``.  ``None`` when ``qed()`` was called with a raw
        ndarray rather than a ``QuartetTopologyResult``.

    global_names : list of str or None
        ``global_names[gid]`` = taxon name.  Required by ``to_frame()``.
        ``None`` when ``qed()`` was called with a raw ndarray.
    """

    scores: np.ndarray  # float64, (n_quartets, n_pairs)
    groups: List[str]
    group_pairs: np.ndarray  # int32, (n_pairs, 2)
    quartets: Optional["Quartets"]
    global_names: Optional[List[str]]

    # ------------------------------------------------------------------
    # Array protocol — lets existing code treat the result as an ndarray.
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, key):
        return self.scores[key]

    def __iter__(self):
        return iter(self.scores)

    def __array__(self, dtype=None, copy=None):
        return self.scores if dtype is None else self.scores.astype(dtype)

    @property
    def shape(self):
        return self.scores.shape

    @property
    def dtype(self):
        return self.scores.dtype

    def __ge__(self, other):
        return self.scores >= other

    def __le__(self, other):
        return self.scores <= other

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_frame(self, form: str = "long", deduplicate: bool = True) -> "pl.DataFrame":
        """
        Convert to a Polars DataFrame.

        Parameters
        ----------
        form : {'long', 'wide'}, default 'long'
            Shape of the output.

            ``'long'``
                One row per (quartet, group pair).  Columns:
                ``quartet_idx``, ``a``, ``b``, ``c``, ``d`` (taxon names),
                ``group_a``, ``group_b``, ``qed``.
                Total rows: n_quartets × n_pairs.

                Join with ``QuartetTopologyResult.to_frame('long')`` on
                ``'quartet_idx'`` (1-to-(n_groups × 3)); filter the
                topology frame by ``group`` after joining.

            ``'wide'``
                One row per quartet.  Columns ``quartet_idx``, ``a``–``d``
                followed by one QED column per group pair named
                ``{group_a}_vs_{group_b}``.  Total rows: n_quartets.

                Join with ``QuartetTopologyResult.to_frame('wide')`` on
                ``'quartet_idx'`` (1-to-1).

        deduplicate : { True, False }, default True
            Remove duplicated rows before returning.

        Returns
        -------
        polars.DataFrame

        Raises
        ------
        RuntimeError
            If this result was constructed from a raw ndarray (quartet and
            taxon metadata are unavailable).  Pass a
            ``QuartetTopologyResult`` to ``Forest.qed()`` to enable
            ``to_frame()``.
        ValueError
            If ``form`` is not ``'long'`` or ``'wide'``.
        """
        if self.quartets is None or self.global_names is None:
            raise RuntimeError(
                "to_frame() requires quartet and taxon metadata. "
                "Pass a QuartetTopologyResult to Forest.qed() instead of a raw ndarray."
            )

        import polars as pl

        n_q, n_pairs = self.scores.shape

        # Quartet identity columns — cached in the Quartets object.
        q_df = self.quartets.to_frame()  # n_q rows: quartet_idx, a, b, c, d

        # String labels for each group pair (small; one entry per pair)
        pair_ga = [self.groups[int(self.group_pairs[pi, 0])] for pi in range(n_pairs)]
        pair_gb = [self.groups[int(self.group_pairs[pi, 1])] for pi in range(n_pairs)]

        if form == "long":
            # Expand each quartet row n_pairs times (C-order: qi major, pi minor).
            base = q_df.select([
                pl.col(c).repeat_by(n_pairs).explode() for c in q_df.columns
            ])

            pi = np.tile(np.arange(n_pairs, dtype=np.int32), n_q)

            data: dict = {
                **{c: base[c] for c in base.columns},
                "group_a": pl.Series(pair_ga).gather(pl.Series(pi)),
                "group_b": pl.Series(pair_gb).gather(pl.Series(pi)),
                "qed":     pl.Series(self.scores.ravel()),
            }
            df = pl.DataFrame(data)
            return df.unique() if deduplicate else df

        elif form == "wide":
            # Wide: one row per quartet — q_df is already the right shape.
            data = {c: q_df[c] for c in q_df.columns}
            for pi in range(n_pairs):
                data[f"{pair_ga[pi]}_vs_{pair_gb[pi]}"] = pl.Series(self.scores[:, pi])
            df = pl.DataFrame(data)
            return df.unique() if deduplicate else df

        else:
            raise ValueError(f"form must be 'long' or 'wide', got {form!r}")
