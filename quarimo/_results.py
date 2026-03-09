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

    counts: np.ndarray              # int32, (n_quartets, n_groups, 3)
    steiner: Optional[np.ndarray]   # float64, (n_quartets, n_groups, 3) or None
    groups: List[str]
    quartets: "Quartets"
    global_names: List[str]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_frame(self, form: str = "long") -> "pl.DataFrame":
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
                ``count``, and optionally ``steiner_sum``.
                topology=3 means unresolved (polytomy).
                Total rows: n_quartets × n_groups × 4.

            ``'wide'``
                One row per quartet.  Columns ``quartet_idx``, ``a``–``d``
                are followed by one count column per (group, topology)
                combination named ``{group}_t{k}`` (k=0–3), and optionally
                one Steiner column per combination named
                ``{group}_steiner_t{k}``.
                Total rows: n_quartets.

        Join key
        --------
        ``quartet_idx`` is a combinadic integer that uniquely and stably
        identifies each quartet by its four global taxon IDs (see
        :meth:`~quarimo._quartets.Quartets.quartet_index`).  Use it as the
        join key when combining this DataFrame with a :class:`QEDResult`
        DataFrame — both forms produce a 1-to-1 join on ``quartet_idx``
        for wide output; for long output each QED row matches n_groups × 3
        topology rows (filter by ``group`` after joining).

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

        # Materialise quartet global IDs and combinadic indices
        ids = np.array(list(self.quartets), dtype=np.int32)
        idx = self.quartets.index_array()
        gnames = np.asarray(self.global_names)

        idx_dtype = pl.Int128 if idx.dtype == object else pl.Int64

        if form == "long":
            qi = np.repeat(np.arange(n_q), n_g * 4)
            gi = np.tile(np.repeat(np.arange(n_g), 4), n_q)
            ti = np.tile(np.arange(4), n_q * n_g)

            data: dict = {
                "quartet_idx": pl.Series(
                    "quartet_idx", np.repeat(idx, n_g * 4).tolist(), dtype=idx_dtype
                ),
                "a": gnames[ids[qi, 0]].tolist(),
                "b": gnames[ids[qi, 1]].tolist(),
                "c": gnames[ids[qi, 2]].tolist(),
                "d": gnames[ids[qi, 3]].tolist(),
                "group": [self.groups[g] for g in gi.tolist()],
                "topology": ti.tolist(),
                "count": self.counts.ravel().tolist(),
            }
            if self.steiner is not None:
                data["steiner_sum"] = self.steiner.ravel().tolist()
            return pl.DataFrame(data)

        elif form == "wide":
            data = {
                "quartet_idx": pl.Series(
                    "quartet_idx", idx.tolist(), dtype=idx_dtype
                ),
                "a": gnames[ids[:, 0]].tolist(),
                "b": gnames[ids[:, 1]].tolist(),
                "c": gnames[ids[:, 2]].tolist(),
                "d": gnames[ids[:, 3]].tolist(),
            }
            for gi, group in enumerate(self.groups):
                for k in range(4):
                    data[f"{group}_t{k}"] = self.counts[:, gi, k].tolist()
                    if self.steiner is not None:
                        data[f"{group}_steiner_t{k}"] = self.steiner[:, gi, k].tolist()
            return pl.DataFrame(data)

        else:
            raise ValueError(f"form must be 'long' or 'wide', got {form!r}")


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

    scores: np.ndarray              # float64, (n_quartets, n_pairs)
    groups: List[str]
    group_pairs: np.ndarray         # int32, (n_pairs, 2)
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

    def to_frame(self, form: str = "long") -> "pl.DataFrame":
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
        ids = np.array(list(self.quartets), dtype=np.int32)
        idx = self.quartets.index_array()
        gnames = np.asarray(self.global_names)

        idx_dtype = pl.Int128 if idx.dtype == object else pl.Int64

        # Pre-compute string labels for each pair
        pair_ga = [self.groups[int(self.group_pairs[pi, 0])] for pi in range(n_pairs)]
        pair_gb = [self.groups[int(self.group_pairs[pi, 1])] for pi in range(n_pairs)]

        if form == "long":
            # C-order ravel: qi major, pi minor → matches scores.ravel()
            qi = np.repeat(np.arange(n_q), n_pairs)
            pi = np.tile(np.arange(n_pairs), n_q)

            data: dict = {
                "quartet_idx": pl.Series(
                    "quartet_idx", np.repeat(idx, n_pairs).tolist(), dtype=idx_dtype
                ),
                "a": gnames[ids[qi, 0]].tolist(),
                "b": gnames[ids[qi, 1]].tolist(),
                "c": gnames[ids[qi, 2]].tolist(),
                "d": gnames[ids[qi, 3]].tolist(),
                "group_a": [pair_ga[p] for p in pi.tolist()],
                "group_b": [pair_gb[p] for p in pi.tolist()],
                "qed": self.scores.ravel().tolist(),
            }
            return pl.DataFrame(data)

        elif form == "wide":
            data = {
                "quartet_idx": pl.Series(
                    "quartet_idx", idx.tolist(), dtype=idx_dtype
                ),
                "a": gnames[ids[:, 0]].tolist(),
                "b": gnames[ids[:, 1]].tolist(),
                "c": gnames[ids[:, 2]].tolist(),
                "d": gnames[ids[:, 3]].tolist(),
            }
            for pi in range(n_pairs):
                col = f"{pair_ga[pi]}_vs_{pair_gb[pi]}"
                data[col] = self.scores[:, pi].tolist()
            return pl.DataFrame(data)

        else:
            raise ValueError(f"form must be 'long' or 'wide', got {form!r}")
