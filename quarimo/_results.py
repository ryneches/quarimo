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
    counts : np.ndarray, int32, shape (n_quartets, n_groups, 3)
        counts[qi, gi, k] = number of trees in group gi where quartet qi
        has topology k.  Trees where any of the four taxa are absent do not
        contribute.

    steiner : np.ndarray or None, float64, shape (n_quartets, n_groups, 3)
        steiner[qi, gi, k] = summed Steiner spanning length for group gi,
        topology k.  None when ``steiner=False`` was passed to
        ``quartet_topology()``.  Mean Steiner per tree:
        ``steiner[qi, gi, k] / max(counts[qi, gi, k], 1)``.

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
                ``a``, ``b``, ``c``, ``d`` (taxon names, sorted by global
                ID), ``group``, ``topology`` (int 0–2), ``count``, and
                optionally ``steiner_sum``.
                Total rows: n_quartets × n_groups × 3.

            ``'wide'``
                One row per quartet.  Columns ``a``–``d`` are followed by
                one count column per (group, topology) combination named
                ``{group}_t{k}``, and optionally one Steiner column per
                combination named ``{group}_steiner_t{k}``.
                Total rows: n_quartets.

        Topology encoding
        -----------------
        For a quartet with taxa sorted by global ID as (a, b, c, d):

          topology 0 → (a, b) | (c, d)
          topology 1 → (a, c) | (b, d)
          topology 2 → (a, d) | (b, c)

        Returns
        -------
        polars.DataFrame
        """
        import polars as pl

        n_q, n_g, _ = self.counts.shape

        # Materialise quartet global IDs: (n_quartets, 4)
        ids = np.array(list(self.quartets), dtype=np.int32)
        gnames = np.asarray(self.global_names)

        taxon_cols = {
            "a": gnames[ids[:, 0]].tolist(),
            "b": gnames[ids[:, 1]].tolist(),
            "c": gnames[ids[:, 2]].tolist(),
            "d": gnames[ids[:, 3]].tolist(),
        }

        if form == "long":
            # Index arrays for the flattened (n_q * n_g * 3) rows
            qi = np.repeat(np.arange(n_q), n_g * 3)
            gi = np.tile(np.repeat(np.arange(n_g), 3), n_q)
            ti = np.tile(np.arange(3), n_q * n_g)

            data: dict = {
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
            data = {k: v for k, v in taxon_cols.items()}
            for gi, group in enumerate(self.groups):
                for k in range(3):
                    data[f"{group}_t{k}"] = self.counts[:, gi, k].tolist()
                    if self.steiner is not None:
                        data[f"{group}_steiner_t{k}"] = self.steiner[:, gi, k].tolist()
            return pl.DataFrame(data)

        else:
            raise ValueError(f"form must be 'long' or 'wide', got {form!r}")
