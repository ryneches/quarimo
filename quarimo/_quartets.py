"""
Quartets class for quartet sampling and generation.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Iterator

# Maximum global taxon ID for which the staged int64 combinadic computation
# is safe from intermediate overflow.  For d < this value every intermediate
# product in the staged division fits in int64 (max intermediate ≈ 9.1×10^18
# at d = 86 000, safely below int64 max ≈ 9.22×10^18).  Forests larger than
# this fall back to Python-int (object) arithmetic.
_QUARTET_IDX_INT64_THRESHOLD = 86_000


class Quartets:
    """
    Deterministic sequence of quartets for phylogenetic analysis.
    
    Represents a window [offset : offset+count] into an infinite sequence where:
      - Indices [0, n_seed): The seed quartets themselves
      - Indices [n_seed, ∞): Pseudo-randomly generated quartets
    
    The seed quartets serve dual purposes:
    1. Explicit quartets to process (when offset < n_seed)
    2. Source of randomness for RNG (hashed to generate rng_seed)
    
    Parameters
    ----------
    forest : Forest
        The forest instance (provides namespace and validation)
    seed : quartet, list of quartets, or None
        Seed quartet(s). Each quartet must be one of:
        - Tuple of 4 taxon names (all str)
        - Tuple of 4 global indices (all int)
        All quartets in one call must use the same type (all str or all int).
        Mixing str and int within a quartet or across quartets raises TypeError.
        If None, uses default [(0, 1, 2, 3)].
    offset : int, default 0
        Starting index in the infinite sequence
    count : int
        Number of quartets to generate
    
    Examples
    --------
    User-provided quartets (by name):
    
    >>> quartets = Quartets(forest, 
    ...     seed=[('A','B','C','D'), ('E','F','G','H')],
    ...     offset=0, 
    ...     count=2)
    
    Random sampling (provide one seed quartet):
    
    >>> import random
    >>> seed_quartet = tuple(random.sample(forest.global_names, 4))
    >>> quartets = Quartets(forest, seed=seed_quartet, offset=1, count=1_000_000)
    
    Default random sampling (uses canonical seed):
    
    >>> quartets = Quartets.random(forest, count=1_000_000)
    
    Attributes
    ----------
    forest : Forest
        The associated forest instance
    seed : list of tuples
        Normalized seed quartets as global namespace indices
    offset : int
        Starting position in the infinite sequence
    count : int
        Number of quartets in this window
    rng_seed : int
        Hash of seed quartets, used to initialize RNG
    """
    
    def __init__(
        self,
        forest,
        seed: Optional[Union[Tuple, List[Tuple]]] = None,
        offset: int = 0,
        count: Optional[int] = None
    ):
        if count is None:
            raise ValueError("count must be specified")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if count <= 0:
            raise ValueError("count must be positive")
        
        self.forest = forest
        self.offset = offset
        self.count = count
        
        # Normalize and validate seed
        self.seed = self._normalize_seed(seed)
        
        # Derive RNG seed from seed quartets (computed on CPU once)
        self.rng_seed = self._hash_seed(self.seed)
    
    def _normalize_seed(self, seed) -> List[Tuple[int, int, int, int]]:
        """
        Convert seed to list of (global_id, global_id, global_id, global_id).

        Disambiguates a single quartet from a list of quartets by inspecting
        the type of seed[0]: if it is a str or int the whole seed is treated
        as one quartet; if it is itself a sequence, seed is treated as a list
        of quartets.

        All quartet elements across the entire call must be the same type —
        either all str (taxon names) or all int (global IDs).  Mixing types
        raises TypeError.

        Returns
        -------
        list of tuples
            Each tuple is 4 global namespace indices, sorted.
        """
        if seed is None:
            # Default seed: first 4 taxa in global namespace
            if self.forest.n_global_taxa < 4:
                raise ValueError(
                    "Forest must have at least 4 taxa for quartet generation"
                )
            return [(0, 1, 2, 3)]

        # Materialise once so we can safely inspect seed[0]
        seed = list(seed)

        if len(seed) == 0:
            raise ValueError("seed must not be empty")

        # Disambiguate: single quartet vs. list of quartets
        first = seed[0]
        if isinstance(first, (str, int, np.integer)):
            # seed IS the single quartet (first element is a scalar)
            quartet_list = [tuple(seed)]
        elif hasattr(first, '__iter__') or hasattr(first, '__len__'):
            # seed is a list/tuple of quartets
            quartet_list = [tuple(q) for q in seed]
        else:
            raise TypeError(
                f"seed elements must be str or int, got {type(first).__name__}"
            )

        # Determine expected element type from first element of first quartet
        first_quartet = quartet_list[0]
        if len(first_quartet) == 0:
            raise ValueError("Quartet must have exactly 4 elements, got 0")
        first_elem = first_quartet[0]
        if isinstance(first_elem, str):
            use_names = True
        elif isinstance(first_elem, (int, np.integer)):
            use_names = False
        else:
            raise TypeError(
                f"Quartet elements must be str (taxon names) or int (global IDs), "
                f"got {type(first_elem).__name__}"
            )

        # Validate length, type consistency, and convert each quartet
        normalized = []
        for quartet in quartet_list:
            if len(quartet) != 4:
                raise ValueError(
                    f"Quartet must have exactly 4 elements, got {len(quartet)}: {quartet}"
                )
            normalized.append(self._normalize_quartet(quartet, use_names))

        return normalized
    
    def _normalize_quartet(self, quartet, use_names: bool) -> Tuple[int, int, int, int]:
        """
        Convert a single length-validated quartet to sorted global IDs.

        Parameters
        ----------
        quartet : tuple of 4 elements
            Length already confirmed to be 4 by _normalize_seed.
        use_names : bool
            True  — all elements must be str taxon names; map to global IDs.
            False — all elements must be int global namespace indices; validate
                    range.

        Returns
        -------
        tuple of 4 ints
            Global namespace indices, sorted.

        Raises
        ------
        TypeError
            If any element has the wrong type (catches within-quartet mixing).
        ValueError
            If a taxon name is absent from the forest namespace, a global ID is
            out of range, or the quartet contains duplicate taxa.
        """
        # Validate element types — catches within-quartet str/int mixing
        for taxon in quartet:
            if use_names:
                if not isinstance(taxon, str):
                    raise TypeError(
                        f"Mixed types in quartet: expected all str (taxon names) "
                        f"but got {type(taxon).__name__} for {taxon!r} in {quartet}. "
                        f"Do not mix str and int within a single quartet."
                    )
            else:
                if not isinstance(taxon, (int, np.integer)):
                    raise TypeError(
                        f"Mixed types in quartet: expected all int (global IDs) "
                        f"but got {type(taxon).__name__} for {taxon!r} in {quartet}. "
                        f"Do not mix str and int within a single quartet."
                    )

        # Check for duplicates
        if len(set(quartet)) != 4:
            raise ValueError(f"Quartet contains duplicates: {quartet}")

        # Convert to global IDs
        global_ids = []
        if use_names:
            for taxon in quartet:
                try:
                    global_id = self.forest.global_names.index(taxon)
                except ValueError:
                    raise ValueError(
                        f"Taxon '{taxon}' not found in forest namespace. "
                        f"Available taxa: {self.forest.global_names[:10]}..."
                    )
                global_ids.append(global_id)
        else:
            for taxon in quartet:
                taxon_int = int(taxon)
                if not 0 <= taxon_int < self.forest.n_global_taxa:
                    raise ValueError(
                        f"Global ID {taxon_int} out of range "
                        f"[0, {self.forest.n_global_taxa})"
                    )
                global_ids.append(taxon_int)

        return tuple(sorted(global_ids))
    
    def _hash_seed(self, seed_quartets: List[Tuple]) -> int:
        """
        Derive integer RNG seed from seed quartets.
        
        Uses a simple hash to ensure determinism across platforms.
        
        Parameters
        ----------
        seed_quartets : list of tuples
            The normalized seed quartets
        
        Returns
        -------
        int
            32-bit unsigned integer for RNG initialization
        """
        h = 0x12345678  # Start with non-zero constant
        for quartet in seed_quartets:
            for taxon_id in quartet:
                # Simple multiplicative hash
                h = (h * 31 + taxon_id) & 0xFFFFFFFF
        return h
    
    def __len__(self) -> int:
        """Number of quartets in this window."""
        return self.count
    
    def __iter__(self) -> Iterator[Tuple[int, int, int, int]]:
        """
        Generate quartets on CPU (for testing/validation).
        
        Uses same algorithm as GPU kernel.
        
        Yields
        ------
        tuple of 4 ints
            Global namespace indices, sorted
        """
        for i in range(self.offset, self.offset + self.count):
            yield self._get_quartet(i)
    
    def _get_quartet(self, absolute_idx: int) -> Tuple[int, int, int, int]:
        """
        Get quartet at absolute index in the infinite sequence.
        
        Parameters
        ----------
        absolute_idx : int
            Index in [0, ∞)
        
        Returns
        -------
        tuple of 4 ints
            Global namespace indices, sorted
        """
        if absolute_idx < len(self.seed):
            # Return seed quartet
            return self.seed[absolute_idx]
        else:
            # Generate random quartet
            rng_state = self._init_rng(self.rng_seed, absolute_idx - len(self.seed))
            return self._sample_quartet(rng_state, self.forest.n_global_taxa)
    
    def _init_rng(
        self, 
        base_seed: int, 
        offset: int
    ) -> np.ndarray:
        """
        Initialize XorShift128 RNG state for given offset.
        
        Must match GPU implementation exactly.
        
        Parameters
        ----------
        base_seed : int
            The rng_seed from hash_seed
        offset : int
            Offset into random sequence (absolute_idx - n_seed)
        
        Returns
        -------
        ndarray, shape (4,), dtype uint32
            RNG state
        """
        # XorShift128 requires 4 words of state
        # Initialize deterministically from seed + offset
        state = np.array([
            (base_seed + offset) & 0xFFFFFFFF,
            ((base_seed + offset) >> 32) & 0xFFFFFFFF,
            0x9e3779b9,  # Golden ratio constant
            0x7f4a7c13   # Arbitrary constant
        ], dtype=np.uint32)
        return state
    
    def _sample_quartet(
        self, 
        rng_state: np.ndarray, 
        n_taxa: int
    ) -> Tuple[int, int, int, int]:
        """
        Sample 4 unique taxa indices using XorShift128.
        
        Must match GPU implementation exactly.
        
        Parameters
        ----------
        rng_state : ndarray, shape (4,), dtype uint32
            XorShift128 state (modified in place)
        n_taxa : int
            Namespace size
        
        Returns
        -------
        tuple of 4 ints
            Sampled indices, sorted
        """
        samples = []
        while len(samples) < 4:
            # XorShift128 step
            t = rng_state[3]
            s = rng_state[0]
            rng_state[3] = rng_state[2]
            rng_state[2] = rng_state[1]
            rng_state[1] = s
            
            t = (t ^ ((t << 11) & 0xFFFFFFFF)) & 0xFFFFFFFF
            t = (t ^ (t >> 8)) & 0xFFFFFFFF
            rng_state[0] = (t ^ s ^ (s >> 19)) & 0xFFFFFFFF
            
            # Sample with rejection
            candidate = int(rng_state[0] % n_taxa)
            if candidate not in samples:
                samples.append(candidate)
        
        return tuple(sorted(samples))
    
    @classmethod
    def from_list(cls, forest, quartets: List[Tuple]):
        """
        Create from explicit list of quartets.
        
        Parameters
        ----------
        forest : Forest
        quartets : list of quartets
            Each quartet is (name, name, name, name) or (id, id, id, id)
        
        Returns
        -------
        Quartets
            Window [0 : len(quartets)]
        
        Examples
        --------
        >>> quartets_list = [('A','B','C','D'), ('E','F','G','H')]
        >>> q = Quartets.from_list(forest, quartets_list)
        >>> len(q)
        2
        """
        return cls(forest, seed=quartets, offset=0, count=len(quartets))
    
    @classmethod
    def random(
        cls,
        forest,
        count: int,
        seed: Optional[Union[Tuple, int]] = None
    ):
        """
        Create random quartet sampler.
        
        Parameters
        ----------
        forest : Forest
        count : int
            Number of random quartets to sample
        seed : quartet, int, or None
            - Tuple: use as seed quartet
            - Int: use to generate seed quartet deterministically
            - None: use default seed [(0,1,2,3)]
        
        Returns
        -------
        Quartets
            Window [1 : count+1] (skips the seed itself)
        
        Examples
        --------
        Default random sampling:
        
        >>> q = Quartets.random(forest, count=1_000_000)
        
        Reproducible with integer seed:
        
        >>> q = Quartets.random(forest, count=1_000_000, seed=42)
        
        Custom seed quartet:
        
        >>> q = Quartets.random(forest, count=1_000_000, seed=(5,10,15,20))
        """
        if isinstance(seed, int):
            # Generate seed quartet deterministically from integer
            rng = np.random.RandomState(seed)
            seed_quartet = tuple(sorted(
                rng.choice(forest.n_global_taxa, 4, replace=False)
            ))
            obj = cls(forest, seed=seed_quartet, offset=1, count=count)
            # Override rng_seed with the integer for reproducibility
            obj.rng_seed = seed & 0xFFFFFFFF
            return obj
        else:
            # seed is a quartet or None
            return cls(forest, seed=seed, offset=1, count=count)
    
    # ------------------------------------------------------------------
    # Quartet indexing
    # ------------------------------------------------------------------

    @staticmethod
    def quartet_index(a: int, b: int, c: int, d: int) -> int:
        """
        Combinadic (combinatorial number system) rank of a sorted quartet.

        Returns ``C(a,1) + C(b,2) + C(c,3) + C(d,4)``, a bijection from
        4-element subsets of non-negative integers to the natural numbers.
        The result is a unique, stable integer index for the quartet ``(a, b,
        c, d)`` with ``a < b < c < d``.

        The index depends only on the four global taxon ID values, not on
        forest size or any other context.  It is therefore stable across
        multiple result objects produced from the same forest, and across any
        two forests that share the same global namespace.

        Uses Python int arithmetic and is exact for any input size.

        Parameters
        ----------
        a, b, c, d : int
            Global taxon IDs, must satisfy ``a < b < c < d``.

        Returns
        -------
        int

        Examples
        --------
        >>> Quartets.quartet_index(0, 1, 2, 3)
        0
        >>> Quartets.quartet_index(0, 1, 2, 4)
        1
        >>> Quartets.quartet_index(1, 2, 3, 4)
        4
        """
        return (
            a
            + b * (b - 1) // 2
            + c * (c - 1) * (c - 2) // 6
            + d * (d - 1) * (d - 2) * (d - 3) // 24
        )

    def index_array(self) -> np.ndarray:
        """
        Combinadic indices for all quartets in this object.

        Each index is the value of :meth:`quartet_index` for the
        corresponding quartet.

        Returns
        -------
        np.ndarray
            Shape ``(n_quartets,)``.  dtype is ``int64`` for forests with
            fewer than ``_QUARTET_IDX_INT64_THRESHOLD`` global taxa (fast
            vectorised path); Python-int ``object`` dtype for larger forests
            (avoids intermediate overflow).  The ``object`` dtype is
            transparently handled by :meth:`~._results.QuartetTopologyResult.to_frame`
            and :meth:`~._results.QEDResult.to_frame` as a Polars ``Int128``
            column.

        Notes
        -----
        Calling this method materialises the full quartet sequence; for large
        random windows this may be slow.  The result is not cached.
        """
        ids = np.array(list(self), dtype=np.int64)
        if len(ids) == 0:
            return np.empty(0, dtype=np.int64)

        a, b, c, d = ids[:, 0], ids[:, 1], ids[:, 2], ids[:, 3]

        if int(d.max()) < _QUARTET_IDX_INT64_THRESHOLD:
            # Fast path: staged integer division.
            # Each // step is exact (product of k consecutive integers is
            # divisible by k!), and no intermediate value exceeds int64 max.
            b2 = b * (b - 1) // 2
            c3 = c * (c - 1) // 2 * (c - 2) // 3
            d4 = d * (d - 1) // 2 * (d - 2) // 3 * (d - 3) // 4
            return (a + b2 + c3 + d4).astype(np.int64)
        else:
            # Slow path: Python int arithmetic — arbitrary precision.
            # Returns object dtype; callers that produce Polars DataFrames
            # use pl.Int128 for this column.
            a_o = a.astype(object)
            b_o = b.astype(object)
            c_o = c.astype(object)
            d_o = d.astype(object)
            return (
                a_o
                + b_o * (b_o - 1) // 2
                + c_o * (c_o - 1) * (c_o - 2) // 6
                + d_o * (d_o - 1) * (d_o - 2) * (d_o - 3) // 24
            )

    def __repr__(self) -> str:
        """String representation."""
        n_seed = len(self.seed)
        if self.offset >= n_seed:
            mode = "random"
            start = self.offset - n_seed
        elif self.offset + self.count <= n_seed:
            mode = "explicit"
            start = self.offset
        else:
            mode = "mixed"
            start = self.offset
        
        return (
            f"Quartets({mode}, offset={start}, count={self.count}, "
            f"n_seed={n_seed}, rng_seed=0x{self.rng_seed:08x})"
        )
