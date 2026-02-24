"""
Quartets class for quartet sampling and generation.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Iterator


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
        Seed quartet(s). Each quartet can be:
        - Tuple of 4 taxon names (str)
        - Tuple of 4 global indices (int)
        If None, uses default [(0, 1, 2, 3)]
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
        
        Returns
        -------
        list of tuples
            Each tuple is 4 global namespace indices, sorted
        """
        if seed is None:
            # Default seed: first 4 taxa in global namespace
            if self.forest.n_global_taxa < 4:
                raise ValueError(
                    "Forest must have at least 4 taxa for quartet generation"
                )
            return [(0, 1, 2, 3)]
        
        # Single quartet → list
        if isinstance(seed, tuple) and len(seed) == 4:
            seed = [seed]
        
        # Validate is iterable
        if not hasattr(seed, '__iter__'):
            raise TypeError("seed must be a quartet or list of quartets")
        
        # Convert each quartet
        normalized = []
        for quartet in seed:
            normalized.append(self._normalize_quartet(quartet))
        
        return normalized
    
    def _normalize_quartet(self, quartet) -> Tuple[int, int, int, int]:
        """
        Convert single quartet to (global_id, global_id, global_id, global_id).
        
        Parameters
        ----------
        quartet : tuple of 4 elements
            Each element is either:
            - str: taxon name (will be mapped to global ID)
            - int: global namespace index (will be validated)
        
        Returns
        -------
        tuple of 4 ints
            Global namespace indices, sorted
        """
        if len(quartet) != 4:
            raise ValueError(
                f"Quartet must have exactly 4 elements, got {len(quartet)}"
            )
        
        # Check for duplicates
        if len(set(quartet)) != 4:
            raise ValueError(f"Quartet contains duplicates: {quartet}")
        
        # Convert to global IDs
        global_ids = []
        for taxon in quartet:
            if isinstance(taxon, str):
                # Map name to global ID
                try:
                    global_id = self.forest.global_names.index(taxon)
                except ValueError:
                    raise ValueError(
                        f"Taxon '{taxon}' not found in forest namespace. "
                        f"Available taxa: {self.forest.global_names[:10]}..."
                    )
                global_ids.append(global_id)
            elif isinstance(taxon, (int, np.integer)):
                # Validate global ID
                if not 0 <= taxon < self.forest.n_global_taxa:
                    raise ValueError(
                        f"Global ID {taxon} out of range "
                        f"[0, {self.forest.n_global_taxa})"
                    )
                global_ids.append(int(taxon))
            else:
                raise TypeError(
                    f"Quartet element must be str (taxon name) or "
                    f"int (global ID), got {type(taxon)}"
                )
        
        # Return sorted tuple
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
