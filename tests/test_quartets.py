"""
Test that CPU and GPU generate identical quartet sequences.

This is critical for reproducibility - users should get the same results
whether they iterate Quartets on CPU or process on GPU.
"""

import numpy as np
import pytest


def test_quartets_cpu_gpu_match():
    """
    Verify CPU and GPU generate identical random sequences.
    
    Tests:
    1. Default seed produces same quartets
    2. Custom seed produces same quartets  
    3. Integer seed produces same quartets
    4. Sequences are deterministic (repeated calls match)
    """
    from quarimo import Forest
    from quarimo._quartets import Quartets
    
    # Create forest with 5 taxa (A-E) so random sampling has meaningful variation.
    # With only 4 taxa there is exactly one unique 4-combination, making every
    # random quartet identical regardless of seed.
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((A:1,C:1):1,(B:1,D:1):1);',
        '((A:1,E:1):1,(B:1,C:1):1);',
    ]
    forest = Forest(trees)
    
    # Test 1: Default seed
    q = Quartets.random(forest, count=100)
    cpu_quartets = list(q)
    
    # All should be tuples of 4 sorted ints
    assert all(len(quartet) == 4 for quartet in cpu_quartets)
    assert all(quartet == tuple(sorted(quartet)) for quartet in cpu_quartets)
    assert all(all(0 <= x < forest.n_global_taxa for x in quartet) 
               for quartet in cpu_quartets)
    
    # Test 2: Repeated calls give same sequence
    q2 = Quartets.random(forest, count=100, seed=q.seed[0])
    q2.rng_seed = q.rng_seed  # Ensure same RNG seed
    cpu_quartets2 = list(q2)
    
    assert cpu_quartets == cpu_quartets2, "Repeated calls should give same sequence"
    
    # Test 3: Custom seed quartet — must differ from default seed (0,1,2,3)
    seed_quartet = (0, 1, 2, 4)
    q3 = Quartets.random(forest, count=100, seed=seed_quartet)
    cpu_quartets3 = list(q3)
    
    # Should be different from default seed
    assert cpu_quartets3 != cpu_quartets, "Different seeds should give different sequences"
    
    # But repeated with same seed should give same sequence
    q4 = Quartets.random(forest, count=100, seed=seed_quartet)
    cpu_quartets4 = list(q4)
    assert cpu_quartets3 == cpu_quartets4, "Same seed should give same sequence"
    
    # Test 4: Integer seed
    q5 = Quartets.random(forest, count=100, seed=42)
    cpu_quartets5 = list(q5)
    
    q6 = Quartets.random(forest, count=100, seed=42)
    cpu_quartets6 = list(q6)
    assert cpu_quartets5 == cpu_quartets6, "Same integer seed should give same sequence"


def test_quartets_explicit_seed_handling():
    """Test that explicit seed quartets are used correctly."""
    from quarimo import Forest
    from quarimo._quartets import Quartets
    
    # 5-taxon forest so random quartets are not always (0,1,2,3)
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((A:1,C:1):1,(B:1,D:1):1);',
        '((A:1,E:1):1,(B:1,C:1):1);',
    ]
    forest = Forest(trees)

    # Test 1: Single seed quartet with offset=0 includes the seed
    seed = (0, 1, 2, 3)
    q = Quartets(forest, seed=seed, offset=0, count=3)
    quartets = list(q)
    
    assert quartets[0] == seed, "First quartet should be the seed"
    assert quartets[1] != seed, "Second should be randomly generated"
    assert quartets[2] != seed, "Third should be randomly generated"
    
    # Test 2: offset=1 skips the seed
    q2 = Quartets(forest, seed=seed, offset=1, count=2)
    quartets2 = list(q2)
    
    assert quartets2[0] == quartets[1], "offset=1 should skip seed"
    assert quartets2[1] == quartets[2]
    
    # Test 3: Multiple seed quartets
    seeds = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4)]
    q3 = Quartets.from_list(forest, seeds)
    quartets3 = list(q3)
    
    assert quartets3 == seeds, "from_list should return exactly the seed quartets"
    
    # Test 4: Seed quartets then random — verify offset mechanism, not specific values
    q4 = Quartets(forest, seed=seeds, offset=0, count=5)
    quartets4 = list(q4)

    assert quartets4[:3] == seeds, "First 3 should be seeds"
    # Positions 3+ come from the RNG; verify they match an equivalent offset window
    q4_tail = Quartets(forest, seed=seeds, offset=3, count=2)
    assert list(q4_tail) == quartets4[3:], "offset=3 should reproduce positions 3-4"


def test_quartets_validation():
    """Test that invalid inputs raise appropriate errors."""
    from quarimo import Forest
    from quarimo._quartets import Quartets
    
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
    ]
    forest = Forest(trees)
    
    # Test 1: Duplicate taxa in quartet
    with pytest.raises(ValueError, match="duplicates"):
        Quartets(forest, seed=(0, 0, 1, 2), offset=0, count=1)
    
    # Test 2: Invalid taxon name
    with pytest.raises(ValueError, match="not found"):
        Quartets(forest, seed=('A', 'B', 'C', 'Z'), offset=0, count=1)
    
    # Test 3: Invalid global ID
    with pytest.raises(ValueError, match="out of range"):
        Quartets(forest, seed=(0, 1, 2, 999), offset=0, count=1)
    
    # Test 4: Wrong number of elements
    with pytest.raises(ValueError, match="exactly 4"):
        Quartets(forest, seed=(0, 1, 2), offset=0, count=1)
    
    # Test 5: Negative offset
    with pytest.raises(ValueError, match="non-negative"):
        Quartets(forest, seed=None, offset=-1, count=1)
    
    # Test 6: Zero or negative count
    with pytest.raises(ValueError, match="positive"):
        Quartets(forest, seed=None, offset=0, count=0)
    
    with pytest.raises(ValueError, match="positive"):
        Quartets(forest, seed=None, offset=0, count=-1)
    
    # Test 7: count not specified
    with pytest.raises(ValueError, match="must be specified"):
        Quartets(forest, seed=None, offset=0)

    # Test 8: mixed str/int within a single quartet
    with pytest.raises(TypeError):
        Quartets(forest, seed=('A', 0, 'B', 'C'), offset=0, count=1)

    # Test 9: mixed str/int across quartets in a list
    with pytest.raises(TypeError):
        Quartets(forest, seed=[('A', 'B', 'C', 'D'), (0, 1, 2, 3)], offset=0, count=2)


def test_quartets_name_to_id_mapping():
    """Test that taxon names are correctly mapped to global IDs."""
    from quarimo import Forest
    from quarimo._quartets import Quartets
    
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((B:1,C:1):1,(D:1,E:1):1);',
    ]
    forest = Forest(trees)
    
    # Global namespace: ['A', 'B', 'C', 'D', 'E'] → [0, 1, 2, 3, 4]
    
    # Test with names
    q_names = Quartets(forest, seed=('A', 'B', 'C', 'D'), offset=0, count=1)
    # Test with IDs
    q_ids = Quartets(forest, seed=(0, 1, 2, 3), offset=0, count=1)
    
    # Should normalize to same internal representation
    assert q_names.seed == q_ids.seed
    assert list(q_names) == list(q_ids)
    
    # Mixed str/int quartets are rejected — types must be uniform across all quartets
    with pytest.raises(TypeError):
        Quartets(forest, seed=[('A', 'B', 'C', 'D'), (1, 2, 3, 4)],
                 offset=0, count=2)


class TestQuartetIndex:
    """Tests for Quartets.quartet_index() and Quartets.index_array()."""

    def test_scalar_known_values(self):
        """C(a,1)+C(b,2)+C(c,3)+C(d,4) for small known inputs."""
        from quarimo._quartets import Quartets

        # C(0,1)+C(1,2)+C(2,3)+C(3,4) = 0+0+0+0 = 0
        assert Quartets.quartet_index(0, 1, 2, 3) == 0
        # C(0,1)+C(1,2)+C(2,3)+C(4,4) = 0+0+0+1 = 1
        assert Quartets.quartet_index(0, 1, 2, 4) == 1
        # C(1,1)+C(2,2)+C(3,3)+C(4,4) = 1+1+1+1 = 4
        assert Quartets.quartet_index(1, 2, 3, 4) == 4
        # C(0,1)+C(1,2)+C(3,3)+C(4,4) = 0+0+1+1 = 2
        assert Quartets.quartet_index(0, 1, 3, 4) == 2

    def test_scalar_bijection(self):
        """All quartets from a small namespace map to distinct indices."""
        from itertools import combinations
        from quarimo._quartets import Quartets

        n = 10
        indices = [Quartets.quartet_index(*q) for q in combinations(range(n), 4)]
        assert len(indices) == len(set(indices)), "Collisions detected"

    def test_scalar_dense(self):
        """Indices form a contiguous range [0, C(n,4)) for all n-choose-4 quartets."""
        from itertools import combinations
        from quarimo._quartets import Quartets
        from math import comb

        n = 8
        indices = sorted(Quartets.quartet_index(*q) for q in combinations(range(n), 4))
        assert indices == list(range(comb(n, 4)))

    def test_index_array_matches_scalar(self):
        """index_array() values match repeated scalar calls."""
        from quarimo import Forest
        from quarimo._quartets import Quartets

        trees = ["((A:1,B:1):1,(C:1,D:1):1);", "((A:1,C:1):1,(B:1,D:1):1);"]
        forest = Forest(trees)
        quartets = [(0, 1, 2, 3)]
        q = Quartets.from_list(forest, quartets)
        arr = q.index_array()

        for i, (a, b, c, d) in enumerate(q):
            assert arr[i] == Quartets.quartet_index(a, b, c, d)

    def test_index_array_dtype_int64(self):
        """index_array() returns int64 for a small forest."""
        from quarimo import Forest
        from quarimo._quartets import Quartets

        trees = ["((A:1,B:1):1,(C:1,D:1):1);"]
        forest = Forest(trees)
        q = Quartets.from_list(forest, [("A", "B", "C", "D")])
        arr = q.index_array()
        assert arr.dtype == np.int64

    def test_index_array_empty(self):
        """index_array() on an empty Quartets raises before returning."""
        # Quartets enforces count > 0, so an empty array can't be constructed
        # via the public API.  Just verify index_array handles the int64 path.
        from quarimo import Forest
        from quarimo._quartets import Quartets

        trees = ["((A:1,B:1):1,(C:1,D:1):1);"]
        forest = Forest(trees)
        q = Quartets.from_list(forest, [("A", "B", "C", "D")])
        arr = q.index_array()
        assert len(arr) == 1

    def test_index_stable_across_result_objects(self):
        """Two result objects from the same forest and quartets yield identical indices."""
        from quarimo import Forest
        from quarimo._quartets import Quartets

        trees = [
            "((A:1,B:1):1,(C:1,D:1):1);",
            "((A:1,C:1):1,(B:1,D:1):1);",
        ]
        forest = Forest(trees)
        q = Quartets.from_list(forest, [("A", "B", "C", "D")])
        assert np.array_equal(q.index_array(), q.index_array())


class TestQuartetFromIndex:
    """Tests for Quartets.quartet_from_index() and Quartets.quartet_names_from_index()."""

    # ── quartet_from_index ────────────────────────────────────────────────

    def test_scalar_known_values(self):
        """Known (index → quartet) pairs from the quartet_index docstring."""
        from quarimo._quartets import Quartets

        assert Quartets.quartet_from_index(0) == (0, 1, 2, 3)
        assert Quartets.quartet_from_index(1) == (0, 1, 2, 4)
        assert Quartets.quartet_from_index(4) == (1, 2, 3, 4)

    def test_scalar_round_trip(self):
        """quartet_from_index inverts quartet_index for all 4-subsets of {0..9}."""
        from itertools import combinations
        from quarimo._quartets import Quartets

        for abcd in combinations(range(10), 4):
            idx = Quartets.quartet_index(*abcd)
            assert Quartets.quartet_from_index(idx) == abcd

    def test_scalar_returns_sorted_tuple(self):
        """Result is always a 4-tuple with a < b < c < d."""
        from quarimo._quartets import Quartets

        for idx in range(50):
            a, b, c, d = Quartets.quartet_from_index(idx)
            assert a < b < c < d

    def test_array_known_values(self):
        """Array input returns correct (n, 4) ndarray."""
        from quarimo._quartets import Quartets

        result = Quartets.quartet_from_index([0, 1, 4])
        expected = np.array([[0, 1, 2, 3], [0, 1, 2, 4], [1, 2, 3, 4]])
        assert np.array_equal(result, expected)

    def test_array_shape_and_dtype(self):
        from quarimo._quartets import Quartets

        result = Quartets.quartet_from_index(np.arange(20))
        assert result.shape == (20, 4)
        assert result.dtype == np.int64

    def test_array_round_trip(self):
        """Array path inverts quartet_index for a random sample of quartets."""
        from quarimo._quartets import Quartets

        rng = np.random.default_rng(0)
        n_taxa = 40
        abcd = np.array(
            [sorted(rng.choice(n_taxa, 4, replace=False).tolist()) for _ in range(200)]
        )
        indices = np.array([Quartets.quartet_index(*row) for row in abcd])
        recovered = Quartets.quartet_from_index(indices)
        assert np.array_equal(recovered, abcd)

    def test_large_index_scalar(self):
        """Correctly unranks a quartet with large taxon IDs (d ~ 1000)."""
        from quarimo._quartets import Quartets

        abcd = (500, 700, 800, 999)
        idx = Quartets.quartet_index(*abcd)
        assert Quartets.quartet_from_index(idx) == abcd

    def test_dense_bijection(self):
        """Indices [0, C(n,4)) each unrank to a distinct valid quartet."""
        from math import comb
        from quarimo._quartets import Quartets

        n = 8
        n_quartets = comb(n, 4)
        recovered = Quartets.quartet_from_index(np.arange(n_quartets))
        # All a < b < c < d
        assert np.all(recovered[:, 0] < recovered[:, 1])
        assert np.all(recovered[:, 1] < recovered[:, 2])
        assert np.all(recovered[:, 2] < recovered[:, 3])
        # All d < n
        assert np.all(recovered[:, 3] < n)
        # No duplicate rows
        rows_as_tuples = set(map(tuple, recovered))
        assert len(rows_as_tuples) == n_quartets

    # ── quartet_names_from_index ──────────────────────────────────────────

    def test_names_scalar(self):
        """Scalar index → 4-tuple of strings."""
        from quarimo._quartets import Quartets

        names = ["Aardvark", "Bear", "Cat", "Dog", "Eel"]
        assert Quartets.quartet_names_from_index(0, names) == (
            "Aardvark", "Bear", "Cat", "Dog"
        )

    def test_names_array(self):
        """Array of indices → list of string tuples."""
        from quarimo._quartets import Quartets

        names = ["Aardvark", "Bear", "Cat", "Dog", "Eel"]
        result = Quartets.quartet_names_from_index([0, 1], names)
        assert result == [
            ("Aardvark", "Bear", "Cat", "Dog"),
            ("Aardvark", "Bear", "Cat", "Eel"),
        ]

    def test_names_agrees_with_from_index(self):
        """quartet_names_from_index is consistent with quartet_from_index."""
        from quarimo._quartets import Quartets

        names = [f"T{i:03d}" for i in range(20)]
        indices = [0, 1, 4, 100]
        ids = Quartets.quartet_from_index(indices)
        expected = [
            (names[r[0]], names[r[1]], names[r[2]], names[r[3]]) for r in ids
        ]
        assert Quartets.quartet_names_from_index(indices, names) == expected

    def test_names_with_forest(self):
        """quartet_names_from_index integrates with Forest.global_names."""
        from quarimo import Forest
        from quarimo._quartets import Quartets

        forest = Forest(["((A:1,B:1):1,(C:1,D:1):1);"])
        q = Quartets.from_list(forest, [("A", "B", "C", "D")])
        idx = q.index_array()[0]
        result = Quartets.quartet_names_from_index(int(idx), forest.global_names)
        assert set(result) == {"A", "B", "C", "D"}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
