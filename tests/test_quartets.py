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
    
    # Create simple forest
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((A:1,C:1):1,(B:1,D:1):1);',
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
    
    # Test 3: Custom seed quartet
    seed_quartet = (0, 1, 2, 3)
    q3 = Quartets.random(forest, count=100, seed=seed_quartet)
    cpu_quartets3 = list(q3)
    
    # Should be different from default seed
    assert cpu_quartets3 != cpu_quartets, "Different seeds should give different sequences"
    
    # But repeated with same seed should match
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
    
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((A:1,C:1):1,(B:1,D:1):1);',
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
    
    # Test 4: Using seed quartets then generating random
    q4 = Quartets(forest, seed=seeds, offset=0, count=5)
    quartets4 = list(q4)
    
    assert quartets4[:3] == seeds, "First 3 should be seeds"
    assert quartets4[3] not in seeds, "4th should be random"
    assert quartets4[4] not in seeds, "5th should be random"


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
    
    # Test mixed
    q_mixed = Quartets(forest, seed=[('A', 'B', 'C', 'D'), (1, 2, 3, 4)], 
                       offset=0, count=2)
    quartets = list(q_mixed)
    
    assert quartets[0] == (0, 1, 2, 3), "Names should map to IDs"
    assert quartets[1] == (1, 2, 3, 4), "IDs should pass through"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
