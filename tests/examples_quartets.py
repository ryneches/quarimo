"""
Examples: Using the Quartets class for on-GPU quartet generation
==================================================================

The Quartets class provides a unified interface for both explicit
quartet lists and random sampling, with optimized GPU generation.
"""

import numpy as np
from quarimo import Forest
from quarimo._quartets import Quartets


# ============================================================================
# Example 1: Explicit quartet list (user-provided)
# ============================================================================

def example_explicit_quartets():
    """Process a specific list of quartets."""
    
    # Create forest
    trees = [
        '((A:1,B:1):1,(C:1,D:1):1);',
        '((A:1,C:1):1,(B:1,D:1):1);',
        '((A:1,D:1):1,(B:1,C:1):1);',
    ]
    forest = Forest(trees)
    
    # Define specific quartets to analyze (by name)
    my_quartets = [
        ('A', 'B', 'C', 'D'),
        ('A', 'B', 'C', 'E'),  # Will raise error - E not in forest
    ]
    
    # This will validate and map names to global IDs
    try:
        q = Quartets.from_list(forest, my_quartets)
        counts = forest.quartet_topology(q)
        print(f"Topology counts: {counts}")
    except ValueError as e:
        print(f"Validation error: {e}")
    
    # Using global IDs instead
    # Global namespace: ['A', 'B', 'C', 'D'] → [0, 1, 2, 3]
    my_quartets_ids = [(0, 1, 2, 3)]
    q = Quartets.from_list(forest, my_quartets_ids)
    counts = forest.quartet_topology(q)
    
    print(f"Analyzed {len(q)} quartets")
    print(f"Topology counts:\n{counts}")


# ============================================================================
# Example 2: Random sampling (default seed)
# ============================================================================

def example_random_sampling():
    """Sample quartets randomly with default seed."""
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 100
    forest = Forest(trees)
    
    # Sample 1 million random quartets
    q = Quartets.random(forest, count=1_000_000)
    
    print(f"Sampling {len(q)} random quartets")
    print(f"Seed quartet: {q.seed[0]}")
    print(f"RNG seed: 0x{q.rng_seed:08x}")
    
    # Process on GPU (if available)
    counts = forest.quartet_topology(q, backend='cuda')
    
    # Compute frequencies
    frequencies = counts.sum(axis=0) / counts.sum()
    print(f"Topology frequencies: {frequencies}")


# ============================================================================
# Example 3: Reproducible random sampling
# ============================================================================

def example_reproducible_sampling():
    """Use specific seed for reproducible results."""
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 100
    forest = Forest(trees)
    
    # Option 1: Integer seed (simple)
    q1 = Quartets.random(forest, count=100, seed=42)
    q2 = Quartets.random(forest, count=100, seed=42)
    
    counts1 = forest.quartet_topology(q1)
    counts2 = forest.quartet_topology(q2)
    
    print(f"Same seed gives same results: {np.array_equal(counts1, counts2)}")
    
    # Option 2: Custom seed quartet
    import random
    random.seed(42)
    seed_quartet = tuple(sorted(random.sample(range(forest.n_global_taxa), 4)))
    
    q3 = Quartets.random(forest, count=100, seed=seed_quartet)
    print(f"Custom seed quartet: {seed_quartet}")


# ============================================================================
# Example 4: Testing - verify CPU and GPU match
# ============================================================================

def example_verify_cpu_gpu():
    """Verify that CPU iteration matches GPU processing."""
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 10
    forest = Forest(trees)
    
    # Create random sampler
    q = Quartets.random(forest, count=100, seed=42)
    
    # Generate quartets on CPU
    cpu_quartets = list(q)
    print(f"Generated {len(cpu_quartets)} quartets on CPU")
    print(f"First 3: {cpu_quartets[:3]}")
    
    # Process on GPU (generates quartets internally)
    gpu_counts = forest.quartet_topology(q, backend='cuda')
    
    # Process CPU-generated quartets on CPU for verification
    q_explicit = Quartets.from_list(forest, cpu_quartets)
    cpu_counts = forest.quartet_topology(q_explicit, backend='cpu-parallel')
    
    print(f"CPU and GPU match: {np.array_equal(cpu_counts, gpu_counts)}")


# ============================================================================
# Example 5: Mixed explicit + random
# ============================================================================

def example_mixed_mode():
    """Use specific quartets as seeds, then generate random ones."""
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 100
    forest = Forest(trees)
    
    # Start with interesting quartets
    interesting = [
        (0, 1, 2, 3),
        (0, 1, 2, 4),
        (0, 1, 3, 4),
    ]
    
    # Process these 3, plus 1 million random quartets
    q = Quartets(forest, seed=interesting, offset=0, count=3 + 1_000_000)
    
    print(f"Processing {len(q)} quartets:")
    print(f"  - First 3: explicit (from seed)")
    print(f"  - Next 1M: random (generated from seed hash)")
    
    counts = forest.quartet_topology(q)
    
    # First 3 rows are the explicit quartets
    print(f"\nExplicit quartet counts:")
    print(counts[:3])
    
    # Remaining are random samples
    print(f"\nRandom sample summary:")
    print(f"  Total sampled: {counts[3:].sum()}")
    print(f"  Frequencies: {counts[3:].sum(axis=0) / counts[3:].sum()}")


# ============================================================================
# Example 6: Batching large analyses
# ============================================================================

def example_batching():
    """Process very large quartet sets in batches."""
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 1000
    forest = Forest(trees)
    
    total_samples = 10_000_000  # 10M quartets
    batch_size = 1_000_000      # Process 1M at a time
    
    # Accumulator for results
    total_counts = np.zeros(3, dtype=np.int64)
    
    # Process in batches
    q_full = Quartets.random(forest, count=total_samples, seed=42)
    
    for batch_start in range(0, total_samples, batch_size):
        batch_count = min(batch_size, total_samples - batch_start)
        
        # Create batch view into the sequence
        q_batch = Quartets(
            forest,
            seed=q_full.seed,
            offset=q_full.offset + batch_start,
            count=batch_count
        )
        q_batch.rng_seed = q_full.rng_seed  # Same RNG seed
        
        # Process batch
        batch_counts = forest.quartet_topology(q_batch)
        total_counts += batch_counts.sum(axis=0)
        
        print(f"Processed batch {batch_start//batch_size + 1}: "
              f"{batch_counts.sum()} quartets")
    
    # Final frequencies
    frequencies = total_counts / total_counts.sum()
    print(f"\nFinal topology frequencies: {frequencies}")


# ============================================================================
# Example 7: Performance comparison
# ============================================================================

def example_performance_comparison():
    """Compare old vs new approach."""
    import time
    
    trees = ['((A:1,B:1):1,(C:1,D:1):1);'] * 3000
    forest = Forest(trees)
    n_samples = 1_000_000
    
    print(f"Processing {n_samples} quartets on {len(trees)} trees")
    print(f"Global namespace: {forest.n_global_taxa} taxa\n")
    
    # Old approach: Generate on CPU, transfer to GPU
    print("Old approach (CPU generation):")
    start = time.time()
    cpu_quartets = [
        tuple(sorted(np.random.choice(forest.n_global_taxa, 4, replace=False)))
        for _ in range(n_samples)
    ]
    gen_time = time.time() - start
    
    start = time.time()
    counts_old = forest.quartet_topology(
        Quartets.from_list(forest, cpu_quartets),
        backend='cuda'
    )
    process_time = time.time() - start
    total_old = gen_time + process_time
    
    print(f"  CPU generation: {gen_time:.3f}s")
    print(f"  GPU processing: {process_time:.3f}s")
    print(f"  Total: {total_old:.3f}s")
    
    # New approach: Generate on GPU
    print("\nNew approach (GPU generation):")
    q = Quartets.random(forest, count=n_samples, seed=42)
    
    start = time.time()
    counts_new = forest.quartet_topology(q, backend='cuda')
    total_new = time.time() - start
    
    print(f"  Total: {total_new:.3f}s")
    print(f"\n  Speedup: {total_old / total_new:.1f}×")
    
    # Verify results match (approximately - different random samples)
    freq_old = counts_old.sum(axis=0) / counts_old.sum()
    freq_new = counts_new.sum(axis=0) / counts_new.sum()
    print(f"\n  Frequency difference: {np.abs(freq_old - freq_new).max():.6f}")


# ============================================================================
# Run examples
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Example 1: Explicit quartet list")
    print("=" * 70)
    example_explicit_quartets()
    
    print("\n" + "=" * 70)
    print("Example 2: Random sampling")
    print("=" * 70)
    example_random_sampling()
    
    print("\n" + "=" * 70)
    print("Example 3: Reproducible sampling")
    print("=" * 70)
    example_reproducible_sampling()
    
    print("\n" + "=" * 70)
    print("Example 4: Verify CPU/GPU match")
    print("=" * 70)
    example_verify_cpu_gpu()
    
    print("\n" + "=" * 70)
    print("Example 5: Mixed explicit + random")
    print("=" * 70)
    example_mixed_mode()
    
    print("\n" + "=" * 70)
    print("Example 6: Batching")
    print("=" * 70)
    example_batching()
    
    print("\n" + "=" * 70)
    print("Example 7: Performance comparison")
    print("=" * 70)
    example_performance_comparison()
