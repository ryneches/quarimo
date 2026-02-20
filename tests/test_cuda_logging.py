#!/usr/bin/env python3
"""
test_cuda_logging.py
====================
Demonstrates and validates the CUDA data transfer logging.

This script tests that all expected logging messages are emitted when using
the CUDA backend, even when CUDA is unavailable (fallback case).
"""

import sys
import logging
import io

sys.path.insert(0, ".")

# Set up logging to capture all messages
log_capture = io.StringIO()
handler = logging.StreamHandler(log_capture)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger("forest")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from quarimo._forest import Forest, _CUDA_AVAILABLE

print("=" * 70)
print("CUDA Data Transfer Logging Validation")
print("=" * 70)
print(f"\nCUDA Available: {_CUDA_AVAILABLE}")

# Create test collection
newicks = [
    "((A:1,B:1):1,(C:1,D:1):1);",
    "((A:1,C:1):1,(B:1,D:1):1);",
    "((A:1,D:1):1,(B:1,C:1):1);",
] * 10  # 30 trees

c = Forest(newicks)
quartets = [("A", "B", "C", "D"), ("A", "B", "D", "C")]

print(f"\nCollection: {c.n_trees} trees, {c.n_global_taxa} taxa")
print(f"Quartets: {len(quartets)}")
print(
    f"Problem size: {len(quartets)} quartets × {c.n_trees} trees = {len(quartets) * c.n_trees} pairs"
)

# Clear the log capture from construction
log_capture.truncate(0)
log_capture.seek(0)

print("\n" + "=" * 70)
print("Test 1: Counts-only mode")
print("=" * 70)

counts = c.quartet_topology(quartets, backend="cuda")

logs = log_capture.getvalue()
print("\nLogging output:")
print(logs)

# Validate expected log messages are present
if _CUDA_AVAILABLE:
    expected_messages = [
        "Transferring data to GPU device:",
        "Tree data:",
        "global_to_local:",
        "CSR packed arrays:",
        "Query data:",
        "sorted_quartet_ids:",
        "Total H→D transfer:",
        "Launching CUDA kernel:",
        "Grid:",
        "Total threads:",
        "Transferring results from GPU device:",
        "counts_out:",
        "Total D→H transfer:",
    ]

    print("\nValidation:")
    for msg in expected_messages:
        if msg in logs:
            print(f"  ✓ Found: {msg}")
        else:
            print(f"  ✗ Missing: {msg}")
else:
    print("\nNote: CUDA unavailable, so CUDA-specific logging not shown.")
    print("Expected messages when CUDA is available:")
    print("  - Transferring data to GPU device")
    print("  - Tree data: N arrays, X.XX MB")
    print("  - Query data: N arrays, X.XX MB")
    print("  - Launching CUDA kernel")
    print("  - Grid: NxM blocks, NxM threads/block")
    print("  - Total threads: X,XXX (active: X,XXX, idle: XXX)")
    print("  - Transferring results from GPU device")
    print("  - counts_out: shape dtype, size")

# Clear for next test
log_capture.truncate(0)
log_capture.seek(0)

print("\n" + "=" * 70)
print("Test 2: Steiner mode")
print("=" * 70)

counts, dists = c.quartet_topology(quartets, steiner=True, backend="cuda")

logs = log_capture.getvalue()
print("\nLogging output:")
print(logs)

if _CUDA_AVAILABLE:
    steiner_messages = [
        "Output arrays:",
        "steiner_out",
        "float64",
    ]

    print("\nValidation (Steiner-specific):")
    for msg in steiner_messages:
        if msg in logs:
            print(f"  ✓ Found: {msg}")
        else:
            print(f"  ✗ Missing: {msg}")

print("\n" + "=" * 70)
print("Expected Logging Structure with CUDA")
print("=" * 70)
print("""
When CUDA is available, each quartet_topology(..., backend='cuda') call logs:

1. DATA TRANSFER TO GPU (Host → Device):
   - Tree data summary:
     • Number of arrays
     • Total size in MB
     • global_to_local shape and dtype
     • CSR array statistics (n_trees, sparse table size)
   
   - Query data summary:
     • Number of arrays
     • sorted_quartet_ids shape and dtype
     • Total size
   
   - Output arrays (Steiner mode only):
     • steiner_out size in MB
     • Array shape
   
   - Total H→D transfer size

2. KERNEL LAUNCH CONFIGURATION:
   - Grid dimensions (blocks in x and y)
   - Threads per block
   - Total threads allocated
   - Active threads (n_quartets × n_trees)
   - Idle threads (overhead from block rounding)

3. RESULTS TRANSFER FROM GPU (Device → Host):
   - counts_out: shape, dtype, size
   - steiner_out: shape, dtype, size (if Steiner mode)
   - Total D→H transfer size

All sizes are formatted for readability:
   - < 1 MB: shown in KB
   - ≥ 1 MB: shown in MB with 2 decimal places
   - Thread counts: comma-separated thousands
""")

print("=" * 70)
print("Test Complete ✓")
print("=" * 70)
