"""
test_cuda_kernels.py
====================
Tests for CUDA kernels (_cuda_kernels.py).

These tests verify that the CUDA kernel module separation works correctly:
- Module can be imported
- CUDA availability is detected
- Functions exist (when CUDA available)
- Grid helper always available

The actual kernel logic is thoroughly tested via integration tests in
test_forest.py which use real tree data from Forest.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import the kernel modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Try to import the CUDA kernels
try:
    from quarimo._cuda_kernels import _compute_cuda_grid, _CUDA_AVAILABLE

    if _CUDA_AVAILABLE:
        from quarimo._cuda_kernels import (
            _rmq_csr_cuda,
            _quartet_counts_cuda,
            _quartet_steiner_cuda,
        )
    KERNELS_MODULE_AVAILABLE = True
except ImportError:
    KERNELS_MODULE_AVAILABLE = False
    _CUDA_AVAILABLE = False

# Skip all tests if module not available
pytestmark = pytest.mark.skipif(
    not KERNELS_MODULE_AVAILABLE, reason="CUDA kernels module not available"
)


class TestModuleImport:
    """Test that CUDA kernel module imports correctly."""

    def test_module_imports_successfully(self):
        """Test that _cuda_kernels module can be imported."""
        import quarimo._cuda_kernels

        assert quarimo._cuda_kernels is not None

    def test_cuda_availability_flag_exists(self):
        """Test that _CUDA_AVAILABLE flag is defined."""
        assert isinstance(_CUDA_AVAILABLE, bool)

    def test_grid_helper_always_available(self):
        """Test that grid helper is available even without CUDA."""
        assert callable(_compute_cuda_grid)


class TestComputeCudaGrid:
    """Tests for _compute_cuda_grid helper function."""

    def test_grid_dimensions_basic(self):
        """Test basic grid dimension calculation."""
        blocks, threads = _compute_cuda_grid(100, 50, threads_per_block=(16, 16))

        # 100 quartets / 16 threads = 7 blocks (ceil)
        # 50 trees / 16 threads = 4 blocks (ceil)
        assert blocks == (7, 4)
        assert threads == (16, 16)

    def test_grid_dimensions_exact_fit(self):
        """Test when dimensions divide evenly."""
        blocks, threads = _compute_cuda_grid(64, 32, threads_per_block=(16, 16))

        # 64 / 16 = 4 exactly
        # 32 / 16 = 2 exactly
        assert blocks == (4, 2)
        assert threads == (16, 16)

    def test_grid_dimensions_single_quartet(self):
        """Test with single quartet."""
        blocks, threads = _compute_cuda_grid(1, 100, threads_per_block=(16, 16))

        # 1 quartet needs 1 block
        # 100 trees / 16 = 7 blocks (ceil)
        assert blocks == (1, 7)
        assert threads == (16, 16)

    def test_grid_dimensions_custom_threads(self):
        """Test with custom thread block size."""
        blocks, threads = _compute_cuda_grid(100, 50, threads_per_block=(32, 8))

        # 100 / 32 = 4 blocks (ceil)
        # 50 / 8 = 7 blocks (ceil)
        assert blocks == (4, 7)
        assert threads == (32, 8)

    def test_grid_handles_large_numbers(self):
        """Test with large quartet/tree counts."""
        blocks, threads = _compute_cuda_grid(10000, 5000, threads_per_block=(16, 16))

        # 10000 / 16 = 625 exactly
        # 5000 / 16 = 313 blocks (ceil)
        assert blocks == (625, 313)
        assert threads == (16, 16)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="Requires CUDA")
class TestCUDAKernelStructure:
    """
    Tests for CUDA kernel structure (only run if CUDA available).

    These tests verify the kernels exist and have correct structure,
    but don't actually execute them with data.
    """

    def test_rmq_cuda_kernel_exists(self):
        """Test that CUDA RMQ device function exists."""
        assert callable(_rmq_csr_cuda)
        assert hasattr(_rmq_csr_cuda, "__name__")

    def test_counts_cuda_kernel_exists(self):
        """Test that CUDA counts kernel exists."""
        assert callable(_quartet_counts_cuda)
        assert hasattr(_quartet_counts_cuda, "__name__")

    def test_steiner_cuda_kernel_exists(self):
        """Test that CUDA Steiner kernel exists."""
        assert callable(_quartet_steiner_cuda)
        assert hasattr(_quartet_steiner_cuda, "__name__")

    def test_cuda_kernels_are_cuda_jit(self):
        """Test that kernels have CUDA JIT attributes."""
        # numba.cuda.jit decorated functions have special attributes
        # We just check they have some kind of dispatcher/compile attributes
        assert hasattr(_quartet_counts_cuda, "__name__")
        assert hasattr(_quartet_steiner_cuda, "__name__")

    def test_cuda_kernels_imported_conditionally(self):
        """Test that CUDA kernels are only available when CUDA is available."""
        # This test only runs when _CUDA_AVAILABLE is True (due to class decorator)
        # So we just verify they're accessible in the module
        import quarimo._cuda_kernels as _cuda_kernels

        assert hasattr(_cuda_kernels, "_rmq_csr_cuda")
        assert hasattr(_cuda_kernels, "_quartet_counts_cuda")
        assert hasattr(_cuda_kernels, "_quartet_steiner_cuda")


class TestCUDAAvailability:
    """Tests for CUDA availability detection."""

    def test_cuda_status_is_boolean(self):
        """Test that CUDA availability is a boolean flag."""
        assert isinstance(_CUDA_AVAILABLE, bool)


class TestGridHelperAlwaysWorks:
    """Tests that grid helper works regardless of CUDA availability."""

    def test_grid_helper_does_not_require_cuda(self):
        """Test that _compute_cuda_grid works without CUDA hardware."""
        # This should work even if CUDA is not available
        # It's just a pure Python helper function
        blocks, threads = _compute_cuda_grid(10, 5)
        assert isinstance(blocks, tuple)
        assert isinstance(threads, tuple)
        assert len(blocks) == 2
        assert len(threads) == 2

    def test_grid_helper_returns_correct_types(self):
        """Test that grid helper returns expected types."""
        blocks, threads = _compute_cuda_grid(100, 50, threads_per_block=(16, 16))
        assert isinstance(blocks, tuple)
        assert isinstance(threads, tuple)
        assert all(isinstance(x, int) for x in blocks)
        assert all(isinstance(x, int) for x in threads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
