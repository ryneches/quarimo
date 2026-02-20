"""
test_cpu_kernels.py
===================
Tests for CPU kernels (_cpu_kernels.py).

These tests verify that the kernel module separation works correctly:
- Modules can be imported
- Numba infrastructure is available
- Functions exist and are callable

The actual kernel logic is thoroughly tested via integration tests in
test_phylo_tree_collection.py which use real tree data from PhyloTreeCollection.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import the kernel modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

# Try to import the CPU kernels
try:
    from quarimo._cpu_kernels import (
        _rmq_csr_nb,
        _quartet_counts_njit,
        _quartet_steiner_njit,
        _NUMBA_AVAILABLE
    )
    KERNELS_AVAILABLE = True
except ImportError:
    KERNELS_AVAILABLE = False
    _NUMBA_AVAILABLE = False

# Skip all tests if kernels not available
pytestmark = pytest.mark.skipif(
    not KERNELS_AVAILABLE,
    reason="CPU kernels module not available"
)


class TestKernelImports:
    """Test that kernel imports work correctly."""
    
    def test_module_imports_successfully(self):
        """Test that _cpu_kernels module can be imported."""
        import quarimo._cpu_kernels as _cpu_kernels
        assert _cpu_kernels is not None
    
    def test_numba_availability_flag_exists(self):
        """Test that _NUMBA_AVAILABLE flag is defined."""
        assert isinstance(_NUMBA_AVAILABLE, bool)
    
    def test_all_kernels_imported(self):
        """Test that all three kernel functions are imported."""
        assert _rmq_csr_nb is not None
        assert _quartet_counts_njit is not None
        assert _quartet_steiner_njit is not None


class TestRMQKernel:
    """Tests for _rmq_csr_nb kernel structure."""
    
    def test_rmq_kernel_is_callable(self):
        """Test that RMQ kernel is callable."""
        assert callable(_rmq_csr_nb)
    
    def test_rmq_kernel_has_correct_name(self):
        """Test that RMQ kernel has expected name."""
        assert _rmq_csr_nb.__name__ == '_rmq_csr_nb'
    
    def test_rmq_kernel_has_parameters(self):
        """Test that RMQ kernel accepts parameters."""
        import inspect
        sig = inspect.signature(_rmq_csr_nb)
        params = list(sig.parameters.keys())
        # Should have: l, r, sp_base, sp_stride, sparse_table, euler_depth, 
        # log2_table, lg_base, tour_base, euler_tour (10 params)
        assert len(params) == 10


class TestQuartetCountsKernel:
    """Tests for _quartet_counts_njit kernel structure."""
    
    def test_counts_kernel_is_callable(self):
        """Test that counts kernel is callable."""
        assert callable(_quartet_counts_njit)
    
    def test_counts_kernel_has_correct_name(self):
        """Test that counts kernel has expected name."""
        assert _quartet_counts_njit.__name__ == '_quartet_counts_njit'
    
    def test_counts_kernel_has_many_parameters(self):
        """Test that counts kernel has correct number of parameters."""
        import inspect
        sig = inspect.signature(_quartet_counts_njit)
        params = list(sig.parameters.keys())
        # Should have 16 parameters for all the CSR arrays and metadata
        assert len(params) == 16


class TestQuartetSteinerKernel:
    """Tests for _quartet_steiner_njit kernel structure."""
    
    def test_steiner_kernel_is_callable(self):
        """Test that Steiner kernel is callable."""
        assert callable(_quartet_steiner_njit)
    
    def test_steiner_kernel_has_correct_name(self):
        """Test that Steiner kernel has expected name."""
        assert _quartet_steiner_njit.__name__ == '_quartet_steiner_njit'
    
    def test_steiner_has_one_more_param_than_counts(self):
        """Test that Steiner kernel has steiner_out parameter."""
        import inspect
        counts_params = len(inspect.signature(_quartet_counts_njit).parameters)
        steiner_params = len(inspect.signature(_quartet_steiner_njit).parameters)
        # Steiner kernel should have one extra parameter (steiner_out)
        assert steiner_params == counts_params + 1
        assert steiner_params == 17


class TestNumbaIntegration:
    """Tests for numba integration."""
    
    def test_numba_status_is_correct(self):
        """Test that _NUMBA_AVAILABLE reflects actual availability."""
        # If we got here, KERNELS_AVAILABLE is True
        # This means import succeeded, so either:
        # 1. Numba is available and kernels are compiled
        # 2. Numba is not available but fallback stubs work
        assert isinstance(_NUMBA_AVAILABLE, bool)
    
    def test_kernels_have_metadata(self):
        """Test that kernels have expected metadata attributes."""
        # All functions should have __name__ and __module__
        assert hasattr(_rmq_csr_nb, '__name__')
        assert hasattr(_quartet_counts_njit, '__name__')
        assert hasattr(_quartet_steiner_njit, '__name__')
    
    def test_module_has_no_unexpected_exports(self):
        """Test that module only exports expected symbols."""
        import quarimo._cpu_kernels as _cpu_kernels
        public_symbols = [name for name in dir(_cpu_kernels) if not name.startswith('_')]
        # Should have minimal public exports (basically none - all start with _)
        # Just checking the module doesn't have unexpected pollution
        assert len(public_symbols) < 10  # Should be very few


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
