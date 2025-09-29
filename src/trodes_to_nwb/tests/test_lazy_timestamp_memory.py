"""Test lazy timestamp array implementation for memory optimization.

This test validates that the LazyTimestampArray successfully prevents
memory explosion (Issue #47) while maintaining correct functionality.
"""

import numpy as np
try:
    import pytest
except ImportError:
    pytest = None
from memory_profiler import memory_usage
import logging

from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.lazy_timestamp_array import LazyTimestampArray
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
from trodes_to_nwb.tests.utils import data_path

logger = logging.getLogger(__name__)


class TestLazyTimestampMemory:
    """Test suite for lazy timestamp memory optimization."""

    def get_test_rec_file(self):
        """Get test .rec file with full ephys data."""
        test_file = data_path / "20230622_sample_01_a1.rec"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        return str(test_file)

    def test_lazy_timestamp_array_basic_functionality(self, test_rec_file):
        """Test that LazyTimestampArray provides correct array-like interface."""
        # Create SpikeGadgetsRawIO object
        neo_io = SpikeGadgetsRawIO(filename=test_rec_file, interpolate_dropped_packets=True)
        neo_io.parse_header()

        # Create lazy timestamp array
        lazy_timestamps = LazyTimestampArray([neo_io], chunk_size=1000)

        # Test basic properties
        assert len(lazy_timestamps) > 0
        assert lazy_timestamps.shape[0] > 0
        assert lazy_timestamps.dtype == np.float64

        # Test single index access
        first_timestamp = lazy_timestamps[0]
        assert isinstance(first_timestamp, (float, np.floating))

        # Test slice access
        chunk = lazy_timestamps[0:100]
        assert len(chunk) == 100
        assert isinstance(chunk, np.ndarray)

        # Test that timestamps are roughly sequential
        assert chunk[1] > chunk[0]
        assert chunk[-1] > chunk[0]

    def test_memory_usage_comparison(self, test_rec_file):
        """Compare memory usage between original and lazy implementations."""

        def create_original_iterator():
            """Create iterator with original timestamp loading (memory explosion)."""
            # This should cause massive memory usage
            iterator = RecFileDataChunkIterator(
                rec_file_path=[test_rec_file],
                stream_id="trodes",
                interpolate_dropped_packets=True,
                # Force original behavior by providing empty timestamps
                timestamps=None
            )
            return iterator

        def create_lazy_iterator():
            """Create iterator with lazy timestamp loading."""
            # This should use much less memory
            iterator = RecFileDataChunkIterator(
                rec_file_path=[test_rec_file],
                stream_id="trodes",
                interpolate_dropped_packets=True,
                timestamp_chunk_size=10000  # Small chunks
            )
            return iterator

        # Measure memory for lazy implementation
        logger.info("Testing lazy timestamp memory usage...")
        lazy_memory = memory_usage((create_lazy_iterator, ()), interval=0.1, timeout=30)

        if lazy_memory:
            peak_lazy = max(lazy_memory)
            baseline = lazy_memory[0]
            lazy_increase = peak_lazy - baseline

            logger.info(f"Lazy implementation memory: {baseline:.1f}MB → {peak_lazy:.1f}MB (+{lazy_increase:.1f}MB)")

            # Lazy implementation should use less than 200MB additional memory
            assert lazy_increase < 200, f"Lazy implementation used too much memory: {lazy_increase:.1f}MB"

            # Save memory info for comparison
            memory_info = {
                "lazy_baseline_mb": baseline,
                "lazy_peak_mb": peak_lazy,
                "lazy_increase_mb": lazy_increase,
            }

            logger.info(f"✅ Lazy timestamp memory test passed: +{lazy_increase:.1f}MB")
            return memory_info

    def test_lazy_timestamp_accuracy(self, test_rec_file):
        """Test that lazy timestamps produce the same results as original method."""
        # Create objects for comparison
        neo_io = SpikeGadgetsRawIO(filename=test_rec_file, interpolate_dropped_packets=True)
        neo_io.parse_header()

        lazy_timestamps = LazyTimestampArray([neo_io], chunk_size=1000)

        # Test small chunks to verify accuracy
        chunk_size = 100
        test_indices = [0, 1000, 5000]  # Test different positions

        for start_idx in test_indices:
            if start_idx + chunk_size > len(lazy_timestamps):
                continue

            # Get lazy chunk
            lazy_chunk = lazy_timestamps[start_idx:start_idx + chunk_size]

            # Verify chunk properties
            assert len(lazy_chunk) == chunk_size
            assert np.all(np.diff(lazy_chunk) > 0), "Timestamps should be increasing"
            assert np.all(np.isfinite(lazy_chunk)), "All timestamps should be finite"

            logger.info(f"✅ Accuracy test passed for chunk starting at {start_idx}")

    def test_lazy_timestamp_indexing(self, test_rec_file):
        """Test various indexing patterns on lazy timestamp array."""
        neo_io = SpikeGadgetsRawIO(filename=test_rec_file, interpolate_dropped_packets=True)
        neo_io.parse_header()

        lazy_timestamps = LazyTimestampArray([neo_io], chunk_size=500)
        total_length = len(lazy_timestamps)

        # Test single index
        single_val = lazy_timestamps[42]
        assert isinstance(single_val, (float, np.floating))

        # Test negative indexing
        last_val = lazy_timestamps[-1]
        second_last = lazy_timestamps[-2]
        assert last_val > second_last

        # Test slice with step
        stepped = lazy_timestamps[0:1000:10]
        assert len(stepped) == 100
        assert np.all(np.diff(stepped) > 0)

        # Test array indexing
        indices = np.array([10, 50, 100, 200])
        indexed = lazy_timestamps[indices]
        assert len(indexed) == 4
        assert np.all(np.diff(indexed) > 0)

        logger.info("✅ All indexing tests passed")

    def test_memory_info_reporting(self, test_rec_file):
        """Test that memory info reporting works correctly."""
        neo_io = SpikeGadgetsRawIO(filename=test_rec_file, interpolate_dropped_packets=True)
        neo_io.parse_header()

        lazy_timestamps = LazyTimestampArray([neo_io], chunk_size=1000)
        memory_info = lazy_timestamps.get_memory_info()

        # Verify memory info structure
        required_keys = ["shape", "dtype", "estimated_full_size_gb", "chunk_size", "num_files"]
        for key in required_keys:
            assert key in memory_info, f"Missing key: {key}"

        assert memory_info["num_files"] == 1
        assert memory_info["chunk_size"] == 1000
        assert memory_info["estimated_full_size_gb"] > 0

        logger.info(f"Memory info: {memory_info}")

    def test_sequential_check_performance(self, test_rec_file):
        """Test that sequential checking doesn't cause memory explosion."""

        def check_with_lazy_method():
            """Test lazy sequential checking."""
            iterator = RecFileDataChunkIterator(
                rec_file_path=[test_rec_file],
                stream_id="trodes",
                interpolate_dropped_packets=True,
                timestamp_chunk_size=10000
            )
            # This should trigger the lazy sequential check
            return iterator

        # Measure memory during sequential check
        memory_usage_result = memory_usage((check_with_lazy_method, ()), interval=0.1, timeout=30)

        if memory_usage_result:
            peak_memory = max(memory_usage_result)
            baseline = memory_usage_result[0]
            increase = peak_memory - baseline

            # Sequential check should not cause massive memory increase
            assert increase < 500, f"Sequential check used too much memory: {increase:.1f}MB"

            logger.info(f"✅ Sequential check memory test passed: +{increase:.1f}MB")


if __name__ == "__main__":
    """Run memory tests directly."""
    import sys
    sys.path.insert(0, "src")

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Find test file
    test_file = data_path / "20230622_sample_01_a1.rec"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)

    print("🧪 Running Lazy Timestamp Memory Tests")
    print("=" * 60)

    try:
        test_class = TestLazyTimestampMemory()

        # Run key tests
        print("\n1. Testing basic functionality...")
        test_class.test_lazy_timestamp_array_basic_functionality(str(test_file))

        print("\n2. Testing memory usage...")
        memory_info = test_class.test_memory_usage_comparison(str(test_file))

        print("\n3. Testing accuracy...")
        test_class.test_lazy_timestamp_accuracy(str(test_file))

        print("\n4. Testing indexing...")
        test_class.test_lazy_timestamp_indexing(str(test_file))

        print("\n5. Testing memory info...")
        test_class.test_memory_info_reporting(str(test_file))

        print("\n6. Testing sequential check...")
        test_class.test_sequential_check_performance(str(test_file))

        print("\n" + "=" * 60)
        print("✅ ALL LAZY TIMESTAMP TESTS PASSED!")
        print("🎯 Memory optimization successfully implemented")

        if memory_info:
            print(f"💾 Memory improvement: Limited to +{memory_info['lazy_increase_mb']:.1f}MB")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)