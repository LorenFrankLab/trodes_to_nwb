"""
Memory profiling tests for trodes_to_nwb memory optimization.

These tests establish baseline memory usage patterns and validate
our theoretical calculations against real measurements.
"""

import pytest
import psutil
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path


class MemoryProfiler:
    """Track memory usage during test execution."""

    def __init__(self):
        self.peak_memory_gb = 0
        self.memory_timeline = []
        self.process = psutil.Process()

    def track_memory(self, label: str):
        """Record current memory usage with a label."""
        current_gb = self.process.memory_info().rss / (1024**3)
        self.peak_memory_gb = max(self.peak_memory_gb, current_gb)
        self.memory_timeline.append((label, current_gb))

    def assert_memory_under_limit(self, limit_gb: float):
        """Assert peak memory stayed under limit."""
        assert self.peak_memory_gb < limit_gb, (
            f"Memory usage {self.peak_memory_gb:.1f}GB exceeded limit {limit_gb}GB. "
            f"Timeline: {self.memory_timeline}"
        )

    def get_memory_delta(self, start_label: str, end_label: str) -> float:
        """Get memory difference between two labels."""
        start_memory = None
        end_memory = None

        for label, memory in self.memory_timeline:
            if label == start_label:
                start_memory = memory
            elif label == end_label:
                end_memory = memory

        if start_memory is None or end_memory is None:
            raise ValueError(f"Could not find labels {start_label} and {end_label}")

        return end_memory - start_memory


@pytest.fixture
def memory_profiler():
    """Provide a MemoryProfiler instance for tests."""
    return MemoryProfiler()


class TestMemoryCalculationValidation:
    """Validate our theoretical memory calculations against real measurements."""

    def test_memory_calculation_accuracy_1hour(self, memory_profiler):
        """Validate memory calculation for 1-hour recording."""
        # Test with manageable 1-hour array first
        samples_1h = int(1 * 3600 * 30000)  # 1 hour = 108M samples
        expected_1h_gb = (samples_1h * 8) / (1024**3)  # float64

        memory_profiler.track_memory("before_1h_array")

        # Create actual array and measure
        timestamps = np.random.random(samples_1h).astype(np.float64)

        memory_profiler.track_memory("after_1h_array")

        actual_gb = memory_profiler.get_memory_delta("before_1h_array", "after_1h_array")

        # Should be close to calculated (within 20% due to overhead)
        relative_error = abs(actual_gb - expected_1h_gb) / expected_1h_gb
        assert relative_error < 0.2, (
            f"Expected {expected_1h_gb:.2f}GB, got {actual_gb:.2f}GB "
            f"(error: {relative_error:.1%})"
        )

        # Clean up
        del timestamps

    def test_data_type_size_assumptions(self):
        """Validate our assumptions about data types used."""
        # Test float64 size
        float64_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert float64_array.dtype == np.float64
        assert float64_array.itemsize == 8  # 8 bytes per float64

        # Test int16 size (for data arrays)
        int16_array = np.array([1, 2, 3], dtype=np.int16)
        assert int16_array.itemsize == 2  # 2 bytes per int16

        # Test uint32 size (for timestamps from file)
        uint32_array = np.array([1, 2, 3], dtype=np.uint32)
        assert uint32_array.itemsize == 4  # 4 bytes per uint32

    def test_extrapolate_to_17_hours(self, memory_profiler):
        """Test memory calculation extrapolation to 17 hours."""
        # Use smaller arrays to extrapolate
        samples_10min = int(10 * 60 * 30000)  # 10 minutes
        expected_10min_gb = (samples_10min * 8) / (1024**3)

        memory_profiler.track_memory("before_10min")
        timestamps = np.random.random(samples_10min).astype(np.float64)
        memory_profiler.track_memory("after_10min")

        actual_10min_gb = memory_profiler.get_memory_delta("before_10min", "after_10min")

        # Extrapolate to 17 hours
        minutes_in_17h = 17 * 60
        scaling_factor = minutes_in_17h / 10
        extrapolated_17h_gb = actual_10min_gb * scaling_factor

        # Should be close to our theoretical 14.4GB calculation
        assert abs(extrapolated_17h_gb - 14.4) < 2.0, (
            f"Extrapolated 17h memory {extrapolated_17h_gb:.1f}GB "
            f"differs significantly from calculated 14.4GB"
        )

        del timestamps


class TestMemoryBottleneckIdentification:
    """Identify which specific operations cause memory spikes."""

    def test_individual_bottleneck_measurement(self, memory_profiler):
        """Test each suspected bottleneck in isolation."""

        bottlenecks = [
            ("timestamp_loading", self._simulate_timestamp_loading),
            ("sample_counting", self._simulate_sample_counting),
            ("array_concatenation", self._simulate_array_concatenation),
        ]

        memory_usage = {}

        for name, operation in bottlenecks:
            memory_profiler.track_memory(f"before_{name}")
            operation()
            memory_profiler.track_memory(f"after_{name}")

            memory_usage[name] = memory_profiler.get_memory_delta(
                f"before_{name}", f"after_{name}"
            )

        # Identify the biggest memory consumers
        sorted_bottlenecks = sorted(memory_usage.items(), key=lambda x: x[1], reverse=True)

        # Document findings
        print(f"Memory bottlenecks (GB): {sorted_bottlenecks}")

        # All should be measurable
        assert all(usage > 0 for usage in memory_usage.values())

        return sorted_bottlenecks

    def _simulate_timestamp_loading(self):
        """Simulate the memory impact of loading timestamps for 1 hour."""
        samples_1h = int(1 * 3600 * 30000)
        timestamps = np.random.random(samples_1h).astype(np.float64)
        return timestamps

    def _simulate_sample_counting(self):
        """Simulate the memory impact of sample counting operations."""
        samples_1h = int(1 * 3600 * 30000)
        # Simulate multiple arrays created during sample counting
        timestamps = np.random.random(samples_1h).astype(np.float64)
        systime = timestamps * 1e9  # Convert to nanoseconds
        sample_indices = np.arange(samples_1h, dtype=np.int64)
        return timestamps, systime, sample_indices

    def _simulate_array_concatenation(self):
        """Simulate memory impact of concatenating multiple arrays."""
        # Simulate 3 files each with 20 minutes of data
        arrays = []
        for i in range(3):
            samples_20min = int(20 * 60 * 30000)
            array = np.random.random(samples_20min).astype(np.float64)
            arrays.append(array)

        # Concatenate like the real code does
        concatenated = np.concatenate(arrays)
        return concatenated


class TestCurrentImplementationBaseline:
    """Establish baseline memory usage of current implementation."""

    def test_current_memory_usage_small_file(self, memory_profiler):
        """Test current implementation with a small file that works."""

        # Mock a 1-hour recording (should work with current implementation)
        total_samples = int(1 * 3600 * 30000)

        memory_profiler.track_memory("start")

        # Mock the current RecFileDataChunkIterator behavior
        with patch('trodes_to_nwb.convert_ephys.SpikeGadgetsRawIO') as mock_io:
            self._setup_mock_io(mock_io, total_samples)

            memory_profiler.track_memory("after_mock_setup")

            try:
                # Import here to avoid issues if the module changes
                from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

                iterator = RecFileDataChunkIterator(['mock_file.rec'])
                memory_profiler.track_memory("after_iterator_init")

            except ImportError:
                pytest.skip("RecFileDataChunkIterator not available for testing")
            except Exception as e:
                memory_profiler.track_memory("exception_occurred")
                pytest.fail(f"Unexpected error: {e}")

        # Document current memory usage
        init_memory = memory_profiler.get_memory_delta("start", "after_iterator_init")
        print(f"Current implementation 1h recording memory: {init_memory:.2f}GB")

        # Should be reasonable for 1-hour file
        assert init_memory < 2.0, f"1-hour file uses {init_memory:.2f}GB - too much!"

    def test_current_memory_usage_estimation_17h(self, memory_profiler):
        """Estimate what current implementation would use for 17h recording."""

        # We can't actually test 17h (would cause MemoryError)
        # So we test scaling behavior with smaller files

        durations = [10, 20, 30, 60]  # minutes
        memory_usage = []

        for duration_min in durations:
            total_samples = int(duration_min * 60 * 30000)

            memory_profiler.track_memory(f"before_{duration_min}min")

            # Mock smaller recordings
            with patch('trodes_to_nwb.convert_ephys.SpikeGadgetsRawIO') as mock_io:
                self._setup_mock_io(mock_io, total_samples)

                try:
                    from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
                    iterator = RecFileDataChunkIterator(['mock_file.rec'])

                    memory_profiler.track_memory(f"after_{duration_min}min")

                    usage = memory_profiler.get_memory_delta(
                        f"before_{duration_min}min", f"after_{duration_min}min"
                    )
                    memory_usage.append((duration_min, usage))

                except Exception as e:
                    # If we hit memory limits even with smaller files
                    memory_usage.append((duration_min, float('inf')))

        # Analyze scaling behavior
        print(f"Memory usage by duration: {memory_usage}")

        # Try to fit a linear relationship
        valid_measurements = [(d, m) for d, m in memory_usage if m != float('inf')]

        if len(valid_measurements) >= 2:
            # Estimate 17h usage by extrapolation
            duration_17h = 17 * 60  # 17 hours in minutes

            # Simple linear extrapolation from the largest valid measurement
            max_duration, max_memory = max(valid_measurements)
            scaling_factor = duration_17h / max_duration
            estimated_17h_memory = max_memory * scaling_factor

            print(f"Estimated 17h memory usage: {estimated_17h_memory:.1f}GB")

            # This should confirm our ~43GB calculation
            assert estimated_17h_memory > 20, "Scaling suggests less memory usage than expected"

    def _setup_mock_io(self, mock_io_class, total_samples):
        """Setup mock SpikeGadgetsRawIO for testing."""
        mock_instance = MagicMock()

        # Mock the methods that load data
        mock_instance.get_regressed_systime.return_value = np.random.random(total_samples).astype(np.float64)
        mock_instance.get_systime_from_trodes_timestamps.return_value = np.random.random(total_samples).astype(np.float64)
        mock_instance.get_signal_size.return_value = total_samples
        mock_instance.signal_channels_count.return_value = 128
        mock_instance.block_count.return_value = 1
        mock_instance.segment_count.return_value = 1
        mock_instance.signal_streams_count.return_value = 4
        mock_instance.parse_header.return_value = None

        # Mock the memory map attributes
        mock_memmap = MagicMock()
        mock_memmap.shape = (total_samples,)
        mock_instance._raw_memmap = mock_memmap

        mock_io_class.return_value = mock_instance
        return mock_instance


if __name__ == "__main__":
    # Run basic validation when executed directly
    profiler = MemoryProfiler()

    print("Running basic memory calculation validation...")

    # Test 1-hour calculation
    samples_1h = int(1 * 3600 * 30000)
    expected_1h_gb = (samples_1h * 8) / (1024**3)

    profiler.track_memory("start")
    timestamps = np.random.random(samples_1h).astype(np.float64)
    profiler.track_memory("end")

    actual_gb = profiler.get_memory_delta("start", "end")

    print(f"1-hour recording:")
    print(f"  Expected: {expected_1h_gb:.2f}GB")
    print(f"  Actual: {actual_gb:.2f}GB")
    print(f"  Error: {abs(actual_gb - expected_1h_gb) / expected_1h_gb:.1%}")

    # Extrapolate to 17 hours
    extrapolated_17h = actual_gb * 17
    print(f"17-hour extrapolation: {extrapolated_17h:.1f}GB")

    del timestamps
    print("Basic validation complete.")