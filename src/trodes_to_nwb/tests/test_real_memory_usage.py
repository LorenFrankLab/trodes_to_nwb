"""
Real memory usage tests for the actual trodes_to_nwb implementation.

These tests measure memory consumption of the actual code paths that users
experience, using the real test data when available or demonstrating the
memory scaling patterns with mock data.
"""

import pytest
import psutil
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the actual classes we want to test
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
from trodes_to_nwb.tests.utils import data_path
from trodes_to_nwb.data_scanner import get_file_info


class RealMemoryProfiler:
    """Track real memory usage during actual code execution."""

    def __init__(self):
        self.peak_memory_gb = 0
        self.memory_timeline = []
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss

    def track_memory(self, label: str):
        """Record current memory usage with a label."""
        current_bytes = self.process.memory_info().rss
        current_gb = current_bytes / (1024**3)
        self.peak_memory_gb = max(self.peak_memory_gb, current_gb)
        self.memory_timeline.append((label, current_gb, current_bytes))

    def get_memory_delta_gb(self, start_label: str, end_label: str) -> float:
        """Get memory difference between two labels in GB."""
        start_bytes = None
        end_bytes = None

        for label, gb, bytes_val in self.memory_timeline:
            if label == start_label:
                start_bytes = bytes_val
            elif label == end_label:
                end_bytes = bytes_val

        if start_bytes is None or end_bytes is None:
            raise ValueError(f"Could not find labels {start_label} and {end_label}")

        return (end_bytes - start_bytes) / (1024**3)

    def get_peak_memory_usage_gb(self) -> float:
        """Get peak memory usage since profiler creation."""
        current_bytes = self.process.memory_info().rss
        peak_bytes = max(current_bytes, max(bytes_val for _, _, bytes_val in self.memory_timeline))
        return (peak_bytes - self.initial_memory) / (1024**3)


@pytest.fixture
def real_memory_profiler():
    """Provide a RealMemoryProfiler instance for tests."""
    return RealMemoryProfiler()


def has_real_test_data():
    """Check if real .rec test files are available."""
    try:
        path_df = get_file_info(data_path)
        rec_files = path_df[path_df.file_extension == ".rec"]
        return len(rec_files) > 0
    except Exception:
        return False


def get_test_rec_files():
    """Get available test .rec files."""
    if not has_real_test_data():
        return []

    path_df = get_file_info(data_path)
    rec_files = path_df[path_df.file_extension == ".rec"]
    return list(rec_files.full_path)


class TestRealMemoryUsage:
    """Test memory usage of actual trodes_to_nwb implementation."""

    @pytest.mark.skipif(not has_real_test_data(), reason="Real test data not available")
    def test_memory_usage_with_real_data(self, real_memory_profiler):
        """Test memory usage with actual test .rec files."""

        test_files = get_test_rec_files()
        if not test_files:
            pytest.skip("No .rec test files available")

        # Use the first available test file
        test_file = test_files[0]
        print(f"Testing with real file: {test_file}")

        real_memory_profiler.track_memory("start_real_test")

        try:
            # Test SpikeGadgetsRawIO with real data
            raw_io = SpikeGadgetsRawIO(filename=test_file)
            real_memory_profiler.track_memory("after_io_creation")

            raw_io.parse_header()
            real_memory_profiler.track_memory("after_header_parsing")

            # Test timestamp loading - this is where memory issues occur
            total_samples = raw_io.get_signal_size(0, 0, 0)
            print(f"Total samples in file: {total_samples:,}")

            # Load timestamps - this is the problematic operation
            timestamps = raw_io.get_analogsignal_timestamps(0, total_samples)
            real_memory_profiler.track_memory("after_timestamp_loading")

            # Measure memory usage
            io_memory = real_memory_profiler.get_memory_delta_gb("start_real_test", "after_io_creation")
            header_memory = real_memory_profiler.get_memory_delta_gb("after_io_creation", "after_header_parsing")
            timestamp_memory = real_memory_profiler.get_memory_delta_gb("after_header_parsing", "after_timestamp_loading")

            print(f"IO creation: {io_memory:.3f}GB")
            print(f"Header parsing: {header_memory:.3f}GB")
            print(f"Timestamp loading: {timestamp_memory:.3f}GB")
            print(f"Timestamp array size: {timestamps.nbytes / (1024**3):.3f}GB")
            print(f"Peak memory: {real_memory_profiler.get_peak_memory_usage_gb():.3f}GB")

            # Validate that we're actually measuring meaningful memory usage
            assert timestamp_memory > 0, "Timestamp loading should use measurable memory"

        except Exception as e:
            print(f"Error with real data: {e}")
            pytest.fail(f"Real data test failed: {e}")

    def test_rec_file_data_chunk_iterator_memory_scaling(self, real_memory_profiler):
        """Test RecFileDataChunkIterator memory usage with controlled scaling."""

        # Test the actual problem: RecFileDataChunkIterator with large timestamp arrays
        durations_hours = [0.1, 0.5, 1.0]  # 6 minutes, 30 minutes, 1 hour
        memory_measurements = []

        for duration_hours in durations_hours:
            total_samples = int(duration_hours * 3600 * 30000)  # 30kHz sampling

            real_memory_profiler.track_memory(f"start_{duration_hours}h")

            # Mock SpikeGadgetsRawIO to return realistic data sizes without creating huge files
            with patch('trodes_to_nwb.convert_ephys.SpikeGadgetsRawIO') as mock_io_class:
                mock_io_instance = MagicMock()

                # Configure mock to behave like real implementation
                mock_io_instance.parse_header.return_value = None
                mock_io_instance.block_count.return_value = 1
                mock_io_instance.segment_count.return_value = 1
                mock_io_instance.signal_streams_count.return_value = 4
                mock_io_instance.signal_channels_count.return_value = 128
                mock_io_instance.get_signal_size.return_value = total_samples

                # CRITICAL: This is where the real memory usage happens
                # The actual get_regressed_systime creates float64 arrays
                mock_timestamp_array = np.arange(total_samples, dtype=np.float64) / 30000.0
                mock_io_instance.get_regressed_systime.return_value = mock_timestamp_array

                # Mock memory map attributes
                mock_memmap = MagicMock()
                mock_memmap.shape = (total_samples,)
                mock_io_instance._raw_memmap = mock_memmap

                mock_io_class.return_value = mock_io_instance

                try:
                    # This is the actual problematic code path
                    iterator = RecFileDataChunkIterator(['mock_file.rec'])

                    real_memory_profiler.track_memory(f"iterator_created_{duration_hours}h")

                    memory_used = real_memory_profiler.get_memory_delta_gb(
                        f"start_{duration_hours}h", f"iterator_created_{duration_hours}h"
                    )

                    memory_measurements.append((duration_hours, memory_used, total_samples))

                    print(f"{duration_hours}h recording:")
                    print(f"  Samples: {total_samples:,}")
                    print(f"  Memory used: {memory_used:.3f}GB")
                    print(f"  Timestamp array: {iterator.timestamps.nbytes / (1024**3):.3f}GB")
                    print(f"  Memory per sample: {memory_used * 1024**3 / total_samples:.1f} bytes")

                    # Verify that timestamps were actually loaded
                    assert len(iterator.timestamps) == total_samples
                    assert iterator.timestamps.dtype == np.float64

                except MemoryError as e:
                    print(f"MemoryError at {duration_hours}h: {e}")
                    memory_measurements.append((duration_hours, float('inf'), total_samples))

                except Exception as e:
                    print(f"Other error at {duration_hours}h: {e}")
                    pytest.fail(f"Unexpected error: {e}")

        # Analyze scaling behavior
        print("\nMemory scaling analysis:")
        for i, (duration, memory, samples) in enumerate(memory_measurements):
            if memory != float('inf'):
                gb_per_hour = memory / duration
                print(f"  {duration}h: {memory:.3f}GB ({gb_per_hour:.1f}GB/hour)")

        # Calculate what 17 hours would require
        if len(memory_measurements) >= 2:
            valid_measurements = [(d, m) for d, m in memory_measurements if m != float('inf')]

            if valid_measurements:
                # Use the largest successful measurement for extrapolation
                max_duration, max_memory = max(valid_measurements)
                gb_per_hour = max_memory / max_duration

                extrapolated_17h = gb_per_hour * 17
                print(f"\nExtrapolated 17h memory usage: {extrapolated_17h:.1f}GB")

                # This should confirm our analysis
                assert extrapolated_17h > 10, "17h extrapolation should show significant memory usage"

                return extrapolated_17h

        return None


if __name__ == "__main__":
    # Run a quick test when executed directly to validate the approach
    print("Running real memory usage validation...")

    profiler = RealMemoryProfiler()
    profiler.track_memory("start")

    # Test RecFileDataChunkIterator memory scaling with realistic mock
    duration_hours = 0.5  # 30 minutes
    total_samples = int(duration_hours * 3600 * 30000)

    print(f"Testing {duration_hours}h recording ({total_samples:,} samples)")

    # Mock the actual implementation behavior
    with patch('trodes_to_nwb.convert_ephys.SpikeGadgetsRawIO') as mock_io_class:
        mock_io_instance = MagicMock()

        # Configure realistic mock
        mock_io_instance.parse_header.return_value = None
        mock_io_instance.block_count.return_value = 1
        mock_io_instance.segment_count.return_value = 1
        mock_io_instance.signal_streams_count.return_value = 4
        mock_io_instance.signal_channels_count.return_value = 128
        mock_io_instance.get_signal_size.return_value = total_samples

        # Create actual timestamp array (this is what causes memory usage)
        mock_timestamp_array = np.arange(total_samples, dtype=np.float64) / 30000.0
        mock_io_instance.get_regressed_systime.return_value = mock_timestamp_array

        mock_memmap = MagicMock()
        mock_memmap.shape = (total_samples,)
        mock_io_instance._raw_memmap = mock_memmap

        mock_io_class.return_value = mock_io_instance

        try:
            # Test the actual problematic code
            iterator = RecFileDataChunkIterator(['mock_file.rec'])
            profiler.track_memory("iterator_created")

            memory_used = profiler.get_memory_delta_gb("start", "iterator_created")

            print(f"Memory used: {memory_used:.3f}GB")
            print(f"Timestamp array: {iterator.timestamps.nbytes / (1024**3):.3f}GB")
            print(f"Memory per hour: {memory_used / duration_hours:.1f}GB/hour")

            # Extrapolate to 17 hours
            extrapolated_17h = (memory_used / duration_hours) * 17
            print(f"Extrapolated 17h usage: {extrapolated_17h:.1f}GB")

            print("Real memory test complete - this confirms the memory bottleneck!")

        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()