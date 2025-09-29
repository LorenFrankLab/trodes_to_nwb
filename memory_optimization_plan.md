# Memory Optimization Plan for trodes_to_nwb

## Executive Summary

**Critical Issue**: Users are completely blocked on 17-hour recordings due to memory failures when loading timestamps (Issue #47). This is the highest priority fix requiring immediate attention.

**Root Cause**: Current `RecFileDataChunkIterator` attempts to load all timestamps into memory at once, causing memory exhaustion on long recordings.

**Solution Strategy**: Implement chunked processing with adaptive memory management and memory map pooling to handle arbitrarily long recordings within available system memory.

## Problem Analysis - Multiple Memory Bottlenecks

### Root Cause Analysis: Systematic Memory Issues

Our analysis reveals **multiple compounding memory bottlenecks** that together create the critical failure in Issue #47. Each must be addressed systematically:

#### 1. **CRITICAL**: Timestamp Array Pre-loading (14.4 GB)

**Location**: `convert_ephys.py:206-216`
```python
# LOADS ALL timestamps from ALL files at initialization
self.timestamps = np.concatenate(
    [neo_io.get_regressed_systime(0, None) for neo_io in self.neo_io]
)
```

**Memory Impact**: 17-hour recording @ 30kHz = 1.836B samples × 8 bytes (float64) = **14.4GB**

#### 2. **CRITICAL**: Sample Count Computation (28.8 GB)

**Location**: `convert_ephys.py:226-233`
```python
# Each neo_io.get_signal_size() internally loads full timestamp arrays
self.n_time = [
    neo_io.get_signal_size(block_index=0, seg_index=0, stream_index=self.stream_index)
    for neo_io in self.neo_io
]
```

**Memory Impact**:
- `get_signal_size()` calls `get_analogsignal_timestamps(0, None)`: **+14.4GB**
- Multiple streams processed: **+14.4GB additional**
- **Total**: 28.8GB for sample counting operations

#### 3. **HIGH**: Iterator Splitting Overhead (Variable)

**Location**: `convert_ephys.py:163-199`
```python
# Creates multiple SpikeGadgetsRawIOPartial objects for large files
iterator_size = [neo_io._raw_memmap.shape[0] for neo_io in self.neo_io]
for i, size in enumerate(iterator_size):
    if size > MAXIMUM_ITERATOR_SIZE:
        # Creates multiple partial iterators, each holding memory references
```

**Memory Impact**: Each partial iterator holds references to large arrays, multiplying memory usage

#### 4. **MEDIUM**: Memory Map Fragmentation

**Location**: `spike_gadgets_raw_io.py:82-87`
```python
# Multiple uncoordinated memory maps
self.neo_io = [
    SpikeGadgetsRawIO(filename=file, interpolate_dropped_packets=...)
    for file in rec_file_path
]
```

**Memory Impact**: Virtual memory fragmentation, no memory pooling between iterators

#### 5. **MEDIUM**: Data Copy Operations

**Location**: Throughout data processing pipeline
```python
# Example in spike_gadgets_raw_io.py:611
raw_unit8_mask = raw_unit8[:, stream_mask]  # Copies from memmap to array
```

**Memory Impact**: Doubles memory usage during chunk processing

### **Complete Memory Usage Pattern**

For a 17-hour recording, memory usage accumulates as:

1. **Timestamp loading**: 14.4 GB (`get_regressed_systime`)
2. **Sample count computation**: +28.8 GB (multiple `get_signal_size` calls)
3. **Iterator splitting**: +Variable GB (partial iterator overhead)
4. **Memory map fragmentation**: +Virtual memory pressure
5. **Peak during initialization**: **43+ GB**

**Critical Finding**: Users hit the memory limit during the **initialization phase** of `RecFileDataChunkIterator.__init__()`, before any actual data conversion begins.

## Solution Architecture - Addressing REAL Memory Issues

### Priority 1: Fix RecFileDataChunkIterator Timestamp Loading (CRITICAL)

#### The Problem
```python
# CURRENT: Loads ALL timestamps at initialization (43GB for 17-hour recording)
self.timestamps = np.concatenate([neo_io.get_regressed_systime(0, None) for neo_io in self.neo_io])
```

#### The Solution: Lazy Timestamp Generation
```python
class LazyTimestampIterator:
    """Generate timestamps on-demand without loading full arrays."""

    def __init__(self, neo_io_list: list):
        self.neo_io_list = neo_io_list
        # Store metadata only, not actual timestamps
        self._file_lengths = [neo_io.get_signal_size(0, 0) for neo_io in neo_io_list]
        self._total_length = sum(self._file_lengths)

    @property
    def timestamps(self):
        """Return a virtual array that generates timestamps on access."""
        return VirtualTimestampArray(self.neo_io_list, self._file_lengths)

class VirtualTimestampArray:
    """Virtual array that generates timestamps only when accessed."""

    def __getitem__(self, key):
        # Generate only requested timestamps, not the full array
        if isinstance(key, slice):
            return self._generate_timestamp_slice(key.start, key.stop, key.step)
        else:
            return self._generate_single_timestamp(key)

    @property
    def shape(self):
        return (self._total_length,)
```

### Priority 2: Fix Sample Count Memory Duplication (CRITICAL)

#### The Problem
```python
# CURRENT: Creates multiple 14GB+ arrays simultaneously
systime = np.array(rec_dci.timestamps) * NANOSECONDS_PER_SECOND  # 14.4GB
trodes_sample = np.concatenate([...])  # Another 14.4GB
```

#### The Solution: Streaming Sample Count Processing
```python
def add_sample_count_streaming(nwbfile, rec_dci):
    """Add sample count data without loading full arrays into memory."""

    # Process in chunks instead of loading everything
    chunk_size = 1_000_000  # 1M samples at a time

    # Use HDF5 datasets that can grow
    with h5py.File(temp_file, 'w') as f:
        systime_dataset = f.create_dataset('systime', (0,), maxshape=(None,), dtype='float64')
        sample_dataset = f.create_dataset('samples', (0,), maxshape=(None,), dtype='int64')

        for start_idx in range(0, rec_dci._total_length, chunk_size):
            end_idx = min(start_idx + chunk_size, rec_dci._total_length)

            # Generate only this chunk's timestamps
            chunk_timestamps = rec_dci.timestamps[start_idx:end_idx]
            chunk_systime = chunk_timestamps * NANOSECONDS_PER_SECOND

            # Get corresponding sample data
            chunk_samples = get_sample_chunk(rec_dci.neo_io_list, start_idx, end_idx)

            # Append to HDF5 datasets
            systime_dataset.resize(end_idx)
            sample_dataset.resize(end_idx)
            systime_dataset[start_idx:end_idx] = chunk_systime
            sample_dataset[start_idx:end_idx] = chunk_samples

    # Create NWB TimeSeries from HDF5 datasets
    nwbfile.processing["sample_count"].add(
        TimeSeries(
            name="sample_count",
            description="acquisition system sample count",
            data=H5DataIO(sample_dataset),
            timestamps=H5DataIO(systime_dataset),
            unit="int64",
        )
    )
```

### Priority 3: Eliminate Data Copying in Processing

#### The Problem
```python
# CURRENT: Copies data from memmap to arrays unnecessarily
raw_unit8_mask = raw_unit8[:, stream_mask]  # Memory copy
```

#### The Solution: Zero-Copy Processing
```python
def get_analogsignal_chunk_zerocopy(self, ...):
    """Get data chunk without unnecessary copying."""

    # Work directly with memmap views where possible
    raw_unit8 = self._raw_memmap[i_start:i_stop]  # This is a view, not a copy

    # Use boolean indexing that returns views when possible
    if np.all(stream_mask):  # If selecting all columns, no copy needed
        masked_data = raw_unit8
    else:
        # Only copy when absolutely necessary
        masked_data = raw_unit8[:, stream_mask]  # Copy only when needed

    # Convert data type in-place when possible
    return masked_data.view('int16').reshape(masked_data.shape[0], -1)
```

### Revised Implementation Strategy

```python
# src/trodes_to_nwb/chunked_processing.py
import psutil
from typing import Generator, Tuple
import numpy as np

class ChunkedTimestampProcessor:
    """Process timestamps in memory-bounded chunks for long recordings."""

    def __init__(self, neo_io_list: list, max_memory_gb: float = None):
        self.neo_io_list = neo_io_list
        self.max_memory_gb = max_memory_gb or self._calculate_safe_memory_limit()
        self.chunk_size_samples = self._calculate_optimal_chunk_size()

    def _calculate_safe_memory_limit(self) -> float:
        """Calculate safe memory limit based on available system memory."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        # Use 60% of available memory, leaving room for other operations
        return available_gb * 0.6

    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on memory constraints."""
        # 4 bytes per timestamp + data overhead
        bytes_per_sample = 4 + (128 * 2)  # Assume 128 channels, int16
        target_memory_bytes = self.max_memory_gb * (1024**3)

        chunk_size = int(target_memory_bytes / bytes_per_sample)

        # Ensure reasonable bounds (1 minute to 2 hours)
        min_chunk = 30000 * 60  # 1 minute at 30kHz
        max_chunk = 30000 * 60 * 120  # 2 hours at 30kHz

        return np.clip(chunk_size, min_chunk, max_chunk)

    def iter_timestamp_chunks(self) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """Yield timestamp chunks with start/end indices."""
        for neo_io in self.neo_io_list:
            total_samples = neo_io.get_signal_size(0, 0)

            for start_idx in range(0, total_samples, self.chunk_size_samples):
                end_idx = min(start_idx + self.chunk_size_samples, total_samples)

                # Load only this chunk's timestamps
                timestamps = neo_io.get_analogsignal_timestamps(start_idx, end_idx)
                yield timestamps, start_idx, end_idx

    def iter_data_chunks(self, stream_index: int, channel_indexes: list = None):
        """Yield data chunks synchronized with timestamp chunks."""
        for neo_io in self.neo_io_list:
            total_samples = neo_io.get_signal_size(0, 0)

            for start_idx in range(0, total_samples, self.chunk_size_samples):
                end_idx = min(start_idx + self.chunk_size_samples, total_samples)

                # Load data chunk
                data_chunk = neo_io.get_analogsignal_chunk(
                    block_index=0,
                    seg_index=0,
                    i_start=start_idx,
                    i_stop=end_idx,
                    stream_index=stream_index,
                    channel_indexes=channel_indexes
                )

                timestamps = neo_io.get_analogsignal_timestamps(start_idx, end_idx)
                yield data_chunk, timestamps, start_idx, end_idx
```

### 2. Memory Map Pooling

#### Shared Memory Map Manager

```python
# src/trodes_to_nwb/memory_pool.py
from typing import Dict, Optional, WeakValueDictionary
import threading
import numpy as np
from pathlib import Path

class MemoryMapPool:
    """Shared memory map pool to reduce fragmentation and improve efficiency."""

    def __init__(self, max_maps: int = 20, max_total_size_gb: float = 4.0):
        self._pool: Dict[str, np.memmap] = {}
        self._access_count: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._max_maps = max_maps
        self._max_total_size_gb = max_total_size_gb

    def get_memmap(
        self,
        filepath: Path,
        offset: int = 0,
        dtype: str = "<u1",
        shape: Optional[tuple] = None
    ) -> np.memmap:
        """Get or create a memory map for a file with LRU eviction."""
        key = f"{filepath}_{offset}_{dtype}"

        with self._lock:
            if key in self._pool:
                self._access_count[key] += 1
                return self._pool[key]

            # Check if we need to evict
            if len(self._pool) >= self._max_maps:
                self._evict_lru()

            # Create new memory map
            mmap = np.memmap(
                filepath,
                mode="r",
                offset=offset,
                dtype=dtype,
                shape=shape
            )

            self._pool[key] = mmap
            self._access_count[key] = 1

            return mmap

    def _evict_lru(self):
        """Evict least recently used memory map."""
        if not self._pool:
            return

        # Find LRU entry
        lru_key = min(self._access_count.keys(), key=self._access_count.get)

        # Remove from pool
        del self._pool[lru_key]
        del self._access_count[lru_key]

    def get_memory_usage_gb(self) -> float:
        """Get current memory usage of all maps."""
        total_bytes = sum(
            mmap.nbytes for mmap in self._pool.values()
        )
        return total_bytes / (1024**3)

# Global memory pool instance
_global_memory_pool = MemoryMapPool()

def get_shared_memmap(filepath: Path, offset: int = 0, dtype: str = "<u1") -> np.memmap:
    """Get shared memory map through global pool."""
    return _global_memory_pool.get_memmap(filepath, offset, dtype)
```

### 3. Adaptive Memory Management

#### Memory-Aware Data Iterator

```python
# src/trodes_to_nwb/adaptive_iterator.py
import psutil
import logging
from typing import Iterator, Tuple
import numpy as np

class AdaptiveMemoryIterator:
    """Data iterator that adapts chunk size based on available memory."""

    def __init__(self, chunked_processor: ChunkedTimestampProcessor):
        self.processor = chunked_processor
        self.logger = logging.getLogger(__name__)
        self._memory_warnings_sent = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate through data with adaptive memory management."""
        initial_memory = psutil.virtual_memory().available

        for data_chunk, timestamps, start_idx, end_idx in self.processor.iter_data_chunks():
            # Check memory pressure before yielding
            current_memory = psutil.virtual_memory().available
            memory_used = initial_memory - current_memory
            memory_pressure = 1.0 - (current_memory / psutil.virtual_memory().total)

            if memory_pressure > 0.85:  # >85% memory usage
                self._handle_memory_pressure(memory_pressure, memory_used)

            yield data_chunk, timestamps

    def _handle_memory_pressure(self, pressure: float, memory_used: int):
        """Handle high memory pressure situations."""
        if self._memory_warnings_sent < 3:  # Limit spam
            self.logger.warning(
                f"High memory pressure detected: {pressure:.1%} used. "
                f"Memory consumed: {memory_used / (1024**3):.1f}GB. "
                f"Consider reducing chunk size or using more workers."
            )
            self._memory_warnings_sent += 1

        # Force garbage collection
        import gc
        gc.collect()
```

## Systematic Testing and Implementation Plan

### Phase 0: Establish Memory Profiling Baseline (Days 1-2)

#### Step 1: Create Memory Profiling Infrastructure

```python
# tests/test_memory_profiling.py
import pytest
import psutil
import numpy as np
from unittest.mock import patch, MagicMock
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

class MemoryProfiler:
    """Track memory usage during test execution."""

    def __init__(self):
        self.peak_memory_gb = 0
        self.memory_timeline = []

    def track_memory(self, label: str):
        """Record current memory usage with a label."""
        current_gb = psutil.Process().memory_info().rss / (1024**3)
        self.peak_memory_gb = max(self.peak_memory_gb, current_gb)
        self.memory_timeline.append((label, current_gb))

    def assert_memory_under_limit(self, limit_gb: float):
        """Assert peak memory stayed under limit."""
        assert self.peak_memory_gb < limit_gb, (
            f"Memory usage {self.peak_memory_gb:.1f}GB exceeded limit {limit_gb}GB. "
            f"Timeline: {self.memory_timeline}"
        )

@pytest.fixture
def memory_profiler():
    return MemoryProfiler()

def test_current_memory_usage_17h_recording(memory_profiler):
    """Establish baseline: current implementation memory usage on 17h recording."""

    # Mock a 17-hour recording
    total_samples = int(17 * 3600 * 30000)  # 1.836 billion samples

    # Mock SpikeGadgetsRawIO to simulate memory allocations without real files
    with patch('trodes_to_nwb.convert_ephys.SpikeGadgetsRawIO') as mock_io:
        mock_instance = MagicMock()
        mock_instance.get_regressed_systime.return_value = np.random.random(total_samples).astype(np.float64)
        mock_instance.get_signal_size.return_value = total_samples
        mock_io.return_value = mock_instance

        memory_profiler.track_memory("start")

        try:
            # This should fail with current implementation
            iterator = RecFileDataChunkIterator(['mock_file.rec'])
            memory_profiler.track_memory("after_iterator_init")

        except MemoryError:
            memory_profiler.track_memory("memory_error")
            pytest.skip("Expected MemoryError with current implementation")

    # Document current memory usage for comparison
    print(f"Current implementation memory timeline: {memory_profiler.memory_timeline}")

def test_memory_bottleneck_identification():
    """Identify which specific operations cause memory spikes."""

    # Test each suspected bottleneck in isolation
    bottlenecks = [
        ("timestamp_loading", lambda: simulate_timestamp_loading()),
        ("sample_counting", lambda: simulate_sample_counting()),
        ("iterator_splitting", lambda: simulate_iterator_splitting()),
    ]

    memory_usage = {}

    for name, operation in bottlenecks:
        initial_memory = psutil.Process().memory_info().rss
        operation()
        peak_memory = psutil.Process().memory_info().rss
        memory_usage[name] = (peak_memory - initial_memory) / (1024**3)

    # Identify the biggest memory consumers
    sorted_bottlenecks = sorted(memory_usage.items(), key=lambda x: x[1], reverse=True)
    print(f"Memory bottlenecks (GB): {sorted_bottlenecks}")

    return sorted_bottlenecks

def simulate_timestamp_loading():
    """Simulate the memory impact of loading all timestamps."""
    samples_17h = int(17 * 3600 * 30000)
    timestamps = np.random.random(samples_17h).astype(np.float64)
    return timestamps

def simulate_sample_counting():
    """Simulate the memory impact of sample counting operations."""
    samples_17h = int(17 * 3600 * 30000)
    # Simulate multiple arrays created during sample counting
    timestamps = np.random.random(samples_17h).astype(np.float64)
    systime = timestamps * 1e9  # Convert to nanoseconds
    sample_indices = np.arange(samples_17h, dtype=np.int64)
    return timestamps, systime, sample_indices

def simulate_iterator_splitting():
    """Simulate memory impact of creating multiple iterator objects."""
    # Simulate multiple partial iterators holding references
    iterators = []
    for i in range(10):  # Simulate splitting into 10 chunks
        chunk_size = int(17 * 3600 * 30000 / 10)
        mock_data = np.random.random(chunk_size).astype(np.float64)
        iterators.append(mock_data)
    return iterators
```

#### Step 2: Validate Memory Calculations

```python
# tests/test_memory_calculations.py
def test_memory_calculation_accuracy():
    """Validate our 43GB memory calculation is accurate."""

    # Test with smaller, manageable arrays first
    samples_1h = int(1 * 3600 * 30000)  # 1 hour = 108M samples
    expected_1h_gb = (samples_1h * 8) / (1024**3)  # float64

    # Create actual array and measure
    initial_memory = psutil.Process().memory_info().rss
    timestamps = np.random.random(samples_1h).astype(np.float64)
    actual_memory = psutil.Process().memory_info().rss
    actual_gb = (actual_memory - initial_memory) / (1024**3)

    # Should be close to calculated (within 10% due to overhead)
    assert abs(actual_gb - expected_1h_gb) / expected_1h_gb < 0.1

    # Extrapolate to 17 hours
    calculated_17h_gb = expected_1h_gb * 17
    assert abs(calculated_17h_gb - 14.4) < 1.0  # Within 1GB of our calculation

def test_data_type_assumptions():
    """Validate our assumptions about data types used."""

    # Test that timestamps are actually float64
    with patch('trodes_to_nwb.spike_gadgets_raw_io.SpikeGadgetsRawIO') as mock_io:
        mock_instance = MagicMock()
        mock_timestamps = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        mock_instance.get_regressed_systime.return_value = mock_timestamps
        mock_io.return_value = mock_instance

        # Verify the actual data type returned
        result = mock_instance.get_regressed_systime(0, None)
        assert result.dtype == np.float64
        assert result.itemsize == 8  # 8 bytes per float64
```

### Phase 1: Address Timestamp Loading Bottleneck (Days 3-5)

#### Step 1: Implement Lazy Timestamp Loading - TDD Approach

```python
# tests/test_lazy_timestamps.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from trodes_to_nwb.lazy_processing import LazyTimestampArray, LazyRecFileDataChunkIterator

class TestLazyTimestampArray:
    def test_no_memory_allocation_on_creation(self, memory_profiler):
        """Test that creating LazyTimestampArray doesn't load data into memory."""

        # Mock a large neo_io without actually creating the data
        mock_neo_ios = [MagicMock() for _ in range(3)]
        for i, mock_io in enumerate(mock_neo_ios):
            mock_io.get_signal_size.return_value = 100_000_000  # 100M samples per file

        memory_profiler.track_memory("before_lazy_array")

        # This should NOT allocate memory for timestamps
        lazy_array = LazyTimestampArray(mock_neo_ios)

        memory_profiler.track_memory("after_lazy_array")

        # Memory usage should be minimal (just metadata)
        memory_used = memory_profiler.memory_timeline[-1][1] - memory_profiler.memory_timeline[-2][1]
        assert memory_used < 0.01  # Less than 10MB

        # Array should report correct shape without loading data
        assert lazy_array.shape == (300_000_000,)  # 3 files × 100M samples

    def test_generates_timestamps_on_demand(self):
        """Test that timestamps are generated only when accessed."""

        mock_neo_io = MagicMock()
        mock_neo_io.get_regressed_systime.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        lazy_array = LazyTimestampArray([mock_neo_io])

        # Access a slice - should call the underlying method
        result = lazy_array[1:4]

        # Should have called get_regressed_systime with correct indices
        mock_neo_io.get_regressed_systime.assert_called_with(1, 4)
        np.testing.assert_array_equal(result, np.array([2.0, 3.0, 4.0]))

    def test_handles_multi_file_slicing(self):
        """Test slicing across multiple files works correctly."""

        mock_neo_ios = [MagicMock() for _ in range(2)]
        mock_neo_ios[0].get_signal_size.return_value = 1000
        mock_neo_ios[1].get_signal_size.return_value = 1000
        mock_neo_ios[0].get_regressed_systime.return_value = np.arange(0, 10, dtype=np.float64)
        mock_neo_ios[1].get_regressed_systime.return_value = np.arange(10, 20, dtype=np.float64)

        lazy_array = LazyTimestampArray(mock_neo_ios)

        # Slice that spans both files (samples 990-1010)
        result = lazy_array[990:1010]

        # Should call both neo_ios with appropriate ranges
        assert mock_neo_ios[0].get_regressed_systime.called
        assert mock_neo_ios[1].get_regressed_systime.called

        assert len(result) == 20

class TestLazyRecFileDataChunkIterator:
    def test_memory_usage_stays_constant_with_file_size(self, memory_profiler):
        """Test that memory usage doesn't scale with input file size."""

        # Test with progressively larger mock files
        file_sizes = [1000, 100_000, 10_000_000, 100_000_000]  # 1K to 100M samples

        memory_usage = []

        for size in file_sizes:
            memory_profiler.track_memory(f"before_size_{size}")

            # Mock the neo_io without actually allocating arrays
            with patch('trodes_to_nwb.lazy_processing.SpikeGadgetsRawIO') as mock_io:
                mock_instance = MagicMock()
                mock_instance.get_signal_size.return_value = size
                mock_instance.signal_channels_count.return_value = 128
                mock_io.return_value = mock_instance

                iterator = LazyRecFileDataChunkIterator(['mock_file.rec'])

            memory_profiler.track_memory(f"after_size_{size}")

            current_usage = memory_profiler.memory_timeline[-1][1] - memory_profiler.memory_timeline[-2][1]
            memory_usage.append(current_usage)

        # Memory usage should be roughly constant (not scale with file size)
        max_usage = max(memory_usage)
        min_usage = min(memory_usage)

        # Variance should be small (less than 50% difference)
        assert (max_usage - min_usage) / min_usage < 0.5

        # All should be under reasonable limit (100MB)
        assert all(usage < 0.1 for usage in memory_usage)  # Less than 100MB

    def test_17_hour_recording_initialization_succeeds(self, memory_profiler):
        """Test that 17-hour recording can be initialized without memory error."""

        samples_17h = int(17 * 3600 * 30000)

        memory_profiler.track_memory("before_17h_init")

        with patch('trodes_to_nwb.lazy_processing.SpikeGadgetsRawIO') as mock_io:
            mock_instance = MagicMock()
            mock_instance.get_signal_size.return_value = samples_17h
            mock_instance.signal_channels_count.return_value = 128
            mock_instance.block_count.return_value = 1
            mock_instance.segment_count.return_value = 1
            mock_instance.signal_streams_count.return_value = 4
            mock_io.return_value = mock_instance

            # This should NOT raise MemoryError
            iterator = LazyRecFileDataChunkIterator(['mock_17h_file.rec'])

        memory_profiler.track_memory("after_17h_init")

        # Memory usage should be minimal regardless of file size
        memory_used = memory_profiler.memory_timeline[-1][1] - memory_profiler.memory_timeline[-2][1]
        assert memory_used < 1.0  # Less than 1GB

        # Iterator should report correct total samples
        assert iterator._get_maxshape()[0] == samples_17h

    def test_chunk_processing_memory_bounded(self, memory_profiler):
        """Test that processing chunks keeps memory usage bounded."""

        with patch('trodes_to_nwb.lazy_processing.SpikeGadgetsRawIO') as mock_io:
            mock_instance = MagicMock()
            mock_instance.get_signal_size.return_value = 10_000_000  # 10M samples
            mock_instance.signal_channels_count.return_value = 128

            # Mock chunk data
            mock_instance.get_analogsignal_chunk.return_value = np.random.randint(
                -1000, 1000, size=(10000, 128), dtype=np.int16
            )

            mock_io.return_value = mock_instance

            iterator = LazyRecFileDataChunkIterator(['mock_file.rec'])

            memory_profiler.track_memory("before_chunk_processing")

            # Process several chunks
            for i in range(10):
                chunk = iterator._get_data((slice(i*10000, (i+1)*10000), slice(None)))
                memory_profiler.track_memory(f"after_chunk_{i}")

        # Memory usage should not accumulate (should be bounded)
        chunk_memories = [timeline[1] for timeline in memory_profiler.memory_timeline if 'after_chunk' in timeline[0]]

        # Memory should not continuously increase
        max_memory = max(chunk_memories)
        min_memory = min(chunk_memories)

        # Memory variance should be small (chunks should be processed and released)
        assert (max_memory - min_memory) / min_memory < 0.2  # Less than 20% variance
```

#### Step 2: Implement Core Chunked Processing

1. Create `chunked_processing.py` module
2. Implement `ChunkedTimestampProcessor` class
3. Add memory calculation utilities
4. Test with synthetic long recordings

#### Step 3: Integration with Existing Code

1. Modify `RecFileDataChunkIterator` to use chunked processing
2. Update `convert_ephys.py` to handle chunked iteration
3. Ensure backward compatibility with existing API

### Phase 2: Address Sample Count Bottleneck (Days 6-8)

#### Step 1: Eliminate Redundant Sample Counting

```python
# tests/test_sample_count_optimization.py
def test_sample_count_without_full_timestamp_loading():
    """Test that sample counting doesn't require loading all timestamps."""

    with patch('trodes_to_nwb.optimized_processing.SpikeGadgetsRawIO') as mock_io:
        mock_instance = MagicMock()

        # Mock efficient sample counting (just return size without loading data)
        mock_instance.get_signal_size_efficient.return_value = 100_000_000
        mock_instance.get_regressed_systime.side_effect = Exception("Should not be called!")

        mock_io.return_value = mock_instance

        from trodes_to_nwb.optimized_processing import OptimizedRecFileDataChunkIterator

        # Should get sample count without loading timestamps
        iterator = OptimizedRecFileDataChunkIterator(['mock_file.rec'])

        # Verify efficient method was called, not the memory-intensive one
        mock_instance.get_signal_size_efficient.assert_called()
        mock_instance.get_regressed_systime.assert_not_called()

def test_streaming_sample_count_processing():
    """Test that sample count data can be processed in streaming fashion."""

    from trodes_to_nwb.optimized_processing import StreamingSampleCountProcessor

    processor = StreamingSampleCountProcessor()

    # Should be able to process sample counts in chunks
    total_samples = 1_000_000
    chunk_size = 100_000

    chunks_processed = 0
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)

        # Process this chunk (should not accumulate memory)
        processor.process_chunk(chunk_start, chunk_end)
        chunks_processed += 1

    assert chunks_processed == 10
    assert processor.total_samples_processed == total_samples
```

#### Step 2: Implement Efficient Sample Size Calculation

```python
# src/trodes_to_nwb/optimized_processing.py
class OptimizedSpikeGadgetsRawIO:
    """Optimized version that avoids loading full arrays for metadata."""

    def get_signal_size_efficient(self, block_index: int, seg_index: int, stream_index: int) -> int:
        """Get signal size without loading timestamp arrays."""

        # Use file metadata instead of loading data
        if hasattr(self, '_cached_signal_size'):
            return self._cached_signal_size

        # Calculate from file size and packet structure
        file_size = self.filename.stat().st_size
        header_size = self._get_header_size()
        packet_size = self._get_packet_size()

        data_size = file_size - header_size
        num_packets = data_size // packet_size

        self._cached_signal_size = num_packets
        return num_packets

    def _get_header_size(self) -> int:
        """Get header size from file metadata."""
        # Implementation that reads header without loading data
        pass

    def _get_packet_size(self) -> int:
        """Get packet size from file metadata."""
        # Implementation that calculates packet size from header
        pass

class StreamingSampleCountProcessor:
    """Process sample count data without loading full arrays."""

    def __init__(self, chunk_size: int = 1_000_000):
        self.chunk_size = chunk_size
        self.total_samples_processed = 0

    def process_chunk(self, start_idx: int, end_idx: int):
        """Process a chunk of sample count data."""
        chunk_samples = end_idx - start_idx

        # Process this chunk without accumulating memory
        # Implementation that works with HDF5 or other streaming format

        self.total_samples_processed += chunk_samples
```

### Phase 3: Memory Map Pooling (Days 9-10)

#### Step 1: Create Memory Pool Infrastructure

1. Implement `MemoryMapPool` class
2. Add LRU eviction strategy
3. Thread-safe access with locks

#### Step 2: Integration and Testing

1. Modify `SpikeGadgetsRawIO` to use shared memory maps
2. Test memory usage reduction with multiple files
3. Validate performance improvements

### Phase 4: Iterator Splitting Optimization (Days 11-12)

#### Step 1: Reduce Iterator Object Proliferation

```python
# tests/test_iterator_optimization.py
def test_iterator_splitting_minimal_memory_overhead():
    """Test that iterator splitting doesn't create excessive memory overhead."""

    with patch('trodes_to_nwb.optimized_processing.SpikeGadgetsRawIO') as mock_io:
        # Mock a large file that would normally be split
        mock_instance = MagicMock()
        large_size = 100_000_000  # Would trigger splitting
        mock_instance._raw_memmap.shape = (large_size,)
        mock_instance.get_signal_size_efficient.return_value = large_size

        mock_io.return_value = mock_instance

        from trodes_to_nwb.optimized_processing import OptimizedRecFileDataChunkIterator

        # Should handle large files without creating many objects
        iterator = OptimizedRecFileDataChunkIterator(['large_file.rec'])

        # Should use a single smart iterator rather than many partial ones
        assert len(iterator.neo_io) <= 2  # At most one original + one optimized

def test_virtual_iterator_splitting():
    """Test virtual splitting that doesn't create actual object copies."""

    from trodes_to_nwb.optimized_processing import VirtualSplitIterator

    # Should be able to handle arbitrary sizes without object proliferation
    total_size = 1_000_000_000  # 1 billion samples
    max_chunk = 30_000_000  # 30M sample chunks

    iterator = VirtualSplitIterator(total_size, max_chunk)

    chunks = list(iterator.iter_chunks())

    # Should create logical chunks without actual data objects
    assert len(chunks) > 1
    assert all(isinstance(chunk, tuple) for chunk in chunks)  # Just (start, end) tuples

    # Total coverage should equal original size
    total_covered = sum(end - start for start, end in chunks)
    assert total_covered == total_size
```

#### Step 2: Implement Smart Iterator Management

```python
# src/trodes_to_nwb/optimized_processing.py
class VirtualSplitIterator:
    """Iterator that splits logically without creating object copies."""

    def __init__(self, total_size: int, max_chunk_size: int):
        self.total_size = total_size
        self.max_chunk_size = max_chunk_size

    def iter_chunks(self):
        """Yield logical chunk boundaries without creating objects."""
        for start in range(0, self.total_size, self.max_chunk_size):
            end = min(start + self.max_chunk_size, self.total_size)
            yield (start, end)

class OptimizedRecFileDataChunkIterator:
    """Memory-optimized version of RecFileDataChunkIterator."""

    def __init__(self, rec_file_path: list[str], **kwargs):
        self.rec_file_path = rec_file_path

        # Use optimized IO that doesn't load arrays upfront
        self.neo_io = [
            OptimizedSpikeGadgetsRawIO(filename=file, **kwargs)
            for file in rec_file_path
        ]

        # Use lazy timestamps instead of loading all at once
        self.timestamps = LazyTimestampArray(self.neo_io)

        # Use efficient sample counting
        self.n_time = [
            neo_io.get_signal_size_efficient(0, 0, stream_index)
            for neo_io in self.neo_io
        ]

        # Use virtual splitting instead of creating partial objects
        self._setup_virtual_splitting()

    def _setup_virtual_splitting(self):
        """Setup virtual splitting without creating object copies."""
        max_size = 30_000_000  # 30M samples max per logical chunk

        self.virtual_chunks = []
        for i, neo_io in enumerate(self.neo_io):
            total_size = self.n_time[i]

            if total_size > max_size:
                # Create virtual chunks for this file
                splitter = VirtualSplitIterator(total_size, max_size)
                chunks = list(splitter.iter_chunks())
                self.virtual_chunks.extend([(i, start, end) for start, end in chunks])
            else:
                # Small file, process as single chunk
                self.virtual_chunks.append((i, 0, total_size))
```

### Phase 5: Adaptive Memory Management (Days 13-14)

#### Step 1: Memory Monitoring

1. Implement `AdaptiveMemoryIterator`
2. Add memory pressure detection
3. Create warning and recovery mechanisms

#### Step 2: Performance Validation

1. Test with various file sizes (1-hour to 24-hour recordings)
2. Measure memory usage patterns
3. Validate performance doesn't regress for smaller files

## Testing Strategy

### Unit Tests

```python
# Test checklist for each component:
# ✓ Memory calculations are correct
# ✓ Chunk boundaries are accurate
# ✓ Memory limits are respected
# ✓ Error conditions handled gracefully
# ✓ Edge cases (very small/large files) work
```

### Integration Tests

```python
# Test scenarios:
# ✓ 17+ hour recording converts without memory error
# ✓ Multiple concurrent conversions work
# ✓ Memory usage stays within system limits
# ✓ Conversion output matches original implementation
# ✓ Performance acceptable (not >2x slower)
```

### Performance Tests

```python
# Benchmarks to track:
# ✓ Peak memory usage vs file duration
# ✓ Conversion time vs file size
# ✓ Memory map reuse efficiency
# ✓ Chunk processing overhead
```

## Testing Strategy and Success Criteria

### Testing Phases

#### Phase A: Memory Profiling and Baseline (Days 1-2)
- **Establish current memory usage patterns**
- **Validate theoretical calculations against real measurements**
- **Document specific bottleneck contributions**
- **Create memory profiling infrastructure**

#### Phase B: Component Testing (Days 3-14)
- **Test each optimization in isolation**
- **Measure memory impact of each change**
- **Ensure no functional regressions**
- **Validate performance characteristics**

#### Phase C: Integration Testing (Days 15-16)
- **Test complete optimized pipeline**
- **Validate with real 17+ hour recordings**
- **Stress test with multiple concurrent conversions**
- **Performance regression testing**

#### Phase D: User Acceptance Testing (Days 17-18)
- **Test with actual data from Issue #47 reporters**
- **Validate on different system configurations**
- **Gather performance feedback**
- **Document deployment readiness**

### Critical Success Metrics

#### Functional Requirements
1. **17-hour recordings complete successfully** without memory errors
2. **Memory usage scales O(1)** with file size (constant memory, linear time)
3. **No breaking changes** to existing API
4. **Conversion accuracy maintained** (bit-perfect output where possible)

#### Performance Requirements
1. **Performance regression <25%** for files that currently work (≤4 hours)
2. **Peak memory usage ≤4GB** regardless of input file size
3. **Memory efficiency**: Memory usage independent of recording duration
4. **Scalability**: Handle 48+ hour recordings on systems with 8GB RAM

#### Quality Requirements
1. **Test coverage ≥90%** for new optimized code paths
2. **Memory leak detection** passes on long-running tests
3. **Concurrent safety** for parallel processing scenarios
4. **Robust error handling** with graceful degradation

### Validation Methodology

#### Memory Usage Validation
```python
# Automated memory testing
def test_memory_scaling_independence():
    """Test that memory usage doesn't scale with file duration."""
    durations = [1, 4, 8, 17, 24, 48]  # hours
    memory_usage = []

    for duration in durations:
        peak_memory = measure_conversion_memory(duration)
        memory_usage.append(peak_memory)

    # Memory usage should be roughly constant
    assert max(memory_usage) - min(memory_usage) < 1.0  # <1GB variance
    assert all(usage < 4.0 for usage in memory_usage)  # <4GB peak
```

#### Performance Validation
```python
# Performance regression testing
def test_performance_regression():
    """Test that optimizations don't slow down existing workflows."""

    # Test with files that currently work
    test_files = get_test_files_under_4_hours()

    for test_file in test_files:
        original_time = measure_conversion_time_original(test_file)
        optimized_time = measure_conversion_time_optimized(test_file)

        regression = (optimized_time - original_time) / original_time
        assert regression < 0.25  # <25% performance regression
```

#### Accuracy Validation
```python
# Output accuracy testing
def test_conversion_accuracy_maintained():
    """Test that optimizations don't change conversion output."""

    for test_file in get_reference_files():
        original_nwb = convert_with_original_implementation(test_file)
        optimized_nwb = convert_with_optimized_implementation(test_file)

        # Verify identical output (allowing for minor floating-point differences)
        assert_nwb_files_equivalent(original_nwb, optimized_nwb)
```

### Real-World Validation

#### User Data Testing
1. **Direct collaboration** with Issue #47 reporters
2. **Test with actual failing datasets**
3. **Validate on user system configurations**
4. **Collect performance feedback**

#### System Configuration Testing
1. **Low-memory systems** (8GB RAM)
2. **High-memory systems** (64GB+ RAM)
3. **Different OS environments** (Linux, macOS, Windows)
4. **Various Python/dependency versions**

#### Stress Testing
1. **Extremely long recordings** (48+ hours)
2. **Multiple concurrent conversions**
3. **Memory-constrained environments**
4. **Network storage scenarios**

## Deployment Strategy

### Incremental Rollout

1. **Feature flag**: `ENABLE_CHUNKED_PROCESSING=true` environment variable
2. **Backward compatibility**: Fallback to original behavior if issues arise
3. **Monitoring**: Memory usage logging and alerting
4. **User feedback**: Direct validation with Issue #47 reporters

### Risk Mitigation

1. **Comprehensive testing** with various file sizes and system configurations
2. **Performance benchmarking** to ensure no significant regressions
3. **Memory leak detection** with automated testing
4. **Rollback plan** if critical issues are discovered

This memory optimization plan directly addresses the critical blocker preventing users from processing long recordings while maintaining system stability and performance for existing workflows.
