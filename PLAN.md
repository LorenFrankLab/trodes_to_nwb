# Implementation Plan: Issue-Driven High-Impact Improvements

Based on code review and analysis of GitHub issues, this plan prioritizes critical production problems first, then performance optimizations that users desperately need. The issues reveal real-world blockers that must be addressed immediately.

## CRITICAL FIX (Week 1) - Hard User Blocker

### 1.1 Memory Optimization for 17-Hour Recordings (Issue #47)
**Impact**: CRITICAL | **Effort**: Medium | **Risk**: Medium

Users are completely blocked on long recordings with no viable workaround. This is the highest priority fix:

```python
# src/trodes_to_nwb/chunked_timestamps.py
class ChunkedTimestampIterator:
    """Avoid loading all timestamps into memory for very long recordings."""

    def __init__(self, neo_io_list: list, chunk_size_hours: float = 1.0):
        self.neo_io_list = neo_io_list
        self.chunk_size = int(chunk_size_hours * 3600 * 30000)  # samples per chunk

    def iter_timestamp_chunks(self):
        """Yield timestamp chunks without loading entire file."""
        for neo_io in self.neo_io_list:
            total_samples = neo_io.get_signal_size(0, 0)

            for start_idx in range(0, total_samples, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_samples)
                timestamps = neo_io.get_analogsignal_timestamps(start_idx, end_idx)
                yield timestamps, start_idx, end_idx

# Replace memory-intensive operations in RecFileDataChunkIterator
def __iter__(self):
    """Modified iterator that processes chunks instead of loading all timestamps."""
    for chunk_timestamps, start_idx, end_idx in self.chunked_iterator.iter_timestamp_chunks():
        data_chunk = self.neo_io[0].get_analogsignal_chunk(
            block_index=self.block_index,
            seg_index=self.seg_index,
            i_start=start_idx,
            i_stop=end_idx,
            stream_index=self.stream_index,
            channel_indexes=self.nwb_hw_channel_order
        )
        yield data_chunk
```

## Phase 1: High-Priority Daily Workflow Fixes (Weeks 2-3)

### 1.1 Improve Error Messages for Config Mismatch (Issue #107)
**Impact**: HIGH | **Effort**: Low | **Risk**: Very Low

Replace generic error with detailed debugging information:

```python
# CURRENT (convert_rec_header.py:127)
if len(hw_channels_in_yaml) != len(hw_channels_in_header):
    raise ValueError("Channel count mismatch")

# IMPROVED
if len(hw_channels_in_yaml) != len(hw_channels_in_header):
    raise ValueError(
        f"Channel count mismatch between metadata and rec header:\n"
        f"  Metadata YAML: {len(hw_channels_in_yaml)} channels\n"
        f"  Rec header: {len(hw_channels_in_header)} channels\n"
        f"  Missing in YAML: {set(hw_channels_in_header) - set(hw_channels_in_yaml)}\n"
        f"  Extra in YAML: {set(hw_channels_in_yaml) - set(hw_channels_in_header)}\n"
        f"  Check your metadata file and rec configuration for consistency."
    )
```

### 1.2 Fix Headstage Sensor Data Units (Issue #19)
**Impact**: HIGH | **Effort**: Medium | **Risk**: Low

Implement proper scaling and units for scientific data integrity:

```python
# src/trodes_to_nwb/sensor_scaling.py
class HeadstageCalibration:
    """Calibration constants for SpikeGadgets headstage sensors."""

    # Accelerometer: ±2G full range, 16-bit resolution
    ACCEL_SCALE_FACTOR = (2 * 2) / (2**16)  # 0.000061 g per step
    ACCEL_UNITS = "g"

    # Gyroscope: ±2000 deg/sec full range, 16-bit resolution
    GYRO_SCALE_FACTOR = (2 * 2000) / (2**16)  # 0.061 deg/sec per step
    GYRO_UNITS = "degrees/s"

def convert_sensor_data(raw_data: np.ndarray, sensor_type: str) -> tuple[np.ndarray, str]:
    """Convert raw sensor data to proper physical units."""
    calibration = HeadstageCalibration()

    if sensor_type == "accelerometer":
        scaled_data = raw_data * calibration.ACCEL_SCALE_FACTOR
        return scaled_data, calibration.ACCEL_UNITS
    elif sensor_type == "gyroscope":
        scaled_data = raw_data * calibration.GYRO_SCALE_FACTOR
        return scaled_data, calibration.GYRO_UNITS
    else:
        return raw_data, "N/A"
```

### 1.3 Memory Optimization for Long Recordings (Issue #47)
**Impact**: HIGH | **Effort**: Medium | **Risk**: Medium

Address the 17-hour recording memory failures with chunked timestamp processing:

```python
# src/trodes_to_nwb/chunked_timestamps.py
class ChunkedTimestampIterator:
    """Avoid loading all timestamps into memory for very long recordings."""

    def __init__(self, neo_io_list: list, chunk_size_hours: float = 1.0):
        self.neo_io_list = neo_io_list
        self.chunk_size = int(chunk_size_hours * 3600 * 30000)  # samples per chunk

    def iter_timestamp_chunks(self):
        """Yield timestamp chunks without loading entire file."""
        for neo_io in self.neo_io_list:
            total_samples = neo_io.get_signal_size(0, 0)

            for start_idx in range(0, total_samples, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_samples)
                timestamps = neo_io.get_analogsignal_timestamps(start_idx, end_idx)
                yield timestamps, start_idx, end_idx
```

### 1.4 Extract Constants and Magic Numbers
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Create a dedicated constants module to centralize configuration values:

```python
# src/trodes_to_nwb/constants.py
from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingConstants:
    # Data processing
    MICROVOLTS_PER_VOLT: float = 1e6
    VOLTS_PER_MICROVOLT: float = 1e-6
    NANOSECONDS_PER_SECOND: float = 1e9

    # Sampling and chunking
    DEFAULT_SAMPLING_RATE: int = 30000
    DEFAULT_CHUNK_TIME_DIM: int = 16384
    DEFAULT_CHUNK_MAX_CHANNEL_DIM: int = 32
    MAX_ITERATOR_MINUTES: int = 30

    # Hardware specifics
    INT_16_CONVERSION: int = 256  # 2^8 for 16-bit to 8-bit conversion
    BITS_PER_BYTE: int = 8
    TIMESTAMP_SIZE_BYTES: int = 4  # uint32

    # Parallel processing
    DEFAULT_THREADS_PER_WORKER: int = 20

CONSTANTS = ProcessingConstants()
```

**Files to update**:
- `convert_ephys.py` - Replace scattered constants
- `spike_gadgets_raw_io.py` - Consolidate hardware constants
- `convert.py` - Use constants for parallel processing

### 1.2 Create Configuration Classes
**Impact**: High | **Effort**: Medium | **Risk**: Low

Replace long parameter lists with structured configuration objects:

```python
# src/trodes_to_nwb/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class ConversionConfig:
    """Configuration for NWB conversion process."""
    path: Path
    output_dir: str = "/stelmo/nwb/raw"

    # Optional paths
    header_reconfig_path: Optional[Path] = None
    device_metadata_paths: Optional[list[Path]] = None
    video_directory: str = ""
    fs_gui_dir: str = ""

    # Processing options
    convert_video: bool = False
    disable_ptp: bool = False
    behavior_only: bool = False

    # Parallel processing
    n_workers: int = 1

    # Filtering
    query_expression: Optional[str] = None

@dataclass
class FileInfo:
    """Structured representation of parsed file information."""
    date: str
    animal_name: str
    epoch: str
    tag: str
    tag_index: str
    extension: str
    full_path: str
```

### 1.3 Standardize Error Handling
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Create consistent error handling patterns:

```python
# src/trodes_to_nwb/exceptions.py
class TrodesToNwbError(Exception):
    """Base exception for trodes_to_nwb package."""
    pass

class MetadataValidationError(TrodesToNwbError):
    """Raised when metadata validation fails."""
    pass

class FileProcessingError(TrodesToNwbError):
    """Raised when file processing fails."""
    pass

class ConversionError(TrodesToNwbError):
    """Raised when conversion process fails."""
    pass
```

## Phase 2: Code Structure Refactoring (Weeks 3-4)

### 2.1 Extract `_create_nwb()` Function
**Impact**: Very High | **Effort**: Medium | **Risk**: Low

Break down the 132-line `_create_nwb()` function into focused, testable components:

```python
# src/trodes_to_nwb/nwb_builder.py
class NwbSessionBuilder:
    """Builder pattern for constructing NWB files from session data."""

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.logger = None
        self.nwb_file = None

    def setup_logging(self, session: tuple) -> None:
        """Initialize session-specific logging."""

    def load_and_validate_files(self, session_df: pd.DataFrame) -> dict:
        """Load rec files and metadata, returning validated file paths."""

    def create_hardware_maps(self, metadata: dict, rec_header) -> tuple:
        """Generate hardware channel and reference electrode maps."""

    def build_nwb_metadata(self, metadata: dict, device_metadata: list) -> None:
        """Populate NWB file with metadata entries."""

    def add_data_streams(self, session_df: pd.DataFrame, maps: tuple) -> None:
        """Add all data streams (ephys, analog, DIOs, position)."""

    def write_and_validate(self, output_path: Path) -> Path:
        """Write NWB file and run validation."""
```

### 2.2 Simplify File Path Processing
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Replace the complex tuple return pattern in `_process_path()`:

```python
def _process_path(path: Path) -> Optional[FileInfo]:
    """Process a file path into structured information."""
    logger = logging.getLogger("convert")

    try:
        if path.suffix == ".yml":
            return _process_metadata_file(path)
        else:
            return _process_data_file(path)
    except ValueError:
        logger.info(f"Invalid file name: {path.stem}. Skipping...")
        return None

def _process_metadata_file(path: Path) -> FileInfo:
    """Process metadata file path."""
    parts = path.stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid metadata file format: {path.stem}")

    date, animal_name, _ = parts
    return FileInfo(
        date=str(int(date)),  # Validate it's numeric
        animal_name=animal_name,
        epoch="1",
        tag="NA",
        tag_index="1",
        extension=path.suffix,
        full_path=str(path.absolute())
    )
```

### 2.3 Improve Type Annotations
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Strengthen type hints throughout the codebase:

```python
# Before
def _get_file_paths(df: pd.DataFrame, file_extension: str) -> list[str]:

# After
def _get_file_paths(df: pd.DataFrame, file_extension: str) -> list[Path]:
    """Get file paths for a given extension."""
    paths = df.loc[df.file_extension == file_extension].full_path.to_list()
    return [Path(path) for path in paths]
```

## Phase 2: File Size Optimization (Week 4) - Major User Impact

### 2.1 Implement HDF5 Compression for Large TimeSeries (Issue #21)
**Impact**: VERY HIGH | **Effort**: Medium | **Risk**: Low

Based on user data showing 87GB → 50GB reduction (42% smaller files):

```python
# src/trodes_to_nwb/hdf5_optimization.py
from hdmf.backends.hdf5 import H5DataIO

class OptimizedDataIO:
    """Optimized HDF5 data I/O with appropriate compression for different data types."""

    @staticmethod
    def create_compressed_dataset(data: np.ndarray, data_type: str) -> H5DataIO:
        """Create compressed HDF5 dataset based on data characteristics."""

        if data_type == "timestamps":
            # 70% compression on timestamps
            return H5DataIO(
                data=data,
                compression="gzip",
                compression_opts=6,
                shuffle=True,
                fletcher32=True,
                chunks=True
            )
        elif data_type == "analog":
            # 99% compression potential on slowly-changing analog data
            return H5DataIO(
                data=data,
                compression="lzf",  # Faster compression for large datasets
                shuffle=True,
                chunks=True
            )
        elif data_type == "ephys":
            # 30% compression on neural data
            return H5DataIO(
                data=data,
                compression="gzip",
                compression_opts=3,  # Lower compression for faster read
                shuffle=True,
                chunks=(min(8192, data.shape[0]), min(64, data.shape[1]))
            )
        else:
            return H5DataIO(data=data, chunks=True)

def add_optimized_timeseries(nwb_file, data, timestamps, name, data_type):
    """Add TimeSeries with optimal compression."""
    compressed_data = OptimizedDataIO.create_compressed_dataset(data, data_type)
    compressed_timestamps = OptimizedDataIO.create_compressed_dataset(timestamps, "timestamps")

    return TimeSeries(
        name=name,
        data=compressed_data,
        timestamps=compressed_timestamps,
        unit="V"
    )
```

### 2.2 Implement Timestamp Linking (Issue #21)
**Impact**: HIGH | **Effort**: Low | **Risk**: Low

Save 4.4GB by linking identical timestamps between streams:

```python
# src/trodes_to_nwb/timestamp_linking.py
def create_shared_timestamps(nwb_file, master_timestamps, stream_name):
    """Create shared timestamp reference to save space."""

    # Create master timestamp dataset
    master_ts_name = f"{stream_name}_master_timestamps"
    if master_ts_name not in nwb_file.acquisition:
        master_data = OptimizedDataIO.create_compressed_dataset(master_timestamps, "timestamps")
        nwb_file.add_acquisition(
            TimeSeries(
                name=master_ts_name,
                data=[0],  # Dummy data
                timestamps=master_data,
                unit="N/A"
            )
        )

    # Return reference for other streams to use
    return nwb_file.acquisition[master_ts_name].timestamps
```

## Phase 3: Performance and Memory Optimizations (Weeks 5-6)

### 3.1 Memory Map Pool and Sharing
**Impact**: Very High | **Effort**: High | **Risk**: Medium

Address the critical memory fragmentation issues identified in large file processing:

```python
# src/trodes_to_nwb/memory_manager.py
from typing import Dict, Optional
import weakref
import numpy as np
from pathlib import Path

class MemoryMapPool:
    """Shared memory map pool to reduce fragmentation and improve efficiency."""

    def __init__(self, max_maps: int = 10):
        self._pool: Dict[str, np.memmap] = {}
        self._refs: Dict[str, int] = {}
        self._max_maps = max_maps

    def get_memmap(self, filepath: Path, offset: int = 0, dtype: str = "<u1") -> np.memmap:
        """Get or create a memory map for a file."""
        key = f"{filepath}_{offset}_{dtype}"

        if key in self._pool:
            self._refs[key] += 1
            return self._pool[key]

        # Clean up if at capacity
        if len(self._pool) >= self._max_maps:
            self._evict_lru()

        mmap = np.memmap(filepath, mode="r", offset=offset, dtype=dtype)
        self._pool[key] = mmap
        self._refs[key] = 1
        return mmap

    def release_memmap(self, filepath: Path, offset: int = 0, dtype: str = "<u1") -> None:
        """Release reference to a memory map."""
        key = f"{filepath}_{offset}_{dtype}"
        if key in self._refs:
            self._refs[key] -= 1
            if self._refs[key] <= 0:
                del self._pool[key]
                del self._refs[key]

# Global memory map pool
_memory_pool = MemoryMapPool()
```

### 3.2 Adaptive Chunking Strategy
**Impact**: High | **Effort**: Medium | **Risk**: Low

Replace fixed chunk sizes with adaptive chunking based on available memory:

```python
# src/trodes_to_nwb/adaptive_chunking.py
import psutil
import numpy as np

class AdaptiveChunker:
    """Dynamically adjusts chunk sizes based on available memory."""

    def __init__(self, target_memory_usage: float = 0.6):
        self.target_memory_usage = target_memory_usage
        self.min_chunk_size = 1024  # Minimum chunk size
        self.max_chunk_size = 1024 * 1024  # Maximum chunk size

    def calculate_optimal_chunk_size(
        self,
        data_shape: tuple,
        dtype: np.dtype,
        n_parallel_streams: int = 1
    ) -> int:
        """Calculate optimal chunk size based on available memory."""
        available_memory = psutil.virtual_memory().available
        target_memory = available_memory * self.target_memory_usage

        # Account for multiple parallel streams
        per_stream_memory = target_memory / n_parallel_streams

        # Calculate bytes per sample
        bytes_per_sample = np.prod(data_shape[1:]) * dtype.itemsize

        # Calculate optimal chunk size
        optimal_chunks = int(per_stream_memory / bytes_per_sample)

        # Clamp to min/max bounds
        return np.clip(optimal_chunks, self.min_chunk_size, self.max_chunk_size)
```

### 3.3 Vectorized File Operations
**Impact**: Medium | **Effort**: Medium | **Risk**: Low

Optimize the DataFrame operations in file scanning:

```python
# src/trodes_to_nwb/fast_scanner.py
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

def fast_file_scan(path: Path) -> pd.DataFrame:
    """Optimized file scanning with vectorized operations."""

    # Single glob call for all extensions
    all_pattern = f"**/*.{{{','.join(VALID_FILE_EXTENSIONS)}}}"
    all_files = list(path.glob(all_pattern))

    # Parallel processing of file paths
    with ThreadPoolExecutor(max_workers=4) as executor:
        file_info_list = list(executor.map(_process_path, all_files))

    # Filter out None values (invalid files)
    valid_info = [info for info in file_info_list if info is not None]

    if not valid_info:
        return pd.DataFrame(columns=COLUMN_NAMES)

    # Create DataFrame directly from list of FileInfo objects
    return pd.DataFrame([info._asdict() for info in valid_info])
```

### 3.4 Stream Fusion for Multi-Stream Processing
**Impact**: High | **Effort**: High | **Risk**: Medium

Process multiple data streams in a single pass to reduce I/O:

```python
# src/trodes_to_nwb/stream_fusion.py
class FusedStreamProcessor:
    """Process multiple data streams in a single file pass."""

    def __init__(self, rec_file_paths: list[Path]):
        self.rec_file_paths = rec_file_paths
        self._stream_configs = {}

    def register_stream(self, stream_id: str, config: StreamConfig):
        """Register a stream for fused processing."""
        self._stream_configs[stream_id] = config

    def process_all_streams(self) -> Dict[str, np.ndarray]:
        """Process all registered streams in a single file pass."""
        results = {stream_id: [] for stream_id in self._stream_configs}

        for chunk in self._iterate_chunks():
            for stream_id, config in self._stream_configs.items():
                stream_data = self._extract_stream_data(chunk, config)
                results[stream_id].append(stream_data)

        # Concatenate results
        return {
            stream_id: np.concatenate(chunks)
            for stream_id, chunks in results.items()
        }
```

## Phase 4: Advanced Performance Features (Weeks 7-8)

### 4.1 Memory Monitoring and Profiling
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Add memory usage monitoring for large file processing:

```python
# src/trodes_to_nwb/performance_monitor.py
import psutil
import warnings
from contextlib import contextmanager
from typing import Generator

class MemoryMonitor:
    """Monitor memory usage and provide warnings for large conversions."""

    def __init__(self, warning_threshold: float = 0.8):
        self.warning_threshold = warning_threshold
        self.peak_memory = 0
        self.initial_memory = 0

    @contextmanager
    def monitor_session(self, session_id: str) -> Generator[None, None, None]:
        """Context manager for monitoring session memory usage."""
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss

        try:
            yield
        finally:
            current_memory = process.memory_info().rss
            memory_delta = current_memory - self.initial_memory

            if memory_delta / psutil.virtual_memory().total > self.warning_threshold:
                warnings.warn(
                    f"Session {session_id} used {memory_delta / 1024**3:.1f} GB memory. "
                    "Consider using smaller chunk sizes or more workers.",
                    PerformanceWarning
                )
```

### 4.2 Parallel I/O Optimization
**Impact**: High | **Effort**: High | **Risk**: Medium

Optimize parallel processing for I/O-bound operations:

```python
# src/trodes_to_nwb/parallel_optimizer.py
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Tuple, Any

class SmartParallelExecutor:
    """Intelligent parallel execution with I/O vs CPU task optimization."""

    def __init__(self):
        self.io_executor = ThreadPoolExecutor(max_workers=8)  # I/O bound
        self.cpu_executor = ProcessPoolExecutor()  # CPU bound

    async def process_sessions_optimally(
        self,
        sessions: List[Tuple[str, Any]],
        file_sizes: List[int]
    ):
        """Distribute work based on file sizes and system characteristics."""

        # Sort sessions by file size for better load balancing
        sorted_sessions = sorted(
            zip(sessions, file_sizes),
            key=lambda x: x[1],
            reverse=True
        )

        # Process large files first with fewer workers to avoid memory pressure
        large_files = [s for s, size in sorted_sessions if size > 1024**3]  # > 1GB
        small_files = [s for s, size in sorted_sessions if size <= 1024**3]

        # Sequential processing for very large files
        for session in large_files[:2]:  # Max 2 large files concurrently
            await self._process_session_async(session)

        # Parallel processing for smaller files
        tasks = [self._process_session_async(session) for session in small_files]
        await asyncio.gather(*tasks)
```

### 4.3 Prefetching and Read-Ahead
**Impact**: Medium | **Effort**: Medium | **Risk**: Low

Implement read-ahead for chunked data processing:

```python
# src/trodes_to_nwb/prefetch_iterator.py
from queue import Queue
from threading import Thread
import numpy as np

class PrefetchIterator:
    """Iterator with read-ahead capability for better I/O performance."""

    def __init__(self, base_iterator, buffer_size: int = 3):
        self.base_iterator = base_iterator
        self.buffer_size = buffer_size
        self._queue = Queue(maxsize=buffer_size)
        self._thread = None
        self._stop_flag = False

    def _prefetch_worker(self):
        """Worker thread for prefetching data."""
        try:
            for chunk in self.base_iterator:
                if self._stop_flag:
                    break
                self._queue.put(chunk)
            self._queue.put(StopIteration)  # Signal end
        except Exception as e:
            self._queue.put(e)

    def __iter__(self):
        self._thread = Thread(target=self._prefetch_worker)
        self._thread.start()
        return self

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, StopIteration):
            raise item
        elif isinstance(item, Exception):
            raise item
        return item
```

### 4.4 Performance Benchmarking Suite
**Impact**: Low | **Effort**: Low | **Risk**: Very Low

Create comprehensive performance benchmarking:

```python
# benchmarks/performance_suite.py
import time
import psutil
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BenchmarkResult:
    """Performance benchmark results."""
    conversion_time: float
    peak_memory_mb: float
    avg_memory_mb: float
    file_size_gb: float
    throughput_mb_per_sec: float
    cpu_utilization_percent: float

class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark_conversion(
        self,
        test_files: List[Path],
        n_workers_list: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, List[BenchmarkResult]]:
        """Benchmark conversion with different worker counts."""

        results = {}
        for n_workers in n_workers_list:
            print(f"Benchmarking with {n_workers} workers...")

            for test_file in test_files:
                result = self._benchmark_single_conversion(test_file, n_workers)
                if str(n_workers) not in results:
                    results[str(n_workers)] = []
                results[str(n_workers)].append(result)

        return results

    def generate_performance_report(self, results: Dict) -> str:
        """Generate detailed performance report."""
        report = ["# Performance Benchmark Report\n"]

        for n_workers, benchmark_results in results.items():
            avg_time = sum(r.conversion_time for r in benchmark_results) / len(benchmark_results)
            avg_throughput = sum(r.throughput_mb_per_sec for r in benchmark_results) / len(benchmark_results)

            report.append(f"## {n_workers} Workers")
            report.append(f"- Average conversion time: {avg_time:.2f}s")
            report.append(f"- Average throughput: {avg_throughput:.2f} MB/s")
            report.append("")

        return "\n".join(report)
```

## Phase 5: Testing and Validation (Week 9)

### 5.1 Performance Regression Testing
**Impact**: High | **Effort**: Medium | **Risk**: Low

Ensure optimizations don't introduce regressions:

```python
# tests/test_performance_regression.py
import pytest
from trodes_to_nwb.convert import create_nwbs
from benchmarks.performance_suite import PerformanceBenchmark

class TestPerformanceRegression:
    """Performance regression test suite."""

    @pytest.mark.performance
    def test_conversion_speed_regression(self, test_data_small):
        """Ensure conversion speed doesn't regress."""
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_single_conversion(test_data_small, n_workers=1)

        # Allow 10% performance variance
        assert result.throughput_mb_per_sec > BASELINE_THROUGHPUT * 0.9

    @pytest.mark.performance
    def test_memory_usage_regression(self, test_data_large):
        """Ensure memory usage doesn't regress."""
        benchmark = PerformanceBenchmark()
        result = benchmark.benchmark_single_conversion(test_data_large, n_workers=1)

        # Ensure memory usage is reasonable for large files
        assert result.peak_memory_mb < test_data_large.file_size_gb * 1000 * 2  # Max 2x file size
```

### 5.2 Memory Leak Detection
**Impact**: Medium | **Effort**: Low | **Risk**: Very Low

Add automated memory leak detection:

```python
# tests/test_memory_leaks.py
import gc
import psutil
import pytest

def test_no_memory_leaks_batch_conversion():
    """Ensure batch conversions don't leak memory."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Run multiple conversions
    for i in range(10):
        create_nwbs(small_test_file, n_workers=1)
        gc.collect()  # Force garbage collection

    final_memory = process.memory_info().rss
    memory_growth = final_memory - initial_memory

    # Allow some memory growth but not excessive
    assert memory_growth < 100 * 1024 * 1024  # < 100MB growth
```

## REVISED Implementation Strategy - Issue-Driven Priorities

### GitHub Issues Have Changed Everything:

The GitHub issues reveal **critical production blockers** that must be fixed immediately. Users cannot use parallel processing at all (#141), and the package is incompatible with modern PyNWB versions (#124). Real users are hitting memory limits on 17-hour recordings (#47), and there's massive file size reduction potential (#21).

### Revised User-Impact Driven Timeline:

**WEEK 1 - CRITICAL BLOCKER (HARD STOP FOR USERS)**:
1. Memory optimization for 17-hour recordings (#47) - **USERS COMPLETELY BLOCKED**

**WEEK 2-3 - HIGH-PRIORITY DAILY PAIN POINTS**:
1. Improve error messages (#107) - **DAILY DEVELOPER FRUSTRATION**
2. Fix headstage sensor data scaling (#19) - **SCIENTIFIC DATA INTEGRITY**

**WEEK 4-5 - MEDIUM-PRIORITY FIXES (INCONVENIENT BUT WORKABLE)**:
1. Fix parallel processing serialization error (#141) - **HAS WORKAROUND** (`n_workers=1`)
2. PyNWB 3.1.0 compatibility (#124) - **CAN PIN VERSIONS** temporarily

**WEEK 6 - HIGH-VALUE OPTIMIZATION**:
1. HDF5 compression implementation (#21) - **42% FILE SIZE REDUCTION** (87GB → 50GB)
2. Timestamp linking for space saving - **4.4GB SAVINGS PER FILE**

**WEEK 7-8 - PERFORMANCE OPTIMIZATIONS**:
1. Memory map pooling and adaptive chunking
2. Vectorized file operations and DataFrame optimization
3. Stream fusion and prefetching

**WEEK 9 - ENHANCEMENTS AND VALIDATION**:
1. Pydantic validation system (#119)
2. NWB representation improvements (#116, #117)
3. End-to-end testing and performance monitoring

### Critical Performance Metrics to Track:

1. **Memory Efficiency**:
   - Peak memory usage per GB of input data
   - Memory growth over multiple conversions
   - Virtual memory fragmentation metrics

2. **Throughput Performance**:
   - MB/s conversion rate for different file sizes
   - Parallel scaling efficiency (speedup vs workers)
   - I/O wait time vs computation time ratios

3. **Resource Utilization**:
   - CPU utilization during conversion
   - Disk I/O patterns and efficiency
   - Memory map reuse statistics

### Risk Mitigation for Performance Changes:

1. **Baseline Establishment**: Record current performance metrics before changes
2. **Incremental Validation**: Test performance impact at each phase
3. **Feature Flags**: Allow fallback to original algorithms if issues arise
4. **Automated Benchmarks**: Run performance tests in CI/CD pipeline

### Expected Outcomes - Issue-Based Improvements:

**IMMEDIATE USER IMPACT (Weeks 1-4)**:
- ✅ **Parallel processing restored** - Users can again process multiple files simultaneously
- ✅ **Modern PyNWB compatibility** - Package works with latest dependencies
- ✅ **17-hour recordings work** - No more memory failures on long recordings
- ✅ **42% smaller files** - 87GB files become 50GB through compression
- ✅ **Better debugging** - Clear error messages instead of generic failures

**PERFORMANCE IMPROVEMENTS (Weeks 5-6)**:
- 30-50% reduction in peak memory usage for large files
- 20-40% improvement in file scanning and DataFrame operations
- Better parallel scaling with optimized worker management
- Reduced memory fragmentation through pooled memory maps

**LONG-TERM BENEFITS (Weeks 7-9)**:
- Modern validation system with Pydantic
- Better scientific data representation in NWB files
- Comprehensive performance monitoring and regression testing
- Documented performance characteristics for different file sizes

### Risk Mitigation - Production-First Approach:

1. **Critical Fixes First**: Address production blockers before any refactoring
2. **Incremental Validation**: Test each fix with real user data and scenarios
3. **Backward Compatibility**: Ensure changes don't break existing workflows
4. **Performance Baselines**: Measure before optimizing to track improvements
5. **User Feedback Loop**: Validate fixes against actual GitHub issue reporters

### Success Criteria - Prioritized by User Impact:

**CRITICAL SUCCESS (Week 1)**:
- **Issue #47**: 17+ hour recordings convert without memory errors - **UNBLOCKS USERS**

**HIGH-PRIORITY SUCCESS (Weeks 2-3)**:
- **Issue #107**: Error messages provide actionable debugging information - **REDUCES DAILY FRUSTRATION**
- **Issue #19**: Sensor data has correct physical units and scaling - **ENSURES SCIENTIFIC INTEGRITY**

**MEDIUM-PRIORITY SUCCESS (Weeks 4-5)**:
- **Issue #141**: Parallel processing works without serialization errors - **REMOVES WORKAROUND NEED**
- **Issue #124**: Package compatible with PyNWB 3.1.0+ - **ENABLES DEPENDENCY UPDATES**

**HIGH-VALUE SUCCESS (Week 6)**:
- **Issue #21**: File sizes reduced by 30-50% with acceptable read performance - **MASSIVE STORAGE SAVINGS**

This issue-driven plan prioritizes real user pain points over theoretical improvements, ensuring immediate value delivery while building toward comprehensive performance optimization.
