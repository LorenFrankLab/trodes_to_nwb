"""
Lazy timestamp array implementation for memory-efficient timestamp access.

This module provides virtual array-like access to timestamps without loading
the entire timestamp array into memory, solving Issue #47 where 17-hour
recordings require 617GB of memory.
"""

import numpy as np
from typing import Optional, Union, List, Iterator
import logging
from hdmf.data_utils import AbstractDataChunkIterator, DataChunk

logger = logging.getLogger(__name__)

# Constants for regression parameter computation
REGRESSION_SAMPLE_SIZE = 10_000  # Target number of samples for regression
MAX_REGRESSION_POINTS = 1_000    # Maximum points to use for regression


class LazyTimestampArray(AbstractDataChunkIterator):
    """
    Virtual array for lazy timestamp loading.

    Provides array-like interface while computing timestamps on-demand in chunks
    to avoid memory explosion. Based on profiling analysis showing that
    `get_regressed_systime()` causes 617GB memory usage for 17-hour recordings.

    Key optimizations:
    - Chunked computation: Process timestamps in configurable chunks (default 1M samples)
    - Lazy regression: Cache regression parameters after first computation
    - Virtual indexing: Support numpy-style indexing without loading full array
    - Memory-mapped base: Use underlying memory-mapped file data efficiently

    Performance constraints (from profiling analysis):
    - Accept +25% computation time for 90%+ memory reduction
    - Maintain <2min total processing time for normal operations
    - Avoid operations that significantly impact numpy.diff performance (80% of time)
    """

    def __init__(self, neo_io_list: List, chunk_size: int = 1_000_000):
        """
        Initialize lazy timestamp array.

        Parameters
        ----------
        neo_io_list : List[SpikeGadgetsRawIO]
            List of SpikeGadgetsRawIO objects to get timestamps from
        chunk_size : int, optional
            Size of chunks for timestamp computation (default: 1M samples)
            Balance between memory usage and computation overhead
        """
        self.neo_io_list = neo_io_list
        self.chunk_size = chunk_size

        # Cache for regression parameters to avoid recomputation
        self._regression_cache = {}

        # Compute total length and file boundaries
        self._compute_boundaries()

        # Iterator state
        self._current_position = 0

        logger.info(f"LazyTimestampArray initialized: {self.shape[0]:,} samples, "
                   f"chunk_size={chunk_size:,}")

    def _compute_boundaries(self):
        """Compute file boundaries and total shape efficiently."""
        self._file_boundaries = []
        total_samples = 0

        for i, neo_io in enumerate(self.neo_io_list):
            # Set up interpolation if needed before getting signal size
            if neo_io.interpolate_dropped_packets and neo_io.interpolate_index is None:
                # Need to trigger interpolation setup by accessing timestamps once
                # This is unavoidable but we'll do it efficiently
                logger.debug(f"Setting up interpolation for file {i}")
                _ = neo_io.get_analogsignal_timestamps(0, 100)  # Small sample to trigger setup

            # Get signal size - this should work now
            n_samples = neo_io.get_signal_size(
                block_index=0, seg_index=0, stream_index=0
            )
            self._file_boundaries.append((total_samples, total_samples + n_samples, i))
            total_samples += n_samples

        self.shape = (total_samples,)

    def _get_file_and_local_index(self, global_index: int) -> tuple:
        """
        Convert global index to file index and local index within that file.

        Parameters
        ----------
        global_index : int
            Global index across all files

        Returns
        -------
        tuple
            (file_index, local_start, local_stop) where file contains the index
        """
        for start, stop, file_idx in self._file_boundaries:
            if start <= global_index < stop:
                return file_idx, global_index - start, global_index - start + 1

        raise IndexError(f"Index {global_index} out of bounds for array of size {self.shape[0]}")

    def _compute_timestamp_chunk(self, neo_io, i_start: int, i_stop: int) -> np.ndarray:
        """
        Compute timestamps for a chunk using cached regression parameters.

        This is the optimized version that handles both sysClock regression
        and Trodes timestamp conversion, avoiding loading entire files.

        Parameters
        ----------
        neo_io : SpikeGadgetsRawIO
            The neo IO object to get timestamps from
        i_start : int
            Start index for the chunk
        i_stop : int
            Stop index for the chunk

        Returns
        -------
        np.ndarray
            Computed timestamps for the chunk
        """
        file_id = id(neo_io)

        # Check which timestamp method to use
        if neo_io.sysClock_byte:
            # Use regressed systime method
            return self._compute_regressed_systime_chunk(neo_io, i_start, i_stop)
        else:
            # Use Trodes timestamp method
            return self._compute_trodes_systime_chunk(neo_io, i_start, i_stop)

    def _compute_regressed_systime_chunk(self, neo_io, i_start: int, i_stop: int) -> np.ndarray:
        """Compute regressed systime timestamps for a chunk."""
        NANOSECONDS_PER_SECOND = 1e9
        file_id = id(neo_io)

        if file_id not in self._regression_cache:
            # First time for this file - compute regression parameters
            # Use a smaller sample (every 1000th point) to avoid memory explosion
            logger.debug(f"Computing regression parameters for file {file_id}")

            # Sample strategy: Take every nth sample to avoid loading entire file
            sample_stride = max(1, neo_io.get_signal_size(0, 0, 0) // REGRESSION_SAMPLE_SIZE)
            sample_indices = np.arange(0, neo_io.get_signal_size(0, 0, 0), sample_stride)

            # Get sampled timestamps and sysclock - this loads much less data
            sampled_trodes = []
            sampled_sys = []

            for idx in sample_indices[:MAX_REGRESSION_POINTS]:
                trodes_chunk = neo_io.get_analogsignal_timestamps(idx, idx + 1)
                sys_chunk = neo_io.get_sys_clock(idx, idx + 1)
                sampled_trodes.extend(trodes_chunk.astype(np.float64))
                sampled_sys.extend(sys_chunk)

            # Perform regression on sampled data
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(sampled_trodes, sampled_sys)

            self._regression_cache[file_id] = {
                "slope": slope,
                "intercept": intercept
            }

            logger.debug(f"Regression parameters cached: slope={slope:.6f}, intercept={intercept:.6f}")

        # Use cached parameters for timestamp computation
        params = self._regression_cache[file_id]
        slope, intercept = params["slope"], params["intercept"]

        # Get Trodes timestamps for this specific chunk only
        trodestime = neo_io.get_analogsignal_timestamps(i_start, i_stop)
        trodestime_index = np.asarray(trodestime, dtype=np.float64)

        # Apply cached regression
        adjusted_timestamps = intercept + slope * trodestime_index
        return adjusted_timestamps / NANOSECONDS_PER_SECOND

    def _compute_trodes_systime_chunk(self, neo_io, i_start: int, i_stop: int) -> np.ndarray:
        """Compute Trodes-based systime timestamps for a chunk."""
        # This method should mirror get_systime_from_trodes_timestamps but for chunks
        # For now, delegate to the original method for the chunk
        return neo_io.get_systime_from_trodes_timestamps(i_start, i_stop)

    def __getitem__(self, key) -> Union[float, np.ndarray]:
        """
        Get timestamp(s) by index with lazy computation.

        Supports:
        - Single index: timestamps[i]
        - Slice: timestamps[start:stop:step]
        - Array indexing: timestamps[array]

        Parameters
        ----------
        key : int, slice, or array-like
            Index specification

        Returns
        -------
        float or np.ndarray
            Timestamp value(s) at the specified index/indices
        """
        if isinstance(key, int):
            # Single index access
            if key < 0:
                key = self.shape[0] + key

            file_idx, local_start, local_stop = self._get_file_and_local_index(key)
            neo_io = self.neo_io_list[file_idx]

            chunk = self._compute_timestamp_chunk(neo_io, local_start, local_stop)
            return chunk[0]

        elif isinstance(key, slice):
            # Slice access
            start, stop, step = key.indices(self.shape[0])

            if step != 1:
                # For non-unit steps, fall back to array indexing
                indices = np.arange(start, stop, step)
                return self[indices]

            # Efficient slice implementation
            result = []
            current_pos = start

            while current_pos < stop:
                # Find which file contains current_pos
                file_idx, local_start, _ = self._get_file_and_local_index(current_pos)
                neo_io = self.neo_io_list[file_idx]

                # Compute how much we can get from this file
                file_start, file_stop, _ = self._file_boundaries[file_idx]
                local_stop = min(local_start + (stop - current_pos), file_stop - file_start)

                # Get chunk from this file
                chunk = self._compute_timestamp_chunk(neo_io, local_start, local_stop)
                result.append(chunk)

                current_pos += len(chunk)

            return np.concatenate(result) if result else np.array([])

        elif hasattr(key, '__iter__'):
            # Array indexing
            indices = np.asarray(key)
            result = np.empty(indices.shape, dtype=self.dtype)

            for i, idx in enumerate(indices.flat):
                result.flat[i] = self[int(idx)]

            return result

        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __len__(self) -> int:
        """Return total number of timestamps."""
        return self.shape[0]

    def __array__(self) -> np.ndarray:
        """
        Convert to numpy array - WARNING: This loads all timestamps!

        This method exists for compatibility but defeats the purpose of lazy loading.
        Only use for small arrays or when absolutely necessary.
        """
        logger.warning("Converting LazyTimestampArray to numpy array - this loads all timestamps!")
        return self[:]

    @property
    def nbytes(self) -> int:
        """Estimated memory usage if fully loaded."""
        return self.shape[0] * np.dtype(self.dtype).itemsize

    def compute_chunk(self, start: int, size: int) -> np.ndarray:
        """
        Compute a specific chunk of timestamps.

        Parameters
        ----------
        start : int
            Start index
        size : int
            Chunk size

        Returns
        -------
        np.ndarray
            Computed timestamp chunk
        """
        stop = min(start + size, self.shape[0])
        return self[start:stop]

    def get_memory_info(self) -> dict:
        """Get memory usage information with cache efficiency metrics."""
        estimated_full_size = self.nbytes / (1024**3)  # GB

        # Calculate cache efficiency
        cache_efficiency = len(self._regression_cache) / len(self.neo_io_list) if self.neo_io_list else 0

        return {
            "shape": self.shape,
            "dtype": self.dtype,
            "estimated_full_size_gb": estimated_full_size,
            "chunk_size": self.chunk_size,
            "num_files": len(self.neo_io_list),
            "regression_cache_size": len(self._regression_cache),
            "cache_efficiency": cache_efficiency,
        }

    # AbstractDataChunkIterator required methods
    @property
    def dtype(self) -> np.dtype:
        """Return data type."""
        return np.dtype(np.float64)

    @property
    def maxshape(self) -> tuple:
        """Return maximum shape."""
        return self.shape

    def __iter__(self) -> Iterator[np.ndarray]:
        """Return iterator over data chunks."""
        self._current_position = 0
        return self

    def __next__(self) -> DataChunk:
        """Return next data chunk."""
        if self._current_position >= self.shape[0]:
            raise StopIteration

        # Get next chunk
        start = self._current_position
        stop = min(start + self.chunk_size, self.shape[0])
        chunk_data = self[start:stop]

        # Create DataChunk with proper selection info
        selection = slice(start, stop)
        data_chunk = DataChunk(data=chunk_data, selection=selection)

        self._current_position = stop
        return data_chunk

    def recommended_chunk_shape(self) -> tuple:
        """Return recommended chunk shape for HDF5 storage."""
        return (min(self.chunk_size, self.shape[0]),)

    def recommended_data_shape(self) -> tuple:
        """Return recommended data shape for HDF5 storage."""
        return self.shape

    def reset(self) -> None:
        """Reset iterator to beginning for reuse."""
        self._current_position = 0