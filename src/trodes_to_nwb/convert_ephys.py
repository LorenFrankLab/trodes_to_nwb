"""Handles the conversion of raw electrophysiology (ephys) data from Trodes .rec files
into an NWB ElectricalSeries object. Includes a DataChunkIterator for efficient reading.
"""

import logging
from warnings import warn

import numpy as np
from hdmf.backends.hdf5 import H5DataIO
from hdmf.data_utils import GenericDataChunkIterator
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries

from trodes_to_nwb import convert_rec_header

from .spike_gadgets_raw_io import SpikeGadgetsRawIO, SpikeGadgetsRawIOPartial

MICROVOLTS_PER_VOLT = 1e6
VOLTS_PER_MICROVOLT = 1e-6
MILLISECONDS_PER_SECOND = 1e3
NANOSECONDS_PER_SECOND = 1e9

DEFAULT_SAMPLING_RATE = 30000  # 30 kHz
SECONDS_PER_MINUTE = 60
MAX_ITERATOR_MINUTES = 30

# maximum size of the iterator in samples
# Just a guess, but should be large enough to not cause issues
MAXIMUM_ITERATOR_SIZE = int(
    DEFAULT_SAMPLING_RATE * SECONDS_PER_MINUTE * MAX_ITERATOR_MINUTES
)  # 30 min of data at 30 kHz
DEFAULT_CHUNK_TIME_DIM = 16384
DEFAULT_CHUNK_MAX_CHANNEL_DIM = 32


def _is_strictly_increasing(values, chunk_size: int = 1_000_000) -> bool:
    """Return whether ``values`` is strictly increasing, streaming in chunks.

    Equivalent to ``np.all(np.diff(values) > 0)`` but never allocates the full
    diff array -- for the ~15 GB timestamp array at 17 h that diff would double
    peak memory (#47). Reads at most ``chunk_size`` elements at a time and checks
    the within-chunk diffs plus each chunk boundary, returning early on the first
    non-increase.
    """
    n = len(values)
    if n < 2:
        return True
    previous = values[0]
    for start in range(1, n, chunk_size):
        block = np.asarray(values[start : start + chunk_size])
        if block[0] <= previous or not np.all(np.diff(block) > 0):
            return False
        previous = block[-1]
    return True


class _LazyTimestamps:
    """Virtual, read-only 1-D ``float64`` timestamps spanning all rec files.

    Replaces the eager
    ``np.concatenate([io.get_regressed_systime(0, None) for io in neo_io])`` that
    held the whole timestamps array (~14.7 GB at 17 h @30 kHz) resident for the
    entire conversion (#47). Each requested slice / index is computed on demand
    from the per-file counter->system-clock regression (or the Trodes-timestamp
    fallback for non-sysClock files), so the full array is never materialised.

    Values are **byte-identical** to the old concatenation: every timestamp is
    ``intercept + slope * unwrapped_counter`` (sysClock path) or
    ``(counter - counter0) / rate + t_creation`` (fallback). The computation is
    **chunk-independent** -- any sub-range read equals the whole-file read sliced,
    including across uint32 wraps (``get_regressed_systime`` restores the wrap
    offset per slice via ``prior_wraps``). Both readers reuse the cached
    regression *parameters* (``regressed_systime_parameters``), so a per-file
    sub-range read is cheap. (Note: those readers are also ``lru_cache(maxsize=1)``
    on the full returned array, so a one-off ``(0, None)`` call would retain a
    whole-file array -- the lazy path here never makes that call.)

    Supported access -- one per consumer of ``RecFileDataChunkIterator.timestamps``:

    - ``ts[a:b]``        contiguous slice  -> chunked writes; the monotonicity scan
    - ``ts[int_array]``  integer fancy index -> decimated headstage sensor timestamps
    - ``ts[i]``          scalar
    - ``np.asarray(ts)`` full materialise -> generic NumPy fallback only; after
      step 3 no hot path uses it (the non-PTP position path now indexes lazily)

    For chunk-by-chunk HDF5 writes use :meth:`as_data_chunk_iterator` (a
    ``GenericDataChunkIterator``). pynwb materialises a plain array-like in one
    shot via ``data[:]`` (calling ``__array__``); it writes only an
    ``AbstractDataChunkIterator`` iteratively -- hence that wrapper.
    """

    def __init__(self, neo_io: list, use_sysclock: bool):
        self._neo_io = list(neo_io)
        self._use_sysclock = use_sysclock
        # Per-file length is the file's packet count -- exactly the length of the
        # old per-file ``get_regressed_systime(0, None)``. For interpolating
        # iterators this must be read *after* interpolation is resolved (the
        # caller triggers it; _get_signal_size raises otherwise).
        lengths = [io._raw_memmap.shape[0] for io in self._neo_io]
        self._starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._total = int(self._starts[-1])

    # --- numpy-array-like surface -------------------------------------------
    def __len__(self) -> int:
        return self._total

    @property
    def shape(self) -> tuple[int]:
        return (self._total,)

    @property
    def size(self) -> int:
        return self._total

    @property
    def ndim(self) -> int:
        return 1

    @property
    def dtype(self) -> np.dtype:
        return np.dtype("float64")

    # --- on-demand computation ----------------------------------------------
    def _read_file(self, file_index: int, lo: int, hi: int) -> np.ndarray:
        """Timestamps for file ``file_index`` over its local range ``[lo, hi)``."""
        io = self._neo_io[file_index]
        if self._use_sysclock:
            values = io.get_regressed_systime(lo, hi)
        else:
            values = io.get_systime_from_trodes_timestamps(lo, hi)
        return np.asarray(values, dtype=np.float64)

    def _slice(self, start: int, stop: int) -> np.ndarray:
        """Contiguous values for the global half-open range ``[start, stop)``."""
        start = max(0, min(start, self._total))
        stop = max(start, min(stop, self._total))
        if stop == start:
            return np.empty(0, dtype=np.float64)
        first = int(np.searchsorted(self._starts, start, side="right") - 1)
        parts = []
        for file_index in range(first, len(self._neo_io)):
            file_start = int(self._starts[file_index])
            if file_start >= stop:
                break
            file_stop = int(self._starts[file_index + 1])
            lo = max(start, file_start) - file_start
            hi = min(stop, file_stop) - file_start
            parts.append(self._read_file(file_index, lo, hi))
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def _gather(self, indices: np.ndarray) -> np.ndarray:
        """Fancy index: load only the spanned contiguous range, then index into it.

        Headstage update indices are sparse but confined to a single file, so the
        spanned range is one file's worth of timestamps (a few MB), never the
        whole recording.
        """
        indices = np.asarray(indices)
        if indices.dtype == bool:
            indices = np.flatnonzero(indices)
        indices = indices.astype(np.int64, copy=False)
        if indices.size == 0:
            return np.empty(0, dtype=np.float64)
        if (indices < 0).any():
            indices = np.where(indices < 0, indices + self._total, indices)
        lo = int(indices.min())
        hi = int(indices.max())
        if lo < 0 or hi >= self._total:
            raise IndexError(
                f"fancy index out of bounds [{lo}, {hi}] for length {self._total}"
            )
        block = self._slice(lo, hi + 1)
        return block[indices - lo]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._total)
            if step == 1:
                return self._slice(start, stop)
            # Non-unit step (never used on the big arrays in practice): load the
            # spanned range once, then stride, so we still never materialise more
            # than the covered span.
            return self._gather(np.arange(start, stop, step))
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self._total
            if not 0 <= i < self._total:
                raise IndexError(f"index {key} out of range for length {self._total}")
            return self._slice(i, i + 1)[0]
        # integer (or boolean) array fancy indexing
        return self._gather(key)

    def __iter__(self):
        # Without this, Python's sequence-protocol fallback (via __getitem__)
        # would iterate element-by-element, each a separate on-disk read -- a
        # silent O(n) trap. Force callers to materialise explicitly instead.
        raise TypeError(
            "refusing to iterate _LazyTimestamps lazily (each step is a separate "
            "disk read); use np.asarray(ts) or ts.as_data_chunk_iterator()"
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        # A virtual array can only ever produce a freshly-built array, so an
        # explicit copy=False ("give me a no-copy view") is unsatisfiable; match
        # NumPy 2's contract and refuse loudly rather than silently copying.
        if copy is False:
            raise ValueError("cannot return a no-copy view of a lazy virtual array")
        out = self._slice(0, self._total)
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def as_data_chunk_iterator(self, **kwargs) -> "_LazyTimestampsChunkIterator":
        """A ``GenericDataChunkIterator`` for streaming these timestamps to HDF5."""
        return _LazyTimestampsChunkIterator(self, **kwargs)


class _LazyTimestampsChunkIterator(GenericDataChunkIterator):
    """Stream a :class:`_LazyTimestamps` into HDF5 one chunk at a time.

    pynwb writes a plain array-like via ``data[:]`` -- one shot, re-materialising
    the whole ~14.7 GB-at-17h timestamps array. Only an
    ``AbstractDataChunkIterator`` is written iteratively, so the big-timestamp
    datasets (e-series / ECU analog / sample_count) are wrapped here to be
    written without ever holding the full array resident (#47).
    """

    def __init__(self, lazy: "_LazyTimestamps", **kwargs):
        self._lazy = lazy
        super().__init__(**kwargs)

    def _get_data(self, selection: tuple) -> np.ndarray:
        return self._lazy[selection[0]]

    def _get_maxshape(self) -> tuple[int]:
        return (len(self._lazy),)

    def _get_dtype(self) -> np.dtype:
        return np.dtype("float64")


def _timestamps_for_write(timestamps):
    """Return a chunked iterator for lazy timestamps, else the value unchanged.

    A :class:`_LazyTimestamps` must be written through a ``DataChunkIterator`` so
    pynwb streams the dataset instead of calling ``__array__`` and re-materialising
    the whole array (#47). A plain array (e.g. the small decimated sensor
    timestamps, or timestamps supplied by the caller) is written as-is.
    """
    if isinstance(timestamps, _LazyTimestamps):
        return timestamps.as_data_chunk_iterator()
    return timestamps


class RecFileDataChunkIterator(GenericDataChunkIterator):
    """Data chunk iterator for SpikeGadgets rec files."""

    def __init__(
        self,
        rec_file_path: list[str],
        nwb_hw_channel_order=None,
        conversion: float = 1.0,
        stream_index: int = None,  # TODO use the stream name instead of the index
        stream_id: str = None,
        is_analog: bool = False,
        interpolate_dropped_packets: bool = False,
        timestamps=None,  # Use this if you already have timestamps from intializing another rec iterator on the same files
        behavior_only: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        rec_file_path : list[str]
            list of paths to rec files
        nwb_hw_channel_order : list, optional
            order of hw channels in the nwb file, by default []
        conversion : float, optional
            conversion factor from raw data to volts, by default 1.0
        stream_index : int, optional
            index of stream to use. If both this and stream_id provided values must match in rec header, by default None
        stream_id : str, optional
            id name of stream to use. If both this and stream_index provided values must match in rec header, by default None
        is_analog : bool, optional
            whether this is an analog stream, by default False
        interpolate_dropped_packets : bool, optional
            whether to interpolate single dropped packets, by default False
        timestamps : [type], optional
            timestamps to use. Can provide efficiency improvements by skipping recalculating timestamps from rec files, by default None
        behavior_only : bool, optional
            indicate if file contains only behavior data (no e-phys), by default False
        kwargs : dict
            additional arguments to pass to GenericDataChunkIterator
        """
        if not rec_file_path:
            raise FileNotFoundError("Must provide at least one rec file path")
        logger = logging.getLogger("convert")
        self.conversion = conversion
        self.is_analog = is_analog
        self.neo_io = [
            SpikeGadgetsRawIO(
                filename=file, interpolate_dropped_packets=interpolate_dropped_packets
            )
            for file in rec_file_path
        ]  # get all streams for all files
        logger.info("Parsing headers")
        [neo_io.parse_header() for neo_io in self.neo_io]
        # TODO see what else spikeinterface does and whether it is necessary
        logger.info("Parsing header COMPLETE")
        # for now, make sure that there is only one block, one segment, and four streams:
        # Controller_DIO_digital
        # ECU_digital
        # ECU_analog
        # trodes
        assert all(neo_io.block_count() == 1 for neo_io in self.neo_io)
        assert all(neo_io.segment_count(0) == 1 for neo_io in self.neo_io)
        assert all(
            neo_io.signal_streams_count() == 4 - behavior_only for neo_io in self.neo_io
        ), (
            "Unexpected number of signal streams. "
            + "Confirm whether behavior_only is set correctly for this recording"
        )

        self.block_index = 0
        self.seg_index = 0

        # resolve stream index and id based on rec header and provided info
        if stream_id is not None:  # if stream id is provided
            if (
                stream_index is None
            ):  # if stream index is not provided, get from the SpikegadgetsRawIO object
                stream_index = self.neo_io[0].get_stream_index_from_id(stream_id)
            # if both provided, check that they agree
            elif self.neo_io[0].get_stream_id_from_index(stream_index) != stream_id:
                raise ValueError(
                    f"Provided stream index {stream_index} does not match provided stream id {stream_id}"
                )
        else:  # if stream id is not provided
            stream_id = self.neo_io[0].get_stream_id_from_index(stream_index)

        if behavior_only and stream_id == "trodes":
            raise ValueError(
                "Behavior only recordings do not contain a `trodes` stream"
            )
        self.stream_id = stream_id
        self.stream_index = stream_index

        # check that all files have the same number of channels.
        if (
            len(
                {
                    neo_io.signal_channels_count(stream_index=self.stream_index)
                    for neo_io in self.neo_io
                }
            )
            > 1
        ):
            raise ValueError("All files must have the same number of signal channels.")
        self.n_channel = self.neo_io[0].signal_channels_count(
            stream_index=self.stream_index
        )
        self.n_multiplexed_channel = 0
        if self.is_analog:
            self.n_multiplexed_channel += len(self.neo_io[0].multiplexed_channel_xml)

        # order that the hw channels are in within the nwb table
        if nwb_hw_channel_order is None:
            nwb_hw_channel_order = []
        if len(nwb_hw_channel_order) == 0:  # TODO: raise error instead?
            self.nwb_hw_channel_order = np.arange(self.n_channel)
        else:
            self.nwb_hw_channel_order = nwb_hw_channel_order

        if (
            self.stream_id == "trodes"
            and len(self.nwb_hw_channel_order) < self.n_channel
        ):
            self.n_channel = len(self.nwb_hw_channel_order)
        """split excessively large iterators into smaller ones
        """
        iterator_size = [neo_io._raw_memmap.shape[0] for neo_io in self.neo_io]
        iterator_size.reverse()
        for i, size in enumerate(
            iterator_size
        ):  # iterate backwards so can insert new iterators
            if size > MAXIMUM_ITERATOR_SIZE:
                # split into smaller iterators
                sub_iterators = []
                j = 0
                previous_multiplex_state = None
                iterator_loc = len(iterator_size) - i - 1
                # calculate systime regression on full epoch, parameters stored and inherited by partial iterators
                if self.neo_io[iterator_loc].sysClock_byte:
                    self.neo_io[iterator_loc].get_regressed_systime(0, None)
                while j < size:
                    sub_iterators.append(
                        SpikeGadgetsRawIOPartial(
                            self.neo_io[iterator_loc],
                            start_index=j,
                            stop_index=j + MAXIMUM_ITERATOR_SIZE,
                            previous_multiplex_state=previous_multiplex_state,
                        )
                    )
                    if self.n_multiplexed_channel > 0:
                        partial_size = sub_iterators[-1]._raw_memmap.shape[0]
                        previous_multiplex_state = sub_iterators[
                            -1
                        ].get_analogsignal_multiplexed_partial(
                            i_start=partial_size - 10,
                            i_stop=partial_size,
                            padding=30000,
                        )[
                            -1
                        ]
                    j += MAXIMUM_ITERATOR_SIZE
                self.neo_io.pop(iterator_loc)
                self.neo_io[iterator_loc:iterator_loc] = sub_iterators
        logger.info(f"# iterators: {len(self.neo_io)}")
        # Resolve dropped-packet interpolation up front. Before #47 this happened
        # as a side effect of materialising the timestamps below; now that the
        # timestamps are lazy we trigger it explicitly so each file's interpolated
        # length is final before _LazyTimestamps and self.n_time read it
        # (_get_signal_size raises if interpolation is enabled but unresolved).
        # This first call scans only the uint32 counter, never the float64
        # timestamps. Split (partial) iterators already resolve it in __init__.
        for neo_io in self.neo_io:
            if getattr(neo_io, "interpolate_dropped_packets", False) and (
                getattr(neo_io, "interpolate_index", None) is None
            ):
                neo_io.get_analogsignal_timestamps(0, 1)

        # Timestamps are built lazily (#47): a virtual array that computes each
        # requested slice / index on demand from the per-file clock regression,
        # instead of concatenating every file's full timestamps (~14.7 GB at 17 h)
        # up front. Values are byte-identical to the old concatenation.
        if timestamps is not None:
            self.timestamps = timestamps
        elif self.neo_io[0].sysClock_byte:  # use this if have sysClock
            self.timestamps = _LazyTimestamps(self.neo_io, use_sysclock=True)
        else:  # use this to convert Trodes timestamps into systime based on sampling rate
            self.timestamps = _LazyTimestamps(self.neo_io, use_sysclock=False)

        logger.info("Reading timestamps COMPLETE")
        # Must be strictly increasing. `np.all(np.diff(...))` only checks that no
        # two timestamps are *equal* (nonzero diff); it silently accepts a
        # backward jump (negative diff), which is exactly what a clock reset or
        # an out-of-order file concatenation would produce.
        is_timestamps_sequential = _is_strictly_increasing(self.timestamps)
        if not is_timestamps_sequential:
            warn(
                "Timestamps are not strictly increasing. This may cause problems with some software or data analysis.",
                stacklevel=2,
            )

        self.n_time = [
            neo_io.get_signal_size(
                block_index=self.block_index,
                seg_index=self.seg_index,
                stream_index=self.stream_index,
            )
            for neo_io in self.neo_io
        ]

        # The lazy timestamps and the data must describe the same number of
        # samples. Both ultimately read each file's post-interpolation packet
        # count (_LazyTimestamps via _raw_memmap.shape[0], n_time via
        # get_signal_size), so this holds by construction -- assert it so a
        # future change that desynchronises them fails loud here rather than
        # silently writing mismatched-length timestamps.
        if isinstance(self.timestamps, _LazyTimestamps):
            assert self.timestamps._total == sum(self.n_time), (
                f"lazy timestamps length {self.timestamps._total} != data length "
                f"{sum(self.n_time)}"
            )

        super().__init__(**kwargs)

    def _get_data(self, selection: tuple[slice]) -> np.ndarray:
        """Get data chunk from the electrophysiology files.

        Parameters
        ----------
        selection : tuple[slice]
            Tuple of slices for (time, channel) selection.

        Returns
        -------
        np.ndarray, shape (n_time_selected, n_channels_selected)
            Array containing the selected electrophysiology data.
        """
        # selection is (time, channel)
        assert selection[0].step is None

        # slice to indices
        # DCI will want channels 0 to X first to put into the array in that order
        # those are stored in the file as channel IDs
        # make into list form passed to neo_io
        selection_list = list(selection)
        if self.is_analog:
            selection_list[1] = slice(
                selection[1].start,
                min(selection[1].stop, self.n_channel),
                selection[1].step,
            )
        channel_ids = [str(x) for x in self.nwb_hw_channel_order[selection_list[1]]]
        # what global index each file starts at
        file_start_ind = np.append(np.zeros(1), np.cumsum(self.n_time))
        # the time indexes we want
        time_index = np.arange(selection_list[0].start, selection_list[0].stop)[
            :: selection_list[0].step
        ]
        data = []
        i = time_index[0]
        while i < min(time_index[-1], self._get_maxshape()[0]):
            # find the stream where this piece of slice begins
            io_stream = np.argmin(i >= file_start_ind) - 1
            # get the data from that stream
            data.append(
                self.neo_io[io_stream].get_analogsignal_chunk(
                    block_index=self.block_index,
                    seg_index=self.seg_index,
                    i_start=int(i - file_start_ind[io_stream]),
                    i_stop=int(
                        min(
                            time_index[-1] - file_start_ind[io_stream],
                            self.n_time[io_stream],
                        )
                    )
                    + 1,
                    stream_index=self.stream_index,
                    channel_ids=channel_ids,
                )
            )
            i += min(
                self.n_time[io_stream]
                - (i - file_start_ind[io_stream]),  # if added up to the end of stream
                time_index[-1] - i,  # if finished in this stream
            )

        data = (np.concatenate(data) * self.conversion).astype("int16")
        # Handle the appended multiplex data
        if (
            self.neo_io[0].header["signal_streams"][self.stream_index]["id"]
            == "ECU_analog"
        ) and self.is_analog:
            multiplex_keys = self.neo_io[0].multiplexed_channel_xml.keys()
            n_multiplex = len(multiplex_keys)
            n_analog = (
                self.n_channel
            )  # number of non-multiplexed channels in the dataset
            n_analog_selected = data.shape[1] - n_multiplex
            return_indices = np.arange(
                n_analog_selected
            )  # include all non-multiplexed channels pulled
            # determine which multiplex channels are being requested
            if (
                selection[1].stop > n_analog
            ):  # if multiplexed channels are being requested
                requested_multiplex = np.arange(n_multiplex) + n_analog_selected
                multiplex_slice = slice(
                    max(selection[1].start - n_analog, 0),
                    max(selection[1].stop - n_analog, 0),
                    selection[1].step,
                )
                requested_multiplex = requested_multiplex[multiplex_slice]
                return_indices = np.append(return_indices, requested_multiplex)
            data = data[:, return_indices]

        return data

    def _get_maxshape(self) -> tuple[int, int]:
        """Get the maximum shape of the data array.

        Returns
        -------
        tuple[int, int]
            Maximum shape as (n_time_total, n_channels_total).
        """
        return (
            np.sum(self.n_time),
            self.n_channel + self.n_multiplexed_channel,
        )  # TODO: Is this right for maxshape @rly

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")


def add_raw_ephys(
    nwbfile: NWBFile,
    recfile: list[str],
    electrode_row_indices: list[int],
    metadata: dict = None,
) -> None:
    """Adds the raw ephys data to a NWB file. Must be called after add_electrode_groups

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : list[str]
        ordered list of file paths to all recfiles with session's data
    electrode_row_indices : list
        which electrodes to add to table
    metadata : dict, optional
        metadata dictionary, useed only for conversion if not in rec, by default None
    """

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=electrode_row_indices,
        description="electrodes used in raw e-series recording",
    )
    # get hw channel order
    nwb_hw_chan_order = [
        int(x) for x in list(nwbfile.electrodes.to_dataframe()["hwChan"])
    ]
    # get conversion factor from rec file
    rec_header = convert_rec_header.read_header(recfile[0])
    spike_config = rec_header.find("SpikeConfiguration")
    if "rawScalingToUv" in spike_config[0].attrib:
        conversion = float(spike_config[0].attrib["rawScalingToUv"])
    else:
        conversion = (
            metadata["raw_data_to_volts"] * MICROVOLTS_PER_VOLT
        )  # Use metadata-provided conversion if not available in rec file

    # make the data iterator
    rec_dci = RecFileDataChunkIterator(
        recfile,
        nwb_hw_channel_order=nwb_hw_chan_order,
        conversion=conversion,
        interpolate_dropped_packets=True,
        stream_id="trodes",
    )  # can set buffer_gb if needed

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(
        rec_dci,
        chunks=(
            DEFAULT_CHUNK_TIME_DIM,
            min(rec_dci.n_channel, DEFAULT_CHUNK_MAX_CHANNEL_DIM),
        ),
    )

    # do we want to pull the timestamps from the rec file? or is there another source?
    eseries = ElectricalSeries(
        name="e-series",
        data=data_data_io,
        # Stream the timestamps dataset chunk-by-chunk (#47); a bare array here
        # would re-materialise the whole ~14.7 GB-at-17h timestamps on write.
        timestamps=_timestamps_for_write(rec_dci.timestamps),
        electrodes=electrode_table_region,  # TODO
        conversion=VOLTS_PER_MICROVOLT,
        offset=0.0,  # TODO
    )

    nwbfile.add_acquisition(eseries)
