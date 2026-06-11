"""Module for creating and adding epoch intervals (start/stop times) and
sample count information (mapping Trodes timestamps to system time) to the NWB file.
"""

import logging

import numpy as np
import pandas as pd
from hdmf.data_utils import GenericDataChunkIterator
from pynwb import NWBFile, TimeSeries

from trodes_to_nwb.convert_ephys import (
    RecFileDataChunkIterator,
    _timestamps_for_write,
)
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO

MILLISECONDS_PER_SECOND = 1e3
NANOSECONDS_PER_SECOND = 1e9


class _TrodesSampleCountIterator(GenericDataChunkIterator):
    """Stream the Trodes sample counts as one virtual 1-D ``uint32`` array.

    The sample-count data spans every rec file in the session. Concatenating all
    of them up front materialises the whole array (~7 GB at 17 h @30 kHz); this
    iterator instead reads each requested chunk on demand from the files'
    memmaps, so the counts are never all resident at once (issue #47). The values
    and their order are identical to
    ``np.concatenate([io.get_analogsignal_timestamps(0, None) for io in neo_io])``.
    """

    def __init__(self, neo_io: list[SpikeGadgetsRawIO], **kwargs):
        self._neo_io = list(neo_io)
        lengths = [io._raw_memmap.shape[0] for io in self._neo_io]
        # global index of the first sample of each file (plus a final total)
        self._file_starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._total = int(self._file_starts[-1])
        super().__init__(**kwargs)

    def _get_data(self, selection: tuple) -> np.ndarray:
        start, stop, _ = selection[0].indices(self._total)
        out = np.empty(stop - start, dtype=np.uint32)
        for i, io in enumerate(self._neo_io):
            file_start = int(self._file_starts[i])
            file_stop = int(self._file_starts[i + 1])
            lo, hi = max(start, file_start), min(stop, file_stop)
            if lo < hi:
                out[lo - start : hi - start] = io.get_analogsignal_timestamps(
                    lo - file_start, hi - file_start
                )
        return out

    def _get_maxshape(self) -> tuple:
        return (self._total,)

    def _get_dtype(self) -> np.dtype:
        return np.dtype("uint32")


def add_epochs(
    nwbfile: NWBFile,
    session_df: pd.DataFrame,
    neo_io: list[SpikeGadgetsRawIO],
):
    """Add epochs to nwbfile.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to add epochs to.
    session_df : pd.DataFrame
        DataFrame with session file information.
    neo_io : list[SpikeGadgetsRawIO]
        List of neo_io iterators for each rec file. Contains time information.
    """
    logger = logging.getLogger("convert")
    for epoch in set(session_df.epoch):
        rec_file_list = session_df[
            (session_df.epoch == epoch) & (session_df.file_extension == ".rec")
        ]
        if len(rec_file_list) == 0:
            logger.info(f"no rec files for epoch {epoch}, No epoch interval created")
            continue
        start_time = None
        end_time = None
        logger.info(list(rec_file_list.full_path))
        for io in neo_io:
            if io.filename in list(rec_file_list.full_path):
                file_start_time = (
                    float(io.system_time_at_creation) / MILLISECONDS_PER_SECOND
                )
                if start_time is None or file_start_time < start_time:
                    start_time = file_start_time
                n_time = io._raw_memmap.shape[0]
                if io.sysClock_byte:
                    file_end_time = (
                        np.max(io.get_sys_clock(n_time - 1, n_time))
                        / NANOSECONDS_PER_SECOND
                    )
                else:
                    file_end_time = np.max(
                        io.get_systime_from_trodes_timestamps(n_time - 1, n_time)
                    )
                if end_time is None or file_end_time > end_time:
                    end_time = float(file_end_time)

        tag = f"{epoch:02d}_{rec_file_list.tag.iloc[0]}"
        nwbfile.add_epoch(start_time, end_time, tag)


def add_sample_count(
    nwbfile: NWBFile,
    rec_dci: RecFileDataChunkIterator,
):
    """add sample counts to nwbfile
    nwbfile : NWBFile
        nwbfle to add sample counts to
    rec_dci: RecFileDataChunkIterator
        rec file iterator with all the rec files for the session already in it
    """
    if "sample_count" in nwbfile.processing:
        try:
            raise ValueError("sample_count already exists in nwbfile.processing")
        except ValueError as e:
            logger = logging.getLogger("convert")
            logger.error(e)
            raise

    # make the objects to add to the nwb file
    nwbfile.create_processing_module(
        name="sample_count",
        description="corespondence between sample count and timestamps",
    )

    # Reference the already-lazy ephys timestamps directly rather than copying
    # them -- the copy duplicated the whole array (~15 GB at 17 h). Wrap them in a
    # DataChunkIterator so the dataset is streamed to disk chunk-by-chunk instead
    # of re-materialising on write (#47), matching how add_raw_ephys / add_analog
    # now stream their timestamps too.
    systime = _timestamps_for_write(rec_dci.timestamps)
    # Stream the sample counts instead of concatenating every file's into one
    # array (~7 GB at 17 h); they are written chunk-by-chunk from the memmaps.
    trodes_sample = _TrodesSampleCountIterator(rec_dci.neo_io)

    # insert into nwbfile
    nwbfile.processing["sample_count"].add(
        TimeSeries(
            name="sample_count",
            description="acquisition system sample count",
            data=trodes_sample,
            timestamps=systime,
            unit="int64",
        )
    )
