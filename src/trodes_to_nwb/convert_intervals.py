import logging
from typing import List

import numpy as np
import pandas as pd
from pynwb import NWBFile, TimeSeries
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO

MILLISECONDS_PER_SECOND = 1e3
NANOSECONDS_PER_SECOND = 1e9


def add_epochs(
    nwbfile: NWBFile,
    session_df: pd.DataFrame,
    neo_io: List[SpikeGadgetsRawIO],
):
    """add epochs to nwbfile

    Parameters
    ----------
    nwbfile : NWBFile
        nwbfle to add epochs to
    file_info : pd.DataFrame
        dataframe with file info
    date : int
        date of session
    animal : str
        animal name
    neo_io: List[SpikeGadgetsRawIO]
        neo_io iterators for each rec file. Contains time information
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
    return


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

    # get the systime information
    systime = np.array(rec_dci.timestamps) * 1e9
    # get the sample count information
    trodes_sample = np.concatenate(
        [neo_io.get_analogsignal_timestamps(0, None) for neo_io in rec_dci.neo_io]
    )

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
