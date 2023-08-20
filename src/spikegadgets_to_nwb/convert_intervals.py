import pandas as pd
import numpy as np
from pynwb import NWBFile
from typing import List
from spikegadgets_to_nwb.convert_rec_header import read_header
from spikegadgets_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO


def add_epochs(
    nwbfile: NWBFile,
    file_info: pd.DataFrame,
    date: int,
    animal: str,
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
    session_info = file_info[(file_info.date == date) & (file_info.animal == animal)]
    for epoch in set(session_info.epoch):
        rec_file_list = session_info[
            (session_info.epoch == epoch) & (session_info.file_extension == ".rec")
        ]
        start_time = None
        end_time = None
        print(list(rec_file_list.full_path))
        for io in neo_io:
            if io.filename in list(rec_file_list.full_path):
                file_start_time = float(io.system_time_at_creation) / 1000.0
                if start_time is None or file_start_time < start_time:
                    start_time = file_start_time
                n_time = io._raw_memmap.shape[0]
                if io.sysClock_byte:
                    file_end_time = np.max(io.get_sys_clock(n_time - 1, n_time)) / 1e9
                else:
                    file_end_time = np.max(
                        io._get_systime_from_trodes_timestamps(n_time - 1, n_time)
                    )
                if end_time is None or file_end_time > end_time:
                    end_time = float(file_end_time)

        tag = f"{epoch:02d}_{rec_file_list.tag.iloc[0]}"
        nwbfile.add_epoch(start_time, end_time, tag)
    return


def get_file_start_time(rec_file: str) -> float:
    header = read_header(rec_file)
    gconf = header.find("GlobalConfiguration")
    return float(gconf.attrib["systemTimeAtCreation"].strip()) / 1000.0
