import pandas as pd
from pynwb import NWBFile
from spikegadgets_to_nwb.convert_rec_header import read_header


def add_epochs(nwbfile: NWBFile, file_info: pd.DataFrame, date: int, animal: str):
    session_info = file_info[(file_info.date == date) & (file_info.animal == animal)]
    for epoch in set(session_info.epoch):
        rec_file_list = session_info[
            (session_info.epoch == epoch) & (session_info.file_extension == ".rec")
        ]
        start_time = None
        end_time = None
        for rec_path in rec_file_list.full_path:
            file_start_time = get_file_start_time(rec_path)
            if start_time is None or file_start_time < start_time:
                start_time = file_start_time
            end_time = 0.0

        tag = f"{epoch:02d}_{rec_file_list.tag.iloc[0]}"
        nwbfile.add_epoch(start_time, end_time, tag)
    return


def get_file_start_time(rec_file: str) -> float:
    header = read_header(rec_file)
    gconf = header.find("GlobalConfiguration")
    return float(gconf.attrib["systemTimeAtCreation"].strip()) / 1000.0
