from pathlib import Path

import pandas as pd

from spikegadgets_to_nwb.data_scanner import (
    check_has_all_file_extensions,
    get_file_info,
)


def get_file_paths(epoch_df: pd.DataFrame, file_extension: str) -> list[str]:
    return epoch_df.loc[epoch_df.file_extension == file_extension].full_path.to_list()


def create_nwb(path: Path):
    if not isinstance(path, Path):
        path = Path(path)

    file_info = get_file_info(path)
    is_good_dataset = check_has_all_file_extensions(file_info)

    for session, session_df in file_info.groupby(["date", "animal"]):
        print(f"Creating NWB file for session: {session}")
        # nwb file creation code goes here
        for epoch, epoch_df in session_df.groupby(["date", "animal", "epoch"]):
            print(f"\tProcessing epoch: {epoch}")
            rec_filepaths = get_file_paths(epoch_df, ".rec")
            position_tracking_filepaths = get_file_paths(
                epoch_df, ".videoPositionTracking"
            )
            position_timestamps_filepaths = get_file_paths(epoch_df, ".cameraHWSync")
            state_script_log_filepaths = get_file_paths(epoch_df, ".stateScriptLog")
            video_filepaths = get_file_paths(epoch_df, ".h264")
            metadata_filepaths = get_file_paths(epoch_df, ".yaml")

            # nwb epoch specific creation code goes here
