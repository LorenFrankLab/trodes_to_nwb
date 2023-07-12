from pathlib import Path

import pandas as pd

from spikegadgets_to_nwb.data_scanner import (
    get_file_info,
)


def _get_file_paths(epoch_df: pd.DataFrame, file_extension: str) -> list[str]:
    return epoch_df.loc[epoch_df.file_extension == file_extension].full_path.to_list()


def create_nwb(path: Path):
    if not isinstance(path, Path):
        path = Path(path)

    file_info = get_file_info(path)

    for session, session_df in file_info.groupby(["date", "animal"]):
        print(f"Creating NWB file for session: {session}")
        # nwb file creation code goes here
        for epoch, epoch_df in session_df.groupby(["date", "animal", "epoch"]):
            print(f"\tProcessing epoch: {epoch}")

            rec_filepaths = _get_file_paths(epoch_df, ".rec")
            print(f"\t\trec_filepaths: {rec_filepaths}")

            metadata_filepaths = _get_file_paths(epoch_df, ".yaml")
            print(f"\t\tmetadata_filepaths: {metadata_filepaths}")

            position_timestamps_filepaths = _get_file_paths(epoch_df, ".cameraHWSync")
            print(f"\t\tposition_timestamps_filepaths: {position_timestamps_filepaths}")

            position_tracking_filepaths = _get_file_paths(
                epoch_df, ".videoPositionTracking"
            )
            print(f"\t\tposition_tracking_filepaths: {position_tracking_filepaths}")

            video_filepaths = _get_file_paths(epoch_df, ".h264")
            print(f"\t\tvideo_filepaths: {video_filepaths}")

            state_script_log_filepaths = _get_file_paths(epoch_df, ".stateScriptLog")
            print(f"\t\tstate_script_log_filepaths: {state_script_log_filepaths}")

            # nwb epoch specific creation code goes here
