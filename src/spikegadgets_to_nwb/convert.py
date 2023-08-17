from pathlib import Path

import pandas as pd

from spikegadgets_to_nwb.convert_dios import add_dios
from spikegadgets_to_nwb.convert_ephys import add_raw_ephys
from spikegadgets_to_nwb.convert_position import add_position
from spikegadgets_to_nwb.convert_rec_header import (
    add_header_device,
    make_hw_channel_map,
    make_ref_electrode_map,
    read_header,
)
from spikegadgets_to_nwb.convert_yaml import (
    add_acquisition_devices,
    add_associated_files,
    add_cameras,
    add_electrode_groups,
    add_subject,
    add_tasks,
    initialize_nwb,
    load_metadata,
)
from spikegadgets_to_nwb.data_scanner import get_file_info


def _get_file_paths(df: pd.DataFrame, file_extension: str) -> list[str]:
    """Get the file paths for a given file extension

    Parameters
    ----------
    df : pd.DataFrame
        File info for a given epoch
    file_extension : str
        File extension to get file paths for

    Returns
    -------
    file_paths : list[str]
        File paths for the given file extension
    """
    return df.loc[df.file_extension == file_extension].full_path.to_list()


def create_nwbs(
    path: Path,
    header_reconfig_path: Path | None = None,
    probe_metadata_paths: list[Path] | None = None,
):
    if not isinstance(path, Path):
        path = Path(path)

    file_info = get_file_info(path)

    for session, session_df in file_info.groupby(["date", "animal"]):
        _create_nwb(session, session_df, header_reconfig_path, probe_metadata_paths)


def _create_nwb(
    session: tuple[str, str, str],
    session_df: pd.DataFrame,
    header_reconfig_path: Path | None = None,
    probe_metadata_paths: list[Path] | None = None,
):
    print(f"Creating NWB file for session: {session}")

    rec_filepaths = _get_file_paths(session_df, ".rec")
    print(f"\trec_filepaths: {rec_filepaths}")

    if header_reconfig_path is not None:
        pass

    rec_header = read_header(rec_filepaths[0])

    metadata_filepaths = _get_file_paths(session_df, ".yml")
    if len(metadata_filepaths) != 1:
        raise ValueError("There must be exactly one metadata file per session")
    else:
        metadata_filepaths = metadata_filepaths[0]
    print(f"\tmetadata_filepath: {metadata_filepaths}")

    metadata, probe_metadata = load_metadata(
        metadata_filepaths, probe_metadata_paths=probe_metadata_paths
    )

    hw_channel_map = make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    nwb_file = initialize_nwb(metadata, first_epoch_config=rec_header)
    add_subject(nwb_file, metadata)
    add_cameras(nwb_file, metadata)
    add_acquisition_devices(nwb_file, metadata)
    add_tasks(nwb_file, metadata)
    add_associated_files(nwb_file, metadata)
    add_electrode_groups(
        nwb_file, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )
    # add_associated_video_files(
    #     nwb_file, metadata, video_directory, raw_data_path, convert_timestamps
    # )
    add_dios(nwb_file, metadata)

    add_header_device(nwb_file, rec_header)

    ### add rec file data ###
    map_row_ephys_data_to_row_electrodes_table = list(
        range(len(nwb_file.electrodes))
    )  # TODO: Double check this

    add_raw_ephys(
        nwb_file,
        rec_filepaths,
        map_row_ephys_data_to_row_electrodes_table,
    )
    ### add position ###
    add_position(nwb_file, metadata, session_df, rec_header)
