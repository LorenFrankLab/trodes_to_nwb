import logging
import os
from pathlib import Path

import pandas as pd
from pynwb import NWBHDF5IO

from spikegadgets_to_nwb.convert_analog import add_analog_data
from spikegadgets_to_nwb.convert_dios import add_dios
from spikegadgets_to_nwb.convert_ephys import RecFileDataChunkIterator, add_raw_ephys
from spikegadgets_to_nwb.convert_intervals import add_epochs, add_sample_count
from spikegadgets_to_nwb.convert_position import add_position
from spikegadgets_to_nwb.convert_rec_header import (
    add_header_device,
    make_hw_channel_map,
    make_ref_electrode_map,
    read_header,
    validate_yaml_header_electrode_map,
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


def setup_logger(name_logfile: str, path_logfile: str) -> logging.Logger:
    """Sets up a logger for each function that outputs
    to the console and to a file

    Parameters
    ----------
    name_logfile : str
        Name of the logfile
    path_logfile : str
        Path to the logfile

    Returns
    -------
    logger : logging.Logger
        Logger object
    """
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%d-%b-%y %H:%M:%S"
    )
    fileHandler = logging.FileHandler(path_logfile, mode="w")
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


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
    file_paths : list[s
        File paths for the given file extension
    """
    return df.loc[df.file_extension == file_extension].full_path.to_list()


def create_nwbs(
    path: Path,
    header_reconfig_path: Path | None = None,
    probe_metadata_paths: list[Path] | None = None,
    output_dir: str = "/home/stelmo/nwb/raw",
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
    output_dir: str = "/home/stelmo/nwb/raw",
):
    # create loggers
    logger = setup_logger(
        "convert", f"{output_dir}/{session[1]}{session[0]}_convert.log"
    )
    for module in [
        "convert_yaml",
        "convert_rec_header",
        "convert_ephys",
        "convert_position",
        "convert_analog",
        "convert_dios",
        "convert_intervals",
        "data_scanner",
    ]:
        setup_logger(
            f"convert.{module}",
            f"{output_dir}/{session[1]}{session[0]}_convert.{module}.log",
        )

    logger.info(f"Creating NWB file for session: {session}")
    rec_filepaths = _get_file_paths(session_df, ".rec")
    logger.info(f"\trec_filepaths: {rec_filepaths}")

    logger.info("CREATING REC DATA ITERATORS")
    # make generic rec file data chunk iterator to pass to functions
    rec_dci = RecFileDataChunkIterator(rec_filepaths, interpolate_dropped_packets=True)

    rec_header = read_header(rec_filepaths[0])
    reconfig_header = rec_header
    if header_reconfig_path is not None:
        reconfig_header = read_header(header_reconfig_path)

    metadata_filepaths = _get_file_paths(session_df, ".yml")
    if len(metadata_filepaths) != 1:
        try:
            raise ValueError("There must be exactly one metadata file per session")
        except ValueError as e:
            logger.exception("ERROR:")
            raise e
    else:
        metadata_filepaths = metadata_filepaths[0]
    logger.info(f"\tmetadata_filepath: {metadata_filepaths}")

    metadata, probe_metadata = load_metadata(
        metadata_filepaths, probe_metadata_paths=probe_metadata_paths
    )

    logger.info("CREATING HARDWARE MAPS")
    # test that yaml and headder are compatible hardware maps
    validate_yaml_header_electrode_map(
        metadata, reconfig_header.find("SpikeConfiguration")
    )
    # make necessary maps for ephys channels
    hw_channel_map = make_hw_channel_map(
        metadata, reconfig_header.find("SpikeConfiguration")
    )
    ref_electrode_map = make_ref_electrode_map(
        metadata, reconfig_header.find("SpikeConfiguration")
    )
    logger.info("CREATING METADATA ENTRIES")
    # make the nwbfile with the basic entries
    nwb_file = initialize_nwb(metadata, first_epoch_config=rec_header)
    add_subject(nwb_file, metadata)
    add_cameras(nwb_file, metadata)
    add_acquisition_devices(nwb_file, metadata)
    add_tasks(nwb_file, metadata)
    add_associated_files(nwb_file, metadata)
    add_electrode_groups(
        nwb_file, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )
    add_header_device(nwb_file, rec_header)

    logger.info("ADDING EPHYS AND REC DATA")
    ### add rec file data ###
    map_row_ephys_data_to_row_electrodes_table = list(
        range(len(nwb_file.electrodes))
    )  # TODO: Double check this
    add_raw_ephys(
        nwb_file,
        rec_filepaths,
        map_row_ephys_data_to_row_electrodes_table,
        metadata,
    )
    add_dios(nwb_file, rec_filepaths, metadata)
    add_analog_data(nwb_file, rec_filepaths)
    add_sample_count(nwb_file, rec_dci)
    logger.info("ADDING POSITION")
    ### add position ###
    add_position(
        nwb_file, metadata, session_df, rec_header, video_directory=""
    )  # TODO: video_directory needs passed from create function

    # add epochs
    add_epochs(
        nwbfile=nwb_file,
        file_info=session_df,
        date=session[0],
        animal=session[1],
        neo_io=rec_dci.neo_io,
    )

    # write file
    logger.info(f"WRITING: {output_dir}/{session[1]}{session[0]}.nwb")
    with NWBHDF5IO(f"{output_dir}/{session[1]}{session[0]}.nwb", "w") as io:
        io.write(nwb_file)
