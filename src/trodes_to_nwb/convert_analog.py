"""Module for handling the conversion of ECU analog and headstage sensor data streams from Trodes .rec files to NWB format."""

from xml.etree import ElementTree

import h5py
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
import pynwb
from pynwb import NWBFile

from trodes_to_nwb import convert_rec_header
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

DEFAULT_CHUNK_TIME_DIM = 16384
DEFAULT_CHUNK_MAX_CHANNEL_DIM = 32


def _get_ecu_analog_channel_ids(rec_file_path: str) -> list[str]:
    """Returns the ordered list of ECU analog channel IDs from the rec file header."""
    root = convert_rec_header.read_header(rec_file_path)
    return get_analog_channel_names(root)


def add_analog_data(
    nwbfile: NWBFile,
    rec_file_path: list[str],
    timestamps: np.ndarray = None,
    behavior_only: bool = False,
    **kwargs,
) -> None:
    """Adds analog streams to the nwb file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file being assembled.
    rec_file_path : list[str]
        Ordered list of file paths to all recfiles with session's data.
    timestamps : np.ndarray, optional, shape (n_samples,)
        Array of timestamps for the analog data.
    behavior_only : bool, optional
        Whether to process only behavior data, by default False.
    **kwargs
        Additional keyword arguments.
    """
    # TODO: ADD HEADSTAGE DATA

    # get the ids of the analog channels from the first rec file header
    analog_channel_ids = _get_ecu_analog_channel_ids(rec_file_path[0])

    # make the data chunk iterator
    # TODO use the stream name instead of the stream index to be more robust
    rec_dci = RecFileDataChunkIterator(
        rec_file_path,
        nwb_hw_channel_order=analog_channel_ids,
        stream_id="ECU_analog",
        is_analog=True,
        timestamps=timestamps,
        behavior_only=behavior_only,
    )

    # add headstage channel IDs to the list of analog channel IDs
    analog_channel_ids.extend(rec_dci.neo_io[0].multiplexed_channel_xml.keys())

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(
        rec_dci,
        chunks=(
            DEFAULT_CHUNK_TIME_DIM,
            min(len(analog_channel_ids), DEFAULT_CHUNK_MAX_CHANNEL_DIM),
        ),
    )

    # make the objects to add to the nwb file
    nwbfile.create_processing_module(
        name="analog", description="Contains all analog data"
    )
    analog_events = pynwb.behavior.BehavioralEvents(name="analog")
    analog_events.add_timeseries(
        pynwb.TimeSeries(
            name="analog",
            description=__merge_row_description(
                analog_channel_ids
            ),  # NOTE: matches rec_to_nwb system
            data=data_data_io,
            timestamps=rec_dci.timestamps,
            unit="-1",
        )
    )
    # add it to the nwb file
    nwbfile.processing["analog"].add(analog_events)


_NWB_ANALOG_DATA_PATH = "processing/analog/analog/analog/data"
_NWB_ANALOG_TIMESTAMPS_PATH = "processing/analog/analog/analog/timestamps"


def update_analog_data(
    nwb_file_path: str,
    rec_file_path: list[str],
    timestamps: np.ndarray = None,
    behavior_only: bool = False,
) -> None:
    """Updates the analog signal data in an existing NWB file in-place.

    Use this function to fix NWB files created before the analog demuxing bug
    was corrected (where ``interleavedDataIDByte`` was not offset by the device
    start byte, causing multiplexed channels to be read incorrectly).

    Parameters
    ----------
    nwb_file_path : str
        Path to the existing NWB file to update in-place.
    rec_file_path : list[str]
        Ordered list of file paths to all rec files with the session's data.
        Must be the same files used during the original conversion.
    timestamps : np.ndarray, optional, shape (n_samples,)
        Array of timestamps for the analog data. If ``None``, timestamps are
        read from the existing NWB file.
    behavior_only : bool, optional
        Whether to process only behavior data, by default False.

    Raises
    ------
    ValueError
        If the shape of the correctly-read data does not match the shape of the
        data already stored in the NWB file.
    """
    # Reconstruct the same analog channel ID list used in the original conversion
    analog_channel_ids = _get_ecu_analog_channel_ids(rec_file_path[0])

    # Read timestamps from the existing NWB file if not provided
    if timestamps is None:
        with h5py.File(nwb_file_path, "r") as f:
            timestamps = f[_NWB_ANALOG_TIMESTAMPS_PATH][:]

    # Build the iterator with the corrected demuxing logic
    rec_dci = RecFileDataChunkIterator(
        rec_file_path,
        nwb_hw_channel_order=analog_channel_ids,
        stream_id="ECU_analog",
        is_analog=True,
        timestamps=timestamps,
        behavior_only=behavior_only,
    )

    n_samples, n_channels = rec_dci.maxshape
    with h5py.File(nwb_file_path, "r+") as f:
        dataset = f[_NWB_ANALOG_DATA_PATH]
        existing_shape = dataset.shape
        expected_shape = (n_samples, n_channels)
        if existing_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: existing data has shape {existing_shape} but "
                f"re-read data has shape {expected_shape}. "
                "Ensure the same rec files and settings are used."
            )
        # Write data chunk-by-chunk to avoid loading the full dataset into memory
        for chunk in rec_dci:
            dataset[chunk.selection] = chunk.data


def __merge_row_description(row_ids: list[str]) -> str:
    return "   ".join(row_ids) + "   "


def get_analog_channel_names(header: ElementTree) -> list[str]:
    """Returns a list of the names of the analog channels in the rec file.

    Parameters
    ----------
    header : ElementTree
        The root element of the rec file header

    Returns
    -------
    list[str]
        List of the names of the analog channels in the rec file

    Raises
    ------
    ValueError
        If no ECU device is found in the rec file header.
    """
    hconf = header.find("HardwareConfiguration")
    if hconf is None:
        raise ValueError(
            "Rec file header missing HardwareConfiguration element. "
            "Cannot extract analog channel IDs."
        )
    ecu_conf = None
    # find the ECU configuration
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    if ecu_conf is None:
        raise ValueError(
            "No ECU device found in rec file header. Cannot extract analog channel IDs."
        )
    return [
        channel.attrib["id"]
        for channel in ecu_conf
        if channel.attrib["dataType"] == "analog"
    ]
