"""Module for handling the conversion of ECU analog and headstage sensor data streams from Trodes .rec files to NWB format."""

import re
from xml.etree import ElementTree

import h5py
import numpy as np
import pynwb
from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile

from trodes_to_nwb import convert_rec_header
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

DEFAULT_CHUNK_TIME_DIM = 16384
DEFAULT_CHUNK_MAX_CHANNEL_DIM = 32


def _get_ecu_analog_channel_ids(rec_file_path: str) -> list[str]:
    """Returns the ordered list of ECU analog channel IDs from the rec file header."""
    root = convert_rec_header.read_header(rec_file_path)
    hconf = root.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    return [
        channel.attrib["id"]
        for channel in ecu_conf
        if channel.attrib["dataType"] == "analog"
    ]


# Sensor type registry. ``conversion`` is the NWB ``TimeSeries.conversion`` factor
# (stored_int16 * conversion = value in ``unit``), NOT a pre-multiplier applied to
# the array. Patterns are anchored with ``$`` so the axis / Ain-number group is the
# whole channel-name suffix (``Headstage_AccelXfoo`` does not match). The headstage
# IMU scaling factors are the SpikeGadgets sensor sensitivities:
# 0.000061 g/LSB = 1/16384 (accelerometer, +/-2 g full scale) and
# 0.061 deg/s/LSB = 2000/32768 (gyroscope, +/-2000 deg/s full scale).
SENSOR_TYPE_CONFIG: dict[str, dict] = {
    "accelerometer": {
        "pattern": r"Headstage_Accel[XYZ]$",
        "conversion": 0.000061,
        "unit": "g",
        "description": "Headstage accelerometer",
    },
    "gyroscope": {
        "pattern": r"Headstage_Gyro[XYZ]$",
        "conversion": 0.061,
        "unit": "d/s",
        "description": "Headstage gyroscope",
    },
    "magnetometer": {
        "pattern": r"Headstage_Mag[XYZ]$",
        "conversion": 1.0,  # no calibrated magnetometer scaling is defined
        "unit": "unspecified",
        "description": "Headstage magnetometer",
    },
    "analog_input": {
        "pattern": r"(ECU_Ain\d+|Controller_Ain\d+)$",
        "conversion": 1.0,  # raw counts; no counts->volts factor is defined
        "unit": "unspecified",
        "description": "ECU analog input",
    },
}

# Used directly (not a SENSOR_TYPE_CONFIG entry) for channels matching no pattern.
_OTHER_CONFIG = {
    "conversion": 1.0,
    "unit": "unspecified",
    "description": "Uncategorized analog channel",
}


def _categorize_sensor_channels(channel_names: list[str]) -> dict[str, list[str]]:
    """Group channel names by sensor type using ``SENSOR_TYPE_CONFIG`` patterns.

    Parameters
    ----------
    channel_names : list[str]
        Analog channel names to classify.

    Returns
    -------
    dict[str, list[str]]
        Insertion-ordered mapping of sensor type to the matching channel names,
        in the order they appear in ``channel_names``. Channels matching no
        pattern are grouped under the key ``"other"``. Sensor types with no
        matching channels are omitted; an empty input returns ``{}``.
    """
    categorized: dict[str, list[str]] = {}
    for sensor_type, config in SENSOR_TYPE_CONFIG.items():
        pattern = config["pattern"]
        matching = [name for name in channel_names if re.match(pattern, name)]
        if matching:
            categorized[sensor_type] = matching

    assigned = {name for names in categorized.values() for name in names}
    uncategorized = [name for name in channel_names if name not in assigned]
    if uncategorized:
        categorized["other"] = uncategorized

    return categorized


def _resolve_sensor_unit(
    sensor_type: str, default_unit: str, metadata: dict | None
) -> str:
    """Resolve the unit *label* for a sensor type.

    Returns ``metadata["sensor_units"][sensor_type]`` when present, otherwise
    ``default_unit``. The override changes only the label string; it does not
    change the numeric ``conversion`` factor (that always comes from
    ``SENSOR_TYPE_CONFIG``).

    Parameters
    ----------
    sensor_type : str
        Bare sensor type key (e.g. ``"accelerometer"``, ``"analog_input"``).
    default_unit : str
        Unit to use when no override is supplied.
    metadata : dict or None
        Metadata dictionary that may contain a ``"sensor_units"`` mapping.

    Returns
    -------
    str
        The resolved unit label.
    """
    if metadata and sensor_type in metadata.get("sensor_units", {}):
        return metadata["sensor_units"][sensor_type]
    return default_unit


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
    """
    hconf = header.find("HardwareConfiguration")
    ecu_conf = None
    # find the ECU configuration
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    # get the names of the analog channels
    analog_channel_names = []
    for channel in ecu_conf:
        if channel.attrib["dataType"] == "analog":
            analog_channel_names.append(channel.attrib["id"])
    return analog_channel_names
