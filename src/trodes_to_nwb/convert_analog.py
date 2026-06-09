"""Module for handling the conversion of ECU analog and headstage sensor data streams from Trodes .rec files to NWB format."""

from dataclasses import dataclass
import logging
import re
from xml.etree import ElementTree

import h5py
from hdmf.backends.hdf5 import H5DataIO
from hdmf.data_utils import GenericDataChunkIterator
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


@dataclass(frozen=True)
class SensorConfig:
    """Scaling/unit/naming for one analog sensor type.

    ``conversion`` is the NWB ``TimeSeries.conversion`` factor
    (``stored_int16 * conversion = value in unit``), NOT a pre-multiplier applied
    to the array. ``pattern`` is an anchored regex matching this sensor's channel
    names, or ``None`` for the catch-all ("other") bucket that matches no pattern.
    """

    conversion: float
    unit: str
    description: str
    pattern: str | None = None


# Sensor type registry. Patterns are anchored with ``$`` so the axis / Ain-number
# group is the whole channel-name suffix (``Headstage_AccelXfoo`` does not match).
# The headstage IMU scaling factors are the SpikeGadgets sensor sensitivities:
# 0.000061 g/LSB = 1/16384 (accelerometer, +/-2 g full scale) and
# 0.061 deg/s/LSB = 2000/32768 (gyroscope, +/-2000 deg/s full scale).
SENSOR_TYPE_CONFIG: dict[str, SensorConfig] = {
    "accelerometer": SensorConfig(
        conversion=0.000061,
        unit="g",
        description="Headstage accelerometer",
        pattern=r"Headstage_Accel[XYZ]$",
    ),
    "gyroscope": SensorConfig(
        conversion=0.061,
        unit="d/s",
        description="Headstage gyroscope",
        pattern=r"Headstage_Gyro[XYZ]$",
    ),
    "magnetometer": SensorConfig(
        conversion=1.0,  # no calibrated magnetometer scaling is defined
        unit="unspecified",
        description="Headstage magnetometer",
        pattern=r"Headstage_Mag[XYZ]$",
    ),
    "analog_input": SensorConfig(
        conversion=1.0,  # raw counts; no counts->volts factor is defined
        unit="unspecified",
        description="ECU analog input",
        pattern=r"(ECU_Ain\d+|Controller_Ain\d+)$",
    ),
}

# Used for channels matching no pattern (pattern=None marks the catch-all).
_OTHER_CONFIG = SensorConfig(
    conversion=1.0,
    unit="unspecified",
    description="Uncategorized analog channel",
)


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
        pattern = config.pattern
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


class _AnalogChannelSubsetIterator(GenericDataChunkIterator):
    """Lazily yield a fixed subset of channel columns from a shared analog iterator.

    Reuses the source ``RecFileDataChunkIterator``'s multi-file, demux-correct
    reads. Stores raw int16 values unchanged; physical scaling is carried by the
    owning ``TimeSeries.conversion`` field, so nothing is upcast or materialized
    here.

    Parameters
    ----------
    source : RecFileDataChunkIterator
        Shared iterator over the combined analog stream (ECU analog channels
        followed by multiplexed headstage channels).
    column_indices : list[int]
        Column positions, into the combined stream, for this sensor's channels,
        in output order.
    """

    def __init__(self, source, column_indices):
        self._source = source
        self._column_indices = list(column_indices)
        self._n_time, self._n_source_cols = source.maxshape
        invalid = [i for i in self._column_indices if not 0 <= i < self._n_source_cols]
        if invalid:
            raise ValueError(
                f"column_indices {invalid} are out of range for a source with "
                f"{self._n_source_cols} columns"
            )
        super().__init__()

    def _get_data(self, selection: tuple[slice, slice]) -> np.ndarray:
        # Read this time-chunk across all source columns (cheap: tens of columns
        # from one packet stream), then pick out the columns for this sensor.
        # selection bounds are concrete integers supplied by the base class.
        full = self._source._get_data((selection[0], slice(0, self._n_source_cols)))
        subset = full[:, self._column_indices]
        return subset[:, selection[1]]

    def _get_maxshape(self) -> tuple[int, int]:
        return (self._n_time, len(self._column_indices))

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")


def add_analog_data(
    nwbfile: NWBFile,
    rec_file_path: list[str],
    timestamps: np.ndarray = None,
    behavior_only: bool = False,
    metadata: dict | None = None,
    **kwargs,
) -> None:
    """Adds analog streams as separate acquisition TimeSeries with physical units.

    Headstage IMU sensors (accelerometer, gyroscope, magnetometer) and ECU
    analog inputs are written as individual ``TimeSeries`` in
    ``nwbfile.acquisition``, one per sensor type, with the physical unit and
    scaling carried by each ``TimeSeries.conversion`` field. Data stays lazy and
    chunked via ``H5DataIO``: raw int16 samples are stored unchanged and scaling
    is applied on read (``stored * conversion``), so memory use is independent of
    session length.

    Channels matching no known sensor pattern are written, unscaled, to an
    ``"other"`` acquisition stream and logged at WARNING.

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
    metadata : dict, optional
        Metadata dictionary. A ``"sensor_units"`` mapping may override the unit
        *label* of any sensor type (it does not change the numeric conversion).
    **kwargs
        Additional keyword arguments.
    """
    logger = logging.getLogger("convert")

    # ECU analog channels come first in the combined stream, then the multiplexed
    # headstage sensor channels appended by RecFileDataChunkIterator.
    ecu_analog_ids = _get_ecu_analog_channel_ids(rec_file_path[0])
    rec_dci = RecFileDataChunkIterator(
        rec_file_path,
        nwb_hw_channel_order=ecu_analog_ids,
        stream_id="ECU_analog",
        is_analog=True,
        timestamps=timestamps,
        behavior_only=behavior_only,
    )
    multiplexed_ids = list(rec_dci.neo_io[0].multiplexed_channel_xml.keys())
    all_channel_ids = ecu_analog_ids + multiplexed_ids
    if not all_channel_ids:
        logger.info(
            "No analog channels found in %s; skipping analog data.", rec_file_path[0]
        )
        return

    groups = _categorize_sensor_channels(all_channel_ids)

    # Warn on sensor_units overrides that name an unknown sensor type, so a typo
    # (e.g. "accel" instead of "accelerometer") is not silently ignored.
    if metadata and "sensor_units" in metadata:
        valid_sensor_types = set(SENSOR_TYPE_CONFIG) | {"other"}
        unknown = set(metadata["sensor_units"]) - valid_sensor_types
        if unknown:
            logger.warning(
                "metadata['sensor_units'] has unrecognized sensor type(s) %s; "
                "those unit overrides are ignored. Valid keys: %s",
                sorted(unknown),
                sorted(valid_sensor_types),
            )

    # Cap the time-axis chunk at the session length so short sessions (fewer than
    # DEFAULT_CHUNK_TIME_DIM samples) get a valid HDF5 chunk shape.
    n_time = rec_dci.maxshape[0]
    chunk_time_dim = min(DEFAULT_CHUNK_TIME_DIM, n_time)

    # All sensor streams share one timestamps dataset: the first TimeSeries owns
    # it; the rest link to that object so pynwb stores it only once.
    shared_timestamps = rec_dci.timestamps
    first_ts = None
    for sensor_type, channel_names in groups.items():
        if sensor_type == "other":
            logger.warning(
                "Analog channels matched no known sensor pattern and are stored "
                "raw (unit='unspecified') under acquisition['other']: %s",
                channel_names,
            )
            config = _OTHER_CONFIG
        else:
            config = SENSOR_TYPE_CONFIG[sensor_type]

        column_indices = [all_channel_ids.index(name) for name in channel_names]
        data_iter = _AnalogChannelSubsetIterator(rec_dci, column_indices)
        data_io = H5DataIO(
            data_iter,
            chunks=(
                chunk_time_dim,
                min(len(column_indices), DEFAULT_CHUNK_MAX_CHANNEL_DIM),
            ),
        )
        unit = _resolve_sensor_unit(sensor_type, config.unit, metadata)
        description = f"{config.description}: {', '.join(channel_names)}"

        timeseries = pynwb.TimeSeries(
            name=sensor_type,
            description=description,
            data=data_io,
            unit=unit,
            conversion=config.conversion,
            timestamps=(first_ts if first_ts is not None else shared_timestamps),
        )
        nwbfile.add_acquisition(timeseries)
        if first_ts is None:
            first_ts = timeseries


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

    This targets the *legacy* file layout, where analog data lives in a single
    combined ``processing/analog/analog/analog/data`` stream. Files written by
    the current :func:`add_analog_data` instead store per-sensor TimeSeries in
    ``acquisition`` and are not handled here.

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

    # Guard: this tool only repairs the legacy combined-analog layout. Files
    # written by the current add_analog_data store per-sensor TimeSeries in
    # acquisition and have no processing/analog stream to repair.
    with h5py.File(nwb_file_path, "r") as f:
        if _NWB_ANALOG_DATA_PATH not in f:
            raise ValueError(
                f"{nwb_file_path!r} has no legacy combined analog stream at "
                f"{_NWB_ANALOG_DATA_PATH!r}. update_analog_data only repairs files "
                "written by the pre-sensor-separation layout; files written by the "
                "current add_analog_data store per-sensor TimeSeries in acquisition "
                "and need no demux repair."
            )

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
