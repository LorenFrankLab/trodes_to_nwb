"""Module for handling the conversion of ECU analog and headstage sensor data streams from Trodes .rec files to NWB format."""

import re
from xml.etree import ElementTree

import numpy as np
import pynwb
from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile, TimeSeries

from trodes_to_nwb import convert_rec_header
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator

DEFAULT_CHUNK_TIME_DIM = 16384
DEFAULT_CHUNK_MAX_CHANNEL_DIM = 32

# Sensor type definitions with scaling factors and units
SENSOR_TYPE_CONFIG = {
    'accelerometer': {
        'pattern': r'Headstage_Accel[XYZ]',
        'scaling_factor': 0.000061,  # Convert to g units
        'unit': 'g',
        'description': 'Headstage accelerometer data'
    },
    'gyroscope': {
        'pattern': r'Headstage_Gyro[XYZ]',
        'scaling_factor': 0.061,  # Convert to degrees/second
        'unit': 'd/s', 
        'description': 'Headstage gyroscope data'
    },
    'magnetometer': {
        'pattern': r'Headstage_Mag[XYZ]',
        'scaling_factor': 1.0,  # No scaling specified in issue
        'unit': 'unspecified',
        'description': 'Headstage magnetometer data'
    },
    'analog_input': {
        'pattern': r'(ECU_Ain\d+|Controller_Ain\d+)',
        'scaling_factor': 1.0,
        'unit': 'unspecified',
        'description': 'Analog input channel'
    }
}


def _categorize_sensor_channels(channel_names: list[str]) -> dict[str, list[str]]:
    """Categorize sensor channels by type based on naming patterns.
    
    Parameters
    ----------
    channel_names : list[str]
        List of channel names to categorize
        
    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping sensor types to lists of channel names
    """
    categorized = {}
    
    for sensor_type, config in SENSOR_TYPE_CONFIG.items():
        pattern = config['pattern']
        matching_channels = [name for name in channel_names if re.match(pattern, name)]
        if matching_channels:
            categorized[sensor_type] = matching_channels
    
    # Handle uncategorized channels
    categorized_flat = [name for channels in categorized.values() for name in channels]
    uncategorized = [name for name in channel_names if name not in categorized_flat]
    if uncategorized:
        categorized['other'] = uncategorized
    
    return categorized


def _create_sensor_timeseries(
    sensor_type: str,
    channel_names: list[str], 
    data: np.ndarray,
    timestamps: np.ndarray,
    metadata: dict = None
) -> TimeSeries:
    """Create a TimeSeries object for a specific sensor type.
    
    Parameters
    ----------
    sensor_type : str
        Type of sensor (accelerometer, gyroscope, etc.)
    channel_names : list[str]
        Names of channels for this sensor type
    data : np.ndarray
        Raw sensor data 
    timestamps : np.ndarray
        Timestamps for the data
    metadata : dict, optional
        Metadata dictionary for custom units/scaling
        
    Returns
    -------
    TimeSeries
        Configured TimeSeries object for the sensor type
    """
    config = SENSOR_TYPE_CONFIG.get(sensor_type, {
        'scaling_factor': 1.0,
        'unit': 'unspecified', 
        'description': f'{sensor_type} data'
    })
    
    # Apply scaling factor
    scaled_data = data * config['scaling_factor']
    
    # Create description with channel names
    description = f"{config['description']}: {', '.join(channel_names)}"
    
    # Use custom units from metadata if available
    unit = config['unit']
    if metadata and 'sensor_units' in metadata and sensor_type in metadata['sensor_units']:
        unit = metadata['sensor_units'][sensor_type]
    
    return TimeSeries(
        name=sensor_type,
        description=description,
        data=scaled_data,
        unit=unit,
        timestamps=timestamps,
    )


def add_analog_data(
    nwbfile: NWBFile,
    rec_file_path: list[str],
    timestamps: np.ndarray = None,
    behavior_only: bool = False,
    metadata: dict = None,
    separate_sensor_data: bool = True,
    **kwargs,
) -> None:
    """Adds analog streams to the nwb file as separate TimeSeries objects for each sensor type.

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    rec_file_path : list[str]
        ordered list of file paths to all recfiles with session's data
    timestamps : np.ndarray, optional
        timestamps for the data
    behavior_only : bool, optional
        if True, only include behavioral data
    metadata : dict, optional
        metadata dictionary for custom units and scaling
    separate_sensor_data : bool, optional
        if True, create separate TimeSeries for each sensor type (new behavior)
        if False, use legacy combined TimeSeries approach
    """
    
    # Legacy behavior for backwards compatibility
    if not separate_sensor_data:
        return _add_analog_data_legacy(nwbfile, rec_file_path, timestamps, behavior_only, **kwargs)
    
    # New behavior: separate sensor TimeSeries
    # Get the ids of the analog channels from the first rec file header
    root = convert_rec_header.read_header(rec_file_path[0])
    hconf = root.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    
    # Get ECU analog channel IDs
    ecu_analog_channel_ids = []
    if ecu_conf is not None:
        for channel in ecu_conf:
            if channel.attrib["dataType"] == "analog":
                ecu_analog_channel_ids.append(channel.attrib["id"])

    # Make the data chunk iterator for ECU analog data
    if ecu_analog_channel_ids:
        rec_dci = RecFileDataChunkIterator(
            rec_file_path,
            nwb_hw_channel_order=ecu_analog_channel_ids,
            stream_id="ECU_analog",
            is_analog=True,
            timestamps=timestamps,
            behavior_only=behavior_only,
        )

        # Get headstage sensor channel IDs from multiplexed channels
        headstage_channel_ids = list(rec_dci.neo_io[0].multiplexed_channel_xml.keys()) if rec_dci.neo_io else []

        # Process ECU analog channels
        if ecu_analog_channel_ids:
            # Get ECU analog data (without headstage data)
            ecu_data = rec_dci._get_data((slice(None), slice(0, len(ecu_analog_channel_ids))))
            
            # Categorize ECU analog channels
            ecu_categorized = _categorize_sensor_channels(ecu_analog_channel_ids)
            
            # Create TimeSeries for each ECU sensor type
            for sensor_type, channel_names in ecu_categorized.items():
                channel_indices = [ecu_analog_channel_ids.index(name) for name in channel_names]
                sensor_data = ecu_data[:, channel_indices]
                
                timeseries = _create_sensor_timeseries(
                    sensor_type=f"ecu_{sensor_type}",
                    channel_names=channel_names,
                    data=sensor_data,
                    timestamps=rec_dci.timestamps,
                    metadata=metadata
                )
                
                # Add to acquisition
                nwbfile.add_acquisition(timeseries)

        # Process headstage sensor channels if any exist  
        if headstage_channel_ids:
            # Get headstage sensor data
            headstage_data = rec_dci.neo_io[0].get_analogsignal_multiplexed(headstage_channel_ids)
            
            # Categorize headstage channels by sensor type
            headstage_categorized = _categorize_sensor_channels(headstage_channel_ids)
            
            # Create separate TimeSeries for each sensor type
            for sensor_type, channel_names in headstage_categorized.items():
                channel_indices = [headstage_channel_ids.index(name) for name in channel_names]
                sensor_data = headstage_data[:, channel_indices]
                
                timeseries = _create_sensor_timeseries(
                    sensor_type=sensor_type,
                    channel_names=channel_names,
                    data=sensor_data,
                    timestamps=rec_dci.timestamps,
                    metadata=metadata
                )
                
                # Add to acquisition  
                nwbfile.add_acquisition(timeseries)
    else:
        # If no ECU analog channels, create a minimal iterator to get headstage data
        try:
            from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
            neo_io = SpikeGadgetsRawIO(filename=rec_file_path[0])
            neo_io.parse_header()
            
            # Get headstage sensor channel IDs from multiplexed channels
            headstage_channel_ids = list(neo_io.multiplexed_channel_xml.keys()) if hasattr(neo_io, 'multiplexed_channel_xml') else []
            
            if headstage_channel_ids:
                # Get headstage sensor data
                headstage_data = neo_io.get_analogsignal_multiplexed(headstage_channel_ids)
                
                # Create timestamps if not provided
                if timestamps is None:
                    timestamps = neo_io.get_analogsignal_timestamps(0, headstage_data.shape[0])
                
                # Categorize headstage channels by sensor type
                headstage_categorized = _categorize_sensor_channels(headstage_channel_ids)
                
                # Create separate TimeSeries for each sensor type
                for sensor_type, channel_names in headstage_categorized.items():
                    channel_indices = [headstage_channel_ids.index(name) for name in channel_names]
                    sensor_data = headstage_data[:, channel_indices]
                    
                    timeseries = _create_sensor_timeseries(
                        sensor_type=sensor_type,
                        channel_names=channel_names,
                        data=sensor_data,
                        timestamps=timestamps[:sensor_data.shape[0]],  # Ensure same length
                        metadata=metadata
                    )
                    
                    # Add to acquisition  
                    nwbfile.add_acquisition(timeseries)
        except Exception as e:
            # If headstage processing fails, log warning but don't crash
            import logging
            logger = logging.getLogger("convert")
            logger.warning(f"Could not process headstage sensor data: {e}")


def _add_analog_data_legacy(
    nwbfile: NWBFile,
    rec_file_path: list[str],
    timestamps: np.ndarray = None,
    behavior_only: bool = False,
    **kwargs,
) -> None:
    """Legacy function for adding analog data as a single combined TimeSeries.
    
    This preserves the original behavior for backwards compatibility.
    """
    # get the ids of the analog channels from the first rec file header
    root = convert_rec_header.read_header(rec_file_path[0])
    hconf = root.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    analog_channel_ids = []
    if ecu_conf is not None:
        for channel in ecu_conf:
            if channel.attrib["dataType"] == "analog":
                analog_channel_ids.append(channel.attrib["id"])

    if not analog_channel_ids:
        return  # No analog channels to process

    # make the data chunk iterator
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
