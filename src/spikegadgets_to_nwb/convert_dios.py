import numpy as np
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents

from .spike_gadgets_raw_io import SpikeGadgetsRawIO


def _get_channel_name_map(metadata: dict) -> dict[str, str]:
    """Parses behavioral events metadata from the yaml file

    Parameters
    ----------
    metadata : dict
        metadata from the yaml generator

    Returns
    -------
    channel_name_map : dict
        Parsed behavioral events metadata mapping hardware event name to human-readable name
    """
    dio_metadata = metadata["behavioral_events"]
    channel_name_map = {}
    for dio_event in dio_metadata:
        if dio_event["description"] in channel_name_map:
            raise ValueError(
                f"Duplicate channel name {dio_event['description']} in metadata YAML"
            )
        channel_name_map[dio_event["description"]] = dio_event["name"]
    return channel_name_map


def add_dios(nwbfile: NWBFile, recfile: list[str], metadata: dict) -> None:
    """Adds DIO event information and data to nwb file

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : list[str]
        list of paths to rec files
    metadata : dict
        metadata from the yaml generator
    """

    # TODO remove redundancy with convert_ephys.py
    neo_io = [
        SpikeGadgetsRawIO(filename=file) for file in recfile
    ]  # get all streams for all files
    [neo_io.parse_header() for neo_io in neo_io]

    # Make a processing module for behavior and add to the nwbfile
    if not "behavior" in nwbfile.processing:
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )

    # Make BehavioralEvents object to hold DIO data
    beh_events = BehavioralEvents(name="behavioral_events")

    # Map hardware event name (encoded in `description` in metadata YAML)
    # to a human-readable name (encoded in `name`)
    channel_name_map = _get_channel_name_map(metadata)

    # Loop through the channels from the metadata YAML and add a TimeSeries for each one
    stream_name = "ECU_digital"
    # Address issue where some Trodes verions have ECU_ prefix and some don't
    prefix = ""
    for chan_id in enumerate(neo_io[0]._mask_channels_ids[stream_name]):
        if "ECU_" in chan_id[1]:
            prefix = "ECU_"
            break

    for channel_name in channel_name_map:
        # merge streams from multiple files
        all_timestamps = []
        all_state_changes = []
        for io in neo_io:
            timestamps, state_changes = io.get_digitalsignal(
                stream_name, prefix + channel_name
            )
            all_timestamps.append(timestamps)
            all_state_changes.append(state_changes)
        all_timestamps = np.concatenate(all_timestamps)
        all_state_changes = np.concatenate(all_state_changes)
        assert isinstance(all_timestamps[0], np.float64)
        assert isinstance(all_timestamps, np.ndarray)
        ts = TimeSeries(
            name=channel_name_map[channel_name],
            description=channel_name,
            data=all_state_changes,
            unit="-1",  # TODO change to "N/A",
            timestamps=all_timestamps,  # TODO adjust timestamps
        )
        beh_events.add_timeseries(ts)

    # Add the BehavioralEvents object to the file
    nwbfile.processing["behavior"].add(beh_events)
