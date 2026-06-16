"""Module for converting Digital Input/Output (DIO) event data (state changes)
from Trodes .rec files into NWB TimeSeries within a BehavioralEvents container.
"""

import logging

import numpy as np
from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents

from .convert_rec_header import read_header
from .spike_gadgets_raw_io import SpikeGadgetsRawIO


def _get_digital_channel_input_map(recfile: str) -> dict[str, str | None]:
    """Map each digital channel's header id to its ``input`` flag.

    Reads the ``.rec`` header and returns ``{channel_id: input}`` for every
    digital ``<Channel>``, where ``input`` is ``"1"`` for a digital input and
    ``"0"`` for a digital output. Used to record the specific hardware channel
    and its direction in each DIO TimeSeries description (issues #116, #117).

    Parameters
    ----------
    recfile : str
        Path to a ``.rec`` file whose header is read.

    Returns
    -------
    dict[str, str | None]
        ``{header channel id -> input flag}`` (e.g. ``{"ECU_Din1": "1"}``); the
        flag is ``None`` for the rare channel with no ``input`` attribute.
    """
    hardware_config = read_header(recfile).find("HardwareConfiguration")
    channel_input_map = {}
    if hardware_config is not None:
        for device in hardware_config:
            for channel in device:
                if channel.attrib.get("dataType") == "digital":
                    channel_input_map[channel.attrib["id"]] = channel.attrib.get(
                        "input"
                    )
    return channel_input_map


def _get_channel_name_map(metadata: dict) -> dict[str, dict]:
    """Parses behavioral events metadata from the yaml file.

    Parameters
    ----------
    metadata : dict
        Metadata from the yaml generator.

    Returns
    -------
    channel_name_map : dict
        Parsed behavioral events metadata mapping hardware event name to human-readable name.
    """
    dio_metadata = metadata["behavioral_events"]
    channel_name_map = {}
    for dio_event in dio_metadata:
        if dio_event["description"] in channel_name_map:
            raise ValueError(
                f"Duplicate channel name {dio_event['description']} in metadata YAML"
            )
        channel_name_map[dio_event["description"]] = {
            "name": dio_event["name"],
            "comments": (dio_event.get("comments", "no comments")),
        }
    return channel_name_map


def add_dios(nwbfile: NWBFile, recfile: list[str], metadata: dict) -> None:
    """Adds DIO event information and data to nwb file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file being assembled.
    recfile : list[str]
        List of paths to rec files.
    metadata : dict
        Metadata from the yaml generator.
    """

    # TODO remove redundancy with convert_ephys.py
    neo_io = [
        SpikeGadgetsRawIO(filename=file) for file in recfile
    ]  # get all streams for all files
    [neo_io.parse_header() for neo_io in neo_io]

    # Make a processing module for behavior and add to the nwbfile
    if "behavior" not in nwbfile.processing:
        nwbfile.create_processing_module(
            name="behavior", description="Contains all behavior-related data"
        )

    # Make BehavioralEvents object to hold DIO data
    beh_events = BehavioralEvents(name="behavioral_events")

    # Map hardware event name (encoded in `description` in metadata YAML)
    # to a human-readable name (encoded in `name`)
    channel_name_map = _get_channel_name_map(metadata)

    # Map each digital channel's header id to its input flag, for the per-channel
    # description (#116 traceability + #117 direction).
    channel_input_map = _get_digital_channel_input_map(recfile[0])

    # Loop through the channels from the metadata YAML and add a TimeSeries for each one
    stream_name = "ECU_digital"
    # Address issue where some Trodes verions have ECU_ prefix and some don't
    prefix = ""
    for chan_id in enumerate(neo_io[0]._mask_channels_ids[stream_name]):
        if "ECU_" in chan_id[1]:
            prefix = "ECU_"
            break

    all_timestamps = [[] for _ in channel_name_map]
    all_state_changes = [[] for _ in channel_name_map]
    # Loop through io objects and get timestamps and state changes for each channel
    for io in neo_io:
        for i, channel_name in enumerate(channel_name_map):
            timestamps, state_changes = io.get_digitalsignal(
                stream_name, prefix + channel_name
            )
            all_timestamps[i].append(timestamps)
            all_state_changes[i].append(state_changes)
    for channel_name, state_changes, timestamps in zip(
        channel_name_map, all_state_changes, all_timestamps, strict=True
    ):
        timestamps = np.concatenate(timestamps)
        state_changes = np.concatenate(state_changes)
        assert isinstance(timestamps[0], np.float64)
        assert isinstance(timestamps, np.ndarray)
        # Describe the channel by its specific hardware id (#116) and its
        # direction from the header `input` flag -- 1 = digital input, 0 = digital
        # output -- giving both the verbatim attribute and a human gloss (#117).
        full_channel_id = prefix + channel_name
        input_flag = channel_input_map.get(full_channel_id)
        if input_flag is None:
            # The channel resolved for data extraction (get_digitalsignal above
            # would have raised otherwise) but the header carries no `input`
            # attribute, so we can't state the direction -- keep the id and warn
            # rather than silently omitting it.
            logging.getLogger("convert").warning(
                "DIO channel %s has no 'input' flag in the header; recording the "
                "channel id without a direction.",
                full_channel_id,
            )
            description = full_channel_id
        else:
            direction = "input" if input_flag == "1" else "output"
            description = f"{full_channel_id}, input={input_flag} (digital {direction})"
        ts = TimeSeries(
            name=channel_name_map[channel_name]["name"],
            comments=channel_name_map[channel_name]["comments"],
            description=description,
            data=state_changes,
            unit="N/A",
            timestamps=timestamps,  # TODO adjust timestamps
        )
        beh_events.add_timeseries(ts)

    # Add the BehavioralEvents object to the file
    nwbfile.processing["behavior"].add(beh_events)
