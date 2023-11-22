from xml.etree import ElementTree

import numpy as np
import pynwb
from hdmf.backends.hdf5 import H5DataIO
from pynwb import NWBFile

from trodes_to_nwb import convert_rec_header
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator


def add_analog_data(
    nwbfile: NWBFile, rec_file_path: list[str], timestamps: np.ndarray = None, **kwargs
) -> None:
    """Adds analog streams to the nwb file.

    Parameters
    ----------
    nwbfile : NWBFile
        nwb file being assembled
    recfile : list[str]
        ordered list of file paths to all recfiles with session's data
    """
    # TODO: ADD HEADSTAGE DATA

    # get the ids of the analog channels from the first rec file header
    root = convert_rec_header.read_header(rec_file_path[0])
    hconf = root.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    analog_channel_ids = []
    for channel in ecu_conf:
        if channel.attrib["dataType"] == "analog":
            analog_channel_ids.append(channel.attrib["id"])

    # make the data chunk iterator
    # TODO use the stream name instead of the stream index to be more robust
    rec_dci = RecFileDataChunkIterator(
        rec_file_path,
        nwb_hw_channel_order=analog_channel_ids,
        stream_index=2,
        is_analog=True,
        timestamps=timestamps,
    )

    # add headstage channel IDs to the list of analog channel IDs
    analog_channel_ids.extend(rec_dci.neo_io[0].multiplexed_channel_xml.keys())

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(rec_dci, chunks=(16384, min(len(analog_channel_ids), 32)))

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
    description = ""
    for id in row_ids:
        description += id + "   "
    return description


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
