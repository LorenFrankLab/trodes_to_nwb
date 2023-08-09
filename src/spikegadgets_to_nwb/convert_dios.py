from pynwb import NWBFile, TimeSeries
from pynwb.behavior import BehavioralEvents

from .spike_gadgets_raw_io import SpikeGadgetsRawIO

def add_dios(
    nwbfile: NWBFile,
    recfile: list[str],
    channel_name_map: dict[str, str],
) -> None:

    # TODO remove redundancy with convert_ephys.py
    neo_io = [
        SpikeGadgetsRawIO(filename=file) for file in recfile
    ]  # get all streams for all files
    [neo_io.parse_header() for neo_io in neo_io]

    # TODO get all channel names
    stream_name = "ECU"
    channel_name = "ECU_Din2"
    timestamps, state_changes = neo_io[0].get_digitalsignal(stream_name, channel_name)

    # TODO concatenate timestamps and data across files

    mapped_channel_name = channel_name_map[channel_name]
    # TODO update this after consistency check with rec_to_nwb
    ts = TimeSeries(
        name=mapped_channel_name,
        data=state_changes,
        timestamps=timestamps,
        unit="-1",
    )

    beh_events = BehavioralEvents()
    beh_events.add_timeseries(ts)
    nwbfile.processing["behavior"].add_data_interface(beh_events)
