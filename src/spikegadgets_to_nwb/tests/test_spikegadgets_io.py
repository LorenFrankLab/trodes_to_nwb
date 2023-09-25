import os
from pathlib import Path

from spikegadgets_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO

import numpy as np

path = os.path.dirname(os.path.abspath(__file__))


def test_spikegadgets_raw_io_interpolation():
    # Interpolation of dropped timestamp only done for ephys data and systime

    # get the path to the rec file
    try:
        # running on github
        data_path = Path(os.environ.get("DOWNLOAD_DIR"))
    except (TypeError, FileNotFoundError):
        # running locally
        data_path = Path(path + "/test_data")
    rec_file = data_path / "20230622_sample_01_a1.rec"

    # create the SpikeGadgetsRawIO object
    neo_io = SpikeGadgetsRawIO(
        rec_file, interpolate_dropped_packets=True
    )  # unaltered file
    neo_io_dropped = SpikeGadgetsRawIO(
        rec_file, interpolate_dropped_packets=True
    )  # dropped packet file

    # parse the header
    neo_io.parse_header()
    neo_io_dropped.parse_header()

    # manually edit the memap to remove the second packet
    neo_io_dropped._raw_memmap = np.delete(neo_io._raw_memmap, 1, axis=0)

    # get the trodes timestamps from each to compare
    trodes_timestamps = neo_io.get_analogsignal_timestamps(0, 10)
    trodes_timestamps_dropped = neo_io_dropped.get_analogsignal_timestamps(0, 10)
    assert len(trodes_timestamps) == len(trodes_timestamps_dropped)
    assert np.isclose(
        trodes_timestamps, trodes_timestamps_dropped, atol=1e-6, rtol=0
    ).all()
    # assert trodes_timestamps[0] == trodes_timestamps_dropped[1]
    # make sure systime behaves expectedly
    systime = neo_io.get_sys_clock(0, 10)
    systime_dropped = neo_io_dropped.get_sys_clock(0, 10)
    assert len(systime) == len(systime_dropped)
    assert np.isclose(systime[2:], systime_dropped[2:], atol=1e-6, rtol=0).all()
    # get ephys data and check
    channel = "1"
    ephys_data = neo_io.get_analogsignal_chunk(
        i_start=0,
        i_stop=10,
        channel_ids=[channel],
        block_index=0,
        seg_index=0,
        stream_index=3,
    )
    ephys_data_dropped = neo_io_dropped.get_analogsignal_chunk(
        i_start=0,
        i_stop=10,
        channel_ids=[channel],
        block_index=0,
        seg_index=0,
        stream_index=3,
    )
    assert len(ephys_data) == len(ephys_data_dropped)
    assert np.isclose(ephys_data[2:], ephys_data_dropped[2:], atol=1e-6, rtol=0).all()
    assert ephys_data[0] == ephys_data_dropped[1]
