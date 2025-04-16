import numpy as np
import pytest

from trodes_to_nwb.spike_gadgets_raw_io import InsertedMemmap, SpikeGadgetsRawIO
from trodes_to_nwb.tests.utils import data_path


def test_spikegadgets_raw_io_interpolation():
    # Interpolation of dropped timestamp only done for ephys data and systime

    # get the path to the rec file
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

    # get the trodes timestamps from each to compare. This also generates the interpolation
    trodes_timestamps = neo_io.get_analogsignal_timestamps(0, 10)
    trodes_timestamps_dropped = neo_io_dropped.get_analogsignal_timestamps(0, 10)
    trodes_timestamps_dropped_secondary = neo_io_dropped.get_analogsignal_timestamps(
        0, 10
    )

    # check that the interpolated memmap returns the same shape value
    assert isinstance(neo_io_dropped._raw_memmap, InsertedMemmap)
    assert neo_io_dropped._raw_memmap.inserted_locations == [0]
    assert neo_io._raw_memmap.shape[0] == neo_io_dropped._raw_memmap.shape[0]

    # check the returned timestamps
    assert len(trodes_timestamps) == len(trodes_timestamps_dropped)
    assert np.allclose(trodes_timestamps, trodes_timestamps_dropped, atol=1e-6, rtol=0)
    assert np.allclose(
        trodes_timestamps_dropped,
        trodes_timestamps_dropped_secondary,
        atol=1e-6,
        rtol=0,
    )
    # make sure systime behaves expectedly
    systime = neo_io.get_sys_clock(0, 10)
    systime_dropped = neo_io_dropped.get_sys_clock(0, 10)
    assert len(systime) == len(systime_dropped)
    assert np.allclose(systime[2:], systime_dropped[2:], atol=1e-6, rtol=0)
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
    assert np.allclose(ephys_data[2:], ephys_data_dropped[2:], atol=1e-6, rtol=0)
    assert ephys_data[0] == ephys_data_dropped[1]

    # check that all timestamps are properly accessible
    chunk = 10000
    i = 0
    while i < neo_io_dropped._get_signal_size(1, 1, 3):
        t = neo_io_dropped.get_analogsignal_timestamps(i, i + chunk)
        assert t.size == chunk or t.size == neo_io_dropped._get_signal_size(1, 1, 3) - i
        i += chunk


def test_produce_ephys_channel_ids():
    """
    Test the static method _produce_ephys_channel_ids for various configurations.

    Verifies the correct interleaved channel ID order is generated for different
    total channel counts and channels per chip, including edge cases and
    error handling.
    """

    # --- Test Cases ---

    # Case 1: Standard 128 channels, 32 per chip (4 chips)
    n_total_1 = 128
    n_per_chip_1 = 32
    expected_1 = [
        0,
        32,
        64,
        96,  # ch 0 across chips
        1,
        33,
        65,
        97,  # ch 1 across chips
        2,
        34,
        66,
        98,  # ch 2 across chips
        # ... many channels omitted for brevity ...
        29,
        61,
        93,
        125,  # ch 29 across chips
        30,
        62,
        94,
        126,  # ch 30 across chips
        31,
        63,
        95,
        127,  # ch 31 across chips
    ]
    # Manually calculate the full expected list for assertion
    full_expected_1 = []
    for k in range(n_per_chip_1):
        full_expected_1.extend(
            [k + i * n_per_chip_1 for i in range(n_total_1 // n_per_chip_1)]
        )
    result_1 = SpikeGadgetsRawIO._produce_ephys_channel_ids(n_total_1, n_per_chip_1)
    assert result_1 == full_expected_1
    # Check a few key elements explicitly too
    assert result_1[0:8] == expected_1[0:8]
    assert result_1[-8:] == expected_1[-8:]
    assert len(result_1) == n_total_1

    # Case 2: Smaller case - 64 channels, 32 per chip (2 chips)
    n_total_2 = 64
    n_per_chip_2 = 32
    expected_2 = [
        0,
        32,  # ch 0
        1,
        33,  # ch 1
        # ...
        30,
        62,  # ch 30
        31,
        63,  # ch 31
    ]
    full_expected_2 = []
    for k in range(n_per_chip_2):
        full_expected_2.extend(
            [k + i * n_per_chip_2 for i in range(n_total_2 // n_per_chip_2)]
        )
    result_2 = SpikeGadgetsRawIO._produce_ephys_channel_ids(n_total_2, n_per_chip_2)
    assert result_2 == full_expected_2
    assert len(result_2) == n_total_2

    # Case 3: Different chip size - 64 channels, 16 per chip (4 chips)
    n_total_3 = 64
    n_per_chip_3 = 16
    expected_3 = [
        0,
        16,
        32,
        48,  # ch 0
        1,
        17,
        33,
        49,  # ch 1
        # ...
        14,
        30,
        46,
        62,  # ch 14
        15,
        31,
        47,
        63,  # ch 15
    ]
    full_expected_3 = []
    for k in range(n_per_chip_3):
        full_expected_3.extend(
            [k + i * n_per_chip_3 for i in range(n_total_3 // n_per_chip_3)]
        )
    result_3 = SpikeGadgetsRawIO._produce_ephys_channel_ids(n_total_3, n_per_chip_3)
    assert result_3 == full_expected_3
    assert len(result_3) == n_total_3

    # Case 4: Single chip - 32 channels, 32 per chip (1 chip)
    n_total_4 = 32
    n_per_chip_4 = 32
    expected_4 = list(range(32))  # Should just be 0, 1, 2, ..., 31
    result_4 = SpikeGadgetsRawIO._produce_e_phys_channel_ids(n_total_4, n_per_chip_4)
    assert result_4 == expected_4
    assert len(result_4) == n_total_4

    # --- Edge Cases ---

    # Case 5: Zero total channels
    result_5 = SpikeGadgetsRawIO._produce_ephys_channel_ids(0, 32)
    assert result_5 == []

    # Case 6: Zero channels per chip
    result_6 = SpikeGadgetsRawIO._produce_ephys_channel_ids(128, 0)
    assert result_6 == []

    # Case 7: Zero for both
    result_7 = SpikeGadgetsRawIO._produce_ephys_channel_ids(0, 0)
    assert result_7 == []

    # --- Error Cases ---

    # Case 8: Total channels not a multiple of channels per chip
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(127, 32)
    assert "multiple of channels per chip" in str(excinfo.value)

    # Case 9: Another non-multiple case
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(65, 16)
    assert "multiple of channels per chip" in str(excinfo.value)
