import numpy as np
import pytest

from trodes_to_nwb.spike_gadgets_raw_io import (
    InsertedMemmap,
    SpikeGadgetsRawIO,
    SpikeGadgetsRawIOPartial,
)
from trodes_to_nwb.tests.utils import data_path

# --- Fixture for SpikeGadgetsRawIO instance ---

# Define the expected path to the test file
# Adjust filename if necessary based on your test_data content
TEST_REC_FILE = data_path / "20230622_sample_01_a.rec"


@pytest.fixture(scope="module")
def raw_io():
    """Fixture to create and parse header for SpikeGadgetsRawIO instance."""
    if not TEST_REC_FILE.exists():
        pytest.skip(
            f"Test data file not found: {TEST_REC_FILE}"
        )  # Skip if file missing

    io = SpikeGadgetsRawIO(filename=str(TEST_REC_FILE))
    io._parse_header()  # Parse header once for the fixture
    # Add cleanup if necessary, e.g., io.close() if implemented
    return io


@pytest.fixture(scope="module")
def raw_io_interpolated():
    """Fixture for SpikeGadgetsRawIO instance with interpolation enabled."""
    if not TEST_REC_FILE.exists():
        pytest.skip(f"Test data file not found: {TEST_REC_FILE}")

    io = SpikeGadgetsRawIO(
        filename=str(TEST_REC_FILE), interpolate_dropped_packets=True
    )
    io._parse_header()
    # Trigger interpolation index calculation and memmap wrapping
    # This is needed before tests that rely on the interpolated state
    _ = io.get_analogsignal_timestamps(0, 10)  # Read a small chunk to trigger it
    return io


# --- Test Functions ---


def test_io_initialization(raw_io):
    """Test if SpikeGadgetsRawIO initializes correctly."""
    assert raw_io is not None
    assert raw_io.filename == str(TEST_REC_FILE)


def test_parse_header_properties(raw_io):
    """Test properties derived from the header parsing."""
    # These expected values depend on the specific TEST_REC_FILE header
    # You may need to adjust them after inspecting your test file's header
    expected_sampling_rate = 30000.0
    expected_num_ephys_channels = 32  # Example from file
    expected_stream_ids = [
        "ECU_digital",
        "ECU_analog",
        "Controller_DIO_digital",
        "trodes",
    ]  # Example from file

    assert raw_io._sampling_rate == expected_sampling_rate
    assert raw_io.header is not None
    assert "signal_streams" in raw_io.header
    assert "signal_channels" in raw_io.header

    # Check stream IDs
    actual_stream_ids = raw_io.header["signal_streams"]["id"].tolist()
    # Use set comparison to ignore order if necessary
    assert set(actual_stream_ids) == set(expected_stream_ids)

    # Check total ephys channels ('trodes' stream)
    trodes_channels = raw_io.header["signal_channels"][
        raw_io.header["signal_channels"]["stream_id"] == "trodes"
    ]
    assert len(trodes_channels) == expected_num_ephys_channels

    # Check if sysClock is present (depends on test file)
    # This test assumes the sample file *does* have SysClock based on other code parts
    assert raw_io.sysClock_byte is not False
    assert isinstance(raw_io.sysClock_byte, int)


def test_get_signal_size(raw_io):
    """Test retrieving the signal size for non-interpolated data."""
    # Assuming block 0, segment 0
    signal_size = raw_io._get_signal_size(
        0, 0, stream_index=0
    )  # Check size for first stream
    assert isinstance(signal_size, int)
    assert signal_size > 0  # Should have some data
    # Store original size for comparison with interpolated size later if needed
    # Note: This requires raw_io fixture to have higher scope or be session-scoped if used across tests
    # For simplicity, we'll recalculate in the interpolation test


def test_get_segment_times(raw_io):
    """Test segment start and stop times."""
    # Assuming block 0, segment 0
    t_start = raw_io._segment_t_start(0, 0)
    t_stop = raw_io._segment_t_stop(0, 0)
    # Get size directly from underlying memmap for comparison
    # Accessing internal _raw_memmap is okay in tests for verification
    original_signal_size = raw_io._raw_memmap.shape[0]

    assert t_start == 0.0
    assert t_stop == pytest.approx(original_signal_size / raw_io._sampling_rate)
    assert t_stop > t_start


# --- Timestamp Tests ---


def test_get_analogsignal_timestamps(raw_io):
    """Test reading a slice of Trodes timestamps without interpolation."""
    n_samples_to_read = 100
    timestamps = raw_io.get_analogsignal_timestamps(0, n_samples_to_read)

    assert isinstance(timestamps, np.ndarray)
    assert timestamps.dtype == np.uint32
    assert timestamps.shape == (n_samples_to_read,)
    # Check if timestamps are generally increasing (might have drops if not interpolated)
    # Allow for single drops (diff == 2) or stays same (diff == 0) or increases (diff == 1)
    diffs = np.diff(timestamps)
    assert np.all(diffs >= 0)  # Should not decrease


def test_get_sys_clock(raw_io):
    """Test reading a slice of SysClock timestamps (if available)."""
    if not raw_io.sysClock_byte:
        pytest.skip("SysClock not available in this test file.")

    n_samples_to_read = 100
    sys_clock = raw_io.get_sys_clock(0, n_samples_to_read)

    assert isinstance(sys_clock, np.ndarray)
    assert sys_clock.dtype == np.int64
    assert sys_clock.shape == (n_samples_to_read,)
    # Check if timestamps are generally increasing
    assert np.all(np.diff(sys_clock) >= 0)


def test_get_regressed_systime(raw_io):
    """Test calculating system time via regression (if sysClock available)."""

    n_samples_to_read = 100
    # Ensure calculation runs without error
    regressed_time_s = raw_io.get_regressed_systime(0, n_samples_to_read)

    assert isinstance(regressed_time_s, np.ndarray)
    assert regressed_time_s.dtype == np.float64
    assert regressed_time_s.shape == (n_samples_to_read,)
    # Check if times are plausible (e.g., positive and increasing)
    assert np.all(regressed_time_s >= 0)
    assert np.all(
        np.diff(regressed_time_s) >= -1e-9
    )  # Allow for small float inaccuracies


def test_get_systime_from_trodes_timestamps(raw_io):
    """Test calculating system time via sampling rate."""
    n_samples_to_read = 100
    systime_s = raw_io.get_systime_from_trodes_timestamps(0, n_samples_to_read)

    assert isinstance(systime_s, np.ndarray)
    assert systime_s.dtype == np.float64
    assert systime_s.shape == (n_samples_to_read,)
    # Check if times are plausible (e.g., positive and increasing)
    assert np.all(systime_s >= 0)
    assert np.all(np.diff(systime_s) >= -1e-9)  # Allow for small float inaccuracies

    # Check if start time roughly matches systemTimeAtCreation from header
    expected_start_s = float(raw_io.system_time_at_creation) / 1000.0
    assert systime_s[0] == pytest.approx(
        expected_start_s, abs=1.0
    )  # Allow some tolerance


# --- Data Chunk Tests ---


def test_get_ephys_chunk(raw_io):
    """Test reading a chunk of ephys ('trodes') data."""
    n_samples_to_read = 100
    stream_id = "trodes"
    stream_index = raw_io.get_stream_index_from_id(stream_id)
    num_channels = raw_io.header["signal_channels"][
        raw_io.header["signal_channels"]["stream_id"] == stream_id
    ].shape[0]

    # Read all channels for the stream
    data_chunk = raw_io._get_analogsignal_chunk(
        0, 0, 0, n_samples_to_read, stream_index, channel_indexes=None
    )

    assert isinstance(data_chunk, np.ndarray)
    assert data_chunk.dtype == np.int16
    assert data_chunk.shape == (n_samples_to_read, num_channels)

    # Read a subset of channels
    channel_subset = [0, 5, 10]  # Example subset
    data_chunk_subset = raw_io._get_analogsignal_chunk(
        0, 0, 0, n_samples_to_read, stream_index, channel_indexes=channel_subset
    )
    assert data_chunk_subset.shape == (n_samples_to_read, len(channel_subset))


def test_get_analog_chunk(raw_io):
    """Test reading a chunk of 'ECU_analog' data (includes multiplexed)."""
    n_samples_to_read = 100
    stream_id = "ECU_analog"
    stream_index = raw_io.get_stream_index_from_id(stream_id)
    # Calculate expected number of channels (analog + multiplexed)
    analog_channels = raw_io.header["signal_channels"][
        raw_io.header["signal_channels"]["stream_id"] == stream_id
    ]
    num_analog = len(analog_channels)
    num_multiplexed = len(raw_io.multiplexed_channel_xml)
    expected_num_channels = num_analog + num_multiplexed

    # Read all channels for the stream
    data_chunk = raw_io._get_analogsignal_chunk(
        0, 0, 0, n_samples_to_read, stream_index, channel_indexes=None
    )

    assert isinstance(data_chunk, np.ndarray)
    assert data_chunk.dtype == np.int16
    assert data_chunk.shape == (n_samples_to_read, expected_num_channels)


# --- DIO Test ---


def test_get_digitalsignal(raw_io):
    """Test reading a digital IO channel."""
    # Find a digital stream and channel ID from the header
    digital_streams = raw_io.header["signal_streams"][
        np.char.endswith(raw_io.header["signal_streams"]["id"], "_digital")
    ]
    if len(digital_streams) == 0:
        pytest.skip("No digital streams found in test file header.")

    stream_id = digital_streams[0]["id"]  # Use the first digital stream found
    digital_channels = raw_io.header["signal_channels"][
        raw_io.header["signal_channels"]["stream_id"] == stream_id
    ]
    if len(digital_channels) == 0:
        pytest.skip(f"No channels found in digital stream '{stream_id}'.")

    channel_id = digital_channels[0]["id"]  # Use the first channel found

    timestamps, states = raw_io.get_digitalsignal(stream_id, channel_id)

    assert isinstance(timestamps, np.ndarray)
    assert isinstance(states, np.ndarray)
    assert timestamps.dtype == np.float64  # Should be seconds
    assert states.dtype == np.uint8
    assert timestamps.shape == states.shape  # One state per timestamp event
    assert len(timestamps) > 0  # Expect some events
    assert np.all(np.isin(states, [0, 1]))  # States should only be 0 or 1


# --- Interpolation Tests ---


def test_interpolation_init(raw_io_interpolated):
    """Test initialization with interpolation enabled."""
    assert raw_io_interpolated.interpolate_dropped_packets is True


def test_interpolation_memmap_wrapper(raw_io_interpolated):
    """Test if interpolation wraps the memmap after timestamp access."""
    # Fixture already triggered timestamp access
    assert isinstance(raw_io_interpolated._raw_memmap, InsertedMemmap)


def test_interpolation_signal_size(raw_io, raw_io_interpolated):
    """Test that signal size increases correctly with interpolation."""
    # Get original size
    original_size = raw_io._raw_memmap.shape[0]

    # Get interpolated size (interpolation already triggered in fixture)
    interpolated_io = raw_io_interpolated
    interpolated_size = interpolated_io._get_signal_size(
        0, 0, stream_index=0
    )  # Use correct indices

    # Check the number of insertions made
    num_insertions = len(interpolated_io._raw_memmap.inserted_locations)

    # The new size should be the original size plus the number of insertions
    assert interpolated_size == original_size + num_insertions
    # Ensure size increased if drops were actually found (depends on test file)
    if (
        interpolated_io.interpolate_index is not None
        and len(interpolated_io.interpolate_index) > 0
    ):
        assert num_insertions > 0
        assert interpolated_size > original_size
    else:
        # If no drops were found, size should be the same
        assert num_insertions == 0
        assert interpolated_size == original_size


def test_interpolation_read_timestamps(raw_io_interpolated):
    """Test reading timestamps with interpolation enabled."""
    n_samples_to_read = 100
    interpolated_size = raw_io_interpolated._get_signal_size(0, 0, stream_index=0)
    n_samples_to_read = min(
        n_samples_to_read, interpolated_size
    )  # Adjust if file is too short

    timestamps = raw_io_interpolated.get_analogsignal_timestamps(0, n_samples_to_read)

    assert isinstance(timestamps, np.ndarray)
    assert timestamps.dtype == np.uint32
    # Shape should match the number of samples requested from the *interpolated* size
    assert timestamps.shape == (n_samples_to_read,)

    # Check if timestamps are now strictly increasing (assuming interpolation fixes single drops)
    # Note: This assumes the interpolation logic correctly handles the drops.
    # If the test file has no drops, diffs >= 0 is still expected.
    diffs = np.diff(timestamps)
    if len(raw_io_interpolated._raw_memmap.inserted_locations) > 0:
        # If drops were interpolated, expect strictly increasing or same timestamp
        # (depending on how interpolation is done - adding 1 makes it increase)
        assert np.all(diffs >= 0)
        # We might expect diffs == 1 mostly, but could be 0 if original had duplicates
    else:
        # If no drops, same check as non-interpolated
        assert np.all(diffs >= 0)


def test_interpolation_read_data_chunk(raw_io_interpolated):
    """Test reading a data chunk with interpolation enabled."""
    n_samples_to_read = 100
    stream_id = "trodes"
    stream_index = raw_io_interpolated.get_stream_index_from_id(stream_id)
    num_channels = raw_io_interpolated.header["signal_channels"][
        raw_io_interpolated.header["signal_channels"]["stream_id"] == stream_id
    ].shape[0]

    interpolated_size = raw_io_interpolated._get_signal_size(
        0, 0, stream_index=stream_index
    )
    n_samples_to_read = min(
        n_samples_to_read, interpolated_size
    )  # Adjust if file is too short

    # Read data chunk
    data_chunk = raw_io_interpolated._get_analogsignal_chunk(
        0, 0, 0, n_samples_to_read, stream_index, channel_indexes=None
    )

    assert isinstance(data_chunk, np.ndarray)
    assert data_chunk.dtype == np.int16
    # Shape should match the number of samples requested from the *interpolated* size
    assert data_chunk.shape == (n_samples_to_read, num_channels)


def test_produce_ephys_channel_ids():
    """
    Test the static method _produce_ephys_channel_ids for various configurations.

    Verifies the correct interleaved channel ID order is generated for different
    total channel counts and channels per chip, including edge cases and
    error handling.
    """
    # Case 1: Standard 128 channels, 32 per chip (4 chips)
    n_total_1 = 128
    n_per_chip_1 = 32
    full_expected_1 = []
    for k in range(n_per_chip_1):
        full_expected_1.extend(
            [k + i * n_per_chip_1 for i in range(n_total_1 // n_per_chip_1)]
        )
    result_1 = SpikeGadgetsRawIO._produce_ephys_channel_ids(
        n_total_1, n_total_1, n_per_chip_1
    )
    assert result_1 == full_expected_1
    assert len(result_1) == n_total_1

    # Case 2: Smaller case - 64 channels, 32 per chip (2 chips)
    n_total_2 = 64
    n_per_chip_2 = 32
    full_expected_2 = []
    for k in range(n_per_chip_2):
        full_expected_2.extend(
            [k + i * n_per_chip_2 for i in range(n_total_2 // n_per_chip_2)]
        )
    result_2 = SpikeGadgetsRawIO._produce_ephys_channel_ids(
        n_total_2, n_total_2, n_per_chip_2
    )
    assert result_2 == full_expected_2
    assert len(result_2) == n_total_2

    # Case 3: Different chip size - 64 channels, 16 per chip (4 chips)
    n_total_3 = 64
    n_per_chip_3 = 16
    full_expected_3 = []
    for k in range(n_per_chip_3):
        full_expected_3.extend(
            [k + i * n_per_chip_3 for i in range(n_total_3 // n_per_chip_3)]
        )

    result_3 = SpikeGadgetsRawIO._produce_ephys_channel_ids(
        n_total_3, n_total_3, n_per_chip_3
    )
    assert result_3 == full_expected_3
    assert len(result_3) == n_total_3

    # Case 4: Single chip - 32 channels, 32 per chip (1 chip)
    n_total_4 = 32
    n_per_chip_4 = 32
    expected_4 = list(range(32))  # Should just be 0, 1, 2, ..., 31
    result_4 = SpikeGadgetsRawIO._produce_ephys_channel_ids(
        n_total_4, n_total_4, n_per_chip_4
    )
    assert result_4 == expected_4
    assert len(result_4) == n_total_4

    # case 5: Not all channels recorded
    n_total_5 = 128
    n_recorded_5 = 127
    n_per_chip_5 = 32
    missing_hw_channel = 2

    full_expected_5 = []
    for k in range(n_per_chip_5):
        full_expected_5.extend(
            [k + i * n_per_chip_5 for i in range(n_total_5 // n_per_chip_5)]
        )
    full_expected_5 = [x for x in full_expected_5 if x != missing_hw_channel]
    hw_channels_recorded_5 = [
        str(x) for x in np.arange(n_total_5) if x != missing_hw_channel
    ]
    result_5 = SpikeGadgetsRawIO._produce_ephys_channel_ids(
        n_total_5,
        n_recorded_5,
        n_per_chip_5,
        hw_channels_recorded=hw_channels_recorded_5,
    )
    assert result_5 == full_expected_5
    assert len(result_5) == n_recorded_5

    # --- Edge Cases ---
    result_6 = SpikeGadgetsRawIO._produce_ephys_channel_ids(0, 0, 32)
    assert result_6 == []
    result_7 = SpikeGadgetsRawIO._produce_ephys_channel_ids(128, 128, 0)
    assert result_7 == []
    result_8 = SpikeGadgetsRawIO._produce_ephys_channel_ids(0, 0, 0)
    assert result_8 == []

    # --- Error Cases ---
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(127, 127, 32)
    assert "multiple of channels per chip" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(65, 65, 16)
    assert "multiple of channels per chip" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(
            64,
            63,
            16,
        )
    assert "hw_channels_recorded must be provided" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        SpikeGadgetsRawIO._produce_ephys_channel_ids(64, 63, 16, ["1", "2", "3"])
    assert "hw_channels_recorded must be provided" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Tests for SpikeGadgetsRawIOPartial.get_systime_from_trodes_timestamps
# ---------------------------------------------------------------------------


class _MockPartial:
    """
    Minimal stand-in for SpikeGadgetsRawIOPartial used to unit-test the
    overridden ``get_systime_from_trodes_timestamps`` method without needing
    a real .rec file.
    """

    def __init__(self, timestamps_uint32, start_index, sampling_rate, system_time_ms):
        self._timestamps = timestamps_uint32.astype(np.uint32)
        self.start_index = start_index
        self._sampling_rate = float(sampling_rate)
        self.system_time_at_creation = str(system_time_ms)

    def get_analogsignal_timestamps(self, i_start, i_stop):
        if i_stop is None:
            i_stop = len(self._timestamps)
        return self._timestamps[i_start:i_stop]

    # Directly borrow the implementation from the real class
    get_systime_from_trodes_timestamps = (
        SpikeGadgetsRawIOPartial.get_systime_from_trodes_timestamps.__wrapped__
    )


def test_partial_systime_first_partial_continuous():
    """Timestamps from the first partial (start_index=0) equal the base result."""
    sampling_rate = 30_000.0
    system_time_ms = 1_575_309_000_000  # arbitrary 2019 Unix ms
    n_samples = 100
    # Timestamps start at 0 and increment by 1 per sample
    ts = np.arange(n_samples, dtype=np.uint32)

    mock = _MockPartial(ts, start_index=0, sampling_rate=sampling_rate,
                        system_time_ms=system_time_ms)
    result = mock.get_systime_from_trodes_timestamps(0, n_samples)

    expected_start = system_time_ms / 1000.0
    assert result[0] == pytest.approx(expected_start, abs=1e-6)
    # Monotonically increasing
    assert np.all(np.diff(result) > 0)


def test_partial_systime_later_partial_continuous():
    """
    Timestamps from a later partial (start_index > 0) must be offset by
    start_index / sampling_rate rather than resetting to system_time_at_creation.
    """
    sampling_rate = 30_000.0
    system_time_ms = 1_575_309_000_000
    chunk_size = 54_000_000   # 30 min at 30 kHz
    start_index = chunk_size  # second partial starts here

    # Timestamps within this partial start at start_index and increment by 1
    ts = np.arange(start_index, start_index + 100, dtype=np.uint32)

    mock = _MockPartial(ts, start_index=start_index, sampling_rate=sampling_rate,
                        system_time_ms=system_time_ms)
    result = mock.get_systime_from_trodes_timestamps(0, 100)

    expected_start = system_time_ms / 1000.0 + start_index / sampling_rate
    # Should NOT reset to system_time_at_creation but be offset correctly
    assert result[0] == pytest.approx(expected_start, abs=1e-6)
    assert np.all(np.diff(result) > 0)


def test_partials_concatenate_monotonically():
    """
    Timestamps from two consecutive partials must form a continuous, monotonic
    sequence when concatenated, reproducing the fix for the issue where
    timestamps reset every 30 minutes.
    """
    sampling_rate = 30_000.0
    system_time_ms = 1_575_309_000_000
    chunk_size = 1000  # small value for the test

    # Partial 1: samples 0 .. chunk_size-1
    ts1 = np.arange(0, chunk_size, dtype=np.uint32)
    partial1 = _MockPartial(ts1, start_index=0, sampling_rate=sampling_rate,
                            system_time_ms=system_time_ms)
    times1 = partial1.get_systime_from_trodes_timestamps(0, chunk_size)

    # Partial 2: samples chunk_size .. 2*chunk_size-1
    ts2 = np.arange(chunk_size, 2 * chunk_size, dtype=np.uint32)
    partial2 = _MockPartial(ts2, start_index=chunk_size, sampling_rate=sampling_rate,
                            system_time_ms=system_time_ms)
    times2 = partial2.get_systime_from_trodes_timestamps(0, chunk_size)

    combined = np.concatenate([times1, times2])
    diffs = np.diff(combined)
    assert np.all(diffs > 0), (
        "Concatenated timestamps from two partials must be monotonically increasing. "
        "This catches the regression where timestamps reset at the 30-min split boundary."
    )

