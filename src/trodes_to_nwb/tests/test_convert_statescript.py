import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from trodes_to_nwb.convert_statescript import (
    StateScriptLogProcessor,
    _interpret_DIO_mask,
    _parse_int,
    parse_statescript_line,
    parse_ts_int_int,
    parse_ts_str,
    parse_ts_str_equals_int,
    parse_ts_str_int,
)

# --- Fixtures ---


@pytest.fixture(scope="module")
def sample_log_content():
    """Provides sample log content for general testing."""
    return """# Test log started
76504 0 0
76566 center_poke
76566 65536 0
100078 counter_handlePoke = 1
100078 4 0
100559 LEFT_PORT 1
Executing this line without timestamp
115030 center_poke
115030 65536 0
115040 0 0
# Test log ended
"""


@pytest.fixture(scope="module")
def empty_log_content():
    """Provides empty log content."""
    return ""


@pytest.fixture(scope="module")
def comment_only_log_content():
    """Provides log content with only comments and whitespace."""
    return """# Start
# Middle line

# End
"""


@pytest.fixture
def processor(sample_log_content):
    """Provides a processor instance initialized with standard sample content."""
    return StateScriptLogProcessor(sample_log_content, source_info="from string")


@pytest.fixture
def empty_processor(empty_log_content):
    """Provides a processor instance initialized with empty content."""
    return StateScriptLogProcessor(empty_log_content, source_info="empty string")


@pytest.fixture
def comment_only_processor(comment_only_log_content):
    """Provides a processor instance initialized with only comments."""
    return StateScriptLogProcessor(
        comment_only_log_content, source_info="comments only string"
    )


@pytest.fixture(scope="module")
def external_times():
    """Provides sample external times for offset calculation tests."""
    # These correspond to the '65536 0' events (ts_int_int) in sample_log_content
    # 76566 ms -> 76.566 s
    # 115030 ms -> 115.030 s
    # Let's assume a base time (e.g., Unix timestamp) for the external system
    base_time = 1678880000.0
    return np.array([base_time + 76.566, base_time + 115.030])


@pytest.fixture(scope="module")
def external_times_for_str_int():
    """Provides sample external times for offset calculation tests using ts_str_int."""
    # These correspond to the 'LEFT_PORT 1' event in sample_log_content
    # 100559 ms -> 100.559 s
    base_time = 1678880000.0
    return np.array(
        [
            base_time + 100.559,
            base_time + 110.0,
            base_time + 120.0,
            base_time + 130.0,
        ]
    )


@pytest.fixture
def temp_log_file(sample_log_content):
    """Creates a temporary log file with standard content and yields its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".stateScriptLog", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(sample_log_content)
        tmp_file_path = tmp_file.name
    yield pathlib.Path(tmp_file_path)
    os.remove(tmp_file_path)


# --- Tests for Level 0 Helpers ---


def test_parse_int():
    """Test the _parse_int helper function."""
    assert _parse_int("123") == 123
    assert _parse_int("-45") == -45
    assert _parse_int("0") == 0
    assert _parse_int("abc") is None
    assert _parse_int("12.3") is None
    assert _parse_int("") is None


def test_interpret_dio_mask():
    """Test the _interpret_DIO_mask function."""
    assert _interpret_DIO_mask(9, max_DIOs=8) == [1, 4]  # Binary 1001
    assert _interpret_DIO_mask(0) == []
    assert _interpret_DIO_mask(None) == []
    assert _interpret_DIO_mask(pd.NA) == []
    assert _interpret_DIO_mask(1) == [1]
    assert _interpret_DIO_mask(65536, max_DIOs=32) == [17]  # 2^16
    assert _interpret_DIO_mask(65535, max_DIOs=16) == list(
        range(1, 17)
    )  # All 16 bits set
    assert _interpret_DIO_mask(65535, max_DIOs=32) == list(
        range(1, 17)
    )  # Check max_DIOs limit
    assert _interpret_DIO_mask("abc") == []  # Invalid input type


# --- Tests for Level 1 Parsers ---


def test_parse_ts_int_int():
    """Test parse_ts_int_int directly."""
    parts = ["8386500", "0", "0"]
    expected = {
        "type": "ts_int_int",
        "timestamp": 8386500,
        "value1": 0,
        "value2": 0,
    }
    assert parse_ts_int_int(parts) == expected

    parts_wrong_len = ["123", "0"]
    assert parse_ts_int_int(parts_wrong_len) is None

    parts_not_int = ["123", "abc", "0"]
    assert parse_ts_int_int(parts_not_int) is None

    parts_float = ["123", "4.5", "0"]
    assert parse_ts_int_int(parts_float) is None


def test_parse_ts_str_int():
    """Test parse_ts_str_int directly."""
    parts = ["8386500", "DOWN", "3"]
    expected = {
        "type": "ts_str_int",
        "timestamp": 8386500,
        "text": "DOWN",
        "value": 3,
    }
    assert parse_ts_str_int(parts) == expected

    parts_wrong_len = ["123", "UP"]
    assert parse_ts_str_int(parts_wrong_len) is None

    # This should be parsed by parse_ts_int_int due to precedence,
    # so parse_ts_str_int should return None here because str part is int.
    parts_str_is_int = ["123", "456", "789"]
    assert parse_ts_str_int(parts_str_is_int) is None

    parts_val_not_int = ["123", "UP", "abc"]
    assert parse_ts_str_int(parts_val_not_int) is None


def test_parse_ts_str_equals_int():
    """Test parse_ts_str_equals_int directly.
    NOTE: The code only handles a single word before '='.
    """
    parts = ["100078", "counter_handlePoke", "=", "1"]
    expected = {
        "type": "ts_str_equals_int",
        "timestamp": 100078,  # Raw timestamp key
        "text": "counter_handlePoke",  # Correctly uses parts[1]
        "value": 1,
    }
    assert parse_ts_str_equals_int(parts) == expected

    # This case is NOT handled by the current implementation (len(parts) != 4)
    parts_multi_word = ["3610855", "total", "rewards", "=", "70"]
    assert parse_ts_str_equals_int(parts_multi_word) is None

    parts_wrong_len = ["123", "=", "1"]
    assert parse_ts_str_equals_int(parts_wrong_len) is None

    parts_no_equals = ["123", "text", "1"]  # len=3 != 4
    assert parse_ts_str_equals_int(parts_no_equals) is None

    parts_wrong_equals_pos = ["123", "text", "1", "="]  # '=' is parts[3], not parts[2]
    assert parse_ts_str_equals_int(parts_wrong_equals_pos) is None

    parts_val_not_int = ["123", "text", "=", "abc"]
    assert parse_ts_str_equals_int(parts_val_not_int) is None


def test_parse_ts_str():
    """Test parse_ts_str directly."""
    parts = ["76566", "center_poke"]
    expected = {
        "type": "ts_str",
        "timestamp": 76566,
        "text": "center_poke",
    }
    assert parse_ts_str(parts) == expected

    parts_multi_word = ["1271815", "some", "multi", "word", "event"]
    expected_multi = {
        "type": "ts_str",
        "timestamp": 1271815,
        "text": "some multi word event",
    }
    assert parse_ts_str(parts_multi_word) == expected_multi

    parts_wrong_len = ["123"]
    assert parse_ts_str(parts_wrong_len) is None

    # Second part is int, should fail this parser (handled by ts_int_int or ts_str_int)
    parts_second_is_int = ["123", "456"]
    assert parse_ts_str(parts_second_is_int) is None


# --- Tests for parse_statescript_line (Covers integration and dispatching) ---


def test_parse_statescript_line_dispatching():
    """Test parse_statescript_line dispatching for various line types."""
    lines_expected = [
        ("8386500 0 0", "ts_int_int", 8386500),
        ("100559 LEFT_PORT 1", "ts_str_int", 100559),
        ("100078 counter_handlePoke = 1", "ts_str_equals_int", 100078),
        ("76566 center_poke", "ts_str", 76566),
        ("Executing trigger function 22", "unknown", None),  # No timestamp
        ("# comment", "comment_or_empty", None),
        ("", "comment_or_empty", None),
        ("   ", "comment_or_empty", None),
        ("123 456 abc", "unknown", None),  # Doesn't fit ts_int_int/ts_str_int/ts_str
        ("123 abc def", "ts_str", 123),  # Fits ts_str
        # Precedence: ts_str_equals_int matches first
        ("456 text = 5", "ts_str_equals_int", 456),
        # Precedence: ts_int_int matches first
        ("8386500 128 512", "ts_int_int", 8386500),
        # Precedence: ts_str_int matches (str 'UP' is not int)
        ("90000 UP 10", "ts_str_int", 90000),
        # Precedence: ts_str matches (str 'some text' is not int)
        ("95000 some text here", "ts_str", 95000),
    ]

    for i, (line, expected_type, expected_ts) in enumerate(lines_expected):
        parsed = parse_statescript_line(line, line_num=i)
        assert parsed["type"] == expected_type, f"Line: {line}"
        assert parsed["raw_line"] == line.strip(), f"Line: {line}"
        assert parsed["line_num"] == i, f"Line: {line}"
        # Check timestamp presence/value based on type
        if expected_type not in ["unknown", "comment_or_empty"]:
            assert "timestamp" in parsed, f"Line: {line}"
            assert parsed["timestamp"] == expected_ts, f"Line: {line}"
        else:
            # Should explicitly contain timestamp: None for these types
            assert parsed.get("timestamp") is None, f"Line: {line}"


# --- Tests for StateScriptLogProcessor ---


def test_init_from_string(processor, sample_log_content):
    """Test initialization from string."""
    assert processor.log_content == sample_log_content
    assert processor.source_description == "from string"
    assert processor.raw_events == []
    assert processor.time_offset is None
    assert processor.processed_events_df is None


def test_init_from_file(temp_log_file, sample_log_content):
    """Test initialization from a file."""
    processor_file = StateScriptLogProcessor.from_file(temp_log_file)
    assert processor_file.log_content == sample_log_content
    assert processor_file.source_description.startswith("from file:")
    assert temp_log_file.name in processor_file.source_description


def test_init_from_file_not_found():
    """Test initialization from a non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        StateScriptLogProcessor.from_file("non_existent_file_qwerty.log")


def test_parse_raw_events(processor, sample_log_content):
    """Test parsing the raw log content into events."""
    events = processor.parse_raw_events()
    assert processor.raw_events is events  # Should store result internally
    assert isinstance(events, list)
    # Count lines in the fixture (includes comments, blanks if any)
    num_lines = len(sample_log_content.strip().splitlines())
    assert len(events) == num_lines

    # Check specific lines based on fixture content
    # Line 0: # Test log started
    assert events[0]["type"] == "comment_or_empty"
    assert events[0]["line_num"] == 0
    assert events[0]["timestamp"] is None
    # Line 1: 76504 0 0
    assert events[1]["type"] == "ts_int_int"
    assert events[1]["timestamp"] == 76504
    assert events[1]["value1"] == 0
    assert events[1]["line_num"] == 1
    assert events[1]["raw_line"] == "76504 0 0"
    # Line 7: Executing this line without timestamp
    assert events[7]["type"] == "unknown"
    assert events[7]["raw_line"] == "Executing this line without timestamp"
    assert events[7]["line_num"] == 7
    assert events[7]["timestamp"] is None
    # Line 11: # Test log ended
    assert events[11]["type"] == "comment_or_empty"
    assert events[11]["line_num"] == 11
    assert events[11]["timestamp"] is None


def test_find_reference_events(processor):
    """Test the internal _find_reference_events method."""
    # Case 1: Find 'ts_str' events ('center_poke' appears twice)
    ref_df_str = processor._find_reference_events(
        event_type="ts_str", conditions={"text": "center_poke"}
    )
    assert isinstance(ref_df_str, pd.DataFrame)
    assert len(ref_df_str) == 2
    # Check raw timestamp column (renamed from 'timestamp' in raw_events)
    pd.testing.assert_series_equal(
        ref_df_str["timestamp"],  # Raw integer timestamp
        pd.Series([76566, 115030], name="timestamp", dtype=int),
        check_names=True,
        check_dtype=True,
    )
    # Check calculated seconds column
    assert "trodes_timestamp_sec" in ref_df_str.columns
    pd.testing.assert_series_equal(
        ref_df_str["trodes_timestamp_sec"],
        pd.Series([76.566, 115.030], name="trodes_timestamp_sec", dtype=float),
        check_names=True,
        check_dtype=True,
    )
    assert ref_df_str["text"].tolist() == ["center_poke", "center_poke"]

    # Case 2: Find 'ts_int_int' events with specific values (appears twice)
    ref_df_int = processor._find_reference_events(
        event_type="ts_int_int", conditions={"value1": 65536, "value2": 0}
    )
    assert len(ref_df_int) == 2
    assert ref_df_int["timestamp"].tolist() == [76566, 115030]
    assert ref_df_int["value1"].tolist() == [65536, 65536]
    assert ref_df_int["value2"].tolist() == [0, 0]
    assert ref_df_int["trodes_timestamp_sec"].tolist() == [76.566, 115.030]

    # Case 3: Find 'ts_str_equals_int' (appears once)
    ref_df_eq = processor._find_reference_events(
        event_type="ts_str_equals_int", conditions={"text": "counter_handlePoke"}
    )
    assert len(ref_df_eq) == 1
    assert ref_df_eq["timestamp"].iloc[0] == 100078
    assert ref_df_eq["text"].iloc[0] == "counter_handlePoke"
    assert ref_df_eq["value"].iloc[0] == 1
    assert ref_df_eq["trodes_timestamp_sec"].iloc[0] == pytest.approx(100.078)

    # Case 4: No matching events found
    ref_df_none = processor._find_reference_events(
        event_type="ts_str", conditions={"text": "nonexistent"}
    )
    assert ref_df_none.empty
    assert isinstance(ref_df_none, pd.DataFrame)  # Should still return DF
    # Check expected columns exist even if empty
    assert "timestamp" in ref_df_none.columns
    assert "trodes_timestamp_sec" in ref_df_none.columns
    assert "text" in ref_df_none.columns  # From conditions

    # Case 5: Ensure processor parses if raw_events is empty
    processor.raw_events = []  # Reset raw events
    assert processor.raw_events == []
    ref_df_reparse = processor._find_reference_events(
        event_type="ts_str", conditions={"text": "center_poke"}
    )
    assert len(processor.raw_events) > 0  # Should have re-parsed
    assert len(ref_df_reparse) == 2  # Should find the events


def test_calculate_time_offset_success(processor, external_times):
    """Test successful time offset calculation."""
    # Use the 'ts_int_int' events matching external_times fixture
    offset = processor.calculate_time_offset(
        external_reference_times=external_times,
        log_event_type="ts_int_int",
        # Use the keys from the raw parsed dict ('value1', 'value2')
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,  # Use both available matching events
    )
    assert offset is not None
    assert processor.time_offset == offset  # Check internal storage
    # Expected offset = external_base_time = 1678880000.0
    # external_times[0] = base + 76.566; log_times_sec[0] = 76.566
    # offset = external - log = base
    assert offset == pytest.approx(1678880000.0)


def test_calculate_time_offset_fail_not_enough_log(
    processor, external_times_for_str_int
):
    """Test offset calculation failure due to insufficient log events."""
    # 'LEFT_PORT 1' only appears once in the log, but default check_n_events=4
    offset = processor.calculate_time_offset(
        external_reference_times=external_times_for_str_int,  # Has 4 times
        log_event_type="ts_str_int",
        log_event_conditions={"text": "LEFT_PORT", "value": 1},
        # check_n_events=4, # Default
    )
    assert offset is None
    assert processor.time_offset is None  # Should remain None


def test_calculate_time_offset_fail_not_enough_external(processor):
    """Test offset calculation failure due to insufficient external times."""
    # Log has 2 '65536 0' events, provide only 1 external time, default check=4
    offset = processor.calculate_time_offset(
        external_reference_times=np.array([1678880076.566]),  # Only 1 time
        log_event_type="ts_int_int",
        log_event_conditions={"value1": 65536, "value2": 0},
        # check_n_events=4, # Default
    )
    assert offset is None
    assert processor.time_offset is None

    # Test again with check_n_events=2 (should still fail, need 2 external)
    offset_check2 = processor.calculate_time_offset(
        external_reference_times=np.array([1678880076.566]),  # Only 1 time
        log_event_type="ts_int_int",
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,
    )
    assert offset_check2 is None
    assert processor.time_offset is None


def test_calculate_time_offset_fail_mismatch(processor, external_times):
    """Test offset calculation failure due to exceeding mismatch threshold."""
    # Shift external times enough to exceed default threshold (0.1) over 2 events
    # Shift each by 0.06 -> total diff = 0.06 + 0.06 = 0.12 > 0.1
    shifted_external_times = external_times + 0.06
    offset = processor.calculate_time_offset(
        external_reference_times=shifted_external_times,
        log_event_type="ts_int_int",
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,
        match_threshold=0.1,  # Explicitly set default for clarity
    )
    assert offset is None
    assert processor.time_offset is None


def test_get_events_dataframe_defaults(processor):
    """Test default behavior: exclude comments/unknown, no offset applied yet."""
    df = processor.get_events_dataframe(apply_offset=False)
    assert processor.processed_events_df is df  # Check internal storage
    assert isinstance(df, pd.DataFrame)
    # Expected: 12 lines total - 2 comments - 1 unknown = 9 valid events
    assert len(df) == 9
    assert df.index.name == "line_num"  # Index should be line_num

    # --- Check Columns ---
    assert "raw_line" in df.columns
    assert "type" in df.columns
    assert "trodes_timestamp" in df.columns
    assert "trodes_timestamp_sec" in df.columns
    assert "text" in df.columns
    assert "value" in df.columns
    assert "active_DIO_inputs_bitmask" in df.columns
    assert "active_DIO_outputs_bitmask" in df.columns
    assert "active_DIO_inputs" in df.columns  # List column
    assert "active_DIO_outputs" in df.columns  # List column
    assert "timestamp_sync" not in df.columns  # Offset not applied

    # --- Check Content and Types (spot check first few rows) ---
    # Row index corresponds to line_num
    # Line 1: 76504 0 0 (type: ts_int_int) -> line_num 1
    assert df.loc[1, "type"] == "ts_int_int"
    assert df.loc[1, "raw_line"] == "76504 0 0"
    assert df.loc[1, "trodes_timestamp"] == 76504
    assert df.loc[1, "trodes_timestamp_sec"] == pytest.approx(76.504)
    assert pd.isna(df.loc[1, "text"])
    assert pd.isna(df.loc[1, "value"])
    assert df.loc[1, "active_DIO_inputs_bitmask"] == 0
    assert df.loc[1, "active_DIO_outputs_bitmask"] == 0
    assert df.loc[1, "active_DIO_inputs"] == []
    assert df.loc[1, "active_DIO_outputs"] == []

    # Line 2: 76566 center_poke (type: ts_str) -> line_num 2
    assert df.loc[2, "type"] == "ts_str"
    assert df.loc[2, "trodes_timestamp"] == 76566
    assert df.loc[2, "text"] == "center_poke"
    assert pd.isna(df.loc[2, "value"])
    assert pd.isna(df.loc[2, "active_DIO_inputs_bitmask"])
    assert pd.isna(df.loc[2, "active_DIO_outputs_bitmask"])
    assert df.loc[2, "active_DIO_inputs"] == []  # Should be empty list from NA mask
    assert df.loc[2, "active_DIO_outputs"] == []  # Should be empty list from NA mask

    # Line 3: 76566 65536 0 (type: ts_int_int) -> line_num 3
    assert df.loc[3, "type"] == "ts_int_int"
    assert df.loc[3, "trodes_timestamp"] == 76566
    assert df.loc[3, "active_DIO_inputs_bitmask"] == 65536  # DIO 17
    assert df.loc[3, "active_DIO_outputs_bitmask"] == 0
    assert df.loc[3, "active_DIO_inputs"] == [17]  # Check interpretation
    assert df.loc[3, "active_DIO_outputs"] == []

    # Line 4: 100078 counter_handlePoke = 1 (type: ts_str_equals_int) -> line_num 4
    assert df.loc[4, "type"] == "ts_str_equals_int"
    assert df.loc[4, "trodes_timestamp"] == 100078
    assert df.loc[4, "text"] == "counter_handlePoke"
    assert df.loc[4, "value"] == 1
    assert pd.isna(df.loc[4, "active_DIO_inputs_bitmask"])

    # Line 6: 100559 LEFT_PORT 1 (type: ts_str_int) -> line_num 6
    assert df.loc[6, "type"] == "ts_str_int"
    assert df.loc[6, "trodes_timestamp"] == 100559
    assert df.loc[6, "text"] == "LEFT_PORT"
    assert df.loc[6, "value"] == 1
    assert pd.isna(df.loc[6, "active_DIO_inputs_bitmask"])

    # --- Check Dtypes ---
    assert df["trodes_timestamp"].dtype == pd.Int64Dtype()  # Nullable int
    assert df["trodes_timestamp_sec"].dtype == "float64"
    assert df["text"].dtype == "object"  # String/mixed
    assert df["value"].dtype == pd.Int64Dtype()
    assert df["active_DIO_inputs_bitmask"].dtype == pd.Int64Dtype()
    assert df["active_DIO_outputs_bitmask"].dtype == pd.Int64Dtype()
    assert df["active_DIO_inputs"].dtype == "object"  # List type
    assert df["active_DIO_outputs"].dtype == "object"  # List type


def test_get_events_dataframe_include_all(processor, sample_log_content):
    """Test including comments and unknown lines."""
    df = processor.get_events_dataframe(
        apply_offset=False, exclude_comments_unknown=False
    )
    assert isinstance(df, pd.DataFrame)
    num_lines = len(sample_log_content.strip().splitlines())
    assert len(df) == num_lines  # All lines included (12)
    assert df.index.name == "line_num"

    # Check specific lines
    # Line 0: Comment
    assert df.loc[0, "type"] == "comment_or_empty"
    assert df.loc[0, "raw_line"] == "# Test log started"
    assert pd.isna(df.loc[0, "trodes_timestamp"])  # Should be NA (Int64Dtype)
    assert np.isnan(df.loc[0, "trodes_timestamp_sec"])  # Should be NaN (float)
    assert pd.isna(df.loc[0, "text"])  # Should be NA
    assert df.loc[0, "active_DIO_inputs"] == []  # Should be empty list for comment

    # Line 7: Unknown
    assert df.loc[7, "type"] == "unknown"
    assert df.loc[7, "raw_line"] == "Executing this line without timestamp"
    assert pd.isna(df.loc[7, "trodes_timestamp"])
    assert np.isnan(df.loc[7, "trodes_timestamp_sec"])
    assert pd.isna(df.loc[7, "text"])
    assert df.loc[7, "active_DIO_inputs"] == []

    # Line 11: Comment
    assert df.loc[11, "type"] == "comment_or_empty"
    assert df.loc[11, "raw_line"] == "# Test log ended"
    assert pd.isna(df.loc[11, "trodes_timestamp"])

    # Check a valid line still looks right
    assert df.loc[1, "type"] == "ts_int_int"
    assert df.loc[1, "trodes_timestamp"] == 76504


def test_get_events_dataframe_with_offset(processor):
    """Test applying offset and check sync timestamp calculation."""
    # Simulate successful offset calculation
    test_offset = 1678880000.0
    processor.time_offset = test_offset
    df = processor.get_events_dataframe(apply_offset=True)  # Default exclude=True
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9  # Excludes comments/unknown
    assert df.index.name == "line_num"
    assert "timestamp_sync" in df.columns
    assert df["timestamp_sync"].dtype == "float64"

    # Check calculation for a few events
    # Line 1: 76504 ms
    expected_sync_1 = (76504 / 1000.0) + test_offset
    assert df.loc[1, "timestamp_sync"] == pytest.approx(expected_sync_1)

    # Line 3: 76566 ms
    expected_sync_3 = (76566 / 1000.0) + test_offset
    assert df.loc[3, "timestamp_sync"] == pytest.approx(expected_sync_3)

    # Line 9: 115030 ms
    expected_sync_9 = (115030 / 1000.0) + test_offset
    assert df.loc[9, "timestamp_sync"] == pytest.approx(expected_sync_9)

    # Check NA value handling in other columns remains correct
    assert pd.isna(df.loc[1, "text"])
    assert df.loc[1, "active_DIO_inputs_bitmask"] == 0
    assert df.loc[3, "active_DIO_inputs"] == [17]


def test_get_events_dataframe_apply_offset_not_calculated(processor, capsys):
    """Test applying offset when offset is None generates warning and no column."""
    processor.time_offset = None  # Ensure no offset is set
    df = processor.get_events_dataframe(apply_offset=True)  # Request offset application
    assert isinstance(df, pd.DataFrame)
    assert "timestamp_sync" not in df.columns  # Sync column should be absent
    assert len(df) == 9  # Should still return the dataframe without the column
    assert df.index.name == "line_num"

    # Check that the warning was printed
    captured = capsys.readouterr()
    assert (
        "Warning: Time offset application requested" in captured.out
        or "Warning: Time offset application requested" in captured.err
    )


def test_get_events_dataframe_no_apply_offset_calculated(processor):
    """Test apply_offset=False ignores existing offset."""
    processor.time_offset = 1000.0  # Set an offset
    df = processor.get_events_dataframe(
        apply_offset=False
    )  # Request NO offset application
    assert isinstance(df, pd.DataFrame)
    assert "timestamp_sync" not in df.columns  # Sync column should be absent
    assert len(df) == 9
    assert df.index.name == "line_num"


def test_empty_log(empty_processor):
    """Test processing an empty log file."""
    events = empty_processor.parse_raw_events()
    assert events == []
    df = empty_processor.get_events_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    # An empty dataframe doesn't have an index name set
    assert df.index.name is None


def test_comment_only_log(comment_only_processor):
    """Test processing a log file with only comments/whitespace."""
    events = comment_only_processor.parse_raw_events()
    assert len(events) == 4  # 4 lines in the fixture
    assert all(e["type"] == "comment_or_empty" for e in events)
    assert all(e["timestamp"] is None for e in events)

    # Default: exclude comments -> empty DataFrame
    df_excluded = comment_only_processor.get_events_dataframe(apply_offset=False)
    assert isinstance(df_excluded, pd.DataFrame)
    assert df_excluded.empty
    assert df_excluded.index.name is None

    # Include comments -> DataFrame with only comment entries
    df_included = comment_only_processor.get_events_dataframe(
        apply_offset=False, exclude_comments_unknown=False
    )
    assert isinstance(df_included, pd.DataFrame)
    assert len(df_included) == 4
    assert df_included.index.name == "line_num"
    assert all(df_included["type"] == "comment_or_empty")
    assert df_included["trodes_timestamp"].isna().all()
    assert df_included["trodes_timestamp_sec"].isna().all()


def test_repr(processor):
    """Test the __repr__ method reflects state."""
    # Initial state
    initial_repr = repr(processor)
    assert isinstance(initial_repr, str)
    assert "<StateScriptLogProcessor" in initial_repr
    assert "status=not parsed" in initial_repr
    assert "no offset calculated" in initial_repr
    assert "DataFrame not generated" in initial_repr
    assert "source='from string'" in initial_repr

    # After parsing
    processor.parse_raw_events()
    num_raw = len(processor.raw_events)
    parsed_repr = repr(processor)
    assert "status=parsed" in parsed_repr
    assert f"raw_events={num_raw}" in parsed_repr
    assert "no offset calculated" in parsed_repr
    assert "DataFrame not generated" in parsed_repr

    # After offset calculation
    processor.time_offset = 1234.5678
    offset_repr = repr(processor)
    assert "offset=1234.5678s" in offset_repr  # Check formatting
    assert "DataFrame not generated" in offset_repr

    # After DataFrame generation
    processor.get_events_dataframe()
    df_repr = repr(processor)
    assert "DataFrame generated" in df_repr


def test_repr_html(processor):
    """Test the _repr_html_ method generates HTML in different states."""
    # Check it runs without error and returns string containing key info

    # Initial state
    html_initial = processor._repr_html_()
    assert isinstance(html_initial, str)
    assert "<h4>StateScriptLogProcessor</h4>" in html_initial
    assert "Status:</strong> Not Parsed" in html_initial
    assert "Offset:</strong> Not Calculated" in html_initial
    assert "DataFrame:</strong> Not Generated" in html_initial
    assert "Source:</strong> from string" in html_initial
    assert "DataFrame Preview" not in html_initial  # No preview yet

    # After parsing
    processor.parse_raw_events()
    num_raw = len(processor.raw_events)
    html_parsed = processor._repr_html_()
    assert isinstance(html_parsed, str)
    assert "Status:</strong> Parsed" in html_parsed
    assert f"({num_raw} raw entries)" in html_parsed
    assert "Offset:</strong> Not Calculated" in html_parsed
    assert "DataFrame:</strong> Not Generated" in html_parsed

    # After offset calculation
    processor.time_offset = 1234.5678
    html_offset = processor._repr_html_()
    assert isinstance(html_offset, str)
    assert "Offset:</strong> 1234.5678s" in html_offset  # Check formatting
    assert "DataFrame:</strong> Not Generated" in html_offset

    # After DataFrame generation
    processor.get_events_dataframe()
    html_df = processor._repr_html_()
    assert isinstance(html_df, str)
    assert "DataFrame:</strong> Generated" in html_df
    assert (
        "<h5>DataFrame Preview (first 5 rows):</h5>" in html_df
    )  # Check for preview section
    assert "<table" in html_df  # Check that a table is likely generated
    assert "trodes_timestamp" in html_df  # Check a column name is present
