import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from trodes_to_nwb.convert_statescript import (
    StateScriptLogProcessor,
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
100559 0 0
Executing this line without timestamp
115030 center_poke
115030 65536 0
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
    # These correspond roughly to the '65536 0' events in sample_log_content
    # 76566 ms -> 76.566 s
    # 115030 ms -> 115.030 s
    # Let's assume a base time for the external system
    base_time = 1678880000.0
    return np.array([base_time + 76.566, base_time + 115.030])


@pytest.fixture
def temp_log_file(sample_log_content):
    """Creates a temporary log file with standard content and yields its path."""
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".stateScriptLog", encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(sample_log_content)
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    os.remove(tmp_file_path)


# --- Tests for Level 1 Parsers ---


def test_parse_int():
    """Test the _parse_int helper function."""
    assert _parse_int("123") == 123
    assert _parse_int("-45") == -45
    assert _parse_int("0") == 0
    assert _parse_int("abc") is None
    assert _parse_int("12.3") is None
    assert _parse_int("") is None
    assert _parse_int("123 ") is None


def test_parse_ts_int_int():
    """Test parse_ts_int_int directly."""
    parts = ["8386500", "0", "0"]
    expected = {
        "type": "ts_int_int",
        "trodes_timestamp": 8386500,
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
        "trodes_timestamp": 8386500,
        "text": "DOWN",
        "value": 3,
    }
    assert parse_ts_str_int(parts) == expected

    parts_wrong_len = ["123", "UP"]
    assert parse_ts_str_int(parts_wrong_len) is None

    parts_str_is_int = ["123", "456", "789"]
    assert parse_ts_str_int(parts_str_is_int) is None  # Should be handled by ts_int_int

    parts_val_not_int = ["123", "UP", "abc"]
    assert parse_ts_str_int(parts_val_not_int) is None


def test_parse_ts_str_equals_int():
    """Test parse_ts_str_equals_int directly."""
    parts = ["100078", "counter_handlePoke", "=", "1"]
    expected = {
        "type": "ts_str_equals_int",
        "trodes_timestamp": 100078,
        "text": "counter_handlePoke",
        "value": 1,
    }
    assert parse_ts_str_equals_int(parts) == expected

    parts_multi_word = ["3610855", "total", "rewards", "=", "70"]
    expected_multi = {
        "type": "ts_str_equals_int",
        "trodes_timestamp": 3610855,
        "text": "total rewards",
        "value": 70,
    }
    assert parse_ts_str_equals_int(parts_multi_word) == expected_multi

    parts_wrong_len = ["123", "=", "1"]
    assert parse_ts_str_equals_int(parts_wrong_len) is None

    parts_no_equals = ["123", "text", "1"]
    assert parse_ts_str_equals_int(parts_no_equals) is None

    parts_val_not_int = ["123", "text", "=", "abc"]
    assert parse_ts_str_equals_int(parts_val_not_int) is None


def test_parse_ts_str():
    """Test parse_ts_str directly."""
    parts = ["76566", "center_poke"]
    expected = {
        "type": "ts_str",
        "trodes_timestamp": 76566,
        "text": "center_poke",
    }
    assert parse_ts_str(parts) == expected

    parts_multi_word = ["1271815", "some", "multi", "word", "event"]
    expected_multi = {
        "type": "ts_str",
        "trodes_timestamp": 1271815,
        "text": "some multi word event",
    }
    assert parse_ts_str(parts_multi_word) == expected_multi

    parts_wrong_len = ["123"]
    assert parse_ts_str(parts_wrong_len) is None

    parts_second_is_int = [
        "123",
        "456",
    ]  # Second part is int, should fail this parser
    assert parse_ts_str(parts_second_is_int) is None


# --- Tests for parse_statescript_line (Covers integration and dispatching) ---


def test_parse_statescript_line_dispatching():
    """Test parse_statescript_line dispatching for various line types."""
    lines_expected_types = [
        ("8386500 0 0", "ts_int_int"),
        ("8386500 DOWN 3", "ts_str_int"),
        ("100078 counter_handlePoke = 1", "ts_str_equals_int"),
        ("76566 center_poke", "ts_str"),
        ("Executing trigger function 22", "unknown"),
        ("# comment", "comment_or_empty"),
        ("", "comment_or_empty"),
        ("   ", "comment_or_empty"),
        ("123 456 abc", "unknown"),  # Doesn't fit ts_int_int because of 'abc'
        ("123 abc def", "ts_str"),  # Fits ts_str
        ("456 123 = 5", "ts_str_equals_int"),  # Fits this specific pattern
    ]

    for line, expected_type in lines_expected_types:
        parsed = parse_statescript_line(line)
        assert parsed["type"] == expected_type
        assert parsed["raw_line"] == line.strip()  # parse_statescript_line strips
        if expected_type not in ["unknown", "comment_or_empty"]:
            assert "trodes_timestamp" in parsed
        else:
            assert "trodes_timestamp" not in parsed or pd.isna(
                parsed.get("trodes_timestamp")
            )


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
    assert pathlib.Path(temp_log_file).name in processor_file.source_description


def test_init_from_file_not_found():
    """Test initialization from a non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        StateScriptLogProcessor.from_file("non_existent_file_qwerty.log")


def test_parse_raw_events(processor, sample_log_content):
    """Test parsing the raw log content into events."""
    events = processor.parse_raw_events()
    assert processor.raw_events is events  # Should store result internally
    assert isinstance(events, list)
    assert len(events) == len(
        sample_log_content.strip().splitlines()
    )  # One dict per line
    assert events[0]["type"] == "comment_or_empty"
    assert events[1]["type"] == "ts_int_int"
    assert events[7]["type"] == "unknown"  # "Executing this line..."
    assert events[9]["type"] == "comment_or_empty"  # Last comment
    assert events[1]["raw_line"] == "76504 0 0"
    assert events[7]["raw_line"] == "Executing this line without timestamp"


def test_find_reference_events(processor):
    """Test the internal _find_reference_events method."""
    # Case 1: Find 'ts_str' events
    ref_df_str = processor._find_reference_events(
        event_type="ts_str", conditions={"text": "center_poke"}
    )
    assert isinstance(ref_df_str, pd.DataFrame)
    assert len(ref_df_str) == 2
    pd.testing.assert_series_equal(
        ref_df_str["trodes_timestamp"],
        pd.Series([76566, 115030], name="trodes_timestamp"),
        check_dtype=False,
    )
    assert "trodes_timestamp_sec" in ref_df_str.columns
    assert ref_df_str["trodes_timestamp_sec"].iloc[0] == pytest.approx(76.566)

    # Case 2: Find 'ts_int_int' events with specific values
    ref_df_int = processor._find_reference_events(
        event_type="ts_int_int", conditions={"value1": 4, "value2": 0}
    )
    assert len(ref_df_int) == 1
    assert ref_df_int["trodes_timestamp"].iloc[0] == 100078

    # Case 3: No matching events found
    ref_df_none = processor._find_reference_events(
        event_type="ts_str_int", conditions={"text": "nonexistent"}
    )
    assert ref_df_none.empty
    assert isinstance(ref_df_none, pd.DataFrame)  # Should still return DF

    # Case 4: Ensure processor parses if raw_events is empty
    processor.raw_events = []
    ref_df_reparse = processor._find_reference_events(
        event_type="ts_str", conditions={"text": "center_poke"}
    )
    assert len(ref_df_reparse) == 2  # Should re-parse automatically


def test_calculate_time_offset_success(processor, external_times):
    """Test successful time offset calculation."""
    offset = processor.calculate_time_offset(
        external_reference_times=external_times,
        log_event_type="ts_int_int",  # Use the events corresponding to external_times
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,  # Use both events for matching
    )
    assert offset is not None
    assert processor.time_offset == offset  # Check internal storage
    # Expected offset = external_base_time = 1678880000.0
    # external_times[0] = base + 76.566; log_times[0] = 76.566
    assert offset == pytest.approx(1678880000.0)


def test_calculate_time_offset_fail_not_enough_log(processor, external_times):
    """Test offset calculation failure due to insufficient log events."""
    # 'counter_handlePoke' only appears once, need 2 events
    offset = processor.calculate_time_offset(
        external_reference_times=external_times,
        log_event_type="ts_str_equals_int",
        log_event_conditions={"text": "counter_handlePoke"},
        check_n_events=2,
    )
    assert offset is None
    assert processor.time_offset is None  # Should remain None


def test_calculate_time_offset_fail_not_enough_external(processor):
    """Test offset calculation failure due to insufficient external times."""
    # Only one external time provided, need 2 events
    offset = processor.calculate_time_offset(
        external_reference_times=np.array([1678880076.566]),
        log_event_type="ts_int_int",
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,
    )
    assert offset is None
    assert processor.time_offset is None


def test_calculate_time_offset_fail_mismatch(processor, external_times):
    """Test offset calculation failure due to exceeding mismatch threshold."""
    # Shift external times slightly more than default threshold (0.1)
    shifted_external_times = external_times + 0.06  # Total shift 0.12 over 2 events
    offset = processor.calculate_time_offset(
        external_reference_times=shifted_external_times,
        log_event_type="ts_int_int",
        log_event_conditions={"value1": 65536, "value2": 0},
        check_n_events=2,
        match_threshold=0.1,  # Default threshold
    )
    assert offset is None
    assert processor.time_offset is None


def test_get_events_dataframe_defaults(processor):
    """Test default behavior: exclude comments/unknown, no offset applied yet."""
    df = processor.get_events_dataframe(apply_offset=False)
    assert processor.processed_events_df is df  # Check internal storage
    assert isinstance(df, pd.DataFrame)
    # Expected: 11 lines total - 3 comments - 1 unknown = 7 valid events
    assert len(df) == 7
    assert "raw_line" in df.columns
    assert "trodes_timestamp" in df.columns
    assert "trodes_timestamp_sec" in df.columns
    assert "timestamp_sync" not in df.columns  # Offset not applied
    # Check content and types
    assert df["type"].iloc[0] == "ts_int_int"
    assert df["raw_line"].iloc[0] == "76504 0 0"
    assert pd.isna(df["text"].iloc[0])  # text NA for ts_int_int
    assert df["value1"].iloc[0] == 0
    assert df["trodes_timestamp"].dtype == "int64"
    assert df["trodes_timestamp_sec"].dtype == "float64"
    assert df["value"].dtype == pd.Int64Dtype()  # Nullable Integer


def test_get_events_dataframe_include_all(processor):
    """Test including comments and unknown lines."""
    df = processor.get_events_dataframe(
        apply_offset=False, exclude_comments_unknown=False
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10  # All lines included
    assert df["type"].iloc[0] == "comment_or_empty"
    assert df["type"].iloc[7] == "unknown"
    assert df["raw_line"].iloc[7] == "Executing this line without timestamp"
    # Check that timestamp is NA/0 for lines without one
    assert (
        pd.isna(df["trodes_timestamp"].iloc[0]) or df["trodes_timestamp"].iloc[0] == 0
    )
    assert (
        pd.isna(df["trodes_timestamp"].iloc[7]) or df["trodes_timestamp"].iloc[7] == 0
    )
    assert pd.isna(df["trodes_timestamp_sec"].iloc[0]) or np.isnan(
        df["trodes_timestamp_sec"].iloc[0]
    )
    assert pd.isna(df["trodes_timestamp_sec"].iloc[7]) or np.isnan(
        df["trodes_timestamp_sec"].iloc[7]
    )


def test_get_events_dataframe_with_offset(processor):
    """Test applying offset and check sync timestamp calculation."""
    # Simulate successful offset calculation
    processor.time_offset = 1678880000.0
    df = processor.get_events_dataframe(apply_offset=True)  # Default exclude=True
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7  # Excludes comments/unknown
    assert "timestamp_sync" in df.columns
    # Check calculation for the first valid event (76504 ms)
    expected_sync_time = (76504 / 1000.0) + 1678880000.0
    assert df["timestamp_sync"].iloc[0] == pytest.approx(expected_sync_time)
    # Check NA value handling in other columns remains correct
    assert pd.isna(df["text"].iloc[0])
    assert df["value1"].iloc[0] == 0
    assert df["timestamp_sync"].dtype == "float64"


def test_get_events_dataframe_offset_not_calculated(processor, capsys):
    """Test applying offset when offset is None."""
    processor.time_offset = None  # Ensure no offset is set
    df = processor.get_events_dataframe(apply_offset=True)
    assert isinstance(df, pd.DataFrame)
    assert "timestamp_sync" not in df.columns  # Sync column should be absent
    assert len(df) == 7  # Should still return the dataframe without the column

    # Check that the warning was printed to stderr/stdout
    captured = capsys.readouterr()
    assert (
        "Warning: Time offset requested but not calculated" in captured.out
        or "Warning: Time offset requested but not calculated" in captured.err
    )


def test_empty_log(empty_processor):
    """Test processing an empty log file."""
    events = empty_processor.parse_raw_events()
    assert events == []
    df = empty_processor.get_events_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_comment_only_log(comment_only_processor):
    """Test processing a log file with only comments/whitespace."""
    events = comment_only_processor.parse_raw_events()
    assert len(events) == 4  # 4 lines in the fixture
    assert all(e["type"] == "comment_or_empty" for e in events)

    # Default: exclude comments -> empty DataFrame
    df_excluded = comment_only_processor.get_events_dataframe(apply_offset=False)
    assert isinstance(df_excluded, pd.DataFrame)
    assert df_excluded.empty

    # Include comments -> DataFrame with only comment entries
    df_included = comment_only_processor.get_events_dataframe(
        apply_offset=False, exclude_comments_unknown=False
    )
    assert isinstance(df_included, pd.DataFrame)
    assert len(df_included) == 4
    assert all(df_included["type"] == "comment_or_empty")
    assert (
        pd.isna(df_included["trodes_timestamp"].iloc[0])
        or df_included["trodes_timestamp"].iloc[0] == 0
    )


def test_repr(processor):
    """Test the __repr__ method."""
    # Initial state
    initial_repr = repr(processor)
    assert isinstance(initial_repr, str)
    assert "StateScriptLogProcessor" in initial_repr
    assert "not parsed" in initial_repr
    assert "no offset" in initial_repr
    assert "not generated" in initial_repr

    # After parsing
    processor.parse_raw_events()
    parsed_repr = repr(processor)
    assert "parsed" in parsed_repr
    assert f"raw_events={len(processor.raw_events)}" in parsed_repr
    assert "no offset" in parsed_repr
    assert "not generated" in parsed_repr

    # After offset calculation
    processor.time_offset = 1000.0
    offset_repr = repr(processor)
    assert "offset=1000.0" in offset_repr
    assert "not generated" in offset_repr

    # After DataFrame generation
    processor.get_events_dataframe()
    df_repr = repr(processor)
    assert "DataFrame generated" in df_repr


def test_repr_html(processor):
    """Test the _repr_html_ method."""
    # Check it runs without error in different states and returns string
    html_initial = processor._repr_html_()
    assert isinstance(html_initial, str)
    assert "StateScriptLogProcessor" in html_initial
    assert "Not Parsed" in html_initial

    processor.parse_raw_events()
    html_parsed = processor._repr_html_()
    assert isinstance(html_parsed, str)
    assert "Parsed" in html_parsed
    assert f"({len(processor.raw_events)} raw entries)" in html_parsed

    processor.time_offset = 1000.0
    html_offset = processor._repr_html_()
    assert isinstance(html_offset, str)
    assert "Offset:</strong> 1000.0" in html_offset

    processor.get_events_dataframe()
    html_df = processor._repr_html_()
    assert isinstance(html_df, str)
    assert "DataFrame:</strong> Generated" in html_df
    assert "DataFrame Preview" in html_df  # Check for preview section
