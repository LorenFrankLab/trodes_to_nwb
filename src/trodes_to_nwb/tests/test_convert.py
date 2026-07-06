import functools
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
import pytest
from pynwb import NWBHDF5IO

from unittest.mock import MagicMock

from trodes_to_nwb.convert import (
    _convert_session,
    check_file_timing,
    create_nwbs,
    get_included_device_metadata_paths,
    setup_logger,
)
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.tests.utils import data_path

MICROVOLTS_PER_VOLT = 1e6


def test_get_file_info():
    path_df = get_file_info(data_path)
    path_df = path_df[
        path_df.animal == "sample"
    ]  # restrict to exclude truncated rec files

    for file_type in [
        ".h264",
        ".stateScriptLog",
        ".cameraHWSync",
        ".videoTimeStamps",
        ".videoPositionTracking",
        ".rec",
        ".trackgeometry",
        ".stateScriptLog",
    ]:
        assert len(path_df[path_df.file_extension == file_type]) == 2

    assert set(path_df.animal) == {"sample"}
    assert set(path_df.date) == {20230622}
    assert set(path_df.epoch) == {1, 2}
    assert (set(path_df.tag) == {"a1"}) or (
        set(path_df.tag) == {"a1", "NA"}
    )  # yamlfiles only added in local testing
    for file in path_df.full_path:
        assert Path(file).exists()


def test_get_included_device_metadata_paths():
    probes = list(get_included_device_metadata_paths())
    assert len(probes) == 19
    assert all(probe.exists() for probe in probes)


def test_get_included_device_metadata_paths_returns_reusable_list():
    # Must be a list, not the rglob generator: it is reused across sessions and
    # pickled for dask workers (issue #141).
    paths = get_included_device_metadata_paths()
    assert isinstance(paths, list)
    assert len(list(paths)) == len(paths)  # not exhausted by a first iteration


def test_convert_session_partial_serializes_with_cloudpickle():
    # The #141 cause was that the per-session function was a *closure* inside
    # create_nwbs, which recent distributed cannot deterministically hash.
    # _convert_session is module-level, so the partial that binds the config
    # serializes cleanly with cloudpickle (the serializer dask actually uses).
    # Serialization alone is a weak proxy (closures also survive cloudpickle);
    # the real guard is the parallel-path test below.
    func = functools.partial(
        _convert_session, output_dir="/tmp", device_metadata_paths=[]
    )
    restored = cloudpickle.loads(cloudpickle.dumps(func))
    assert restored.func is _convert_session
    assert restored.keywords["output_dir"] == "/tmp"


def test_convert_session_reports_success_and_failure(mocker):
    session_item = (("20230101", "rat"), MagicMock())

    # Success -> None
    mocker.patch("trodes_to_nwb.convert._create_nwb", return_value=None)
    assert _convert_session(session_item, output_dir="/tmp") is None

    # Failure -> (session, error repr), not a raised exception (so the batch can
    # continue and aggregate every failure).
    mocker.patch(
        "trodes_to_nwb.convert._create_nwb", side_effect=ValueError("bad metadata")
    )
    result = _convert_session(session_item, output_dir="/tmp")
    assert result[0] == ("20230101", "rat")
    assert "bad metadata" in result[1]


def test_create_nwbs_aggregates_session_failures(mocker, tmp_path):
    # Two sessions, both fail: create_nwbs must not silently succeed (parallel)
    # or abort on the first (serial) -- it collects every failure and raises one
    # summary (issue #141).
    file_info = pd.DataFrame(
        {
            "date": [20230101, 20230102],
            "animal": ["rat", "rat"],
            "epoch": [1, 1],
            "tag": ["a1", "a1"],
            "tag_index": [1, 1],
            "file_extension": [".rec", ".rec"],
            "full_path": ["a.rec", "b.rec"],
        }
    )
    mocker.patch("trodes_to_nwb.convert.get_file_info", return_value=file_info)
    mocker.patch("trodes_to_nwb.convert._create_nwb", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="2 of 2 session.* failed to convert"):
        create_nwbs(tmp_path, output_dir=str(tmp_path), device_metadata_paths=[])


class _FakeFuture:
    """A dask-Future stand-in: .result() returns a value or re-raises."""

    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class _FakeClient:
    """Synchronous stand-in for dask's Client so the n_workers>1 branch runs
    deterministically (no real cluster). map() runs the function inline; a task
    that raises becomes a future that re-raises on .result() (a hard worker
    death)."""

    def __init__(self, *args, **kwargs):
        self.closed = False

    def map(self, func, iterable):
        futures = []
        for item in iterable:
            try:
                futures.append(_FakeFuture(value=func(item)))
            except Exception as e:
                futures.append(_FakeFuture(exc=e))
        return futures

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _session_file_info(dates):
    n = len(dates)
    return pd.DataFrame(
        {
            "date": list(dates),
            "animal": ["rat"] * n,
            "epoch": [1] * n,
            "tag": ["a1"] * n,
            "tag_index": [1] * n,
            "file_extension": [".rec"] * n,
            "full_path": [f"{d}.rec" for d in dates],
        }
    )


def test_create_nwbs_parallel_path_aggregates_and_closes(mocker, tmp_path):
    # The n_workers>1 branch (the literal #141 bug) run via a synchronous fake
    # Client: failures must aggregate (not be swallowed) and the client must be
    # closed.
    created = []

    def make_client(*a, **k):
        client = _FakeClient(*a, **k)
        created.append(client)
        return client

    mocker.patch(
        "trodes_to_nwb.convert.get_file_info",
        return_value=_session_file_info([20230101, 20230102]),
    )
    mocker.patch("trodes_to_nwb.convert.Client", make_client)
    mocker.patch("trodes_to_nwb.convert._create_nwb", side_effect=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="2 of 2 session.* failed to convert"):
        create_nwbs(
            tmp_path, output_dir=str(tmp_path), device_metadata_paths=[], n_workers=2
        )

    assert created and created[0].closed is True  # client/cluster torn down


def test_create_nwbs_parallel_surfaces_worker_death_with_other_failures(
    mocker, tmp_path
):
    # A worker that dies hard makes future.result() raise; that session must still
    # become a failure entry, and must NOT discard the other session's failure
    # (silent-failure review C1).
    mocker.patch(
        "trodes_to_nwb.convert.get_file_info",
        return_value=_session_file_info([20230101, 20230102]),
    )
    mocker.patch("trodes_to_nwb.convert.Client", _FakeClient)

    def convert_side_effect(session_item, **kwargs):
        session = session_item[0]
        if session[0] == 20230102:  # date is an int from get_file_info
            raise RuntimeError("KilledWorker")  # hard death: escapes the task
        return (session, "boom")  # ordinary failure, returned

    mocker.patch(
        "trodes_to_nwb.convert._convert_session", side_effect=convert_side_effect
    )

    with pytest.raises(RuntimeError) as excinfo:
        create_nwbs(
            tmp_path, output_dir=str(tmp_path), device_metadata_paths=[], n_workers=2
        )
    message = str(excinfo.value)
    assert "2 of 2 session" in message
    assert "20230101" in message  # ordinary failure NOT lost
    assert "worker died" in message  # hard death surfaced as a failure


def test_create_nwbs_reports_only_failed_sessions(mocker, tmp_path):
    # Partial batch: only the failed session is named, and the success count is
    # reported so a caller knows the rest were written.
    mocker.patch(
        "trodes_to_nwb.convert.get_file_info",
        return_value=_session_file_info([20230101, 20230102, 20230103]),
    )

    def create_side_effect(session, session_df, **kwargs):
        if session[0] == 20230102:  # date is an int from get_file_info
            raise RuntimeError("boom")
        return None

    mocker.patch("trodes_to_nwb.convert._create_nwb", side_effect=create_side_effect)

    with pytest.raises(RuntimeError) as excinfo:
        create_nwbs(tmp_path, output_dir=str(tmp_path), device_metadata_paths=[])
    message = str(excinfo.value)
    assert "1 of 3 session" in message
    assert "2 succeeded" in message
    assert "20230102" in message  # the failed session is named
    assert "20230101" not in message  # successes are not listed


def test_create_nwbs_reuses_metadata_paths_across_sessions(mocker, tmp_path):
    # A generator passed as device_metadata_paths must not be exhausted after the
    # first session -- it is materialized to a list so session 2+ still gets the
    # full set (#141).
    mocker.patch(
        "trodes_to_nwb.convert.get_file_info",
        return_value=_session_file_info([20230101, 20230102]),
    )
    spy = mocker.patch("trodes_to_nwb.convert._create_nwb", return_value=None)
    paths_generator = (Path(p) for p in ["x.yml", "y.yml"])

    create_nwbs(
        tmp_path, output_dir=str(tmp_path), device_metadata_paths=paths_generator
    )

    seen = [call.kwargs["device_metadata_paths"] for call in spy.call_args_list]
    assert len(seen) == 2
    assert all(len(paths) == 2 for paths in seen)  # 2nd session not exhausted


def test_convert_full():
    device_metadata = get_included_device_metadata_paths()

    video_directory = data_path / "temp_video_directory_full_convert"
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    exclude_reconfig_yaml = str(data_path / "20230622_sample_metadataProbeReconfig.yml")
    create_nwbs(
        path=data_path,
        device_metadata_paths=device_metadata,
        output_dir=str(data_path),
        n_workers=1,
        query_expression=f"animal == 'sample' and full_path != '{exclude_reconfig_yaml}'",
        fs_gui_dir=data_path,
    )

    output_file_path = data_path / "sample20230622.nwb"
    assert output_file_path.exists()

    rec_to_nwb_file = data_path / "minirec20230622_.nwb"
    with NWBHDF5IO(output_file_path) as io:
        nwbfile = io.read()
        with NWBHDF5IO(rec_to_nwb_file) as io2:
            old_nwbfile = io2.read()
            # run nwb comparison
            compare_nwbfiles(nwbfile, old_nwbfile)

    output_report_path = data_path / "sample20230622_nwbinspector_report.txt"
    assert os.path.isfile(output_report_path)

    # cleanup
    os.remove(output_file_path)
    os.remove(output_report_path)
    shutil.rmtree(video_directory)


def test_convert_full_partial_iterators():
    with patch("trodes_to_nwb.convert_ephys.MAXIMUM_ITERATOR_SIZE", new=5000):
        test_convert_full()


def test_convert_full_with_inspector_error(mocker):
    def do_nothing(nwbfile, metadata_dict):
        pass

    mocker.patch("trodes_to_nwb.convert.add_subject", do_nothing)

    device_metadata = get_included_device_metadata_paths()

    video_directory = data_path / "temp_video_directory_full_convert"
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    exclude_reconfig_yaml = str(data_path / "20230622_sample_metadataProbeReconfig.yml")
    create_nwbs(
        path=data_path,
        device_metadata_paths=device_metadata,
        output_dir=str(data_path),
        fs_gui_dir=data_path,
        n_workers=1,
        query_expression=f"animal == 'sample' and full_path != '{exclude_reconfig_yaml}'",
    )

    output_file_path = data_path / "sample20230622.nwb"

    output_report_path = data_path / "sample20230622_nwbinspector_report.txt"
    assert os.path.isfile(output_report_path)

    with open(output_report_path) as f:
        assert "Importance.CRITICAL: check_subject_exists" in f.read()

    # TODO check that the error is printed to stdout
    # 0.0  Importance.CRITICAL: check_subject_exists - 'NWBFile' object at location '/'
    #    Message: Subject is missing.

    # cleanup
    os.remove(output_file_path)
    os.remove(output_report_path)
    shutil.rmtree(video_directory)


def check_module_entries(test, reference):
    todo = [
        "camera_sample_frame_counts",
        # "video_files",
    ]  # TODO: known missing entries
    for entry in reference:
        if entry in todo:
            continue
        assert entry in test


def compare_nwbfiles(nwbfile, old_nwbfile, truncated_size=False):
    """Compare two nwbfiles, checking that all the same entries are present and that the data matches

    Parameters
    ----------
    nwbfile : pynwb.NWBFile
        The nwbfile to be tested
    old_nwbfile : pynwb.NWBFile
        The reference nwbfile (generated by rec_to_nwb)
    truncated_size : bool, optional
        Whether the new nwbfile only contains a subset of the data, by default False
    """

    # check existence of contents
    check_module_entries(nwbfile.processing, old_nwbfile.processing)
    check_module_entries(nwbfile.acquisition, old_nwbfile.acquisition)
    check_module_entries(nwbfile.devices, old_nwbfile.devices)
    assert nwbfile.subject
    assert nwbfile.session_description
    assert nwbfile.session_id
    assert nwbfile.session_start_time
    assert nwbfile.electrodes
    assert nwbfile.experiment_description
    assert nwbfile.experimenter
    assert nwbfile.file_create_date
    assert nwbfile.identifier
    assert nwbfile.institution
    assert nwbfile.lab

    # check ephys data values
    conversion = nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
    assert (
        (nwbfile.acquisition["e-series"].data[0, :] * conversion).astype("int16")
        == old_nwbfile.acquisition["e-series"].data[0, :]
    ).all()
    # check data shapes match if untruncated
    assert (
        nwbfile.acquisition["e-series"].data.shape
        == old_nwbfile.acquisition["e-series"].data.shape
    ) or truncated_size
    ephys_size = nwbfile.acquisition["e-series"].data.shape[0]
    # check all values of one of the streams
    old_data = old_nwbfile.acquisition["e-series"].data[:, 0]
    ind = np.where(np.abs(old_data[:ephys_size]) > 0)[
        0
    ]  # Ignore the artifact zero valued points from rec_to_nwb_conversion
    assert (
        (nwbfile.acquisition["e-series"].data[ind, 0] * conversion).astype("int16")
        == old_data[ind]
    ).all()
    # check that timestamps are less than one sample different
    assert np.allclose(
        nwbfile.acquisition["e-series"].timestamps[:],
        old_nwbfile.acquisition["e-series"].timestamps[:ephys_size],
        rtol=0,
        atol=1.0 / 30000,
    )

    # check analog data
    # get index mapping of channels
    id_order = nwbfile.processing["analog"]["analog"]["analog"].description.split(
        "   "
    )[:-1]
    old_id_order = old_nwbfile.processing["analog"]["analog"][
        "analog"
    ].description.split("   ")[:-1]
    # TODO check that all the same channels are present
    if (
        old_nwbfile.processing["analog"]["analog"]["analog"].data.size > 0
    ):  # analog data not included in all old files. Shouldn't fail because we include it now
        # compare analog data on channels present in rec conversion
        if "timestamps" in old_id_order:
            old_id_order.remove("timestamps")
        index_order = [id_order.index(id) for id in old_id_order]

        assert (
            nwbfile.processing["analog"]["analog"]["analog"].data.shape[0]
            == old_nwbfile.processing["analog"]["analog"]["analog"].data.shape[0]
        ) or truncated_size
        analog_size = nwbfile.processing["analog"]["analog"]["analog"].data.shape[0]
        # compare matching for first timepoint
        assert (
            nwbfile.processing["analog"]["analog"]["analog"].data[0, :][index_order]
            == old_nwbfile.processing["analog"]["analog"]["analog"].data[0, :]
        ).all()
        # compare one channel across all timepoints
        assert (
            nwbfile.processing["analog"]["analog"]["analog"].data[:, index_order[0]]
            == old_nwbfile.processing["analog"]["analog"]["analog"].data[
                :analog_size, 0
            ]
        ).all()

    # compare dio data
    for dio_name in old_nwbfile.processing["behavior"]["behavioral_events"].time_series:
        old_dio = old_nwbfile.processing["behavior"]["behavioral_events"][dio_name]
        current_dio = nwbfile.processing["behavior"]["behavioral_events"][dio_name]
        # check that timeseries match
        dio_size = current_dio.data.shape[0]
        np.testing.assert_array_equal(current_dio.data[:], old_dio.data[:dio_size])
        assert np.allclose(
            current_dio.timestamps[:],
            old_dio.timestamps[:dio_size],
            rtol=0,
            atol=1.0 / 30000,
        )
        # unit is now "N/A" and the description records the header channel id +
        # input flag (#116, #117), so neither still matches the old rec_to_nwb
        # reference; the data/timestamp equivalence above is the real check.
        assert current_dio.unit == "N/A"
        assert ", input=" in current_dio.description

    # Compare position data
    for series in nwbfile.processing["behavior"]["position"].spatial_series:
        # check series in new nwbfile
        assert series in nwbfile.processing["behavior"]["position"].spatial_series
        # find the corresponding data in the old file
        validated = False
        for old_series in old_nwbfile.processing["behavior"]["position"].spatial_series:
            # check that led number matches
            if series.split("_")[1] != old_series.split("_")[1]:
                continue
            # check if timestamps end the same
            timestamps = nwbfile.processing["behavior"]["position"][series].timestamps[
                :
            ]
            old_timestamps = old_nwbfile.processing["behavior"]["position"][
                old_series
            ].timestamps[:]
            if np.allclose(
                timestamps[-30:],
                old_timestamps[-30:],
                rtol=0,
                atol=np.mean(np.diff(old_timestamps[-30:])),
            ):
                pos = nwbfile.processing["behavior"]["position"][series].data[:]
                old_pos = old_nwbfile.processing["behavior"]["position"][
                    old_series
                ].data[:]
                # check that the data is the same
                assert np.allclose(pos[-30:], old_pos[-30:], rtol=0, atol=1e-6)
                validated = True
                break
        assert validated, f"Could not find matching series for {series}"


def _make_mock_io(start_ns, end_ns, filename="file.rec", sys_clock=True):
    """Helper to create a mock SpikeGadgetsRawIO-like object."""
    mock_io = MagicMock()
    mock_io.get_sys_clock.side_effect = lambda start, end: (
        [start_ns] if end == 1 else [end_ns]
    )
    mock_io._raw_memmap.shape = (100, 1)
    mock_io._raw_memmap.filename = filename
    mock_io.sysClock_byte = sys_clock
    mock_io.system_time_at_creation = start_ns / 1e6  # in milliseconds
    return mock_io


def test_check_file_timing_valid_single_file():
    """check_file_timing should not raise for a single valid file."""
    start_ns = int(1e18)  # ~31 years in nanoseconds, valid Unix time in seconds
    end_ns = start_ns + int(60 * 1e9)  # 60 seconds later
    mock_io = _make_mock_io(start_ns, end_ns)

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.return_value = mock_io
        # Should not raise
        check_file_timing(["file.rec"], setup_logger("test", "test.log"))


def test_check_file_timing_valid_multiple_files():
    """check_file_timing should not raise for multiple ordered valid files."""
    base_ns = int(1e18)
    mock_io1 = _make_mock_io(base_ns, base_ns + int(60 * 1e9), "file1.rec")
    mock_io2 = _make_mock_io(
        base_ns + int(120 * 1e9), base_ns + int(180 * 1e9), "file2.rec"
    )

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.side_effect = [mock_io1, mock_io2]
        # Should not raise
        check_file_timing(["file1.rec", "file2.rec"], setup_logger("test", "test.log"))


def test_check_file_timing_valid_multiple_files_no_sys_clock():
    """check_file_timing should not raise for multiple ordered valid files."""
    base_ns = int(1e18)
    mock_io1 = _make_mock_io(
        base_ns, base_ns + int(60 * 1e9), "file1.rec", sys_clock=False
    )
    mock_io2 = _make_mock_io(
        base_ns + int(120 * 1e9), base_ns + int(180 * 1e9), "file2.rec", sys_clock=False
    )

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.side_effect = [mock_io1, mock_io2]
        # Should not raise
        check_file_timing(["file1.rec", "file2.rec"], setup_logger("test", "test.log"))


def test_check_file_timing_negative_duration_raises():
    """check_file_timing should raise ValueError when end time is before start time."""
    start_ns = int(1e18)
    end_ns = start_ns - int(10 * 1e9)  # end before start
    mock_io = _make_mock_io(start_ns, end_ns)

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.return_value = mock_io
        try:
            check_file_timing(["file.rec"], logger=setup_logger("test", "test.log"))
            assert False, "Expected ValueError for negative duration"
        except ValueError:
            pass


def test_check_file_timing_out_of_order_raises():
    """check_file_timing should raise ValueError when files are out of order."""
    base_ns = int(1e18)
    # file2 starts before file1 ends and before file1 starts
    mock_io1 = _make_mock_io(
        base_ns + int(120 * 1e9), base_ns + int(180 * 1e9), "file1.rec"
    )
    mock_io2 = _make_mock_io(base_ns, base_ns + int(60 * 1e9), "file2.rec")

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.side_effect = [mock_io1, mock_io2]
        try:
            check_file_timing(
                ["file1.rec", "file2.rec"], logger=setup_logger("test", "test.log")
            )
            assert False, "Expected ValueError for out of order files"
        except ValueError:
            pass


def test_check_file_timing_equal_start_times_raises():
    """check_file_timing should raise ValueError when two files have the same start time."""
    base_ns = int(1e18)
    mock_io1 = _make_mock_io(base_ns, base_ns + int(60 * 1e9), "file1.rec")
    mock_io2 = _make_mock_io(base_ns, base_ns + int(120 * 1e9), "file2.rec")

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.side_effect = [mock_io1, mock_io2]
        try:
            check_file_timing(
                ["file1.rec", "file2.rec"], logger=setup_logger("test", "test.log")
            )
            assert False, "Expected ValueError for equal start times"
        except ValueError:
            pass


def test_check_file_timing_empty_list():
    """check_file_timing should not raise for an empty list."""
    check_file_timing([], logger=setup_logger("test", "test.log"))


def test_check_file_timing_parses_header():
    """check_file_timing should call parse_header for each file."""
    base_ns = int(1e18)
    mock_io = _make_mock_io(base_ns, base_ns + int(60 * 1e9))

    with patch("trodes_to_nwb.convert.SpikeGadgetsRawIO") as MockRawIO:
        MockRawIO.return_value = mock_io
        check_file_timing(["file.rec"], logger=setup_logger("test", "test.log"))
        mock_io.parse_header.assert_called_once()
