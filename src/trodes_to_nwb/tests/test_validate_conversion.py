import json
import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from xml.etree.ElementTree import Element, SubElement

import numpy as np
import pandas as pd
import pytest

from trodes_to_nwb.tests.utils import data_path
from trodes_to_nwb.validate_conversion import (
    _build_result_summary,
    _compare_1d_arrays,
    _compare_chunked_time_series,
    _humanize_report,
    _load_validation_context,
    _parse_session_from_metadata_filepath,
    _resolve_header_reconfig_path,
    _resolve_report_filepath,
    _resolve_rec_filepaths,
    validate_conversion,
)

validation_module = importlib.import_module("trodes_to_nwb.validate_conversion")


class _ArraySource:
    def __init__(self, data: np.ndarray):
        self.data = data

    def _get_data(self, selection):
        time_slice, channel_slice = selection
        return self.data[time_slice, channel_slice]


class _CloseTracker:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _BehavioralEvents:
    def __init__(self, time_series):
        self.time_series = time_series

    def __getitem__(self, key):
        return self.time_series[key]


class _FakeRawIO:
    def __init__(self, filename):
        self.filename = filename
        self.parse_header_called = False
        self.multiplexed_channel_xml = {}
        self._mask_channels_ids = {"ECU_digital": []}

    def parse_header(self):
        self.parse_header_called = True


def _make_header(sampling_rate="30000"):
    root = Element("Configuration")
    hardware = SubElement(root, "HardwareConfiguration")
    hardware.attrib["samplingRate"] = sampling_rate
    spike_configuration = SubElement(root, "SpikeConfiguration")
    SubElement(spike_configuration, "SpikeNTrode", rawScalingToUv="0.195")
    global_configuration = SubElement(root, "GlobalConfiguration")
    global_configuration.attrib.update(
        {
            "headstageSerial": "hs",
            "headstageSmartRefOn": "1",
            "realtimeMode": "0",
            "headstageAutoSettleOn": "0",
            "timestampAtCreation": "1",
            "controllerFirmwareVersion": "cfw",
            "controllerSerial": "cs",
            "saveDisplayedChanOnly": "0",
            "headstageFirmwareVersion": "hfw",
            "qtVersion": "qt",
            "compileDate": "2024-01-01",
            "compileTime": "12:00:00",
            "filePrefix": "prefix",
            "headstageGyroSensorOn": "0",
            "headstageMagSensorOn": "0",
            "trodesVersion": "1.0",
            "headstageAccelSensorOn": "0",
            "commitHead": "abc123",
            "systemTimeAtCreation": "1718064000000",
            "filePath": "/tmp/session.rec",
        }
    )
    return root


def test_compare_1d_arrays_reports_mismatch_count():
    result = _compare_1d_arrays(
        expected=np.array([1.0, 2.0, 3.0]),
        actual=np.array([1.0, 2.2, 3.5]),
        atol=0.1,
    )

    assert result["mismatch_count"] == 2
    assert result["max_abs_error"] == pytest.approx(0.5)
    assert "first_mismatch_index=1" in result["messages"][0]


def test_compare_1d_arrays_reports_shape_mismatch():
    result = _compare_1d_arrays(
        expected=np.array([1.0, 2.0]),
        actual=np.array([[1.0, 2.0]]),
        atol=0.0,
    )

    assert result["mismatch_count"] == 1
    assert "shape mismatch" in result["messages"][0]


def test_compare_chunked_time_series_honors_tolerance():
    expected = _ArraySource(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    actual = np.array([[1.0, 2.0], [3.4, 4.0], [5.0, 6.0]])

    result = _compare_chunked_time_series(
        expected_source=expected,
        actual_source=actual,
        n_rows=3,
        n_columns=2,
        chunk_size=2,
        atol=0.5,
    )

    assert result["mismatch_count"] == 0
    assert result["max_abs_error"] == pytest.approx(0.4)
    assert result["messages"] == []


def test_compare_chunked_time_series_reports_shape_mismatch():
    expected = _ArraySource(np.array([[1.0, 2.0], [3.0, 4.0]]))
    actual = np.array([1.0, 2.0, 3.0])

    result = _compare_chunked_time_series(
        expected_source=expected,
        actual_source=actual,
        n_rows=2,
        n_columns=2,
        chunk_size=2,
        atol=0.0,
    )

    assert result["mismatch_count"] == 1
    assert "shape mismatch" in result["messages"][0]


def test_compare_chunked_time_series_reports_first_mismatch():
    expected = _ArraySource(np.array([[1.0, 2.0], [3.0, 4.0]]))
    actual = np.array([[1.0, 2.0], [30.0, 4.0]])

    result = _compare_chunked_time_series(
        expected_source=expected,
        actual_source=actual,
        n_rows=2,
        n_columns=2,
        chunk_size=1,
        atol=0.0,
    )

    assert result["mismatch_count"] == 1
    assert "first_mismatch=(1, 0)" in result["messages"][0]


def test_parse_session_from_metadata_filepath():
    session_date, session_animal = _parse_session_from_metadata_filepath(
        data_path / "20230622_sample_metadata.yml"
    )

    assert session_date == 20230622
    assert session_animal == "sample"


def test_resolve_rec_filepaths_from_data_path(tmp_path):
    metadata_path = tmp_path / "20240609_L14_metadata.yml"
    metadata_path.write_text("session_id: test\n", encoding="utf-8")
    rec_path_1 = tmp_path / "20240609_L14_01_a1.rec"
    rec_path_2 = tmp_path / "nested" / "20240609_L14_02_a1.rec"
    rec_path_2.parent.mkdir()
    rec_path_1.write_bytes(b"")
    rec_path_2.write_bytes(b"")

    rec_paths = _resolve_rec_filepaths(
        rec_filepaths=None,
        data_path=tmp_path,
        metadata_filepath=metadata_path,
    )

    assert rec_paths == [rec_path_1, rec_path_2]


def test_resolve_report_filepath_defaults_to_nwb_directory(tmp_path):
    nwb_path = tmp_path / "session.nwb"
    report_path = _resolve_report_filepath(
        nwb_filepath=nwb_path,
        report_filepath=None,
    )

    assert report_path == tmp_path / "session_conversion_validation_report.json"


def test_resolve_report_filepath_accepts_custom_path(tmp_path):
    nwb_path = tmp_path / "session.nwb"
    custom_path = tmp_path / "reports" / "custom.json"
    report_path = _resolve_report_filepath(
        nwb_filepath=nwb_path,
        report_filepath=custom_path,
    )

    assert report_path == custom_path


def test_resolve_header_reconfig_path_raises_on_conflict():
    with pytest.raises(ValueError, match="do not point to the same file"):
        _resolve_header_reconfig_path(
            header_reconfig_path="a.trodesconf",
            reconfig_header_path="b.trodesconf",
        )


def test_humanize_report_adds_summary_text_for_failure():
    report = _humanize_report(
        {
            "passed": False,
            "summary": {"n_checks": 2, "n_failed": 1},
            "checks": [
                {
                    "name": "sample_count",
                    "passed": False,
                    "details": "mismatch_count=4",
                    "mismatch_count": 4,
                    "max_abs_error": 1.0,
                },
                {
                    "name": "ephys",
                    "passed": True,
                    "details": "OK",
                    "mismatch_count": 0,
                    "max_abs_error": 0.0,
                },
            ],
        }
    )

    assert report["summary_text"] == (
        "Validation failed. 1 of 2 checks reported problems. "
        "Failed checks: Sample Count Time Series."
    )
    assert "result_summary" in report["checks"][0]


def test_build_result_summary_handles_tolerated_differences():
    summary = _build_result_summary(
        {
            "passed": True,
            "details": "OK",
            "mismatch_count": 3,
            "max_abs_error": 0.5,
        },
        "Electrical Series",
    )

    assert "within the accepted tolerance" in summary


def test_validate_conversion_writes_report_and_closes_context(tmp_path, monkeypatch):
    nwb_path = tmp_path / "session.nwb"
    metadata_path = tmp_path / "session_metadata.yml"
    close_tracker = _CloseTracker()
    context = SimpleNamespace(
        rec_filepaths=[tmp_path / "session.rec"],
        nwb_filepath=nwb_path,
        metadata_filepath=metadata_path,
        behavior_only=False,
        nwb_io=close_tracker,
    )

    monkeypatch.setattr(
        validation_module, "_load_validation_context", lambda **_: context
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_header_and_metadata",
        lambda _: {
            "name": "header_and_metadata",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_electrodes",
        lambda _: {
            "name": "electrodes",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_sample_count",
        lambda _: {
            "name": "sample_count",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_dios",
        lambda _: {
            "name": "dios",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_analog",
        lambda _: {
            "name": "analog",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_validate_ephys",
        lambda _: {
            "name": "ephys",
            "passed": True,
            "details": "OK",
            "mismatch_count": 0,
            "max_abs_error": 0.0,
        },
    )

    report = validate_conversion(
        nwb_filepath=nwb_path,
        metadata_filepath=metadata_path,
    )

    report_path = Path(report["report_filepath"])
    assert report["passed"] is True
    assert close_tracker.closed is True
    assert report_path.exists()

    with report_path.open(encoding="utf-8") as stream:
        written = json.load(stream)
    assert written["summary"]["n_checks"] == 6
    assert written["summary_text"].startswith("Validation passed.")


def test_validate_conversion_reports_initialization_failure(tmp_path, monkeypatch):
    nwb_path = tmp_path / "session.nwb"
    metadata_path = tmp_path / "session_metadata.yml"
    monkeypatch.setattr(
        validation_module,
        "_load_validation_context",
        lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    report = validate_conversion(
        nwb_filepath=nwb_path,
        metadata_filepath=metadata_path,
        data_path=tmp_path,
    )

    assert report["passed"] is False
    assert report["checks"][0]["name"] == "initialization"
    assert report["errors"] == ["boom"]
    assert report["inputs"]["data_path"] == str(tmp_path)
    assert Path(report["report_filepath"]).exists()


def test_load_validation_context_uses_reconfig_and_default_tolerance(
    tmp_path, monkeypatch
):
    metadata_path = tmp_path / "session_metadata.yml"
    nwb_path = tmp_path / "session.nwb"
    rec_path = tmp_path / "session.rec"
    reconfig_path = tmp_path / "reconfig.trodesconf"
    for path in (metadata_path, nwb_path, rec_path, reconfig_path):
        path.write_text("x", encoding="utf-8")

    rec_header = _make_header(sampling_rate="20000")
    reconfig_header = _make_header(sampling_rate="25000")
    captured = {}
    fake_nwb = object()

    monkeypatch.setattr(
        validation_module,
        "_resolve_rec_filepaths",
        lambda **_: [rec_path],
    )
    monkeypatch.setattr(
        validation_module,
        "_get_included_device_metadata_paths",
        lambda: [tmp_path / "device.yml"],
    )

    def fake_load_metadata(metadata_arg, device_args):
        captured["load_metadata"] = (metadata_arg, device_args)
        return {"subject": {}}, [{"probe_type": "test"}]

    monkeypatch.setattr(validation_module, "load_metadata", fake_load_metadata)

    def fake_read_header(path):
        if Path(path) == reconfig_path:
            return reconfig_header
        return rec_header

    monkeypatch.setattr(validation_module, "read_header", fake_read_header)

    def fake_validate_yaml_header_electrode_map(metadata, spike_configuration):
        captured["spike_configuration"] = spike_configuration

    monkeypatch.setattr(
        validation_module,
        "validate_yaml_header_electrode_map",
        fake_validate_yaml_header_electrode_map,
    )
    monkeypatch.setattr(validation_module, "SpikeGadgetsRawIO", _FakeRawIO)

    class _FakeNWBHDF5IO:
        def __init__(self, path, mode, load_namespaces):
            captured["nwb_io_init"] = (path, mode, load_namespaces)

        def read(self):
            return fake_nwb

    monkeypatch.setattr(validation_module, "NWBHDF5IO", _FakeNWBHDF5IO)

    context = _load_validation_context(
        rec_filepaths=None,
        nwb_filepath=nwb_path,
        metadata_filepath=metadata_path,
        data_path=tmp_path,
        header_reconfig_path=reconfig_path,
        device_metadata_paths=None,
        behavior_only=False,
        max_ephys_frames_per_chunk=123,
        ephys_tolerance_uv=1.5,
        timestamp_tolerance_s=None,
    )

    assert context.rec_filepaths == [rec_path]
    assert context.rec_header is rec_header
    assert context.effective_header is reconfig_header
    assert context.nwbfile is fake_nwb
    assert context.timestamp_tolerance_s == pytest.approx(1 / 20000)
    assert captured["load_metadata"] == (
        str(metadata_path),
        [str(tmp_path / "device.yml")],
    )
    assert captured["spike_configuration"] is reconfig_header.find("SpikeConfiguration")
    assert captured["nwb_io_init"] == (str(nwb_path), "r", True)
    assert all(io.parse_header_called for io in context.neo_io)


def test_validate_header_and_metadata_reports_mismatches():
    header = _make_header()
    rec_global = header.find("GlobalConfiguration")
    header_device = SimpleNamespace(
        **{
            field_name: (
                rec_global.attrib[header_key]
                if field_name != "controller_serial"
                else "wrong-controller"
            )
            for field_name, header_key in validation_module.HEADER_DEVICE_FIELDS.items()
        }
    )
    nwbfile = SimpleNamespace(
        devices={"header_device": header_device},
        session_start_time=datetime.fromtimestamp(1718064001),
        session_description="wrong description",
        session_id="wrong id",
        experiment_description="wrong experiment",
        lab="wrong lab",
        institution="wrong institution",
        subject=SimpleNamespace(
            subject_id="rat2",
            description="desc2",
            genotype="geno2",
            sex="F",
            species="mouse",
        ),
    )
    context = SimpleNamespace(
        nwbfile=nwbfile,
        metadata={
            "session_description": "expected description",
            "session_id": "expected id",
            "experiment_description": "expected experiment",
            "lab": "expected lab",
            "institution": "expected institution",
            "subject": {
                "subject_id": "rat1",
                "description": "desc1",
                "genotype": "geno1",
                "sex": "M",
                "species": "rat",
            },
        },
        rec_header=header,
    )

    result = validation_module._validate_header_and_metadata(context)

    assert result["passed"] is False
    assert "controller_serial" in result["details"]
    assert "session_start_time" in result["details"]
    assert "subject.subject_id" in result["details"]


def test_validate_electrodes_handles_behavior_only_and_reference_mismatch(monkeypatch):
    nwbfile = SimpleNamespace(
        acquisition={},
        electrodes=SimpleNamespace(
            to_dataframe=lambda: pd.DataFrame(
                [
                    {
                        "group_name": "1",
                        "hwChan": "10",
                        "probe_electrode": 0,
                        "probe_shank": 0,
                        "ref_elect_id": 0,
                    },
                    {
                        "group_name": "2",
                        "hwChan": "20",
                        "probe_electrode": 0,
                        "probe_shank": 0,
                        "ref_elect_id": 0,
                    },
                ]
            )
        ),
    )
    context = SimpleNamespace(
        nwbfile=nwbfile,
        behavior_only=False,
        metadata={},
        effective_header=Element("Configuration"),
        device_metadata=[],
    )
    monkeypatch.setattr(validation_module, "make_hw_channel_map", lambda *_: {})
    monkeypatch.setattr(
        validation_module,
        "make_ref_electrode_map",
        lambda *_: {"1": ("2", 0), "2": (-1, -1)},
    )
    monkeypatch.setattr(
        validation_module,
        "_build_expected_electrode_rows",
        lambda *_: [
            {
                "group_name": "1",
                "hwChan": "10",
                "probe_electrode": 0,
                "probe_shank": 0,
            },
            {
                "group_name": "2",
                "hwChan": "20",
                "probe_electrode": 0,
                "probe_shank": 0,
            },
        ],
    )

    result = validation_module._validate_electrodes(context)
    behavior_result = validation_module._validate_electrodes(
        SimpleNamespace(
            nwbfile=SimpleNamespace(acquisition={"e-series": object()}),
            behavior_only=True,
        )
    )

    assert result["passed"] is False
    assert "ref_elect_id" in result["details"]
    assert behavior_result["passed"] is False
    assert "behavior_only=True" in behavior_result["details"]


def test_validate_sample_count_reports_data_and_timestamp_mismatch(monkeypatch):
    sample_count_series = SimpleNamespace(
        data=np.array([1, 99, 3]),
        timestamps=np.array([0.0, 0.5, 1.0]),
    )
    context = SimpleNamespace(
        nwbfile=SimpleNamespace(
            processing={
                "sample_count": SimpleNamespace(
                    data_interfaces={"sample_count": sample_count_series}
                )
            }
        ),
        neo_io=[
            SimpleNamespace(get_analogsignal_timestamps=lambda *_: np.array([1, 2, 3]))
        ],
        timestamp_tolerance_s=0.01,
        behavior_only=False,
    )
    monkeypatch.setattr(
        validation_module,
        "_get_rec_timestamps",
        lambda *_args, **_kwargs: np.array([0.0, 0.1, 0.2]),
    )
    monkeypatch.setattr(
        validation_module,
        "_get_primary_stream_id",
        lambda *_: "trodes",
    )

    result = validation_module._validate_sample_count(context)

    assert result["passed"] is False
    assert "sample_count data mismatch_count=1" in result["details"]
    assert "first_mismatch_index=1" in result["details"]


def test_validate_dios_reports_missing_series_and_metadata_mismatch(monkeypatch):
    actual_series = SimpleNamespace(
        description="wrong-desc",
        comments="wrong-comments",
        data=np.array([0, 0]),
        timestamps=np.array([0.0, 0.4]),
    )
    behavioral_events = _BehavioralEvents({"dio_a": actual_series})
    fake_io = SimpleNamespace(
        _mask_channels_ids={"ECU_digital": ["ECU_Din12"]},
        get_digitalsignal=lambda *_: (np.array([0.0, 0.1]), np.array([0, 1])),
    )
    context = SimpleNamespace(
        nwbfile=SimpleNamespace(
            processing={
                "behavior": {
                    "behavioral_events": behavioral_events,
                }
            }
        ),
        metadata={},
        neo_io=[fake_io],
        timestamp_tolerance_s=0.01,
    )
    monkeypatch.setattr(
        validation_module,
        "_get_channel_name_map",
        lambda *_: {
            "Din12": {"name": "dio_a", "comments": "expected comments"},
            "Din13": {"name": "dio_missing", "comments": "unused"},
        },
    )

    result = validation_module._validate_dios(context)

    assert result["passed"] is False
    assert "description mismatch" in result["details"]
    assert "comments mismatch" in result["details"]
    assert "state mismatch_count=1" in result["details"]
    assert "Missing DIO timeseries 'dio_missing'" in result["details"]


def test_validate_analog_reports_channel_order_and_data_mismatch(monkeypatch):
    analog_series = SimpleNamespace(
        description="chan_b   mux_1   ",
        data=np.zeros((2, 2)),
        timestamps=np.array([0.0, 0.1]),
    )
    context = SimpleNamespace(
        nwbfile=SimpleNamespace(
            processing={
                "analog": {
                    "analog": {
                        "analog": analog_series,
                    }
                }
            }
        ),
        rec_header=_make_header(),
        neo_io=[SimpleNamespace(multiplexed_channel_xml={"mux_1": object()})],
        rec_filepaths=[Path("a.rec")],
        behavior_only=False,
        max_ephys_frames_per_chunk=10,
        timestamp_tolerance_s=0.01,
    )
    monkeypatch.setattr(
        validation_module,
        "get_analog_channel_names",
        lambda *_: ["chan_a"],
    )

    class _FakeIterator:
        timestamps = np.array([0.0, 0.2])

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def _get_maxshape(self):
            return (2, 2)

    monkeypatch.setattr(validation_module, "RecFileDataChunkIterator", _FakeIterator)
    monkeypatch.setattr(
        validation_module,
        "_get_rec_timestamps",
        lambda *_args, **_kwargs: np.array([0.0, 0.2]),
    )
    monkeypatch.setattr(
        validation_module,
        "_get_primary_stream_id",
        lambda *_: "trodes",
    )
    monkeypatch.setattr(
        validation_module,
        "_compare_chunked_time_series",
        lambda **_: {
            "messages": ["mismatch_count=2, first_mismatch=(1, 0)"],
            "mismatch_count": 2,
            "max_abs_error": 3.0,
        },
    )
    monkeypatch.setattr(
        validation_module,
        "_compare_1d_arrays",
        lambda **_: {
            "messages": ["mismatch_count=1, first_mismatch_index=1"],
            "mismatch_count": 1,
            "max_abs_error": 0.1,
        },
    )

    result = validation_module._validate_analog(context)

    assert result["passed"] is False
    assert "analog channel order mismatch" in result["details"]
    assert result["mismatch_count"] == 3
    assert result["max_abs_error"] == 3.0


def test_validate_ephys_uses_metadata_conversion_fallback(monkeypatch):
    electrode_df = pd.DataFrame([{"hwChan": 10}, {"hwChan": 20}])
    e_series = SimpleNamespace(
        data=np.zeros((2, 2)),
        timestamps=np.array([0.0, 0.1]),
    )
    header = Element("Configuration")
    spike_configuration = SubElement(header, "SpikeConfiguration")
    SubElement(spike_configuration, "SpikeNTrode")
    context = SimpleNamespace(
        nwbfile=SimpleNamespace(
            acquisition={"e-series": e_series},
            electrodes=SimpleNamespace(to_dataframe=lambda: electrode_df),
        ),
        behavior_only=False,
        rec_header=header,
        metadata={"raw_data_to_volts": 0.000000195},
        rec_filepaths=[Path("a.rec")],
        max_ephys_frames_per_chunk=5,
        ephys_tolerance_uv=1.0,
        timestamp_tolerance_s=0.01,
    )
    captured = {}

    class _FakeIterator:
        timestamps = np.array([0.0, 0.1])

        def __init__(self, *args, **kwargs):
            captured["iterator_kwargs"] = kwargs

        def _get_maxshape(self):
            return (2, 2)

    monkeypatch.setattr(validation_module, "RecFileDataChunkIterator", _FakeIterator)
    monkeypatch.setattr(
        validation_module,
        "_compare_chunked_time_series",
        lambda **_: {"messages": [], "mismatch_count": 0, "max_abs_error": 0.0},
    )
    monkeypatch.setattr(
        validation_module,
        "_compare_1d_arrays",
        lambda **_: {"messages": [], "mismatch_count": 0, "max_abs_error": 0.0},
    )

    result = validation_module._validate_ephys(context)
    behavior_result = validation_module._validate_ephys(
        SimpleNamespace(
            behavior_only=True,
            nwbfile=SimpleNamespace(acquisition={"e-series": object()}),
        )
    )

    assert result["passed"] is True
    assert captured["iterator_kwargs"]["conversion"] == pytest.approx(0.195)
    assert behavior_result["passed"] is False
    assert "behavior_only=True" in behavior_result["details"]


@pytest.mark.integration
def test_validate_conversion_generated_nwb(tmp_path):
    from trodes_to_nwb.convert import create_nwbs, get_included_device_metadata_paths

    rec_files = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    metadata_path = data_path / "20230622_sample_metadata.yml"
    if not all(path.exists() for path in rec_files) or not metadata_path.exists():
        pytest.skip(
            "Validation integration fixtures are not available in this environment."
        )

    create_nwbs(
        path=data_path,
        device_metadata_paths=get_included_device_metadata_paths(),
        output_dir=str(tmp_path),
        n_workers=1,
        query_expression=(
            "animal == 'sample' and full_path != "
            f"'{str(data_path / '20230622_sample_metadataProbeReconfig.yml')}'"
        ),
        fs_gui_dir=data_path,
    )

    output_path = tmp_path / "sample20230622.nwb"
    assert output_path.exists()

    report = validate_conversion(
        data_path=data_path,
        nwb_filepath=output_path,
        metadata_filepath=metadata_path,
    )

    assert report["passed"] is True
    assert Path(report["report_filepath"]).exists()
    assert all(check["passed"] for check in report["checks"])
