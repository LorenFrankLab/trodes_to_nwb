import numpy as np
import pytest

from trodes_to_nwb.convert import create_nwbs, get_included_device_metadata_paths
from trodes_to_nwb.tests.utils import data_path
from trodes_to_nwb.validate_conversion import (
    _compare_1d_arrays,
    _compare_chunked_time_series,
    _parse_session_from_metadata_filepath,
    _resolve_report_filepath,
    _resolve_rec_filepaths,
    validate_conversion,
)


class _ArraySource:
    def __init__(self, data: np.ndarray):
        self.data = data

    def _get_data(self, selection):
        time_slice, channel_slice = selection
        return self.data[time_slice, channel_slice]


def test_compare_1d_arrays_reports_mismatch_count():
    result = _compare_1d_arrays(
        expected=np.array([1.0, 2.0, 3.0]),
        actual=np.array([1.0, 2.2, 3.5]),
        atol=0.1,
    )

    assert result["mismatch_count"] == 2
    assert result["max_abs_error"] == pytest.approx(0.5)
    assert "first_mismatch_index=1" in result["messages"][0]


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


@pytest.mark.integration
def test_validate_conversion_generated_nwb(tmp_path):
    rec_files = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    metadata_path = data_path / "20230622_sample_metadata.yml"
    if not all(path.exists() for path in rec_files) or not metadata_path.exists():
        pytest.skip("Validation integration fixtures are not available in this environment.")

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
