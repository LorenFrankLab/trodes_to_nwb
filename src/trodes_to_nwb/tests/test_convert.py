import os
import shutil
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
from trodes_to_nwb.convert import create_nwbs, get_included_probe_metadata_paths
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


def test_get_included_probe_metadat_paths():
    probes = get_included_probe_metadata_paths()
    assert len(probes) == 3
    assert [probe.exists() for probe in probes]


def test_convert_full():
    probe_metadata = [data_path / "tetrode_12.5.yml"]

    video_directory = data_path / "temp_video_directory_full_convert"
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    exclude_reconfig_yaml = str(data_path / "20230622_sample_metadataProbeReconfig.yml")
    create_nwbs(
        path=data_path,
        probe_metadata_paths=probe_metadata,
        output_dir=str(data_path),
        n_workers=1,
        query_expression=f"animal == 'sample' and full_path != '{exclude_reconfig_yaml}'",
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


def test_convert_full_with_inspector_error(mocker):
    def do_nothing(nwbfile, metadata_dict):
        pass

    mocker.patch("trodes_to_nwb.convert.add_subject", do_nothing)

    probe_metadata = [data_path / "tetrode_12.5.yml"]

    video_directory = data_path / "temp_video_directory_full_convert"
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    exclude_reconfig_yaml = str(data_path / "20230622_sample_metadataProbeReconfig.yml")
    create_nwbs(
        path=data_path,
        probe_metadata_paths=probe_metadata,
        output_dir=str(data_path),
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
    for dio_name in old_nwbfile.processing["behavior"][
        "behavioral_events"
    ].time_series.keys():
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
        assert (current_dio.unit == old_dio.unit) or (
            (current_dio.unit == "-1") and (old_dio.unit == "'unspecified'")
        )  # old rec_to_nwb conversions have a different default for unspecified units
        assert current_dio.description == old_dio.description

    # Compare position data
    for series in nwbfile.processing["behavior"]["position"].spatial_series.keys():
        # check series in new nwbfile
        assert (
            series in nwbfile.processing["behavior"]["position"].spatial_series.keys()
        )
        # find the corresponding data in the old file
        validated = False
        for old_series in old_nwbfile.processing["behavior"][
            "position"
        ].spatial_series.keys():
            # check that led number matches
            if not series.split("_")[1] == old_series.split("_")[1]:
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