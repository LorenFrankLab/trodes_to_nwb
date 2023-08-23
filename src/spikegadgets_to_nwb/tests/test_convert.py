from spikegadgets_to_nwb.data_scanner import get_file_info
from spikegadgets_to_nwb.convert import create_nwbs, _create_nwb

import os
import pandas as pd
from pathlib import Path
from pynwb import NWBHDF5IO

path = os.path.dirname(os.path.abspath(__file__))


def test_get_file_info():
    try:
        # running on github
        data_path = Path(os.environ.get("DOWNLOAD_DIR"))
    except (TypeError, FileNotFoundError):
        # running locally
        data_path = Path(path)
    path_df = get_file_info(data_path)

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


def test_convert():
    try:
        # running on github
        data_path = Path(os.environ.get("DOWNLOAD_DIR"))
        yml_data_path = Path(path + "/test_data")
        yml_path_df = get_file_info(yml_data_path)
        yml_path_df = yml_path_df[yml_path_df.file_extension == ".yml"]
        append_yml_df = True
    except (TypeError, FileNotFoundError):
        # running locally
        data_path = Path(path + "/test_data")
        append_yml_df = False
    probe_metadata = [Path(path + "/test_data/tetrode_12.5.yml")]

    # make session_df
    path_df = get_file_info(data_path)
    if append_yml_df:
        path_df = pd.concat([path_df, yml_path_df])
        path_df = path_df[
            path_df.full_path
            != yml_data_path.as_posix() + "/20230622_sample_metadataProbeReconfig.yml"
        ]
    else:
        path_df = path_df[
            path_df.full_path
            != data_path.as_posix() + "/20230622_sample_metadataProbeReconfig.yml"
        ]
    session_df = path_df[(path_df.animal == "sample")]
    assert len(session_df[session_df.file_extension == ".yml"]) == 1
    _create_nwb(
        session=("20230622", "sample", "1"),
        session_df=session_df,
        probe_metadata_paths=probe_metadata,
        output_dir=str(data_path),
    )
    assert "sample20230622.nwb" in os.listdir(str(data_path))
    with NWBHDF5IO(str(data_path) + "/sample20230622.nwb") as io:
        nwbfile = io.read()
        with NWBHDF5IO(str(data_path) + "/minirec20230622_.nwb") as io2:
            old_nwbfile = io2.read()

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
    # cleanup
    os.remove(str(data_path) + "/sample20230622.nwb")


def check_module_entries(test, reference):
    todo = [
        "camera_sample_frame_counts",
        "video_files",
        "dataacq_device0",
    ]  # TODO: known missing entries
    for entry in reference:
        if entry in todo:
            continue
        assert entry in test
