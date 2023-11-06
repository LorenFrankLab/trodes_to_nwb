import logging
import os
import shutil
from datetime import datetime

from hdmf.common.table import DynamicTable, VectorData
from ndx_franklab_novela import CameraDevice, Probe, Shank, ShanksElectrode
from pynwb.file import ProcessingModule, Subject
from trodes_to_nwb.convert_position import add_associated_video_files
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

from trodes_to_nwb import convert, convert_rec_header, convert_yaml


def test_initial_nwb_creation():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    # check that things were added in
    assert nwb_file.experimenter == ["lastname, firstname", "lastname2, firstname2"]
    assert isinstance(nwb_file.session_start_time, datetime)
    assert isinstance(nwb_file.timestamps_reference_time, datetime)
    assert nwb_file.session_description == "test yaml insertion"
    assert nwb_file.session_id == "12345"
    assert nwb_file.lab == "Loren Frank Lab"
    assert nwb_file.experiment_description == "Test Conversion"
    assert nwb_file.institution == "UCSF"
    assert isinstance(nwb_file.identifier, str)
    # confirm that undefined fields are empty
    assert nwb_file.electrodes is None
    assert len(nwb_file.acquisition) == 0
    assert len(nwb_file.processing) == 0
    assert len(nwb_file.devices) == 0


def test_subject_creation():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_yaml.add_subject(nwb_file, metadata)
    subject = nwb_file.subject
    assert isinstance(nwb_file.subject, Subject)
    assert subject.weight == "100 g"
    assert isinstance(subject.sex, str) and len(subject.sex)
    assert subject.description == "Long-Evans Rat"
    assert subject.species == "Rattus pyctoris"
    assert subject.genotype == "Obese Prone CD Rat"
    assert subject.subject_id == "54321"
    assert isinstance(subject.date_of_birth, datetime)


def test_camera_creation():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_yaml.add_cameras(nwb_file, metadata)
    cameras = nwb_file.devices
    assert len(cameras) == 2
    name = "camera_device " + str(metadata["cameras"][0]["id"])
    assert cameras[name].meters_per_pixel == 0.001
    assert cameras[name].model == "model1"
    assert cameras[name].lens == "lens1"
    assert cameras[name].manufacturer == "Allied Vision"


def test_acq_device_creation():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_yaml.add_acquisition_devices(nwb_file, metadata)
    devices = nwb_file.devices
    assert len(devices) == 1
    name = f"dataacq_device0"
    assert devices[name].system == "Main Control Unit"
    assert devices[name].amplifier == "Intan"
    assert devices[name].adc_circuit == "Intan"


def test_electrode_creation():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using rec data
    recfile = data_path / "20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(recfile)
    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    # Call the function to be tested
    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    # Perform assertions to check the results
    # Check if the electrode groups were added correctly
    assert len(nwbfile.electrode_groups) == len(metadata["electrode_groups"])
    for i, group_metadata in enumerate(metadata["electrode_groups"]):
        group = nwbfile.electrode_groups[str(group_metadata["id"])]
        assert group.description == group_metadata["description"]
        assert group.location == group_metadata["targeted_location"]

    # Check if the probes were added correctly
    assert len(nwbfile.devices) == len(metadata["electrode_groups"])
    probe = nwbfile.devices["probe 0"]
    probe_metadata = probe_metadata[0]

    assert isinstance(probe, Probe)
    assert probe.probe_type == probe_metadata["probe_type"]
    assert probe.units == probe_metadata["units"]
    assert probe.probe_description == probe_metadata["probe_description"]
    assert probe.contact_side_numbering == probe_metadata["contact_side_numbering"]
    assert probe.contact_size == probe_metadata["contact_size"]

    # Check if the shanks and electrodes were added correctly
    shank_meta_list = probe_metadata["shanks"]
    assert len(probe.shanks) == len(shank_meta_list)
    electrode_id = 0
    for j, shank_meta in enumerate(shank_meta_list):
        shank = probe.shanks[str(j)]
        assert isinstance(shank, Shank)
        assert len(shank.shanks_electrodes) == len(shank_meta["electrodes"])
        for k, electrode_meta in enumerate(shank_meta["electrodes"]):
            electrode = shank.shanks_electrodes[str(electrode_id)]
            assert isinstance(electrode, ShanksElectrode)
            assert electrode.rel_x == float(electrode_meta["rel_x"])
            assert electrode.rel_y == float(electrode_meta["rel_y"])
            assert electrode.rel_z == float(electrode_meta["rel_z"])
            electrode_id += 1

    # Check if the electrode table was extended correctly
    assert len(nwbfile.electrodes.columns) == 18
    # Check that electrode table hwChan is correct
    assert list(nwbfile.electrodes.to_dataframe()["hwChan"][:4]) == [
        "29",
        "25",
        "28",
        "21",
    ]
    # Check that electrode table reference electrode is correct
    assert list(nwbfile.electrodes.to_dataframe()["ref_elect_id"][:4]) == [0, 0, 0, 0]


def test_electrode_creation_reconfigured():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # swap two channels in the map to test hw channel mapping
    metadata["ntrode_electrode_group_channel_map"][-1]["map"]["30"] = 127
    metadata["ntrode_electrode_group_channel_map"][-1]["map"]["31"] = 126

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)
    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    # Call the function to be tested
    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    # Perform assertions to check the results
    # Check if the electrode groups were added correctly
    assert len(nwbfile.electrode_groups) == len(metadata["electrode_groups"])
    for i, group_metadata in enumerate(metadata["electrode_groups"]):
        group = nwbfile.electrode_groups[str(group_metadata["id"])]
        assert group.description == group_metadata["description"]
        assert group.location == group_metadata["targeted_location"]

    # Check if the probes were added correctly
    assert len(nwbfile.devices) == len(metadata["electrode_groups"])
    probe = nwbfile.devices["probe 0"]
    probe_metadata = probe_metadata[0]

    assert isinstance(probe, Probe)
    assert probe.probe_type == probe_metadata["probe_type"]
    assert probe.units == probe_metadata["units"]
    assert probe.probe_description == probe_metadata["probe_description"]
    assert probe.contact_side_numbering == probe_metadata["contact_side_numbering"]
    assert probe.contact_size == probe_metadata["contact_size"]

    # Check if the shanks and electrodes were added correctly
    shank_meta_list = probe_metadata["shanks"]
    assert len(probe.shanks) == len(shank_meta_list)
    electrode_id = 0
    for j, shank_meta in enumerate(shank_meta_list):
        shank = probe.shanks[str(j)]
        assert isinstance(shank, Shank)
        assert len(shank.shanks_electrodes) == len(shank_meta["electrodes"])
        for electrode_meta in shank_meta["electrodes"]:
            electrode = shank.shanks_electrodes[str(electrode_id)]
            assert isinstance(electrode, ShanksElectrode)
            assert electrode.rel_x == float(electrode_meta["rel_x"])
            assert electrode.rel_y == float(electrode_meta["rel_y"])
            assert electrode.rel_z == float(electrode_meta["rel_z"])
            electrode_id += 1

    # Check if the electrode table was extended correctly
    assert len(nwbfile.electrodes.columns) == 18
    # Check that electrode table hwChan is correct
    assert list(nwbfile.electrodes.to_dataframe()["hwChan"][:4]) == [
        "29",
        "25",
        "28",
        "21",
    ]
    # Check that electrode table reference electrode is correct
    assert list(nwbfile.electrodes.to_dataframe()["ref_elect_id"][:4]) == [0, 0, 0, 0]
    # check that hw channel mapping from channel_map is correct
    assert list(nwbfile.electrodes.to_dataframe()["hwChan"][-2:]) == ["115", "107"]


def test_add_tasks():
    # Set up test data
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # Call the function to be tested
    convert_yaml.add_tasks(nwbfile, metadata)

    # Perform assertions to check the results
    # Check if the processing module was added correctly
    assert "tasks" in nwbfile.processing
    tasks_module = nwbfile.processing["tasks"]
    assert isinstance(tasks_module, ProcessingModule)
    assert tasks_module.name == "tasks"
    assert tasks_module.description == "Contains all tasks information"

    # Check if the tasks were added correctly
    assert len(tasks_module.data_interfaces) == len(metadata["tasks"])
    for i, task_metadata in enumerate(metadata["tasks"]):
        task = tasks_module.data_interfaces[f"task_{i}"]
        assert isinstance(task, DynamicTable)
        assert task.name == f"task_{i}"
        assert task.description == ""
        assert len(task.columns) == 5

        # Check if the task metadata columns were added correctly
        for val in task.columns:
            assert isinstance(val, VectorData)
        for a, b in zip(
            task.colnames,
            (
                "task_name",
                "task_description",
                "camera_id",
                "task_epochs",
                "task_environment",
            ),
        ):
            assert a == b

        # Check if the task metadata values were added correctly
        task_df = task.to_dataframe()
        assert task_df["task_name"][0] == task_metadata["task_name"]
        assert task_df["task_description"][0] == task_metadata["task_description"]
        assert task_df["camera_id"][0] == [int(id) for id in task_metadata["camera_id"]]
        assert task_df["task_epochs"][0] == [
            int(epoch) for epoch in task_metadata["task_epochs"]
        ]
        assert task_df["task_environment"][0] == task_metadata["task_environment"]


def test_add_associated_files(capsys):
    # Create a logger
    logger = convert.setup_logger("convert", "testing.log")
    # Set up test data
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    # Change path of files to be relative to this directory
    for assoc_meta in metadata["associated_files"]:
        assoc_meta["path"] = str(data_path / assoc_meta["name"])
    # call the function to test
    convert_yaml.add_associated_files(nwbfile, metadata)
    assert "associated_files" in nwbfile.processing
    assert len(nwbfile.processing["associated_files"].data_interfaces) == len(
        metadata["associated_files"]
    )
    assert "associated1.txt" in nwbfile.processing["associated_files"].data_interfaces
    assert (
        nwbfile.processing["associated_files"]["associated1.txt"].description
        == "good file"
    )
    assert (
        nwbfile.processing["associated_files"]["associated1.txt"].task_epochs == "1, "
    )
    assert (
        nwbfile.processing["associated_files"]["associated1.txt"].content
        == "test file 1"
    )

    # Test printed errormessage for missing file
    # Change path of files to be relative to this directory
    metadata["associated_files"][0]["path"] = "bad_path.txt"
    metadata["associated_files"][0]["name"] = "bad_path.txt"
    metadata["associated_files"].pop(1)
    convert_yaml.add_associated_files(nwbfile, metadata)
    printed_warning = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_path = handler.baseFilename
            with open(log_file_path, "r") as log_file:
                for line in log_file.readlines():
                    if "ERROR: associated file bad_path.txt does not exist" in line:
                        printed_warning = True
                    break
    assert printed_warning


def test_add_associated_video_files():
    # Set up test data
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    convert_yaml.add_cameras(nwbfile, metadata)

    # make session_df
    path_df = get_file_info(data_path)
    session_df = path_df[(path_df.animal == "sample")]

    # make temp video directory
    video_directory = data_path / "temp_video_directory"
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)

    # Call the function to be tested
    add_associated_video_files(
        nwbfile, metadata, session_df, video_directory=str(video_directory)
    )
    assert "video_files" in nwbfile.processing
    assert "video" in nwbfile.processing["video_files"].data_interfaces
    assert len(nwbfile.processing["video_files"]["video"].time_series) == 2

    for video, video_meta in zip(
        nwbfile.processing["video_files"]["video"].time_series,
        metadata["associated_video_files"],
    ):
        video = nwbfile.processing["video_files"]["video"][video]
        assert video.name == video_meta["name"]
        assert video.format == "external"
        assert video.timestamps_unit == "seconds"
        assert video.timestamps is not None
        assert isinstance(video.device, CameraDevice)
        assert (video_directory / video.external_file[0]).exists()

    # cleanup
    shutil.rmtree(video_directory)
