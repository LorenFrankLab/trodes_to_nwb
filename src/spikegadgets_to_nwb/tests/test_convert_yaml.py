from spikegadgets_to_nwb import convert_yaml
from datetime import datetime
from pynwb.file import Subject, ProcessingModule
from ndx_franklab_novela import Probe, Shank, ShanksElectrode
from hdmf.common.table import DynamicTable, VectorData

import os

path = os.path.dirname(os.path.abspath(__file__))


def test_initial_nwb_creation():
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata)
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
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata)
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
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata)
    convert_yaml.add_cameras(nwb_file, metadata)
    cameras = nwb_file.devices
    assert len(cameras) == 2
    name = "camera_device " + str(metadata["cameras"][0]["id"])
    assert cameras[name].meters_per_pixel == 0.001
    assert cameras[name].model == "model1"
    assert cameras[name].lens == "lens1"
    assert cameras[name].manufacturer == "Allied Vision"


def test_acq_device_creation():
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwb_file = convert_yaml.initialize_nwb(metadata)
    convert_yaml.add_acquisition_devices(nwb_file, metadata)
    devices = nwb_file.devices
    assert len(devices) == 1
    name = f"dataacq_deviceSpikeGadgets"
    assert devices[name].system == "Main Control Unit"
    assert devices[name].amplifier == "Intan"
    assert devices[name].adc_circuit == "Intan"


def test_electrode_creation():
    metadata_path = path + "/test_data/test_metadata.yml"
    probe_metadata = [
        path + "/test_data/tetrode_12.5.yml",
    ]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata)

    # Call the function to be tested
    convert_yaml.add_electrode_groups(nwbfile, metadata, probe_metadata)

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
    for j, shank_meta in enumerate(shank_meta_list):
        shank = probe.shanks[str(j)]
        assert isinstance(shank, Shank)
        assert len(shank.shanks_electrodes) == len(shank_meta["electrodes"])
        for k, electrode_meta in enumerate(shank_meta["electrodes"]):
            electrode = shank.shanks_electrodes[str(k)]
            assert isinstance(electrode, ShanksElectrode)
            assert electrode.rel_x == float(electrode_meta["rel_x"])
            assert electrode.rel_y == float(electrode_meta["rel_y"])
            assert electrode.rel_z == float(electrode_meta["rel_z"])

    # Check if the electrode table was extended correctly
    assert len(nwbfile.electrodes.columns) == 13


def test_add_tasks():
    # Set up test data
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata)

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


def test_add_associated_files():
    # Set up test data
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata)
    # Change path of files to be relative to this directory
    for assoc_meta in metadata["associated_files"]:
        assoc_meta["path"] = path + "/test_data/"
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
