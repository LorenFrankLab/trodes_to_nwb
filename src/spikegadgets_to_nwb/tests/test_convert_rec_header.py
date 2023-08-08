from spikegadgets_to_nwb import convert_yaml, convert_rec_header

from pynwb.file import NWBFile
from ndx_franklab_novela import HeaderDevice
from xml.etree import ElementTree

import os
import pytest

path = os.path.dirname(os.path.abspath(__file__))


def default_test_xml_tree() -> ElementTree:
    """Function to return a default xml tree for intial nwb generation

    Returns
    -------
    ElementTree
        root xml tree for intial nwb generation
    """
    try:
        # running on github
        trodesconf_file = (
            os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_01_a1.rec"
        )  # "/test_data/reconfig_probeDevice.trodesconf"
        rec_header = convert_rec_header.read_header(trodesconf_file)
    except:
        # running locally
        trodesconf_file = (
            path + "/test_data/20230622_sample_01_a1.rec"
        )  # "/test_data/reconfig_probeDevice.trodesconf"
        rec_header = convert_rec_header.read_header(trodesconf_file)
    return rec_header


def test_add_header_device():
    # Set up test data
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    # Call the function to be tested
    try:
        # running on github
        recfile = os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_01_a1.rec"
        convert_rec_header.add_header_device(
            nwbfile, convert_rec_header.read_header(recfile)
        )
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = path + "/test_data/20230622_sample_01_a1.rec"
        convert_rec_header.add_header_device(
            nwbfile, convert_rec_header.read_header(recfile)
        )

    # Perform assertions to check the results
    # Check if the device was added correctly
    assert "header_device" in nwbfile.devices
    header_device = nwbfile.devices["header_device"]
    assert isinstance(header_device, HeaderDevice)

    # Check if the device attributes were set correctly
    assert header_device.headstage_serial == "01504 00126"
    assert header_device.headstage_smart_ref_on == "0"
    assert header_device.realtime_mode == "0"
    assert header_device.headstage_auto_settle_on == "0"
    assert header_device.timestamp_at_creation == "51493215"
    assert header_device.controller_firmware_version == "3.18"
    assert header_device.controller_serial == "65535 65535"
    assert header_device.save_displayed_chan_only == "1"
    assert header_device.headstage_firmware_version == "4.4"
    assert header_device.qt_version == "6.2.2"
    assert header_device.compile_date == "May 24 2023"
    assert header_device.compile_time == "10:59:15"
    assert header_device.file_prefix == ""
    assert header_device.headstage_gyro_sensor_on == "1"
    assert header_device.headstage_mag_sensor_on == "0"
    assert header_device.trodes_version == "2.4.0"
    assert header_device.headstage_accel_sensor_on == "1"
    assert header_device.commit_head == "heads/Release_2.4.0-0-g499429f3"
    assert header_device.system_time_at_creation == "       1687474797888"
    assert header_device.file_path == ""

    # Check if error raised if improper header file is passed
    recfile = path + "/test_data/bad_header.trodesconf"
    with pytest.raises(
        ValueError,
        match="SpikeGadgets: the xml header does not contain '</Configuration>'",
    ):
        convert_rec_header.read_header(recfile)


def test_detect_ptp():
    assert convert_rec_header.detect_ptp_from_header(default_test_xml_tree())


def test_validate_yaml_header_electrode_map():
    # get metadata and rec_header
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    try:
        # running on github
        recfile = os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_01_a1.rec"
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = path + "/test_data/20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(recfile)

    # correct matching
    convert_rec_header.validate_yaml_header_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    # check if error is raised when there is extra channel map
    new_map_entry = metadata["ntrode_electrode_group_channel_map"][0].copy()
    new_map_entry["ntrode_id"] = 33
    new_map_entry["electrode_group_id"] = 32
    metadata["ntrode_electrode_group_channel_map"].append(new_map_entry)
    with pytest.raises(
        IndexError, match="XML Header contains less ntrodes than the yaml indicates"
    ):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )
    # check if error is raised when there is missing channel map
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    metadata["ntrode_electrode_group_channel_map"].pop(0)
    with pytest.raises(KeyError, match="Missing yaml metadata for ntrodes 1"):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )
    # check if error is raised when channel map has wrong number of channels
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    metadata["ntrode_electrode_group_channel_map"][0]["map"]["4"] = 4
    with pytest.raises(
        ValueError,
        match="Ntrode group 1 does not contain the number of channels indicated by the metadata yaml",
    ):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )
