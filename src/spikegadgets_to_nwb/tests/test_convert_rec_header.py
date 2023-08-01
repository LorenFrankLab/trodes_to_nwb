from spikegadgets_to_nwb import convert_yaml, convert_rec_header

from pynwb.file import NWBFile
from ndx_franklab_novela import HeaderDevice
from xml.etree import ElementTree

import os

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
            os.environ.get("DOWNLOAD_DIR") + "/20230622_155936.rec"
        )  # "/test_data/reconfig_probeDevice.trodesconf"
        rec_header = convert_rec_header.read_header(trodesconf_file)
    except:
        # running locally
        trodesconf_file = (
            path + "/test_data/20230622_155936.rec"
        )  # "/test_data/reconfig_probeDevice.trodesconf"
        rec_header = convert_rec_header.read_header(trodesconf_file)
    return rec_header


def test_add_header_device():
    # Set up test data
    metadata_path = path + "/test_data/test_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    # Call the function to be tested
    try:
        # running on github
        recfile = os.environ.get("DOWNLOAD_DIR") + "/20230622_155936.rec"
        convert_rec_header.add_header_device(
            nwbfile, convert_rec_header.read_header(recfile)
        )
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = path + "/test_data/20230622_155936.rec"
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
