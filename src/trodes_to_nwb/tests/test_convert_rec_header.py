from xml.etree import ElementTree

import pytest
from ndx_franklab_novela import HeaderDevice

from trodes_to_nwb import convert, convert_rec_header, convert_yaml
from trodes_to_nwb.tests.utils import data_path


def default_test_xml_tree() -> ElementTree:
    """Function to return a default xml tree for intial nwb generation

    Returns
    -------
    ElementTree
        root xml tree for intial nwb generation
    """
    trodesconf_file = data_path / "20230622_sample_01_a1.rec"
    # "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)
    return rec_header


def test_add_header_device():
    # Set up test data
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())
    recfile = data_path / "20230622_sample_01_a1.rec"

    # Call the function to be tested
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
    recfile = data_path / "bad_header.trodesconf"
    with pytest.raises(
        ValueError,
        match="SpikeGadgets: the xml header does not contain '</Configuration>'",
    ):
        convert_rec_header.read_header(recfile)


def test_detect_ptp():
    convert.setup_logger("convert", "testing.log")
    assert convert_rec_header.detect_ptp_from_header(default_test_xml_tree())


def test_validate_yaml_header_electrode_map():
    # get metadata and rec_header
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    recfile = data_path / "20230622_sample_01_a1.rec"
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
    with pytest.raises(IndexError, match="Ntrode count mismatch"):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )
    # check if error is raised when there is missing channel map
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    metadata["ntrode_electrode_group_channel_map"].pop(0)
    with pytest.raises(
        KeyError, match="ntrode 1 is present in the rec/reconfig header"
    ):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )
    # check if error is raised when channel map has wrong number of channels
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    metadata["ntrode_electrode_group_channel_map"][0]["map"]["4"] = 4
    with pytest.raises(
        ValueError,
        match="Channel count mismatch for ntrode 1",
    ):
        convert_rec_header.validate_yaml_header_electrode_map(
            metadata, rec_header.find("SpikeConfiguration")
        )


def _two_ntrode_spike_config() -> ElementTree.Element:
    """Minimal SpikeConfiguration with two ntrodes (ids 1 and 2), two channels each."""
    return ElementTree.fromstring(
        """
        <SpikeConfiguration>
          <SpikeNTrode id="1"><SpikeChannel/><SpikeChannel/></SpikeNTrode>
          <SpikeNTrode id="2"><SpikeChannel/><SpikeChannel/></SpikeNTrode>
        </SpikeConfiguration>
        """
    )


def _two_ntrode_metadata() -> dict:
    """Metadata whose channel map matches `_two_ntrode_spike_config`."""
    return {
        "ntrode_electrode_group_channel_map": [
            {"ntrode_id": 1, "electrode_group_id": 0, "map": {"0": 0, "1": 1}},
            {"ntrode_id": 2, "electrode_group_id": 1, "map": {"0": 2, "1": 3}},
        ]
    }


def test_validate_yaml_header_electrode_map_error_messages():
    """Issue #107: validation errors should name both sources (rec/reconfig
    header vs. metadata YAML) and report the channel/ntrode counts of each so
    the mismatch can be diagnosed quickly.

    Hermetic: builds a synthetic SpikeConfiguration and metadata so the test
    does not depend on the bulk .rec data downloaded only in CI.
    """
    spike_config = _two_ntrode_spike_config()

    # Sanity check: the unperturbed pair validates without error.
    convert_rec_header.validate_yaml_header_electrode_map(
        _two_ntrode_metadata(), spike_config
    )

    # 1. Channel-count mismatch: header ntrode 1 has 2 channels, YAML lists 3.
    metadata = _two_ntrode_metadata()
    metadata["ntrode_electrode_group_channel_map"][0]["map"]["2"] = 4
    with pytest.raises(ValueError) as excinfo:
        convert_rec_header.validate_yaml_header_electrode_map(metadata, spike_config)
    msg = str(excinfo.value)
    assert "ntrode 1" in msg.lower()
    assert "header" in msg.lower()
    assert "yaml" in msg.lower()
    assert "2" in msg  # count from the header
    assert "3" in msg  # count from the metadata YAML

    # 2. Header has fewer ntrodes than the YAML defines: report both counts.
    metadata = _two_ntrode_metadata()
    metadata["ntrode_electrode_group_channel_map"].append(
        {"ntrode_id": 3, "electrode_group_id": 2, "map": {"0": 4, "1": 5}}
    )
    with pytest.raises(IndexError) as excinfo:
        convert_rec_header.validate_yaml_header_electrode_map(metadata, spike_config)
    msg = str(excinfo.value)
    assert "header" in msg.lower()
    assert "yaml" in msg.lower()
    assert "2" in msg  # ntrode count from the header
    assert "3" in msg  # ntrode count from the metadata YAML

    # 3. Header ntrode missing from the YAML: name both sources and the ntrode.
    metadata = _two_ntrode_metadata()
    metadata["ntrode_electrode_group_channel_map"].pop(0)  # drop ntrode 1
    with pytest.raises(KeyError) as excinfo:
        convert_rec_header.validate_yaml_header_electrode_map(metadata, spike_config)
    msg = str(excinfo.value)
    assert "1" in msg  # the ntrode id present in header but missing from YAML
    assert "header" in msg.lower()
    assert "yaml" in msg.lower()
