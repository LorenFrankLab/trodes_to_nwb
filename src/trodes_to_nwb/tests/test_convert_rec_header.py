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


def _hwchans(ntrode):
    return [channel.attrib["hwChan"] for channel in ntrode]


def test_generate_reconfig_header_reproduces_handmade_reconfig():
    """Issue #113: merging the per-tetrode ntrodes of a raw header into per-probe
    ntrodes should reproduce the hand-edited reconfig .trodesconf exactly."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    # 32 tetrodes -> 4 probes of 8 tetrodes (32 channels) each
    ntrode_groups = [list(range(i, i + 8)) for i in range(1, 33, 8)]

    new_header = convert_rec_header.generate_reconfig_header(raw, ntrode_groups)
    new_ntrodes = list(new_header.find("SpikeConfiguration"))

    assert [nt.attrib["id"] for nt in new_ntrodes] == ["1", "2", "3", "4"]
    assert all(len(nt) == 32 for nt in new_ntrodes)

    # Channel order must match the real hand-made reconfig file.
    reference = list(
        convert_rec_header.read_header(
            data_path / "reconfig_probeDevice.trodesconf"
        ).find("SpikeConfiguration")
    )
    for produced, expected in zip(new_ntrodes, reference):
        assert _hwchans(produced) == _hwchans(expected)

    # The input header must not be mutated.
    assert len(list(raw.find("SpikeConfiguration"))) == 32


def test_generate_reconfig_header_supports_differing_contact_counts():
    """Issue #113: probes can have differing numbers of contacts, so merged
    ntrodes may have different channel counts."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    raw_ntrodes = list(raw.find("SpikeConfiguration"))
    # Unequal groups: sizes 4, 12, 12, 4 tetrodes -> 16, 48, 48, 16 channels
    ntrode_groups = [
        list(range(1, 5)),
        list(range(5, 17)),
        list(range(17, 29)),
        list(range(29, 33)),
    ]

    new_ntrodes = list(
        convert_rec_header.generate_reconfig_header(raw, ntrode_groups).find(
            "SpikeConfiguration"
        )
    )

    assert [len(nt) for nt in new_ntrodes] == [16, 48, 48, 16]
    # Each merged ntrode is the in-order concatenation of its source ntrodes.
    for produced, group in zip(new_ntrodes, ntrode_groups):
        expected = [h for src in group for h in _hwchans(raw_ntrodes[src - 1])]
        assert _hwchans(produced) == expected


def test_write_reconfig_trodesconf_roundtrips_and_validates(tmp_path):
    """Issue #113: a generated reconfig .trodesconf must be re-readable and pass
    the conversion's own header/metadata validation."""
    out_path = tmp_path / "generated_reconfig.trodesconf"
    # 32 tetrodes -> 4 probes of 8, via the uniform-grouping convenience.
    ntrode_groups = convert_rec_header.group_ntrodes_uniformly(32, 8)

    convert_rec_header.write_reconfig_trodesconf(
        data_path / "20230622_sample_01_a1.rec", out_path, ntrode_groups
    )
    assert out_path.exists()

    reread = convert_rec_header.read_header(out_path)
    spike_config = reread.find("SpikeConfiguration")
    assert len(list(spike_config)) == 4

    # Non-SpikeConfiguration sections must survive the write/read round-trip --
    # GlobalConfiguration in particular is required by add_header_device.
    source = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    assert reread.find("GlobalConfiguration") is not None
    assert {child.tag for child in source} == {child.tag for child in reread}

    # The generated header validates against the matching reconfig metadata.
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadataProbeReconfig.yml", []
    )
    convert_rec_header.validate_yaml_header_electrode_map(metadata, spike_config)


def test_group_ntrodes_uniformly():
    """Issue #113: uniform grouping chunks consecutive 1-based source ntrode ids
    into equal-size groups (the equal-shank-probe convenience)."""
    assert convert_rec_header.group_ntrodes_uniformly(32, 8) == [
        list(range(1, 9)),
        list(range(9, 17)),
        list(range(17, 25)),
        list(range(25, 33)),
    ]
    assert convert_rec_header.group_ntrodes_uniformly(8, 2) == [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
    ]
    assert convert_rec_header.group_ntrodes_uniformly(4, 4) == [[1, 2, 3, 4]]

    # Non-uniform / invalid inputs are rejected rather than guessing.
    with pytest.raises(ValueError, match="not a multiple"):
        convert_rec_header.group_ntrodes_uniformly(30, 8)
    with pytest.raises(ValueError, match="positive"):
        convert_rec_header.group_ntrodes_uniformly(0, 8)


def test_generate_reconfig_header_rejects_duplicate_or_dropped_channels():
    """Issue #113 review: a non-partition ntrode_groups would silently duplicate
    or drop channels in the probe map, so it must fail loudly."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")

    # A source ntrode assigned to two groups (one channel -> two electrode groups)
    with pytest.raises(ValueError, match="more than one"):
        convert_rec_header.generate_reconfig_header(raw, [[1, 2], [2, 3]])
    # The same source ntrode twice within one group
    with pytest.raises(ValueError, match="more than one"):
        convert_rec_header.generate_reconfig_header(raw, [[1, 1]])
    # Leaving most of the 32 source ntrodes unassigned (their channels dropped)
    with pytest.raises(ValueError, match="not assigned"):
        convert_rec_header.generate_reconfig_header(raw, [[1, 2]])

    # allow_partial=True permits intentionally dropping the unassigned ntrodes.
    partial = convert_rec_header.generate_reconfig_header(
        raw, [[1, 2]], allow_partial=True
    )
    assert len(list(partial.find("SpikeConfiguration"))) == 1
