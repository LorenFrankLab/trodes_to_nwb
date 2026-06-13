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

    # Must match the real hand-made reconfig file exactly: every SpikeNTrode
    # attribute and every SpikeChannel attribute, in order -- not just hwChan.
    reference = list(
        convert_rec_header.read_header(
            data_path / "reconfig_probeDevice.trodesconf"
        ).find("SpikeConfiguration")
    )
    for produced, expected in zip(new_ntrodes, reference):
        assert produced.attrib == expected.attrib
        produced_channels = list(produced)
        expected_channels = list(expected)
        assert len(produced_channels) == len(expected_channels)
        for produced_ch, expected_ch in zip(produced_channels, expected_channels):
            assert produced_ch.attrib == expected_ch.attrib

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


def _refs(ntrode):
    return ntrode.attrib.get("refNTrodeID"), ntrode.attrib.get("refChan")


def test_generate_reconfig_header_preserves_trivial_references():
    """Issue #113 review #189: the common header has every ntrode referencing
    ntrode 1 / channel 1. Source ntrode 1 stays merged ntrode 1 and its channel 1
    stays channel 1, so the trivial reference must survive the merge unchanged."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    new_header = convert_rec_header.generate_reconfig_header(
        raw, convert_rec_header.group_ntrodes_uniformly(32, 8)
    )
    for nt in new_header.find("SpikeConfiguration"):
        assert _refs(nt) == ("1", "1")


def test_generate_reconfig_header_retargets_references_to_merged_numbering():
    """Issue #113 review #189: refNTrodeID names a *source* ntrode and refChan is
    a 1-based channel within it. After merging+renumbering, both must be rewritten
    to the merged ntrode/channel that now holds the referenced channel, otherwise
    the reference dangles (KeyError) or silently resolves to the wrong merged
    group in make_ref_electrode_map."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    # Point every source ntrode's reference at source ntrode 5, channel 3 (1-based).
    for nt in raw.find("SpikeConfiguration"):
        nt.attrib["refOn"] = "1"
        nt.attrib["refNTrodeID"] = "5"
        nt.attrib["refChan"] = "3"

    new_header = convert_rec_header.generate_reconfig_header(
        raw, convert_rec_header.group_ntrodes_uniformly(32, 8)
    )
    # Source ntrode 5 lives in merged ntrode 1 (group [1..8]); the 4 source
    # tetrodes before it contribute 16 channels, so its channel 3 becomes channel
    # 16 + 3 = 19 of merged ntrode 1.
    for nt in new_header.find("SpikeConfiguration"):
        assert _refs(nt) == ("1", "19")

    # Downstream resolution must now succeed instead of KeyError-ing, and every
    # group must reference merged ntrode 1's nwb electrode group.
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadataProbeReconfig.yml", []
    )
    ref_map = convert_rec_header.make_ref_electrode_map(
        metadata, new_header.find("SpikeConfiguration")
    )
    assert {ref_group for ref_group, _ in ref_map.values()} == {"0"}


def test_generate_reconfig_header_retargets_legacy_refNTrode_index():
    """Issue #113 review #189: default Trodes configs express the reference as the
    legacy ``refNTrode`` (a 1-based position in the ntrode list) with no
    ``refNTrodeID``. It must be retargeted (otherwise the merged header keeps
    out-of-range indices like 9/17/25 with only 4 merged ntrodes, which Trodes
    resolves by an unchecked array index on reopen), and the canonical
    ``refNTrodeID`` must be emitted -- it is what Trodes writes on save and the
    only form make_ref_electrode_map reads, so without it the reference is
    silently dropped from the NWB output."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    sc = raw.find("SpikeConfiguration")
    # Emulate a default-config header: refNTrode self-index, no refNTrodeID.
    for nt in sc:
        nt.attrib["refOn"] = "1"
        nt.attrib["refNTrode"] = nt.attrib["id"]  # ntrode at position i -> i
        del nt.attrib["refNTrodeID"]

    new_header = convert_rec_header.generate_reconfig_header(
        raw, convert_rec_header.group_ntrodes_uniformly(32, 8)
    )
    new_ntrodes = list(new_header.find("SpikeConfiguration"))

    # Each merged ntrode (whose first source ntrode self-references) now points at
    # itself by the merged numbering, in both the legacy and canonical forms.
    assert [nt.attrib["refNTrode"] for nt in new_ntrodes] == ["1", "2", "3", "4"]
    assert [nt.attrib["refNTrodeID"] for nt in new_ntrodes] == ["1", "2", "3", "4"]

    # The reference now survives into the NWB reference map instead of being
    # dropped to (-1, -1).
    metadata, _ = convert_yaml.load_metadata(
        data_path / "20230622_sample_metadataProbeReconfig.yml", []
    )
    ref_map = convert_rec_header.make_ref_electrode_map(
        metadata, new_header.find("SpikeConfiguration")
    )
    assert (-1, -1) not in ref_map.values()
    assert {ref_group for ref_group, _ in ref_map.values()} == {"0"}


def test_generate_reconfig_header_rejects_reference_into_dropped_ntrode():
    """Issue #113 review #189: if a kept ntrode references a source ntrode that
    was dropped (allow_partial), the reference cannot be retargeted, so fail
    loudly rather than emit a dangling reference."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    first = list(raw.find("SpikeConfiguration"))[0]
    first.attrib["refOn"] = "1"
    first.attrib["refNTrodeID"] = "30"  # dropped by the partial grouping below
    first.attrib["refChan"] = "1"

    with pytest.raises(ValueError, match="cannot be retargeted"):
        convert_rec_header.generate_reconfig_header(raw, [[1, 2]], allow_partial=True)


def test_generate_reconfig_header_rejects_out_of_range_refNTrode():
    """Issue #113 review #189: a legacy refNTrode index past the source ntrode
    count cannot be resolved to a merged ntrode, so it must raise rather than
    silently mis-resolve."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    first = list(raw.find("SpikeConfiguration"))[0]
    first.attrib["refOn"] = "1"
    first.attrib["refNTrode"] = "99"  # only 32 source ntrodes exist
    del first.attrib["refNTrodeID"]

    with pytest.raises(ValueError, match="out of range"):
        convert_rec_header.generate_reconfig_header(
            raw, convert_rec_header.group_ntrodes_uniformly(32, 8)
        )


def test_generate_reconfig_header_preserves_no_reference_sentinel():
    """Issue #113 review #189: an ntrode carrying Trodes' no-reference sentinel
    (refNTrodeID <= 0) has nothing to retarget and must be left untouched, not
    turned into a spurious reference."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    for nt in raw.find("SpikeConfiguration"):
        nt.attrib["refNTrodeID"] = "-1"
        nt.attrib.pop("refNTrode", None)

    new_ntrodes = list(
        convert_rec_header.generate_reconfig_header(
            raw, convert_rec_header.group_ntrodes_uniformly(32, 8)
        ).find("SpikeConfiguration")
    )
    assert all(nt.attrib["refNTrodeID"] == "-1" for nt in new_ntrodes)


def test_generate_reconfig_header_validates_inputs():
    """Issue #113: empty groupings and headers without a SpikeConfiguration are
    caller errors and must fail loudly with actionable messages."""
    raw = convert_rec_header.read_header(data_path / "20230622_sample_01_a1.rec")
    with pytest.raises(ValueError, match="at least one group"):
        convert_rec_header.generate_reconfig_header(raw, [])
    with pytest.raises(ValueError, match="non-empty"):
        convert_rec_header.generate_reconfig_header(raw, [[1], []])
    with pytest.raises(ValueError, match="no SpikeConfiguration"):
        convert_rec_header.generate_reconfig_header(
            ElementTree.Element("Configuration"), [[1]]
        )


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
