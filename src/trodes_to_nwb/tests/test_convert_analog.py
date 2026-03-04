import shutil

import h5py
import numpy as np
import pynwb

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.convert_analog import (
    add_analog_data,
    get_analog_channel_names,
    update_analog_data,
)
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.tests.utils import data_path


def test_add_analog_data(tmp_path):
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # comparison file
    rec_header = convert_rec_header.read_header(rec_file)
    # make file with data
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)
    get_analog_channel_names(rec_header)
    add_analog_data(nwbfile, [rec_file])
    # save file
    filename = tmp_path / "test_add_analog.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)
    # read new and rec_to_nwb_file. Compare.
    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "analog" in read_nwbfile.processing
        assert "analog" in read_nwbfile.processing["analog"].data_interfaces
        assert "analog" in read_nwbfile.processing["analog"]["analog"].time_series
        assert read_nwbfile.processing["analog"]["analog"]["analog"].data.chunks == (
            16384,
            22,
        )

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()

            # get index mapping of channels
            id_order = read_nwbfile.processing["analog"]["analog"][
                "analog"
            ].description.split("   ")[:-1]
            old_id_order = old_nwbfile.processing["analog"]["analog"][
                "analog"
            ].description.split("   ")[:-1]
            index_order = [old_id_order.index(id) for id in id_order]

            # check that all the same channels are present
            assert set(id_order) == set(old_id_order)

            # compare data shapes
            new_data = read_nwbfile.processing["analog"]["analog"]["analog"].data[:]
            old_data = old_nwbfile.processing["analog"]["analog"]["analog"].data[:]
            assert new_data.shape == old_data.shape

            # compare ALL channels across ALL timepoints
            old_data_reordered = old_data[:, index_order]
            np.testing.assert_array_equal(new_data, old_data_reordered)

            # check dtype
            assert new_data.dtype == np.int16


def test_update_analog_data(tmp_path):
    """Test that update_analog_data correctly overwrites data in an existing NWB file."""
    rec_files = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]

    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = convert_yaml.load_metadata(metadata_path, [])
    rec_header = convert_rec_header.read_header(rec_files[0])

    # make file with data
    nwbfile = convert_yaml.initialize_nwb(metadata, rec_header)
    get_analog_channel_names(rec_header)
    add_analog_data(nwbfile, rec_files)

    # save file
    ref_filename = tmp_path / "correctly_added_analog.nwb"
    with pynwb.NWBHDF5IO(ref_filename, "w") as io:
        io.write(nwbfile)

    # Copy the reference NWB file so we don't modify the original
    buggy_filename = tmp_path / "test_update_analog_buggy.nwb"
    shutil.copy(ref_filename, buggy_filename)

    # Zero out the analog data in the copy to simulate the pre-fix (buggy) state
    with h5py.File(buggy_filename, "r+") as f:
        analog_hdf5_path = "processing/analog/analog/analog/data"
        f[analog_hdf5_path][...] = np.zeros_like(f[analog_hdf5_path][()])

    # Confirm data was zeroed out
    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        buggy_nwbfile = io.read()
        buggy_data = buggy_nwbfile.processing["analog"]["analog"]["analog"].data[:]
    assert (buggy_data == 0).all(), "Buggy data should be all zeros before update"

    # Run the update function (timestamps default to those already in the NWB file)
    update_analog_data(buggy_filename, rec_files)

    with pynwb.NWBHDF5IO(ref_filename, "r", load_namespaces=True) as io:
        correct_nwbfile = io.read()
        correct_data = correct_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        updated_nwbfile = io.read()
        updated_data = updated_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    # Compare ALL channels across ALL timepoints
    assert correct_data.shape == updated_data.shape
    np.testing.assert_array_equal(correct_data, updated_data)

    # check dtype
    assert updated_data.dtype == np.int16


def test_selection_of_multiplexed_data():
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(rec_file)
    hconf = rec_header.find("HardwareConfiguration")
    ecu_conf = None
    for conf in hconf:
        if conf.attrib["name"] == "ECU":
            ecu_conf = conf
            break
    analog_channel_ids = []
    for channel in ecu_conf:
        if channel.attrib["dataType"] == "analog":
            analog_channel_ids.append(channel.attrib["id"])
    assert (len(analog_channel_ids)) == 12
    rec_dci = RecFileDataChunkIterator(
        [rec_file],
        nwb_hw_channel_order=analog_channel_ids,
        stream_index=2,
        is_analog=True,
    )
    assert len(rec_dci.neo_io[0].multiplexed_channel_xml.keys()) == 10

    slice_ind = [(0, 4), (0, 30), (1, 15), (5, 15), (20, 25)]
    expected_channels = [4, 22, 14, 10, 2]
    for ind, expected in zip(slice_ind, expected_channels, strict=True):
        data = rec_dci._get_data(
            (
                slice(0, 100, None),
                slice(ind[0], ind[1], None),
            )
        )
        # check shape
        assert data.shape[1] == expected
        # check dtype
        assert data.dtype == np.int16
