import os
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


def test_add_analog_data():
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
    filename = "test_add_analog.nwb"
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
            # TODO check that all the same channels are present

            # compare data
            assert (
                read_nwbfile.processing["analog"]["analog"]["analog"].data.shape
                == old_nwbfile.processing["analog"]["analog"]["analog"].data.shape
            )
            # compare matching for first timepoint
            assert (
                read_nwbfile.processing["analog"]["analog"]["analog"].data[0, :]
                == old_nwbfile.processing["analog"]["analog"]["analog"].data[0, :][
                    index_order
                ]
            ).all()
            # compare one channel across all timepoints
            test_index = 14  # channel with non-zero data
            assert (
                read_nwbfile.processing["analog"]["analog"]["analog"].data[
                    :, test_index
                ]
                == old_nwbfile.processing["analog"]["analog"]["analog"].data[
                    :, index_order[test_index]
                ]
            ).all()
    # cleanup
    # os.remove(filename)


def test_update_analog_data():
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
    ref_filename = "correctly_added_analog.nwb"
    with pynwb.NWBHDF5IO(ref_filename, "w") as io:
        io.write(nwbfile)

    # Copy the reference NWB file so we don't modify the original
    buggy_filename = "test_update_analog_buggy.nwb"
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

    print("buggy file name: \n", buggy_filename)
    with pynwb.NWBHDF5IO(ref_filename, "r", load_namespaces=True) as io:
        correct_nwbfile = io.read()
        correct_data = correct_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    with pynwb.NWBHDF5IO(buggy_filename, "r", load_namespaces=True) as io:
        updated_nwbfile = io.read()
        updated_data = updated_nwbfile.processing["analog"]["analog"]["analog"].data[:]

    # Map channel indices from the updated file into the correct file's ordering
    assert correct_data.shape == updated_data.shape
    # compare one non-zero multiplexed channel across all timepoints
    test_index = 14
    assert (correct_data[:, test_index] == updated_data[:, test_index]).all()

    # cleanup
    os.remove(buggy_filename)
    os.remove(ref_filename)


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
        assert data.shape[1] == expected


def test_ecu_analog_iterator_default_channel_order_uses_header_ids():
    # The default channel order used to be np.arange(n_channel), which only works
    # for streams whose channel IDs are numeric strings. ECU_analog IDs are names
    # like "ECU_Ain1", so default construction should use the header IDs.
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_dci = RecFileDataChunkIterator(
        [rec_file],
        stream_id="ECU_analog",
        is_analog=True,
    )

    assert rec_dci._get_data((slice(0, 1), slice(0, 1))).shape == (1, 1)
    assert rec_dci._get_data((slice(0, 1), slice(0, rec_dci.n_channel))).shape == (
        1,
        rec_dci.n_channel,
    )


def test_ecu_analog_iterator_open_ended_channel_slices():
    rec_file = data_path / "20230622_sample_01_a1.rec"
    rec_dci = RecFileDataChunkIterator(
        [rec_file],
        stream_id="ECU_analog",
        is_analog=True,
    )
    total_time, total_channels = rec_dci._get_maxshape()

    def expected_from_raw(time_slice, channel_slice):
        start, stop, _ = time_slice.indices(total_time)
        requested_channels = np.arange(total_channels)[channel_slice]
        if stop <= start:
            return np.empty((0, len(requested_channels)), dtype=np.int16)

        physical_channels = requested_channels[requested_channels < rec_dci.n_channel]
        channel_ids = [
            str(channel)
            for channel in np.asarray(rec_dci.nwb_hw_channel_order)[physical_channels]
        ]
        raw_data = rec_dci.neo_io[0].get_analogsignal_chunk(
            block_index=rec_dci.block_index,
            seg_index=rec_dci.seg_index,
            i_start=start,
            i_stop=stop,
            stream_index=rec_dci.stream_index,
            channel_ids=channel_ids,
        )

        physical_lookup = {
            int(channel): index for index, channel in enumerate(physical_channels)
        }
        return_indices = [
            (
                physical_lookup[int(channel)]
                if channel < rec_dci.n_channel
                else len(physical_channels) + int(channel) - rec_dci.n_channel
            )
            for channel in requested_channels
        ]
        return (raw_data[:, return_indices] * rec_dci.conversion).astype("int16")

    channel_slices = [
        slice(None),
        slice(0, None),
        slice(rec_dci.n_channel, None),
        slice(None, rec_dci.n_channel),
        slice(None, None, 2),
        slice(rec_dci.n_channel - 1, rec_dci.n_channel + 2),
    ]
    time_slices = [
        slice(0, 3),
        slice(total_time - 1, total_time),
        slice(total_time, total_time),
    ]
    for time_slice in time_slices:
        for channel_slice in channel_slices:
            data = rec_dci._get_data((time_slice, channel_slice))
            np.testing.assert_array_equal(
                data, expected_from_raw(time_slice, channel_slice)
            )
