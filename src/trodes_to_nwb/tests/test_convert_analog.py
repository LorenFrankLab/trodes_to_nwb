import os

import pynwb

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.convert_analog import add_analog_data, get_analog_channel_names
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
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
    analog_channel_names = get_analog_channel_names(rec_header)
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
            assert (
                read_nwbfile.processing["analog"]["analog"]["analog"].data[:, 0]
                == old_nwbfile.processing["analog"]["analog"]["analog"].data[
                    :, index_order[0]
                ]
            ).all()
    # cleanup
    os.remove(filename)


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
    for ind, expected in zip(slice_ind, expected_channels):
        data = rec_dci._get_data(
            (
                slice(0, 100, None),
                slice(ind[0], ind[1], None),
            )
        )
        assert data.shape[1] == expected
    return
