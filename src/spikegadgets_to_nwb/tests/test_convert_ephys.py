import os
from pathlib import Path

import numpy as np
import pynwb
from trodes_to_nwb.convert_ephys import add_raw_ephys
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

from trodes_to_nwb import convert_rec_header, convert_yaml

MICROVOLTS_PER_VOLT = 1e6


def test_add_raw_ephys_single_rec():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "20230622_sample_01_a1.rec"
    # "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [data_path / "20230622_sample_01_a1.rec"]
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = "test_add_raw_ephys_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )
    os.remove(filename)


def test_add_raw_ephys_single_rec_probe_configuration():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [data_path / "20230622_sample_01_a1.rec"]
    rec_to_nwb_file = (
        data_path / "probe_reconfig_20230622_155936.nwb"
    )  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = "test_add_raw_ephys_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )

    os.remove(filename)


def test_add_raw_ephys_two_epoch():
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    rec_to_nwb_file = data_path / "minirec20230622_.nwb"  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = "test_add_raw_ephys_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            print(
                read_nwbfile.acquisition["e-series"].data.shape,
                old_nwbfile.acquisition["e-series"].data.shape,
            )

            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )

    os.remove(filename)
