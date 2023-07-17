import os
import pynwb
from spikegadgets_to_nwb.convert_ephys import add_raw_ephys
from spikegadgets_to_nwb import convert_yaml, convert_rec_header


path = os.path.dirname(os.path.abspath(__file__))


def test_add_raw_ephys_single_rec():
    # load metadata yml and make nwb file
    metadata_path = (
        path + "/test_data/test_metadata.yml"
    )  # "/test_data/test_metadata_probe_reconfig.yml"
    probe_metadata = [
        path
        + "/test_data/tetrode_12.5.yml",  # "/test_data/128c-4s6mm6cm-15um-26um-sl.yml",
    ]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata)

    # create the hw_channel map using the reconfig header
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

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    try:
        # running on github
        recfile = os.environ.get("DOWNLOAD_DIR") + "/20230622_155936.rec"
        rec_to_nwb_file = os.environ.get("DOWNLOAD_DIR") + "/20230622_155936.nwb"
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = path + "/test_data/20230622_155936.rec"
        rec_to_nwb_file = path + "/test_data/20230622_155936.nwb"

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
        conversion=float(metadata["raw_data_to_volts"]),
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
            assert (
                read_nwbfile.acquisition["e-series"].data[0, :]
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                read_nwbfile.acquisition["e-series"].data[:, 0]
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()

    # TODO verify that the data is correct

    # TODO remove the file
