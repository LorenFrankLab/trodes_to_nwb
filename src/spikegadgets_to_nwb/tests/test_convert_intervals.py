import os

import numpy as np
from pynwb import NWBHDF5IO
from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.convert_intervals import add_epochs, add_sample_count
from trodes_to_nwb.convert_yaml import initialize_nwb, load_metadata
from trodes_to_nwb.data_scanner import get_file_info
from trodes_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path


def test_add_epochs():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = load_metadata(metadata_path, [])
    nwbfile = initialize_nwb(metadata, default_test_xml_tree())
    file_info = get_file_info(data_path)
    file_info = file_info[file_info.animal == "sample"]
    file_info = file_info[file_info.date == 20230622]
    rec_to_nwb_file = data_path / "minirec20230622_.nwb"  # comparison file
    # get all streams for all files
    rec_dci = RecFileDataChunkIterator(
        file_info[file_info.file_extension == ".rec"].full_path.to_list()
    )
    add_epochs(nwbfile, file_info, rec_dci.neo_io)
    epochs_df = nwbfile.epochs.to_dataframe()
    # load old nwb version
    io = NWBHDF5IO(rec_to_nwb_file, "r")
    old_nwbfile = io.read()
    old_epochs_df = old_nwbfile.epochs.to_dataframe()

    assert len(epochs_df) == 2
    assert list(epochs_df.index) == [0, 1]
    assert list(epochs_df.tags) == [["01_a1"], ["02_a1"]]
    assert list(epochs_df.start_time) == [1687474797.888, 1687474821.109]
    assert list(epochs_df.stop_time) == list(old_epochs_df.stop_time)


def test_add_sample_count():
    metadata_path = data_path / "20230622_sample_metadata.yml"
    metadata, _ = load_metadata(metadata_path, [])
    nwbfile = initialize_nwb(metadata, default_test_xml_tree())
    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    rec_to_nwb_file = data_path / "minirec20230622_.nwb"  # comparison file

    # make recfile data chunk iterator
    rec_dci = RecFileDataChunkIterator(recfile)

    # add sample counts
    add_sample_count(nwbfile, rec_dci)

    assert "sample_count" in nwbfile.processing

    filename = "test_add_sample_count.nwb"
    with NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)
    with NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        with NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()

            assert (
                read_nwbfile.processing["sample_count"]["sample_count"].data[:]
                == old_nwbfile.processing["sample_count"]["sample_count"].data[:]
            ).all()
            assert np.allclose(
                read_nwbfile.processing["sample_count"]["sample_count"].timestamps[:],
                old_nwbfile.processing["sample_count"]["sample_count"].timestamps[:],
                rtol=0,
                atol=(1.0 / 30000) * 1e9,
            )
    os.remove(filename)
