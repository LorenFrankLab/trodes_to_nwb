import os
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO

from spikegadgets_to_nwb.convert_ephys import RecFileDataChunkIterator
from spikegadgets_to_nwb.convert_intervals import add_epochs, add_sample_count
from spikegadgets_to_nwb.convert_yaml import initialize_nwb, load_metadata
from spikegadgets_to_nwb.data_scanner import get_file_info
from spikegadgets_to_nwb.spike_gadgets_raw_io import SpikeGadgetsRawIO
from spikegadgets_to_nwb.tests.test_convert_rec_header import default_test_xml_tree

path = os.path.dirname(os.path.abspath(__file__))


def test_add_epochs():
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    metadata, _ = load_metadata(metadata_path, [])
    nwbfile = initialize_nwb(metadata, default_test_xml_tree())
    try:
        # running on github
        file_info = get_file_info(Path(os.environ.get("DOWNLOAD_DIR")))
        rec_to_nwb_file = (
            os.environ.get("DOWNLOAD_DIR") + "/probe_reconfig_20230622_155936.nwb"
        )
        rec_to_nwb_file = os.environ.get("DOWNLOAD_DIR") + "/minirec20230622_.nwb"
    except (TypeError, FileNotFoundError):
        # running locally
        file_info = get_file_info(Path(path))
        rec_to_nwb_file = path + "/test_data/minirec20230622_.nwb"
    # get all streams for all files
    neo_io = [
        SpikeGadgetsRawIO(filename=file)
        for file in file_info[file_info.file_extension == ".rec"].full_path
    ]
    [neo_io.parse_header() for neo_io in neo_io]
    add_epochs(nwbfile, file_info, 20230622, "sample", neo_io)
    epochs_df = nwbfile.epochs.to_dataframe()
    # load old nwb versio
    io = NWBHDF5IO(rec_to_nwb_file, "r")
    old_nwbfile = io.read()
    old_epochs_df = old_nwbfile.epochs.to_dataframe()

    assert len(epochs_df) == 2
    assert list(epochs_df.index) == [0, 1]
    assert list(epochs_df.tags) == [["01_a1"], ["02_a1"]]
    assert list(epochs_df.start_time) == [1687474797.888, 1687474821.109]
    assert list(epochs_df.stop_time) == list(old_epochs_df.stop_time)


def test_add_sample_count():
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    metadata, _ = load_metadata(metadata_path, [])
    nwbfile = initialize_nwb(metadata, default_test_xml_tree())
    try:
        # running on github
        recfile = [
            os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_01_a1.rec",
            os.environ.get("DOWNLOAD_DIR") + "/20230622_sample_02_a1.rec",
        ]
        rec_to_nwb_file = os.environ.get("DOWNLOAD_DIR") + "/minirec20230622_.nwb"
    except (TypeError, FileNotFoundError):
        # running locally
        recfile = [
            path + "/test_data/20230622_sample_01_a1.rec",
            path + "/test_data/20230622_sample_02_a1.rec",
        ]
        rec_to_nwb_file = path + "/test_data/minirec20230622_.nwb"

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
