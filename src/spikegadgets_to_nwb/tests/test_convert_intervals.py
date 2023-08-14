from pathlib import Path
import os

from spikegadgets_to_nwb.data_scanner import get_file_info
from spikegadgets_to_nwb.convert_intervals import add_epochs
from spikegadgets_to_nwb.convert_yaml import initialize_nwb, load_metadata
from spikegadgets_to_nwb.tests.test_convert_rec_header import default_test_xml_tree

path = os.path.dirname(os.path.abspath(__file__))


def test_add_epochs():
    metadata_path = path + "/test_data/20230622_sample_metadata.yml"
    metadata, _ = load_metadata(metadata_path, [])
    nwbfile = initialize_nwb(metadata, default_test_xml_tree())
    try:
        # running on github
        file_info = get_file_info(Path(os.environ.get("DOWNLOAD_DIR")))
    except (TypeError, FileNotFoundError):
        # running locally
        file_info = get_file_info(Path(path))
    add_epochs(nwbfile, file_info, 20230622, "sample")
    epochs_df = nwbfile.epochs.to_dataframe()

    assert len(epochs_df) == 2
    assert list(epochs_df.index) == [0, 1]
    assert list(epochs_df.tags) == [["01_a1"], ["02_a1"]]
    assert list(epochs_df.start_time) == [1687474797.888, 1687474821.109]
    assert list(epochs_df.stop_time) == [0.0, 0.0]
