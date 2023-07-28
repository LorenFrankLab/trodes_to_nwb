import pytest
import os
from xml.etree import ElementTree
from spikegadgets_to_nwb import convert_rec_header

# fixtures
@pytest.fixture
def path() -> str:
    path = os.path.dirname(os.path.abspath(__file__))
    return path


@pytest.fixture
def default_test_xml_tree() -> ElementTree:
    """Function to return a default xml tree for initial nwb generation

    Returns
    -------
    ElementTree
        root xml tree for initial nwb generation
    """
    # running in a git action if DOWNLOAD_DIR' in os.environ == true; else running locally
    rec_path = os.environ.get("DOWNLOAD_DIR") if 'DOWNLOAD_DIR' in os.environ else f'{path}../test_data'
    trodesconf_file = f'{rec_path}/20230622_155936.rec'
    rec_header = convert_rec_header.read_header(trodesconf_file)
    return rec_header