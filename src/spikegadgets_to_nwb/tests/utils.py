"""Set the paths to the config files dir and the path to the bulk test data dir"""
import os
from pathlib import Path


yaml_path = Path(__file__).resolve().parent / "test_data"

data_path = os.environ.get("DOWNLOAD_DIR", None)
if data_path is not None:
    # running from the GitHub Action workflow
    data_path = Path(data_path)
else:
    # running locally -- bulk test data is the same directory as the test yaml files
    data_path = yaml_path
