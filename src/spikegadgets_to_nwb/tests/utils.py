"""Set the path to the bulk test data dir and copies the yaml/config files there"""
import os
from pathlib import Path
import shutil


yaml_path = Path(__file__).resolve().parent / "test_data"

data_path = os.environ.get("DOWNLOAD_DIR", None)
if data_path is not None:
    # running from the GitHub Action workflow
    data_path = Path(data_path)
    shutil.copytree(
        yaml_path,
        data_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(*os.listdir(data_path)),  # ignore existing
    )
else:
    # running locally -- bulk test data is the same directory as the test yaml files
    data_path = yaml_path

del yaml_path
