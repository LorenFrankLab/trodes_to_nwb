# trodes_to_nwb

[![Tests](https://github.com/LorenFrankLab/trodes_to_nwb/actions/workflows/test_package_build.yml/badge.svg)](https://github.com/LorenFrankLab/trodes_to_nwb/actions/workflows/test_package_build.yml)
[![codecov](https://codecov.io/gh/LorenFrankLab/trodes_to_nwb/branch/main/graph/badge.svg?token=ZY6X3HSRHE)](https://codecov.io/gh/LorenFrankLab/trodes_to_nwb)
[![PyPI version](https://badge.fury.io/py/trodes-to-nwb.svg)](https://badge.fury.io/py/trodes-to-nwb)

Converts data from SpikeGadgets .rec files to the NWB 2.0+ Data Format.

It then validates the NWB file using the [NWB Inspector](https://github.com/NeurodataWithoutBorders/nwbinspector) for compatilibity for upload to the [DANDI archive](https://dandiarchive.org/).

This replaces [rec_to_nwb](https://github.com/LorenFrankLab/rec_to_nwb).

## Installation

You can install `trodes_to_nwb` in two ways.

For most users, we recommend installing `trodes_to_nwb` in a virtual or conda environment via pip.

Developers should install from source.

1. Install from PyPI

    ```bash
    pip install trodes_to_nwb
    ```

2. Install from source

    _Note: We currently reccomend using mamba as a package manager.  If using conda, please replace mamba with conda in the command below._

    ```bash
    git clone https://github.com/LorenFrankLab/trodes_to_nwb.git

    cd trodes_to_nwb

    mamba env create -f environment.yml

    mamba activate trodes_to_nwb

    pip install -e .
    ```

## Usage

1. Place your files in a folder. Subfolders are okay. The files with the following extensions should be in your directory:
    + `.rec`: binary file containing the ephys recording, accelerometer, gyroscope, magnetometer, DIO data, header
    + `.videoPositionTracking`:  trodes tracked position (optional)
    + `.h264`: video file
    + `.cameraHWSync`: position timestamps
    + `.stateScriptLog`: state script controls the experimenter parameters

    These files need to be named in the following format `{date}_{animal}_{epoch}_{tag}.{extension}` where date is in the `YYYYMMDD` format, epoch is an integer with zero padding (e.g. `02` and not `2`), and tag can be any handy short descriptor. For example, `20230622_randy_02_r1.rec` is the recording file for animal randy, second epoch, run 1 (r1) for June 22, 2023.

    (*Note: By default, Trodes saves video-related files (`.h264`, `videoPositionTracking`, `cameraHWSync`) slightly different from this format as `{date}_{animal}_{epoch}_{tag}.{camera number}.{extension}`. This is accepted by this conversion package, and used to match camera to position tracking in epochs with mulitple cameras*)

2. Create a metadata yaml file for each recording session. See this [example metadata yaml file](src/trodes_to_nwb/tests/test_data/20230622_sample_metadata.yml). We recommend using the [NWB YAML Creator](https://lorenfranklab.github.io/rec_to_nwb_yaml_creator/) to create the metadata yaml file in the correct format.

    The metadata yaml file should be named `{date}_{animal}.metadata.yml` where date is in the `YYYYMMDD` format and placed in the same directory as the `.rec` files.

    Here is an example valid directory structure:

    ```bash
   `-- beans
       |   |
       |   `-- raw
       |       |
       |       `-- 20190718
       |           |-- 20190718_beans_01_s1.1.h264
       |           |-- 20190718_beans_01_s1.1.trackgeometry
       |           |-- 20190718_beans_01_s1.1.videoPositionTracking
       |           |-- 20190718_beans_01_s1.1.videoTimeStamps
       |           |-- 20190718_beans_01_s1.1.videoTimeStamps.cameraHWSync
       |           |-- 20190718_beans_01_s1.rec
       |           |-- 20190718_beans_01_s1.stateScriptLog
       |           `-- 20190718_beans_metadata.yml
       |
       `-- README.md
    ```

3. Run the code in python. This will create a NWB file for each `.rec` file in the output directory.

    ```python
    from trodes_to_nwb.convert import create_nwbs

    path = "/path/to/your/data" # from step 1
    output_dir = "/path/to/your/output/directory"

    create_nwbs(
        path,
        output_dir,
        header_reconfig_path=None,
        probe_metadata_paths=None,
        convert_video=False,
        n_workers=1,
        query_expression=None,
    )
    ```

    For the example directory structure above, the path would look like this:

    ```python
    path = "/path/to/your/data/beans/raw"
    ```

    Note the following optional arguments:
    + `header_reconfig_path`: If you want to change the header information, you can provide a path to a yaml file with the new header information. See this [example header reconfig yaml file](src/trodes_to_nwb/tests/test_data/reconfig_probeDevice.trodesconf). For example, this can be important for data recorded from non-tetrode devices.
    + `probe_metadata_paths`: By default, several common probe device types configurations are included in the package. If you are using a probe that is not included, you can provide a path to a yaml file with the probe metadata. See this [example probe metadata yaml file](src/trodes_to_nwb/probe_metadata/128c-4s6mm6cm-15um-26um-sl.yml) for an example.
    + `convert_video`: Converts the .h264 video file to .mp4. This requires `ffmpeg` to be installed on your system.
    + `n_workers`: Number of workers to use for parallel processing. Defaults to 1.
    + `query_expression`: A query expression to select which files to convert. For example, if you have several animals in your folder, you could write `"animal == 'sample'"` to select only the sample animal. Defaults to `None` which converts all files in the directory.

    For complete example code of the conversion, see the [tutorial notebook](notebooks/conversion_tutorial.ipynb)
