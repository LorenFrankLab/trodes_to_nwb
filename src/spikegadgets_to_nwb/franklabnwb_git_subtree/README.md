# franklabnwb

Example notebook and related code to create NWB files from trodes .rec and associated files assuming franklab
compatible naming system

## Instructions for initial setup and testing: 

### Prerequisites

- [Download](https://bitbucket.org/mkarlsso/trodes/downloads/) and install Trodes export code, and add the installed directory to path.

    For more detailed instructions, see [Installing SpikeGadgets](installing_spikegadgets.md).



### Setting up a python environment

1. Clone this repository to your code source directory:
    
    ```
    cd your_source_directory
    git clone https://github.com/LorenFrankLab/franklabnwb.git
    git clone https://github.com/LorenFrankLab/rec_to_nwb.git
    ```

2. Create the conda environment required for the conversion and install the rec_to_nwb module

    ```
    cd rec_to_nwb
    conda env create -f environment.yml
    conda activate rec_to_nwb
    pip install -e .
    ```

3. Start the notebook server from a directory below the franklabnwb directory:

    ```
    jupyter notebook
    ```

4. Try from notebook: 
    In the notebook, navigate to the `franklabnwb/notebooks` directory and open
    `franklab_nwb_generation.ipynb`.

     Edit the variables in that notebook as required for your data and run all cells. See below for more
    information on the animal metadata file.

5. Try from file:
    Edit 'create_nwb_example.py' to match your data and run that:
    python create_nwb_examply.py


## Animal metadata file:

`rec_to_nwb` requires a metadata file for each day of recording. Details on the content of that file can
be found in the
[documentation](https://novelaneuro.github.io/rec_to_nwb-docs/README.html#how-to-use-it).

Alternatively, you can start with the [franklabnwb/yaml/beans20190718_metadata.yml](yaml/beans20190718_metadata.yml) file as an example.
