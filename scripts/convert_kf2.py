import os
import logging
import sys
import numpy as np
import unittest

from rec_to_nwb.processing.builder.raw_to_nwb_builder import RawToNWBBuilder
from rec_to_nwb.processing.metadata.metadata_manager import MetadataManager

# switches for preprocessing
RUN_PREPROCESSING = True
OVERWRITE = False


def convert_kf_single_day(animal_name: str,
                date: str,
                yaml_path = '../yaml',
                reconfig_file = None
                ):
    dates = [date] # single date for now

    # Specify the paths for the data, the output nwb file, and the video files
    # data are saved in data_path/animal_name/raw/date/*.*
    data_path = '/data2/data1_backup/jason/'
    out_path = '/stelmo/jhbak/nwb/conversion/' # both preprocessing and NWB

    output_path = out_path + animal_name + '/out/'
    video_path = out_path + animal_name + '/video/'

    # Specify metadata files
    animal_metadata_file = '{}_{}_metadata.yml'.format(animal_name, date)
    probe_metadata_files = [
        'tetrode_12.5.yml',
        '32c-2s8mm6cm-20um-40um-dl.yml',
        ]

    # specify the locations of the metadata files for the animal and the probe(s).
    # Note that specifying all possible probes is fine
    animal_metadata = os.path.join(yaml_path, animal_metadata_file)
    probe_metadata = [os.path.join(yaml_path, file) for file in probe_metadata_files]

    # Specify any optional trodes export flags
    if reconfig_file is None:
        trodes_rec_export_args = ()
    else:
        trodes_rec_export_args = ('-reconfig', reconfig_file)


    metadata = MetadataManager(animal_metadata, probe_metadata)

    builder = RawToNWBBuilder(animal_name=animal_name,
                              data_path=data_path,
                              dates=dates,
                              nwb_metadata=metadata,
                              overwrite=OVERWRITE,
                              preprocessing_path=out_path, # new
                              output_path=output_path,
                              video_path=video_path,
                              trodes_rec_export_args = trodes_rec_export_args)

    _ = builder.build_nwb(run_preprocessing=RUN_PREPROCESSING)


if __name__ == '__main__':
    reconfig_file = '../xml/kf2_reconfig.xml'

    # date = '20170120'
    date = '20170201'

    convert_kf_single_day(animal_name='kf2',
                          date=date,
                          yaml_path = '../yaml',
                          reconfig_file = reconfig_file)
