import os
#import loggings
import sys
from rec_to_nwb.processing.builder.nwb_file_builder import NWBFileBuilder
from rec_to_binaries import extract_trodes_rec_file
from rec_to_nwb.processing.builder.raw_to_nwb_builder import RawToNWBBuilder
from rec_to_nwb.processing.metadata.metadata_manager import MetadataManager

def main():
    print('Setting variables', flush=True)
   
    # set the animal name and the date or list of dates to process
    # animal_name = 'beans'
    # date = '20190718'
    # animal_metadata_file = 'beans20190718_metadata.yml'
    # reconfig_file = '/stelmo/loren/beans/Probe_128ch_allnT_DIOs_PTP_reconfig_export_shanks.xml'

    animal_name = 'despereaux'
    date = '20191125'
    animal_metadata_file = 'despereaux20191125.yml'
    reconfig_file = None
    #reconfig_file = '/stelmo/loren/despereaux/raw/20191125/20191125_despereaux_01_s1.rec_header.xml'

    # animal_name = 'Jaq'
    # date = '20190826'
    # animal_metadata_file = 'jaq20190826.yml'
    # reconfig_file = None
    
    yaml_path = '/home/loren/Src/NWB/franklabnwb/yaml'


    probe1_metadata_file = '128c-4s8mm6cm-20um-40um-sl.yml'
    probe2_metadata_file = 'tetrode_12.5.yml'

    #Specify the paths for the data, the output nwb file, and the video files
    data_path = '/stelmo/loren/'
    output_path='/stelmo/loren/'
    video_path='/stelmo/loren/'

    # data_path = '/Users/loren/data/nwb_builder_test_data/'
    # output_path='//Users/loren/data/nwb_builder_test_data/tmp'
    # video_path='/Users/loren/data/nwb_builder_test_data/tmp'


 


    # specify the locations of the metadata files for the animal and the probe(s). 
    # Note that specifying all possible probes is fine
    animal_metadata = os.path.join(yaml_path, animal_metadata_file )
    probe_metadata = [os.path.join(yaml_path, probe1_metadata_file), 
                    os.path.join(yaml_path, probe2_metadata_file)]

    print(probe_metadata)

    # Specify whether data should be reextracted. 
    overwrite=False

# metadata parameters
    


    metadata = MetadataManager(animal_metadata, probe_metadata)
    print(metadata, flush=True)

    print('Creating Builder', flush=True)
    if reconfig_file is not None:
           # Specify any optional trodes export flags
        trodes_rec_export_args = ('-reconfig', reconfig_file) 
        builder = RawToNWBBuilder(animal_name=animal_name,
                            data_path=data_path,
                            dates=[date],
                            nwb_metadata=metadata,
                            overwrite=overwrite,
                            output_path=output_path,
                            video_path=video_path,
                            extract_analog=True,
                            trodes_rec_export_args = trodes_rec_export_args)
    else: 
       builder = RawToNWBBuilder(animal_name=animal_name,
                            data_path=data_path,
                            dates=[date],
                            nwb_metadata=metadata,
                            overwrite=overwrite,
                            output_path=output_path,
                            video_path=video_path,
                            extract_analog=True)

    print('Building and Writing', flush=True)
    builder.build_nwb(run_preprocessing='False', 
                            process_mda_valid_time='False',
                            process_mda_invalid_time='False',
                            process_pos_valid_time='False',
                            process_pos_invalid_time='False')    
    print('Done', flush=True)

if __name__ == '__main__':
    main()
