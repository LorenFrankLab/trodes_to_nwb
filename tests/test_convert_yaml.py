from spikegadgets_to_nwb import convert_yaml
from datetime import datetime
from pynwb.file import Subject


def test_initial_nwb_creation():
    metadata =  {'session_description':'testing session',
                "experimenter_name":'Baggins, Bilbo',
                "lab":'FrankLab',
                "institution":'UCSF',
                # session_start_time=datetime(2000,1,1),#session_start_time, TODO: requires .rec data
                "session_id":'12345',
                # notes=self.link_to_notes, TODO
                "experiment_description":'testing file creation',}
    nwb_file = convert_yaml.initialNWB(metadata)
    
    #check that things were added in
    assert len(nwb_file.experimenter) > 0
    assert type(nwb_file.session_start_time) is datetime
    #confirm that undefined fields are empty
    assert nwb_file.electrodes is None
    assert len(nwb_file.acquisition) == 0
    assert len(nwb_file.processing) == 0
    assert len(nwb_file.devices) == 0
    
def test_subject_creation():
    metadata =  {'session_description':'testing session',
                "experimenter_name":'Baggins, Bilbo',
                "lab":'FrankLab',
                "institution":'UCSF',
                "session_id":'12345',
                "experiment_description":'testing file creation',
                'subject':{'weight': 500}
                }
    nwb_file = convert_yaml.initialNWB(metadata)
    convert_yaml.add_subject(nwb_file,metadata)
    assert type(nwb_file.subject) is Subject
    assert type(nwb_file.subject.weight) is str
    
# def test_camera_creation()