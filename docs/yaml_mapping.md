|Yaml Input|NWBfile Location|Description|Data Type|
| :---        |    :----:   |          ---: | ---:|
| __General__ |
| Experimenter Name| file.experimenter | |_tuple -> str_, Dandi format: lastName, firstName |
| Lab | file.lab | | _str_|
| Institution | file.institution | |_str_|
| Experiment Description | file.experiment_description | |_str_|
| Session Description | file.session_description | |_str_|
| Session ID | file.session_id | |_str_|
| Keywords | ????? |
|__Subject__|
| Description | file.subject.description | Description of subject and where subject came from (e.g., breeder, if animal) |_str_|
| Species | file.subject.species | |_str_|
| Genotype | file.subject.genotype | Genetic strain. If absent, assume Wild Type (WT) |_str_|
| Sex | file.subject.sex | Sex of subject, single letter identifier |_str_|
| Subject ID | file.subject.subject_id|ID of animal/person used/participating in experiment (lab convention) | _str_|
| Date of Birth | file.subject.date_of_birth | Date of birth of subject. | _datetime_ |
| Weight (g) | file.subject.weight | Weight at time of experiment, at time of surgery and at other important times. | _str_|
|__Data Acq Device__|
| Name | names device (device_name = "dataacq_device{'name'}") |typically a number|_str_|
| System | file.devices[device_name].system | system of device | _str_|
| Amplifier | file.devices[device_name].amplifier | | _str_|
| ADC circuit | file.devices[device_name].adc_circuit | | _str_ |
|__Cameras__|
| Cmaera ID | names device (device name = "camera_device {Camera ID}") | typically a number| _str_|
| Meters Per Pixel | file.devices[device_name].meters_per_pixel ||_float_|
| Manufacturer | file.devices[device_name].manufacturer ||_str_|
| model | file.devices[device_name].model | model of this camera|_str_|
| lens | file.devices[device_name].len | info about lens in this camera |_str_|
| Camera Name | file.devices[device_name].camera_name |name of this camera| _str_|
|__Tasks__|
| Task Name | file.processing['tasks']['task_#'].to_dataframe()['task_name'] | e.g. linear track, sleep| _str_|
| Task Description | file.processing['tasks']['task_#'].to_dataframe()['task_description'] | |_str_|
| Task Environment | file.processing['tasks']['task_#'].to_dataframe()['task_environment'] | where the task occurs (e.g. sleep box)| _str_|
| Camera ID | file.processing['tasks']['task_#'].to_dataframe()['camera_id'] | Camera(s) recording this task| _array -> int_
| Task Epochs | file.processing['tasks']['task_#'].to_dataframe()['task_epochs'] | what epochs this task is applied | _array -> int_|
|__Associated Files__|
| Name | names file in nwb | |_str_|
| Description | file.processing['associated_files'][Name].description | description of the file|_str_|
| Path | used to open and load content in nwbfile.processing['associated_files'][Name].content | |_str_|
| Task Epoch | file.processing['associated_files'][Name].task_epochs | what tasks/epochs was this code run for | _str_|
|__Associated Video Files__|
| Name |
| Camera ID | | what camera recorded this video | _str_ |
| Task Epoch | |what epoch was recorded in this video| _str_|
|__Units__|
| Analog | ??? |
| Behavioral Events| ??? |
| Times Period Multiplier | ??? |
| Ephys-to-Volt Conversion Factor | file.acquisition['e-series'].conversion | Scalar to multiply each element in data to convert it to the specified 'unit'. If the data are stored in acquisition system units or other units that require a conversion to be interpretable, multiply the data by 'conversion' to convert the data to the specified 'unit'.| _float_|
| Default Header File Path | ??? |
|__Behavioral Events__|
| Description | file.processing['behavior']['behavioral_events'][Name] | DIO info (eg. Din01)| _str_|
| Name | Names the DIO event | (e.g. light1) |
|__Device__|
| Name |
|__Electrode Groups__|
| ID | Names the electrodeGroup (Name = str(id)) | typically a number| _str_
| Location | ??? |
| Device Type | file.electrode_groups[Name].device_type | Used to match to probe yaml data|_str_|
| Description | file.electrode_groups[Name].description ||_str_|
| Targeted Location | file.electrode_groups[Name].location | Where device is implanted | _str_ |
| ML from Bregma |
| AP from Bregma |
| DV to Cortical Surface |
| Units | file.electrode_groups[Name].units | Distance units defining positioning | _str_|
