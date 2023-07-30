#Test of automatic datajoint schema generation from NWB file
import datajoint as dj
franklab_schema = dj.schema("franklab", locals())
@franklab_schema
class Nwbfile(dj.Manual):
     definition = """
    file_name: varchar(80)
    ---
    """


@franklab_schema
class Subject(dj.Manual):
    definition = """
    subject_id: varchar(80)
    ---
    age: varchar(80)
    description: varchar(80)
    genotype: varchar(80)
    sex: enum('M', 'F', 'U')
    species: varchar(80)
    """


@franklab_schema
class LabMember(dj.Lookup):
    definition = """
    lab_member_name: varchar(80)
    ---
    """


@franklab_schema
class Institution(dj.Manual):
    definition = """
    institution_name: varchar(80)
    ---
    """


@franklab_schema
class Lab(dj.Manual):
     definition = """
    lab_name: varchar(80)
    ---
    """


@franklab_schema
class Device(dj.Manual):
    definition = """
    device_name: varchar(80)
    ---
    """


@franklab_schema
class Apparatus(dj.Manual):
     definition = """
    apparatus_name: varchar(80)
    ---
    -> Nwbfile
    module: varchar(80)
    container: varchar(80)
    """


@franklab_schema
class Session(dj.Manual):
     definition = """
    session_id: varchar(80)
    ---
    -> Nwbfile
    -> Subject
    -> Device
    -> Institution
    -> Lab
    session_description: varchar(80)
    identifier: varchar(80)
    session_start_time: datetime
    timestamps_reference_time: datetime
    experimenter: varchar(80)
    experiment_description: varchar(80)
    """


@franklab_schema
class Experimenter(dj.Manual):
    definition = """
    experimenter_name: varchar(80)
    -> LabMember
    -> Session
    ---
    """


