import yaml

spyglass_error = (
    "This functional is optional and requires the spyglass package. It is only relevant"
    + " for those using spyglass for data management and requires spyglass be installed."
)


def check_for_brain_region(data: dict):
    from spyglass.common import BrainRegion

    missing_regions = []
    for k, v in data.items():
        if isinstance(v, dict):
            missing_regions.extend(check_for_brain_region(v))
            continue
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    missing_regions.extend(check_for_brain_region(item))
            continue
        if "location" in k:
            key = {"region_name": v}
            if not (BrainRegion & key):
                missing_regions.append(v)
    return list(set(missing_regions))


def yaml_database_validation(yaml_path: str):
    """Validate the YAML file against the database tables to reduce duplication and errors.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML file.
    """

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)
    try:
        from spyglass.common import CameraDevice, Task
    except ImportError:
        raise ImportError(spyglass_error)

    # check entries for tables in the database for consistency with the yaml file
    yaml_to_table = {
        "tasks": Task(),
        "cameras": CameraDevice(),
    }
    for yaml_key, table in yaml_to_table.items():
        if yaml_key not in yaml_data:
            continue
        print(f"Checking table `{table.camel_name}` against yaml key `{yaml_key}`")
        for key in yaml_data[yaml_key]:
            _primary = {k: v for k, v in key.items() if k in table.primary_key}
            if not (query := (table & _primary)):
                continue

            table_entry = query.fetch1()
            for k, v in key.items():
                if k not in table_entry:
                    continue
                if table_entry[k] != v:
                    print(
                        f"\tFor entry {_primary} in table {table.camel_name} \n"
                        + f"\t\t Mismatch in {k}: {table_entry[k]} != {v}"
                    )
    if new_regions := check_for_brain_region(yaml_data):
        print(
            "The following brain regions are missing from the BrainRegion table.\n"
            + " Consider using existing BrainRegion entries:"
        )
        for region in new_regions:
            print(f"\t{region}")
