import numpy as np
import pynwb

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.convert_ephys import add_raw_ephys
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

MICROVOLTS_PER_VOLT = 1e6
SAMPLING_RATE = 30_000


def _find_epoch_boundary_rows(timestamps, sampling_rate=SAMPLING_RATE, window=1):
    """Return a boolean mask of rows near epoch boundaries.

    An epoch boundary is detected where the timestamp gap between consecutive
    samples exceeds 10x the expected inter-sample interval.  A window of
    ``window`` rows on each side of the boundary is included to account for
    off-by-one differences in how rec_to_nwb stitches epochs.
    """
    dt = np.diff(timestamps)
    expected_dt = 1.0 / sampling_rate
    boundary_indices = np.where(dt > 10 * expected_dt)[0]

    n_rows = len(timestamps)
    boundary_mask = np.zeros(n_rows, dtype=bool)
    for idx in boundary_indices:
        lo = max(0, idx - window)
        hi = min(
            n_rows, idx + window + 2
        )  # +2 because boundary is between idx and idx+1
        boundary_mask[lo:hi] = True
    return boundary_mask


def _assert_ephys_match_with_epoch_boundary_masking(new_data, old_data, timestamps):
    """Assert ephys data matches, allowing zeros only at epoch boundaries.

    The rec_to_nwb reference files contain zero-valued elements at epoch
    boundaries (where multiple .rec files are stitched together).  Rather than
    masking zeros with an arbitrary percentage guard, this function verifies
    that ALL zeros in the reference data fall within a small window around
    detected epoch boundaries, then compares non-zero elements exactly.
    """
    boundary_mask = _find_epoch_boundary_rows(timestamps)

    # Verify that every zero in old_data falls in a boundary row
    zero_rows = np.where(np.any(old_data == 0, axis=1))[0]
    non_boundary_zeros = zero_rows[~boundary_mask[zero_rows]]
    assert len(non_boundary_zeros) == 0, (
        f"Found {len(non_boundary_zeros)} rows with zeros outside epoch boundaries "
        f"(rows: {non_boundary_zeros[:10]})"
    )

    # Non-boundary rows: exact match, no masking
    if np.any(~boundary_mask):
        np.testing.assert_array_equal(
            new_data[~boundary_mask], old_data[~boundary_mask]
        )

    # Boundary rows: compare only non-zero elements
    if np.any(boundary_mask):
        boundary_old = old_data[boundary_mask]
        boundary_new = new_data[boundary_mask]
        nonzero = boundary_old != 0
        if np.any(nonzero):
            np.testing.assert_array_equal(boundary_new[nonzero], boundary_old[nonzero])


def test_add_raw_ephys_single_rec(tmp_path):
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "20230622_sample_01_a1.rec"
    # "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [data_path / "20230622_sample_01_a1.rec"]
    rec_to_nwb_file = data_path / "20230622_155936.nwb"  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = tmp_path / "test_add_raw_ephys_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # compare ALL channels across ALL timepoints
            new_data = (
                read_nwbfile.acquisition["e-series"].data[:] * conversion
            ).astype("int16")
            old_data = old_nwbfile.acquisition["e-series"].data[:]
            np.testing.assert_array_equal(new_data, old_data)
            # check dtype
            assert read_nwbfile.acquisition["e-series"].data.dtype == np.int16
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )


def test_add_raw_ephys_single_rec_probe_configuration(tmp_path):
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadataProbeReconfig.yml"
    probe_metadata = [data_path / "128c-4s6mm6cm-15um-26um-sl.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "reconfig_probeDevice.trodesconf"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [data_path / "20230622_sample_01_a1.rec"]
    rec_to_nwb_file = (
        data_path / "probe_reconfig_20230622_155936.nwb"
    )  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = tmp_path / "test_add_raw_ephys_probe_reconfig.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # compare ALL channels across ALL timepoints
            new_data = (
                read_nwbfile.acquisition["e-series"].data[:] * conversion
            ).astype("int16")
            old_data = old_nwbfile.acquisition["e-series"].data[:]
            np.testing.assert_array_equal(new_data, old_data)
            # check dtype
            assert read_nwbfile.acquisition["e-series"].data.dtype == np.int16
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )


def test_add_raw_ephys_two_epoch(tmp_path):
    # load metadata yml and make nwb file
    metadata_path = data_path / "20230622_sample_metadata.yml"
    probe_metadata = [data_path / "tetrode_12.5.yml"]
    metadata, probe_metadata = convert_yaml.load_metadata(metadata_path, probe_metadata)
    nwbfile = convert_yaml.initialize_nwb(metadata, default_test_xml_tree())

    # create the hw_channel map using the reconfig header
    trodesconf_file = data_path / "20230622_sample_01_a1.rec"
    rec_header = convert_rec_header.read_header(trodesconf_file)

    hw_channel_map = convert_rec_header.make_hw_channel_map(
        metadata, rec_header.find("SpikeConfiguration")
    )
    ref_electrode_map = convert_rec_header.make_ref_electrode_map(
        metadata, rec_header.find("SpikeConfiguration")
    )

    convert_yaml.add_electrode_groups(
        nwbfile, metadata, probe_metadata, hw_channel_map, ref_electrode_map
    )

    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    rec_to_nwb_file = data_path / "minirec20230622_.nwb"  # comparison file

    map_row_ephys_data_to_row_electrodes_table = list(range(len(nwbfile.electrodes)))

    add_raw_ephys(
        nwbfile,
        recfile,
        map_row_ephys_data_to_row_electrodes_table,
    )

    filename = tmp_path / "test_add_raw_ephys_two_epoch.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # compare ALL channels across ALL timepoints
            # The rec_to_nwb reference file (minirec) has zero-valued artifact
            # elements at epoch boundaries that we fill with real data.
            # Verify zeros only occur near detected epoch boundaries.
            new_data = (
                read_nwbfile.acquisition["e-series"].data[:] * conversion
            ).astype("int16")
            old_data = old_nwbfile.acquisition["e-series"].data[:]
            timestamps = old_nwbfile.acquisition["e-series"].timestamps[:]
            _assert_ephys_match_with_epoch_boundary_masking(
                new_data, old_data, timestamps
            )
            # check dtype
            assert read_nwbfile.acquisition["e-series"].data.dtype == np.int16
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )
