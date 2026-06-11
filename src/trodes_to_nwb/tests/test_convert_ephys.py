import os

import numpy as np
import pynwb
import pytest

from trodes_to_nwb import convert_rec_header, convert_yaml
from trodes_to_nwb.convert_ephys import (
    RecFileDataChunkIterator,
    _is_strictly_increasing,
    _LazyTimestamps,
    add_raw_ephys,
)
from trodes_to_nwb.tests.test_convert_rec_header import default_test_xml_tree
from trodes_to_nwb.tests.utils import data_path

MICROVOLTS_PER_VOLT = 1e6


def _sample_neo_io():
    """neo_io list over the two-epoch sample rec files (for _LazyTimestamps tests)."""
    recfile = [
        data_path / "20230622_sample_01_a1.rec",
        data_path / "20230622_sample_02_a1.rec",
    ]
    return RecFileDataChunkIterator(
        [str(f) for f in recfile], stream_id="trodes"
    ).neo_io


def test_lazy_timestamps_matches_concatenation():
    # _LazyTimestamps must be byte-identical to the old eager concatenation for
    # every access pattern its consumers use -- this is the only *pytest* coverage
    # of the slice/_gather plumbing (the golden master is a separate dev script).
    neo_io = _sample_neo_io()
    lazy = _LazyTimestamps(neo_io, use_sysclock=True)
    expected = np.concatenate([io.get_regressed_systime(0, None) for io in neo_io])
    boundary = neo_io[0]._raw_memmap.shape[0]

    assert len(lazy) == expected.shape[0]
    assert lazy.shape == expected.shape and lazy.dtype == expected.dtype
    # full materialise
    np.testing.assert_array_equal(np.asarray(lazy), expected)
    # contiguous slices, including one straddling the file boundary
    for s in [
        slice(0, 10),
        slice(boundary - 5, boundary + 5),
        slice(len(lazy) - 3, len(lazy)),
        slice(None, None),
    ]:
        np.testing.assert_array_equal(lazy[s], expected[s])
    # integer fancy index: within one file, straddling the boundary, and negative
    for idx in [
        np.array([0, 1, 5, 1000, boundary - 1]),
        np.array([boundary - 2, boundary, boundary + 50, len(lazy) - 1]),
        np.array([-1, -2, 0, boundary]),
    ]:
        np.testing.assert_array_equal(lazy[idx], expected[idx])
    # boolean mask
    mask = np.zeros(len(lazy), dtype=bool)
    mask[[3, boundary - 1, boundary, len(lazy) - 1]] = True
    np.testing.assert_array_equal(lazy[mask], expected[mask])
    # the capped-span (blocked) _gather path, forced via a small max_span, must
    # be byte-identical to indexing the eager array (sparse, spanning both files)
    sparse = np.arange(0, len(lazy), 997)
    np.testing.assert_array_equal(lazy._gather(sparse, max_span=512), expected[sparse])
    # scalars
    for i in [0, boundary - 1, boundary, -1]:
        assert lazy[i] == expected[i]


def test_lazy_timestamps_non_sysclock_fallback_matches():
    # The use_sysclock=False branch (get_systime_from_trodes_timestamps) is never
    # hit by the sample data through the normal path, so exercise its plumbing
    # directly and assert byte-identity to the eager concatenation (#47).
    neo_io = _sample_neo_io()
    lazy = _LazyTimestamps(neo_io, use_sysclock=False)
    expected = np.concatenate(
        [io.get_systime_from_trodes_timestamps(0, None) for io in neo_io]
    )
    boundary = neo_io[0]._raw_memmap.shape[0]
    np.testing.assert_array_equal(np.asarray(lazy), expected)
    np.testing.assert_array_equal(
        lazy[boundary - 5 : boundary + 5], expected[boundary - 5 : boundary + 5]
    )
    np.testing.assert_array_equal(
        lazy[np.array([0, boundary, len(lazy) - 1])],
        expected[np.array([0, boundary, len(lazy) - 1])],
    )


def test_lazy_timestamps_chunk_iterator_roundtrip(tmp_path):
    # The chunked writer is the component whose whole purpose is the #47 memory
    # win; round-trip it through a real HDF5 write and confirm the on-disk dataset
    # equals the lazy array exactly.
    from datetime import datetime

    neo_io = _sample_neo_io()
    lazy = _LazyTimestamps(neo_io, use_sysclock=True)
    expected = np.asarray(lazy)

    nwbfile = pynwb.NWBFile(
        session_description="roundtrip",
        identifier="roundtrip",
        session_start_time=datetime(2023, 1, 1),
    )
    nwbfile.add_acquisition(
        pynwb.TimeSeries(
            name="lazy_ts",
            data=lazy.as_data_chunk_iterator(),
            rate=30000.0,
            unit="seconds",
        )
    )
    path = str(tmp_path / "lazy_ts_roundtrip.nwb")
    with pynwb.NWBHDF5IO(path, "w") as io:
        io.write(nwbfile)
    with pynwb.NWBHDF5IO(path, "r", load_namespaces=True) as io:
        written = io.read().acquisition["lazy_ts"].data[:]
    np.testing.assert_array_equal(written, expected)


def test_lazy_timestamps_guards_fail_loud():
    # Array-like-but-not-an-array surfaces that should raise rather than silently
    # do the wrong/slow thing (#47 review hardening).
    lazy = _LazyTimestamps(_sample_neo_io(), use_sysclock=True)
    n = len(lazy)
    with pytest.raises(TypeError):
        iter(lazy)  # would otherwise be O(n) disk reads via the sequence protocol
    with pytest.raises(IndexError):
        lazy[np.array([0, n])]  # fancy index out of bounds
    with pytest.raises(IndexError):
        lazy[n]  # scalar out of bounds
    with pytest.raises(ValueError):
        np.array(lazy, copy=False)  # no-copy view of a virtual array is impossible


def test_is_strictly_increasing_matches_full_diff():
    # The streaming check (#47, avoids the full np.diff for the ~15 GB timestamp
    # array) must agree with np.all(np.diff(...) > 0) on every case, including
    # violations that land exactly on a chunk boundary.
    def reference(a):
        return bool(np.all(np.diff(a) > 0)) if len(a) > 1 else True

    cases = [
        np.arange(50, dtype=float),  # strictly increasing
        np.array([0.0, 1.0, 1.0, 2.0]),  # equal (not strict)
        np.array([0.0, 1.0, 2.0, 1.5, 3.0]),  # backward jump
        np.array([5.0]),  # single element
        np.array([]),  # empty
    ]
    for values in cases:
        for chunk in (3, 1_000_000):
            assert _is_strictly_increasing(values, chunk) == reference(values)

    # planted equal-value violation straddling a small chunk boundary
    increasing = np.cumsum(np.abs(np.sin(np.arange(5000))) + 0.01)
    assert _is_strictly_increasing(increasing, 1000) is True
    broken = increasing.copy()
    broken[2000] = broken[1999]
    assert _is_strictly_increasing(broken, 1000) is False


def test_add_raw_ephys_single_rec():
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

    filename = "test_add_raw_ephys_single_rec.nwb"
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
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )
    os.remove(filename)


def test_add_raw_ephys_single_rec_probe_configuration():
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

    filename = "test_add_raw_ephys_single_rec.nwb"
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
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )

    os.remove(filename)


def test_add_raw_ephys_two_epoch():
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

    filename = "test_add_raw_ephys_single_rec.nwb"
    with pynwb.NWBHDF5IO(filename, "w") as io:
        io.write(nwbfile)

    with pynwb.NWBHDF5IO(filename, "r", load_namespaces=True) as io:
        read_nwbfile = io.read()
        assert "e-series" in read_nwbfile.acquisition
        assert read_nwbfile.acquisition["e-series"].data.chunks == (16384, 32)

        with pynwb.NWBHDF5IO(rec_to_nwb_file, "r", load_namespaces=True) as io2:
            old_nwbfile = io2.read()
            print(
                read_nwbfile.acquisition["e-series"].data.shape,
                old_nwbfile.acquisition["e-series"].data.shape,
            )

            # check ordering worked correctly
            conversion = (
                read_nwbfile.acquisition["e-series"].conversion * MICROVOLTS_PER_VOLT
            )
            assert (
                (read_nwbfile.acquisition["e-series"].data[0, :] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[0, :]
            ).all()
            # check data shapes match
            assert (
                read_nwbfile.acquisition["e-series"].data.shape
                == old_nwbfile.acquisition["e-series"].data.shape
            )
            # check all values of one of the streams
            assert (
                (read_nwbfile.acquisition["e-series"].data[:, 0] * conversion).astype(
                    "int16"
                )
                == old_nwbfile.acquisition["e-series"].data[:, 0]
            ).all()
            # check that timestamps are less than one sample different
            assert np.allclose(
                read_nwbfile.acquisition["e-series"].timestamps[:],
                old_nwbfile.acquisition["e-series"].timestamps[:],
                rtol=0,
                atol=1.0 / 30000,
            )

    os.remove(filename)
