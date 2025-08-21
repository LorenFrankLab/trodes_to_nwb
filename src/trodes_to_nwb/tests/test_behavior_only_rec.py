import pytest

from trodes_to_nwb.convert_ephys import RecFileDataChunkIterator
from trodes_to_nwb.tests.utils import data_path


def test_behavior_only_rec_file():
    file = data_path / "behavior_only.rec"
    # accessing trodes stream with behavior only file should fail
    with pytest.raises(IndexError):
        RecFileDataChunkIterator(
            rec_file_path=[file],
            interpolate_dropped_packets=True,
            stream_id="trodes",
            behavior_only=True,
        )

    # misspecification of behavior-only should result in mismatched stream number
    with pytest.raises(AssertionError):
        RecFileDataChunkIterator(
            rec_file_path=[file],
            interpolate_dropped_packets=True,
            stream_id="trodes",
            behavior_only=False,
        )

    # correctly build iterator
    rec_dci = RecFileDataChunkIterator(
        rec_file_path=[file],
        interpolate_dropped_packets=True,
        stream_id="ECU_analog",
        behavior_only=True,
    )
    neo_io = rec_dci.neo_io[0]

    # check file streams
    stream_names = [stream[0] for stream in neo_io.header["signal_streams"]]
    assert all(
        [
            x in stream_names
            for x in ["ECU_analog", "ECU_digital", "Controller_DIO_digital"]
        ]
    ), "missing expected stream in iterator"
    assert "trodes" not in stream_names, "unexpected trodes stream in iterator"

    # check data accesses
    assert rec_dci.timestamps.size == 433012
    assert rec_dci.timestamps[-1] == 1751195974.5656028, "unexpected last timestamp"
    assert set(neo_io.multiplexed_channel_xml.keys()) == set(
        [
            "Headstage_AccelX",
            "Headstage_AccelY",
            "Headstage_AccelZ",
            "Headstage_GyroX",
            "Headstage_GyroY",
            "Headstage_GyroZ",
            "Headstage_MagX",
            "Headstage_MagY",
            "Headstage_MagZ",
            "Controller_Ain1",
        ]
    )
    assert neo_io._raw_memmap.shape == (433012, 54)
