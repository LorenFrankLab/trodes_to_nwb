from typing import Tuple
from hdmf.data_utils import GenericDataChunkIterator
from hdmf.backends.hdf5 import H5DataIO
import numpy as np
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from warnings import warn

from .spike_gadgets_raw_io import SpikeGadgetsRawIO

MICROVOLTS_PER_VOLT = 1e6


class RecFileDataChunkIterator(GenericDataChunkIterator):
    """Data chunk iterator for SpikeGadgets rec files."""

    def __init__(self, rec_file_path: str, nwb_hw_channel_order=[], **kwargs):
        self.neo_io = SpikeGadgetsRawIO(filename=rec_file_path)  # get all streams
        self.neo_io.parse_header()
        # TODO see what else spikeinterface does and whether it is necessary

        # for now, make sure that there is only one block, one segment, and two streams
        assert self.neo_io.block_count() == 1
        assert self.neo_io.segment_count(0) == 1
        assert self.neo_io.signal_streams_count() == 2

        self.block_index = 0
        self.seg_index = 0
        self.stream_index = 1  # TODO confirm that the stream index is trodes

        self.n_time = self.neo_io.get_signal_size(
            block_index=self.block_index,
            seg_index=self.seg_index,
            stream_index=self.stream_index,
        )
        self.n_channel = self.neo_io.signal_channels_count(
            stream_index=self.stream_index
        )

        # order that the hw channels are in within the nwb table
        if len(nwb_hw_channel_order) == 0:  # TODO: raise error instead?
            self.nwb_hw_channel_order = np.arange(self.n_channel)
        else:
            self.nwb_hw_channel_order = nwb_hw_channel_order

        # NOTE: this will read all the timestamps from the rec file, which can be slow
        self.timestamps = self.neo_io.get_analogsignal_timestamps(0, self.n_time)
        is_timestamps_sequential = np.all(np.diff(self.timestamps))
        if not is_timestamps_sequential:
            warn(
                "Timestamps are not sequential. This may cause problems with some software or data analysis."
            )

        super().__init__(**kwargs)

    def _get_data(self, selection: Tuple[slice]) -> np.ndarray:
        # selection is (time, channel)
        assert selection[0].step is None

        # slice to indices
        # DCI will want channels 0 to X first to put into the array in that order
        # those are stored in the file as channel IDs
        # make into list form passed to neo_io
        channel_ids = [str(x) for x in self.nwb_hw_channel_order[selection[1]]]

        data = (
            self.neo_io.get_analogsignal_chunk(
                block_index=self.block_index,
                seg_index=self.seg_index,
                i_start=selection[0].start,
                i_stop=selection[0].stop,
                stream_index=self.stream_index,
                channel_ids=channel_ids,
            )
        ).astype("int16")
        return data

    def _get_maxshape(self) -> Tuple[int, int]:
        return (self.n_time, self.n_channel)

    def _get_dtype(self) -> np.dtype:
        return np.dtype("int16")


def add_raw_ephys(
    nwbfile: NWBFile, recfile: str, electrode_row_indices: list, conversion: float
) -> None:
    # TODO handle merging of multiple rec files and their timestamps

    electrode_table_region = nwbfile.create_electrode_table_region(
        region=electrode_row_indices,
        description="electrodes used in raw e-series recording",
    )

    nwb_hw_chan_order = [
        int(x) for x in list(nwbfile.electrodes.to_dataframe()["hwChan"])
    ]
    # nwb_hw_chan_order = []
    rec_dci = RecFileDataChunkIterator(
        recfile,
        nwb_hw_channel_order=nwb_hw_chan_order,
    )  # can set buffer_gb if needed

    # (16384, 32) chunks of dtype int16 (2 bytes) is 1 MB, which is recommended
    # by studies by the NWB team.
    # could also add compression here. zstd/blosc-zstd are recommended by the NWB team, but
    # they require the hdf5plugin library to be installed. gzip is available by default.
    data_data_io = H5DataIO(rec_dci, chunks=(16384, min(rec_dci.n_channel, 32)))

    # do we want to pull the timestamps from the rec file? or is there another source?
    eseries = ElectricalSeries(
        name="e-series",
        data=data_data_io,
        timestamps=rec_dci.timestamps,
        electrodes=electrode_table_region,  # TODO
        conversion=conversion,
        offset=0.0,  # TODO
    )

    nwbfile.add_acquisition(eseries)
