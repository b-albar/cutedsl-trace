"""
Tests for cutedsl-trace host module.
"""

import json
import struct
import pytest
import numpy as np
import tempfile
import os

from cutedsl_trace.types import TraceType, BlockType, TrackType, TrackLevel, LaneType
from cutedsl_trace.host import (
    StaticTraceBuilder,
    DynamicTraceBuilder,
    TraceWriter,
    create_hierarchical_tracks,
)


class TestStaticTraceBuilder:
    """Tests for StaticTraceBuilder."""

    def setup_method(self):
        TraceType.reset_counter()
        BlockType.reset_counter()
        TrackType.reset_counter()

    def test_creation(self):
        """Test basic builder creation."""
        tt = TraceType("Test", "L", "T", 0)

        builder = StaticTraceBuilder(
            num_lanes=8,
            trace_types=[tt] * 8,
            max_events_per_lane=100,
            grid_dims=(16, 1, 1),
        )

        assert builder.num_lanes == 8
        assert builder.max_event_width == 2  # 0 params = width 2
        assert builder.grid_dims == (16, 1, 1)

    def test_event_width_calculation(self):
        """Test max event width computed from trace types."""
        t0 = TraceType("T0", "L", "T", 0)
        t2 = TraceType("T2", "L", "T", 2)  # width 4

        builder = StaticTraceBuilder(
            num_lanes=2,
            trace_types=[t0, t2],
            max_events_per_lane=10,
            grid_dims=(4, 1, 1),
        )

        # Max should be 4 (from t2)
        assert builder.max_event_width == 4

    def test_row_stride(self):
        """Test row stride calculation."""
        tt = TraceType("Test", "L", "T", 0)  # width 2

        builder = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=100,
            grid_dims=(1, 1, 1),
        )

        # (100 + 1 header) * 2 words * 4 bytes = 808 bytes
        assert builder.row_stride_bytes == 101 * 2 * 4

    def test_handle(self):
        """Test get_handle returns valid handle."""
        tt = TraceType("Test", "L", "T", 0)

        builder = StaticTraceBuilder(
            num_lanes=4,
            trace_types=[tt] * 4,
            max_events_per_lane=50,
            grid_dims=(8, 1, 1),
        )

        handle = builder.get_handle()

        assert handle.num_lanes == 4
        assert handle.max_event_width == 2
        assert handle.row_stride_bytes == builder.row_stride_bytes

    def test_track_type_all_lanes(self):
        """Test setting track type for all lanes."""
        tt = TraceType("Test", "L", "T", 0)
        track = TrackType("Track", "Warp {lane}", "Warp")

        builder = StaticTraceBuilder(
            num_lanes=4,
            trace_types=[tt] * 4,
            max_events_per_lane=10,
            grid_dims=(1, 1, 1),
        )

        builder.set_track_type(track)

        assert all(t == track for t in builder._track_types)

    def test_track_type_single_lane(self):
        """Test setting track type for a single lane."""
        tt = TraceType("Test", "L", "T", 0)
        track1 = TrackType("Track1", "T1", "T1")
        track2 = TrackType("Track2", "T2", "T2")

        builder = StaticTraceBuilder(
            num_lanes=4,
            trace_types=[tt] * 4,
            max_events_per_lane=10,
            grid_dims=(1, 1, 1),
        )

        builder.set_track_type(track1)
        builder.set_track_type(track2, lane=2)

        assert builder._track_types[0] == track1
        assert builder._track_types[1] == track1
        assert builder._track_types[2] == track2
        assert builder._track_types[3] == track1

    def test_dynamic_lane_type_rejected(self):
        """Test that DYNAMIC trace types are rejected."""
        tt = TraceType("Test", "L", "T", 0, lane_type=LaneType.DYNAMIC)

        with pytest.raises(ValueError, match="STATIC"):
            StaticTraceBuilder(
                num_lanes=1,
                trace_types=[tt],
                max_events_per_lane=10,
                grid_dims=(1, 1, 1),
            )

    def test_copy_to_host(self):
        """Test copying buffer to host."""
        tt = TraceType("Test", "L", "T", 0)

        builder = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(2, 1, 1),
        )

        host_buffer = builder.copy_to_host()

        assert isinstance(host_buffer, np.ndarray)
        assert host_buffer.dtype == np.uint8


class TestDynamicTraceBuilder:
    """Tests for DynamicTraceBuilder."""

    def setup_method(self):
        TraceType.reset_counter()

    def test_creation(self):
        builder = DynamicTraceBuilder(
            num_lanes=4,
            max_events_per_lane=50,
            grid_dims=(16, 1, 1),
        )

        assert builder.num_lanes == 4
        assert builder.EVENT_WIDTH == 8  # Always 8 for dynamic

    def test_handle(self):
        builder = DynamicTraceBuilder(
            num_lanes=2,
            max_events_per_lane=20,
            grid_dims=(4, 1, 1),
        )

        handle = builder.get_handle()

        assert handle.max_event_width == 8


class TestTraceWriter:
    """Tests for TraceWriter."""

    def setup_method(self):
        TraceType.reset_counter()
        BlockType.reset_counter()
        TrackType.reset_counter()

    def test_creation(self):
        writer = TraceWriter("test_kernel")
        assert writer.kernel_name == "test_kernel"

    def test_set_block_type(self):
        writer = TraceWriter("test_kernel")
        bt = BlockType("Block", "Block {blockLinear}", "Block tooltip")

        writer.set_block_type(bt)

        assert writer.block_type == bt
        assert bt.id in writer.format_descriptors

    def test_add_tensor(self):
        writer = TraceWriter("test_kernel")
        bt = BlockType("Block", "Block", "Block")
        tt = TraceType("Test", "L", "T", 0)
        track = TrackType("Track", "Lane", "Lane")

        writer.set_block_type(bt)

        builder = StaticTraceBuilder(
            num_lanes=2,
            trace_types=[tt] * 2,
            max_events_per_lane=10,
            grid_dims=(4, 1, 1),
        )
        builder.set_track_type(track)

        writer.add_tensor(builder)

        assert len(writer.tensors) == 1
        assert tt.id in writer.format_descriptors
        assert track.id in writer.format_descriptors

    def test_grid_dims_mismatch(self):
        """Test that adding tensors with different grids fails."""
        writer = TraceWriter("test")
        bt = BlockType("B", "B", "B")
        tt = TraceType("T", "L", "T", 0)

        writer.set_block_type(bt)

        b1 = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(4, 1, 1),
        )
        b2 = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(8, 1, 1),  # Different!
        )

        writer.add_tensor(b1)

        with pytest.raises(ValueError, match="Grid dimensions"):
            writer.add_tensor(b2)

    def test_write_requires_tensors(self):
        """Test that write fails without tensors."""
        writer = TraceWriter("test")
        bt = BlockType("B", "B", "B")
        writer.set_block_type(bt)

        with pytest.raises(ValueError, match="No tensors"):
            writer.write("test.nanotrace")

    def test_write_requires_block_type(self):
        """Test that write fails without block type."""
        writer = TraceWriter("test")
        tt = TraceType("T", "L", "T", 0)

        builder = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(1, 1, 1),
        )
        writer.add_tensor(builder)

        with pytest.raises(ValueError, match="Block type"):
            writer.write("test.nanotrace")

    def test_write_empty_trace(self):
        """Test writing an empty trace (no events)."""
        writer = TraceWriter("empty_test")
        bt = BlockType("Block", "B", "B")
        tt = TraceType("Trace", "L", "T", 0)
        track = TrackType("Track", "Lane", "Lane")

        writer.set_block_type(bt)

        builder = StaticTraceBuilder(
            num_lanes=2,
            trace_types=[tt] * 2,
            max_events_per_lane=10,
            grid_dims=(4, 1, 1),
        )
        builder.set_track_type(track)

        writer.add_tensor(builder)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.nanotrace")
            writer.write(filepath, compress=False)

            # Check file exists and isn't empty
            assert os.path.exists(filepath)
            size = os.path.getsize(filepath)
            assert size > 0

            # Check magic header
            with open(filepath, "rb") as f:
                header = f.read(10)
                assert header == b"nanotrace\x00"


class TestHierarchicalTracks:
    """Tests for create_hierarchical_tracks helper."""

    def setup_method(self):
        TraceType.reset_counter()
        TrackType.reset_counter()

    def test_basic_creation(self):
        """Test creating hierarchical warp/thread tracks."""
        warp_tracks, thread_tracks = create_hierarchical_tracks(
            warps_per_block=4,
            threads_per_warp=32,
            sample_threads=8,  # Sample every 8th thread
        )

        # Should have 4 warp tracks
        assert len(warp_tracks) == 4

        # Each warp should have 32/8 = 4 thread tracks
        assert len(thread_tracks) == 4 * 4  # 16 total

    def test_warp_track_properties(self):
        """Test warp track properties."""
        warp_tracks, _ = create_hierarchical_tracks(
            warps_per_block=2,
            sample_threads=16,
        )

        assert warp_tracks[0].level == TrackLevel.WARP
        assert warp_tracks[0].parent_lane is None
        assert "Warp 0" in warp_tracks[0].label_string
        assert warp_tracks[1].level == TrackLevel.WARP

    def test_thread_track_properties(self):
        """Test thread track properties."""
        warp_tracks, thread_tracks = create_hierarchical_tracks(
            warps_per_block=2,
            sample_threads=16,  # 2 threads per warp sampled
        )

        # First warp's threads
        assert thread_tracks[0].level == TrackLevel.THREAD
        assert thread_tracks[0].parent_lane == 0
        assert thread_tracks[1].level == TrackLevel.THREAD
        assert thread_tracks[1].parent_lane == 0

        # Second warp's threads
        assert thread_tracks[2].parent_lane == 1
        assert thread_tracks[3].parent_lane == 1

    def test_all_threads(self):
        """Test creating tracks for all threads (no sampling)."""
        warp_tracks, thread_tracks = create_hierarchical_tracks(
            warps_per_block=1,
            threads_per_warp=32,
            sample_threads=None,  # All threads
        )

        assert len(warp_tracks) == 1
        assert len(thread_tracks) == 32


def _make_numpy_buffer(builder):
    """Replace the builder's GPU buffer with a numpy buffer for testing."""
    if hasattr(builder._buffer, "cpu"):
        size = builder._buffer.shape[0]
        builder._buffer = np.zeros(size, dtype=np.uint8)


def _inject_events(builder, block_id, lane_id, events):
    """Inject synthetic events into a trace builder's numpy buffer.

    Each event is a tuple of (start_ns, end_ns).
    Call _make_numpy_buffer() first if the builder uses a CUDA buffer.
    """
    buf = builder._buffer

    num_lanes = builder.num_lanes
    row_stride = builder.row_stride_bytes

    if isinstance(builder, StaticTraceBuilder):
        ew = builder.max_event_width
    else:
        ew = DynamicTraceBuilder.EVENT_WIDTH

    ew_bytes = ew * 4
    lane_base = (block_id * num_lanes + lane_id) * row_stride

    # Write events starting after the header slot
    write_offset = lane_base + ew_bytes
    for start_ns, end_ns in events:
        if ew == 2:
            data = struct.pack("<II", start_ns, end_ns)
        elif ew == 4:
            data = struct.pack("<IIII", start_ns, end_ns, 0, 0)
        else:
            data = struct.pack("<IIIIIIII", start_ns, end_ns, 0, 0, 0, 0, 0, 0)
        buf[write_offset : write_offset + ew_bytes] = np.frombuffer(data, dtype=np.uint8)
        write_offset += ew_bytes

    # Write header: sm_id=42, write_offset_bytes=current position
    header = struct.pack("<II", 42, write_offset)
    buf[lane_base : lane_base + 8] = np.frombuffer(header, dtype=np.uint8)


class TestWritePerfetto:
    """Tests for TraceWriter.write_perfetto()."""

    def setup_method(self):
        TraceType.reset_counter()
        BlockType.reset_counter()
        TrackType.reset_counter()

    def test_write_perfetto_basic(self):
        """Test basic Perfetto JSON export with synthetic events."""
        tt = TraceType("LoadA", "Load A", "Load matrix A", 0)
        bt = BlockType("CTA", "Block {blockLinear}", "Block tooltip")
        track = TrackType("Warp", "Warp {lane}", "Warp tooltip")

        builder = StaticTraceBuilder(
            num_lanes=2,
            trace_types=[tt] * 2,
            max_events_per_lane=10,
            grid_dims=(2, 1, 1),
        )
        builder.set_track_type(track)
        _make_numpy_buffer(builder)

        # Inject events: block 0 lane 0, block 0 lane 1, block 1 lane 0
        _inject_events(builder, 0, 0, [(1000, 2000), (3000, 4000)])
        _inject_events(builder, 0, 1, [(1500, 2500)])
        _inject_events(builder, 1, 0, [(2000, 3500)])

        writer = TraceWriter("test_kernel")
        writer.set_block_type(bt)
        writer.add_tensor(builder)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "trace.json")
            writer.write_perfetto(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

        # Should be a list of events
        assert isinstance(data, list)

        # Separate metadata and trace events
        meta_events = [e for e in data if e["ph"] == "M"]
        trace_events = [e for e in data if e["ph"] == "X"]

        # 1 process_name + thread_name per track (4 tracks with events)
        process_names = [e for e in meta_events if e["name"] == "process_name"]
        thread_names = [e for e in meta_events if e["name"] == "thread_name"]
        assert len(process_names) == 1
        assert process_names[0]["args"]["name"] == "test_kernel"
        assert len(thread_names) >= 3  # at least 3 tracks with events

        # Total of 4 trace events
        assert len(trace_events) == 4

        # All events should have required Perfetto fields
        for event in trace_events:
            assert "name" in event
            assert "ts" in event
            assert "dur" in event
            assert "pid" in event
            assert "tid" in event
            assert event["cat"] == "gpu"
            assert event["args"]["sm_id"] == 42

        # Timestamps should be in microseconds (original ns / 1000)
        # Min time is 1000ns, so first event should start at ts=0
        first_event = min(trace_events, key=lambda e: e["ts"])
        assert first_event["ts"] == 0.0

    def test_write_perfetto_empty_trace(self):
        """Test Perfetto export with no events."""
        tt = TraceType("Test", "L", "T", 0)
        bt = BlockType("Block", "B", "B")
        track = TrackType("Track", "Lane", "Lane")

        builder = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(1, 1, 1),
        )
        builder.set_track_type(track)

        writer = TraceWriter("empty_kernel")
        writer.set_block_type(bt)
        writer.add_tensor(builder)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "trace.json")
            writer.write_perfetto(filepath)

            with open(filepath, "r") as f:
                data = json.load(f)

        # Only metadata, no trace events
        trace_events = [e for e in data if e["ph"] == "X"]
        assert len(trace_events) == 0

    def test_write_perfetto_requires_tensors(self):
        """Test that write_perfetto fails without tensors."""
        writer = TraceWriter("test")
        bt = BlockType("B", "B", "B")
        writer.set_block_type(bt)

        with pytest.raises(ValueError, match="No tensors"):
            writer.write_perfetto("test.json")

    def test_write_perfetto_requires_block_type(self):
        """Test that write_perfetto fails without block type."""
        writer = TraceWriter("test")
        tt = TraceType("T", "L", "T", 0)

        builder = StaticTraceBuilder(
            num_lanes=1,
            trace_types=[tt],
            max_events_per_lane=10,
            grid_dims=(1, 1, 1),
        )
        writer.add_tensor(builder)

        with pytest.raises(ValueError, match="Block type"):
            writer.write_perfetto("test.json")
