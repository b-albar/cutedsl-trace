"""
Tests for cutedsl-trace types module.
"""

from cutedsl_trace.types import (
    TraceType,
    BlockType,
    TrackType,
    LaneType,
    FormatDescriptor,
    TraceTensorHandle,
    LaneContext,
    DynamicLaneContext,
)


class TestTraceType:
    """Tests for TraceType."""

    def setup_method(self):
        """Reset counter before each test."""
        TraceType.reset_counter()

    def test_creation(self):
        """Test basic TraceType creation."""
        tt = TraceType(
            name="TestTrace",
            label_string="Test",
            tooltip_string="Test tooltip",
            param_count=0,
        )
        assert tt.name == "TestTrace"
        assert tt.label_string == "Test"
        assert tt.param_count == 0
        assert tt.lane_type == LaneType.STATIC

    def test_auto_id(self):
        """Test that IDs are auto-assigned."""
        t1 = TraceType("T1", "L1", "TT1", 0)
        t2 = TraceType("T2", "L2", "TT2", 0)
        t3 = TraceType("T3", "L3", "TT3", 0)

        assert t1.id == 0
        assert t2.id == 1
        assert t3.id == 2

    def test_event_width(self):
        """Test event width calculation."""
        t0 = TraceType("T0", "L", "T", param_count=0)
        t1 = TraceType("T1", "L", "T", param_count=1)
        t2 = TraceType("T2", "L", "T", param_count=2)
        t3 = TraceType("T3", "L", "T", param_count=3)
        t6 = TraceType("T6", "L", "T", param_count=6)

        assert t0.event_width == 2  # start, end
        assert t1.event_width == 4  # start, end, p0, padding
        assert t2.event_width == 4  # start, end, p0, p1
        assert t3.event_width == 8  # start, end, p0-p5
        assert t6.event_width == 8

    def test_descriptor(self):
        """Test format descriptor generation."""
        tt = TraceType("Test", "Label {0}", "Tooltip {0}", param_count=1)
        desc = tt.descriptor

        assert isinstance(desc, FormatDescriptor)
        assert desc.label_string == "Label {0}"
        assert desc.tooltip_string == "Tooltip {0}"
        assert desc.param_count == 1
        assert desc.id == tt.id


class TestBlockType:
    """Tests for BlockType."""

    def setup_method(self):
        BlockType.reset_counter()

    def test_creation(self):
        bt = BlockType(
            name="TestBlock",
            label_string="Block {blockLinear}",
            tooltip_string="Block {blockLinear} on SM",
        )
        assert bt.name == "TestBlock"
        assert bt.id >= 0


class TestTrackType:
    """Tests for TrackType."""

    def setup_method(self):
        TrackType.reset_counter()

    def test_creation(self):
        tt = TrackType(
            name="WarpTrack",
            label_string="Warp {lane}",
            tooltip_string="Warp {lane}",
        )
        assert tt.name == "WarpTrack"
        assert tt.param_count == 0


class TestLaneContext:
    """Tests for LaneContext."""

    def test_creation(self):
        ctx = LaneContext(
            base_offset_bytes=0,
            write_offset_bytes=8,
            max_event_width=2,
            enabled=True,
        )
        assert ctx.base_offset_bytes == 0
        assert ctx.write_offset_bytes == 8
        assert ctx.max_event_width_bytes == 8

    def test_advance(self):
        ctx = LaneContext(
            base_offset_bytes=0,
            write_offset_bytes=8,
            max_event_width=2,
        )
        ctx.advance()
        assert ctx.write_offset_bytes == 16

        ctx.advance()
        assert ctx.write_offset_bytes == 24


class TestDynamicLaneContext:
    """Tests for DynamicLaneContext."""

    def test_fixed_width(self):
        """Dynamic lanes always use width 8."""
        assert DynamicLaneContext.EVENT_WIDTH == 8
        assert DynamicLaneContext.EVENT_WIDTH_BYTES == 32

    def test_advance(self):
        ctx = DynamicLaneContext(
            base_offset_bytes=0,
            write_offset_bytes=32,
        )
        ctx.advance()
        assert ctx.write_offset_bytes == 64


class TestTraceTensorHandle:
    """Tests for TraceTensorHandle."""

    def test_creation(self):
        import numpy as np

        buffer = np.zeros(1024, dtype=np.uint8)

        handle = TraceTensorHandle(
            buffer=buffer,
            row_stride_bytes=256,
            num_lanes=8,
            max_event_width=4,
        )

        assert handle.row_stride_bytes == 256
        assert handle.num_lanes == 8
        assert handle.max_event_width == 4
        assert handle.max_event_width_bytes == 16
