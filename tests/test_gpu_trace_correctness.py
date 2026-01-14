#!/usr/bin/env python3
"""
Test trace file structure with minimal GPU kernel.

This test validates that the trace writer produces correctly formatted
.nanotrace files that match the spec using a real end-to-end GPU kernel.
"""

import struct
import sys
import tempfile
from pathlib import Path

import pytest
import torch

try:
    import cutlass.cute as cute
    from cutlass.cute import Int32

    has_cute = True
except ImportError:
    has_cute = False
    # Mock Int32 for type checking if needed, or just let tests skip
    Int32 = int

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cutedsl_trace import (
    TraceType,
    BlockType,
    DynamicTraceBuilder,
    TraceWriter,
    create_hierarchical_tracks,
    reset_all_counters,
)
from cutedsl_trace.device import (
    begin_lane_dynamic_raw,
    start,
    end_event_dynamic_raw_0,
    finish_lane_dynamic_raw,
)
from cutedsl_trace.host import parse_events_from_buffer


# Only define kernels if we have cute
if has_cute:

    @cute.kernel
    def simple_traced_kernel(
        trace_buffer: cute.Tensor,
        row_stride_bytes: Int32,
        num_lanes: Int32,
        format_id: Int32,
        num_events: Int32,
    ):
        """Simple kernel that records a fixed number of events per warp."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        warp_id = tidx // 32
        lane_in_warp = tidx % 32

        # Only lane 0 of each warp traces
        should_trace = lane_in_warp == 0

        # Initialize lane
        lane = begin_lane_dynamic_raw(
            num_lanes,
            row_stride_bytes,
            Int32(bidx),
            Int32(warp_id),
            should_trace,
        )

        # Record events
        for i in range(num_events):
            s = start()
            # Minimal work: multiply i by 2 (result unused but prevents optimization)
            _ = i * 2
            lane = end_event_dynamic_raw_0(
                s, trace_buffer, row_stride_bytes, lane, format_id
            )

        # Finalize
        finish_lane_dynamic_raw(trace_buffer, lane)

    @cute.jit
    def launch_simple_kernel(
        trace_buffer,
        row_stride_bytes,
        num_lanes,
        format_id,
        num_events,
    ):
        """JIT wrapper for launching the kernel."""
        # Grid/block must be compile-time constants for MLIR
        simple_traced_kernel(
            trace_buffer,
            row_stride_bytes,
            num_lanes,
            format_id,
            num_events,
        ).launch(grid=[2, 1, 1], block=[64, 1, 1])


_kernel_cache = {}


def create_test_types():
    """Create minimal trace types for testing."""
    reset_all_counters()
    T_Work = TraceType("Work", "Work", "Work unit")
    CTA = BlockType("CTA", "Block {blockLinear}", "Block {blockLinear}")
    return T_Work, CTA


def run_simple_kernel(num_blocks: int, num_warps: int, num_events: int):
    """Run the simple traced kernel and return the buffer."""
    if not has_cute:
        pytest.skip("cutlass.cute not available")

    T_Work, CTA = create_test_types()

    warp_tracks, _ = create_hierarchical_tracks(num_warps)

    builder = DynamicTraceBuilder(
        num_lanes=num_warps,
        max_events_per_lane=num_events + 10,
        grid_dims=(num_blocks, 1, 1),
    )

    for i, track in enumerate(warp_tracks):
        builder.set_track_type(track, lane=i)

    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    trace_buffer = torch.zeros(
        num_blocks * num_warps * builder.row_stride_bytes,
        device=device,
        dtype=torch.uint8,
    )

    # Arguments
    row_stride = Int32(builder.row_stride_bytes)
    nl = Int32(builder.num_lanes)
    fmt = Int32(T_Work.id)
    ne = Int32(num_events)

    # Compile and run
    cache_key = (num_blocks, num_warps, num_events)
    if cache_key not in _kernel_cache:
        _kernel_cache[cache_key] = cute.compile(
            launch_simple_kernel,
            trace_buffer,
            row_stride,
            nl,
            fmt,
            ne,
        )

    _kernel_cache[cache_key](
        trace_buffer,
        row_stride,
        nl,
        fmt,
        ne,
    )
    torch.cuda.synchronize()

    return builder, trace_buffer.cpu().numpy(), T_Work, CTA


@pytest.mark.skipif(not has_cute, reason="cutlass.cute not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_trace_structure_end_to_end():
    """
    Complete end-to-end test:
    1. Run GPU kernel
    2. Check buffer layout (device format)
    3. Parse buffer (host format)
    4. Write file
    5. Verify file structure (binary spec)
    """
    num_blocks = 2
    num_warps = 2
    num_events = 3

    # 1. Run Kernel
    builder, buffer, T_Work, CTA = run_simple_kernel(num_blocks, num_warps, num_events)
    builder._buffer = buffer

    # 2. Check Buffer Layout
    # Check each lane's header
    for block_id in range(num_blocks):
        for lane_id in range(num_warps):
            lane_base = (block_id * num_warps + lane_id) * builder.row_stride_bytes
            # sm_id = struct.unpack("<I", buffer[lane_base : lane_base + 4])[0]
            write_offset = struct.unpack("<I", buffer[lane_base + 4 : lane_base + 8])[0]

            events_start = lane_base + 32  # EVENT_WIDTH_BYTES
            if write_offset > 0:
                bytes_written = write_offset - events_start
                event_count = bytes_written // 32
            else:
                event_count = 0

            assert event_count == num_events, (
                f"Lane {block_id}:{lane_id} has {event_count} events, expected {num_events}"
            )

            if event_count > 0:
                ev_off = events_start
                _, _, fmt_id, _ = struct.unpack("<IIII", buffer[ev_off : ev_off + 16])
                assert fmt_id == T_Work.id

    # 3. Parse Buffer
    blocks, tracks, min_time = parse_events_from_buffer(buffer, builder, 0)
    expected_tracks = num_blocks * num_warps
    assert len(tracks) == expected_tracks
    for track in tracks:
        assert len(track.events) == num_events

    # 4. Write File
    with tempfile.NamedTemporaryFile(suffix=".nanotrace", delete=False) as f:
        output_path = f.name

    try:
        writer = TraceWriter("test_kernel")
        writer.set_block_type(CTA)
        writer.add_tensor(builder)
        writer.format_descriptors[T_Work.id] = T_Work.descriptor
        writer.write(output_path, compress=False)

        # 5. Verify File Structure
        data = Path(output_path).read_bytes()

        # Skip header (12) + kernel name string (2+len) + grid(12) + cluster(12) + counts(20)
        # Just find where tracks verify loop starts
        offset = 12
        name_len = struct.unpack("<H", data[offset : offset + 2])[0]
        offset += 2 + name_len + 24

        fmt_count = struct.unpack("<I", data[offset : offset + 4])[0]
        block_count = struct.unpack("<I", data[offset + 4 : offset + 8])[0]
        track_count = struct.unpack("<I", data[offset + 8 : offset + 12])[0]
        total_events = struct.unpack("<Q", data[offset + 12 : offset + 20])[0]

        assert track_count == expected_tracks
        assert total_events == expected_tracks * num_events

        # Skip counts
        offset += 20

        # Skip formats
        for _ in range(fmt_count):
            l_len = struct.unpack("<H", data[offset : offset + 2])[0]
            offset += 2 + l_len
            t_len = struct.unpack("<H", data[offset : offset + 2])[0]
            offset += 2 + t_len
            offset += 1  # param_count

        # Skip blocks
        offset += block_count * 12

        # Verify tracks
        for _ in range(track_count):
            # Track header: block_idx(4), fmt(2), lane(4), event_count(4) = 14 bytes
            # IF using old format it was 19 bytes. We verify it's NOT that.

            # Read header
            offset += 10  # block_idx, fmt, lane
            event_count = struct.unpack("<I", data[offset : offset + 4])[0]
            offset += 4

            assert event_count == num_events, (
                f"Track header has wrong event count: {event_count}"
            )

            # Events
            offset += event_count * 10  # 10 bytes per event (no params)

        assert offset == len(data), "File has extra bytes or misalignment"

    finally:
        if Path(output_path).exists():
            Path(output_path).unlink()


if __name__ == "__main__":
    if has_cute and torch.cuda.is_available():
        test_gpu_trace_structure_end_to_end()
        print("✓ End-to-end GPU trace structure test passed")
    else:
        print("Skipped: GPU or cute not available")
