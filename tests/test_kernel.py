#!/usr/bin/env python3
"""
Real CuteDSL kernel tests with tracing integration.

These tests verify that cutedsl-trace can instrument actual CuteDSL kernels
and produce valid trace files.

Requires CUDA and cutlass.
"""

from pathlib import Path
import warnings

import torch
import cutlass.cute as cute
from cutlass.cute import Int32

from cutedsl_trace import (
    TraceType,
    BlockType,
    TrackType,
    StaticTraceBuilder,
    TraceWriter,
    create_hierarchical_tracks,
    reset_all_counters,
)
from cutedsl_trace.device import (
    read_globaltimer_lo,
    read_smid,
    store_global_wt_v2_u32,
)


# =============================================================================
# Trace Type Definitions
# =============================================================================


def create_vector_add_trace_types():
    """Create trace types for vector add kernel."""
    reset_all_counters()

    TraceCompute = TraceType(
        name="VectorAdd",
        label_string="VectorAdd",
        tooltip_string="Vector addition operation",
        param_count=0,
    )

    CTABlock = BlockType(
        name="CTA",
        label_string="CTA {blockLinear}",
        tooltip_string="Thread Block {blockLinear}",
    )

    WarpTrack = TrackType(
        name="Warp",
        label_string="Warp {lane}",
        tooltip_string="Warp {lane}",
    )

    return TraceCompute, CTABlock, WarpTrack


def create_matmul_trace_types():
    """Create trace types for matmul kernel phases."""
    reset_all_counters()

    TraceLoadA = TraceType(
        name="LoadA",
        label_string="LoadA",
        tooltip_string="Load matrix A from global memory",
        param_count=0,
    )

    TraceLoadB = TraceType(
        name="LoadB",
        label_string="LoadB",
        tooltip_string="Load matrix B from global memory",
        param_count=0,
    )

    TraceMMA = TraceType(
        name="MMA",
        label_string="MMA",
        tooltip_string="Matrix multiply-accumulate operation",
        param_count=0,
    )

    TraceStore = TraceType(
        name="StoreC",
        label_string="StoreC",
        tooltip_string="Store result matrix C",
        param_count=0,
    )

    CTABlock = BlockType(
        name="CTA",
        label_string="CTA {blockLinear}",
        tooltip_string="Thread Block {blockLinear}",
    )

    # Different track types for different phases
    ProducerTrack = TrackType(
        name="Producer",
        label_string="Producer",
        tooltip_string="Producer warp - loads data",
    )

    MathTrack = TrackType(
        name="Math",
        label_string="Math",
        tooltip_string="Math warp - tensor core operations",
    )

    ConsumerTrack = TrackType(
        name="Consumer",
        label_string="Consumer",
        tooltip_string="Consumer warp - stores data",
    )

    return (
        TraceLoadA,
        TraceLoadB,
        TraceMMA,
        TraceStore,
        CTABlock,
        ProducerTrack,
        MathTrack,
        ConsumerTrack,
    )


# =============================================================================
# Simple Traced Kernel: Vector Add with Warp-Level Tracing
# =============================================================================


@cute.kernel
def vector_add_traced_kernel(
    a_ptr: cute.Tensor,
    b_ptr: cute.Tensor,
    c_ptr: cute.Tensor,
    trace_buffer: cute.Tensor,
    n: Int32,
    row_stride: Int32,
    num_warps_per_block: Int32,
):
    """Vector add with trace timing capture."""
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]

    idx = bidx * bdim + tidx
    warp_id = tidx // 32
    lane_in_warp = tidx % 32

    # Only lane 0 of each warp traces
    should_trace = lane_in_warp == 0

    start_time = Int32(0)
    if should_trace:
        start_time = read_globaltimer_lo()

    # Actual work (all threads participate)
    if idx < n:
        a_val = a_ptr[idx]
        b_val = b_ptr[idx]
        c_ptr[idx] = a_val + b_val

    # Capture end time and write trace
    if should_trace:
        end_time = read_globaltimer_lo()
        sm_id = read_smid()

        # Compute trace buffer offset
        lane_offset = (bidx * num_warps_per_block + warp_id) * row_stride

        # Write header: [sm_id, write_offset]
        event_offset = lane_offset + 8  # After header
        write_offset = event_offset + 8  # One event (2 uint32s)

        header_ptr = trace_buffer.iterator + lane_offset
        store_global_wt_v2_u32(header_ptr, sm_id, Int32(write_offset))

        # Write event: [start_time, end_time]
        event_ptr = trace_buffer.iterator + event_offset
        store_global_wt_v2_u32(event_ptr, start_time, end_time)


@cute.jit
def launch_vector_add_traced(a, b, c, trace, n, row_stride, num_warps):
    # Grid/block must be compile-time constants for MLIR
    vector_add_traced_kernel(a, b, c, trace, n, row_stride, num_warps).launch(
        grid=[32, 1, 1], block=[256, 1, 1]
    )


def test_vector_add_traced():
    """Test vector add kernel with tracing at warp level."""
    TraceCompute, CTABlock, WarpTrack = create_vector_add_trace_types()

    # Test data
    n = 8192
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    # Setup trace buffer
    grid = (n + 255) // 256
    num_warps = 8

    trace_builder = StaticTraceBuilder(
        num_lanes=num_warps,
        trace_types=[TraceCompute] * num_warps,
        max_events_per_lane=20,
        grid_dims=(grid, 1, 1),
    )
    trace_builder.set_track_type(WarpTrack)

    trace_buffer = torch.zeros(
        grid * num_warps * trace_builder.row_stride_bytes,
        device="cuda",
        dtype=torch.uint8,
    )

    # Launch kernel - bypassing from_dlpack and passing torch tensors directly
    row_stride = Int32(trace_builder.row_stride_bytes)
    nw = Int32(num_warps)
    n_arg = Int32(n)

    # We still use cute.compile for robust launches
    func = cute.compile(
        launch_vector_add_traced,
        a,
        b,
        c,
        trace_buffer,
        n_arg,
        row_stride,
        nw,
    )
    func(a, b, c, trace_buffer, n_arg, row_stride, nw)

    torch.cuda.synchronize()

    # Verify result
    expected = a + b
    assert torch.allclose(c, expected), "Vector add result mismatch"

    # Copy trace to host and write file
    trace_builder._buffer = trace_buffer.cpu().numpy()
    trace_builder._host_buffer = trace_builder._buffer

    writer = TraceWriter("vector_add_traced")
    writer.set_block_type(CTABlock)
    writer.add_tensor(trace_builder)

    output_path = Path("/tmp/vector_add_traced.nanotrace")
    writer.write(str(output_path), compress=False)

    # Verify trace file exists
    assert output_path.exists()

    print(f"✓ test_vector_add_traced passed ({output_path})")


# =============================================================================
# Thread-Level Tracing Test
# =============================================================================


@cute.kernel
def vector_add_thread_traced(
    a_ptr: cute.Tensor,
    b_ptr: cute.Tensor,
    c_ptr: cute.Tensor,
    trace_buffer: cute.Tensor,
    n: Int32,
    trace_every_n: Int32,  # Trace every N threads to reduce overhead
    row_stride: Int32,
    num_lanes_per_block: Int32,
):
    """Vector add with per-thread tracing (sampled)."""
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]

    idx = bidx * bdim + tidx

    # Sample threads for tracing
    should_trace = (tidx % trace_every_n) == 0
    lane_id = tidx // trace_every_n  # Lane ID for sampled threads

    start_time = Int32(0)
    if should_trace:
        start_time = read_globaltimer_lo()

    if idx < n:
        a_val = a_ptr[idx]
        b_val = b_ptr[idx]
        c_ptr[idx] = a_val + b_val

    if should_trace:
        end_time = read_globaltimer_lo()
        sm_id = read_smid()

        lane_offset = (bidx * num_lanes_per_block + lane_id) * row_stride
        event_offset = lane_offset + 8
        write_offset = event_offset + 8

        header_ptr = trace_buffer.iterator + lane_offset
        store_global_wt_v2_u32(header_ptr, sm_id, Int32(write_offset))

        event_ptr = trace_buffer.iterator + event_offset
        store_global_wt_v2_u32(event_ptr, start_time, end_time)


@cute.jit
def launch_vector_add_thread(a, b, c, trace, n, every_n, row_stride, num_lanes):
    # Grid/block must be compile-time constants
    vector_add_thread_traced(a, b, c, trace, n, every_n, row_stride, num_lanes).launch(
        grid=[16, 1, 1], block=[256, 1, 1]
    )


def test_vector_add_thread_level():
    """Test vector add with per-thread tracing (for thread-level visualization)."""
    reset_all_counters()

    TraceThread = TraceType(
        name="ThreadWork",
        label_string="ThreadWork",
        tooltip_string="Per-thread computation",
        param_count=0,
    )

    CTABlock = BlockType(
        name="CTA",
        label_string="CTA {blockLinear}",
        tooltip_string="Thread Block {blockLinear}",
    )

    ThreadTrack = TrackType(
        name="Thread",
        label_string="Thread {lane}",
        tooltip_string="Thread {lane}",
    )

    n = 4096
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    grid = (n + 255) // 256
    trace_every_n = 32  # Sample every 32 threads
    num_lanes = 256 // trace_every_n

    trace_builder = StaticTraceBuilder(
        num_lanes=num_lanes,
        trace_types=[TraceThread] * num_lanes,
        max_events_per_lane=10,
        grid_dims=(grid, 1, 1),
    )
    trace_builder.set_track_type(ThreadTrack)

    trace_buffer = torch.zeros(
        grid * num_lanes * trace_builder.row_stride_bytes,
        device="cuda",
        dtype=torch.uint8,
    )

    row_stride = Int32(trace_builder.row_stride_bytes)
    nl = Int32(num_lanes)
    every_n = Int32(trace_every_n)
    n_arg = Int32(n)

    func = cute.compile(
        launch_vector_add_thread,
        a,
        b,
        c,
        trace_buffer,
        n_arg,
        every_n,
        row_stride,
        nl,
    )
    func(
        a,
        b,
        c,
        trace_buffer,
        n_arg,
        every_n,
        row_stride,
        nl,
    )

    torch.cuda.synchronize()

    expected = a + b
    assert torch.allclose(c, expected), "Vector add result mismatch"

    trace_builder._buffer = trace_buffer.cpu().numpy()
    trace_builder._host_buffer = trace_builder._buffer

    writer = TraceWriter("vector_add_threads")
    writer.set_block_type(CTABlock)
    writer.add_tensor(trace_builder)

    output_path = Path("/tmp/vector_add_threads.nanotrace")
    writer.write(str(output_path), compress=False)

    assert output_path.exists()
    print(f"✓ test_vector_add_thread_level passed ({output_path})")


# =============================================================================
# Multi-Phase Kernel Test
# =============================================================================


@cute.kernel
def multi_phase_kernel(
    a_ptr: cute.Tensor,
    b_ptr: cute.Tensor,
    c_ptr: cute.Tensor,
    trace_buffer: cute.Tensor,
    n: Int32,
    row_stride: Int32,
    num_lanes_per_block: Int32,
):
    """Simulated multi-phase kernel with per-phase tracing."""
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]
    bdim = cute.arch.block_dim()[0]

    idx = bidx * bdim + tidx
    warp_id = tidx // 32
    lane_in_warp = tidx % 32

    # 4 warps for different phases
    is_producer = warp_id < 2
    is_math = warp_id == 2
    is_consumer = warp_id == 3

    should_trace = lane_in_warp == 0

    start_time = Int32(0)
    lane_id = Int32(0)
    lane_offset = Int32(0)
    if should_trace:
        lane_id = warp_id if warp_id < 4 else 0
        lane_offset = (bidx * num_lanes_per_block + lane_id) * row_stride
        start_time = read_globaltimer_lo()

    # Simulated work per phase
    if is_producer and idx < n:
        _ = a_ptr[idx]  # Load phase
    elif is_math and idx < n:
        _ = a_ptr[idx] + b_ptr[idx]  # Compute phase
    elif is_consumer and idx < n:
        c_ptr[idx] = a_ptr[idx] + b_ptr[idx]  # Store phase

    if should_trace and warp_id < 4:
        end_time = read_globaltimer_lo()
        sm_id = read_smid()

        event_offset = lane_offset + 8
        write_offset = event_offset + 8

        header_ptr = trace_buffer.iterator + lane_offset
        store_global_wt_v2_u32(header_ptr, sm_id, Int32(write_offset))

        event_ptr = trace_buffer.iterator + event_offset
        store_global_wt_v2_u32(event_ptr, start_time, end_time)


@cute.jit
def launch_multi_phase(a, b, c, trace, n, row_stride, num_lanes):
    # Grid/block must be compile-time constants
    multi_phase_kernel(a, b, c, trace, n, row_stride, num_lanes).launch(
        grid=[16, 1, 1], block=[256, 1, 1]
    )


def test_multi_phase_kernel():
    """Test a kernel with multiple traced phases (load, compute, store)."""
    (
        TraceLoadA,
        TraceLoadB,
        TraceMMA,
        TraceStore,
        CTABlock,
        ProducerTrack,
        MathTrack,
        ConsumerTrack,
    ) = create_matmul_trace_types()

    n = 4096
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    c = torch.zeros(n, device="cuda", dtype=torch.float32)

    grid = (n + 255) // 256
    num_lanes = 4

    trace_builder = StaticTraceBuilder(
        num_lanes=num_lanes,
        trace_types=[TraceLoadA, TraceLoadB, TraceMMA, TraceStore],
        max_events_per_lane=10,
        grid_dims=(grid, 1, 1),
    )

    # Set custom track types per lane
    trace_builder.set_track_type(ProducerTrack, lane=0)
    trace_builder.set_track_type(ProducerTrack, lane=1)
    trace_builder.set_track_type(MathTrack, lane=2)
    trace_builder.set_track_type(ConsumerTrack, lane=3)

    trace_buffer = torch.zeros(
        grid * num_lanes * trace_builder.row_stride_bytes,
        device="cuda",
        dtype=torch.uint8,
    )

    row_stride = Int32(trace_builder.row_stride_bytes)
    nl = Int32(num_lanes)
    n_arg = Int32(n)

    func = cute.compile(
        launch_multi_phase,
        a,
        b,
        c,
        trace_buffer,
        n_arg,
        row_stride,
        nl,
    )
    func(a, b, c, trace_buffer, n_arg, row_stride, nl)

    torch.cuda.synchronize()

    trace_builder._buffer = trace_buffer.cpu().numpy()
    trace_builder._host_buffer = trace_builder._buffer

    writer = TraceWriter("multi_phase_kernel")
    writer.set_block_type(CTABlock)
    writer.add_tensor(trace_builder)

    output_path = Path("/tmp/multi_phase.nanotrace")
    writer.write(str(output_path), compress=False)

    assert output_path.exists()
    print(f"✓ test_multi_phase_kernel passed ({output_path})")


# =============================================================================
# Hierarchical Expansion Test
# =============================================================================


@cute.kernel
def hierarchical_kernel(
    a_ptr: cute.Tensor,
    trace_buffer: cute.Tensor,
    n: Int32,
    row_stride: Int32,
    num_lanes_per_block: Int32,
):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]

    warp_id = tidx // 32
    lane_in_warp = tidx % 32

    # 1. Warp tracing (only lane 0 of the warp)
    if lane_in_warp == 0:
        start_time = read_globaltimer_lo()
        sm_id = read_smid()

        # Simulated warp-level work
        _ = a_ptr[bidx * 256 + tidx]

        end_time = read_globaltimer_lo()

        lane_id = warp_id  # Lanes 0-7 are warps
        lane_offset = (bidx * num_lanes_per_block + lane_id) * row_stride

        header_ptr = trace_buffer.iterator + lane_offset
        store_global_wt_v2_u32(header_ptr, sm_id, Int32(lane_offset + 16))

        event_ptr = trace_buffer.iterator + lane_offset + 8
        store_global_wt_v2_u32(event_ptr, start_time, end_time)

    # 2. Thread tracing (all threads)
    start_time_th = read_globaltimer_lo()
    _ = a_ptr[bidx * 256 + tidx]
    end_time_th = read_globaltimer_lo()
    sm_id_th = read_smid()

    lane_id_th = 8 + tidx  # Lanes 8-263 are threads
    lane_offset_th = (bidx * num_lanes_per_block + lane_id_th) * row_stride

    header_ptr_th = trace_buffer.iterator + lane_offset_th
    store_global_wt_v2_u32(header_ptr_th, sm_id_th, Int32(lane_offset_th + 16))

    event_ptr_th = trace_buffer.iterator + lane_offset_th + 8
    store_global_wt_v2_u32(event_ptr_th, start_time_th, end_time_th)


@cute.jit
def launch_hierarchical(a, trace, n, row_stride, num_lanes):
    # Grid/block must be compile-time constants
    hierarchical_kernel(a, trace, n, row_stride, num_lanes).launch(
        grid=[1, 1, 1], block=[256, 1, 1]
    )


def test_hierarchical_expansion():
    """Test hierarchical warp->thread expansion."""
    TraceOp, CTABlock, _ = create_vector_add_trace_types()

    n = 256
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    grid = 1

    # Setup hierarchy with 8 warps and all 256 threads
    warp_tracks, thread_tracks = create_hierarchical_tracks(
        8, threads_per_warp=32, sample_threads=1
    )
    all_tracks = warp_tracks + thread_tracks

    trace_builder = StaticTraceBuilder(
        num_lanes=len(all_tracks),
        trace_types=[TraceOp] * len(all_tracks),
        max_events_per_lane=1,
        grid_dims=(grid, 1, 1),
    )

    for i, track in enumerate(all_tracks):
        trace_builder.set_track_type(track, lane=i)

    trace_buffer = torch.zeros(
        grid * len(all_tracks) * trace_builder.row_stride_bytes,
        device="cuda",
        dtype=torch.uint8,
    )

    row_stride = Int32(trace_builder.row_stride_bytes)
    nl = Int32(len(all_tracks))
    n_arg = Int32(n)

    func = cute.compile(
        launch_hierarchical,
        a,
        trace_buffer,
        n_arg,
        row_stride,
        nl,
    )
    func(a, trace_buffer, n_arg, row_stride, nl)

    torch.cuda.synchronize()

    trace_builder._buffer = trace_buffer.cpu().numpy()
    trace_builder._host_buffer = trace_builder._buffer

    writer = TraceWriter("hierarchical_trace")
    writer.set_block_type(CTABlock)
    writer.add_tensor(trace_builder)

    output_path = Path("/tmp/hierarchical.nanotrace")
    writer.write(str(output_path), compress=False)

    print(f"✓ test_hierarchical_expansion passed ({output_path})")
    assert output_path.exists()


# =============================================================================
# Main
# =============================================================================


def run_all_tests():
    """Run all real kernel tests."""
    # Suppress the DSL warning about dynamic block size
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Dynamic variable in block size.*"
    )
    print("\n" + "=" * 60)
    print("cutedsl-trace Real Kernel Tests")
    print("=" * 60 + "\n")

    test_vector_add_traced()
    test_vector_add_thread_level()
    test_multi_phase_kernel()
    test_hierarchical_expansion()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
