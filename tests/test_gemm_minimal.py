#!/usr/bin/env python3
"""
Minimal GEMM kernel with tracing.
"""

import sys
import warnings
from pathlib import Path
import torch

# Ensure we can import from the source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cutlass.cute as cute
from cutlass.cute import Int32

from cutedsl_trace import (
    TraceType,
    BlockType,
    StaticTraceBuilder,
    TraceWriter,
    reset_all_counters,
)
from cutedsl_trace.device import (
    read_globaltimer_lo,
    read_smid,
    store_global_wt_v2_u32,
)


def create_gemm_trace_types():
    reset_all_counters()
    T_Compute = TraceType("Compute", "Compute", "Compute GEMM")
    CTABlock = BlockType("CTA", "Block {blockLinear}", "Block {blockLinear}")
    return T_Compute, CTABlock


@cute.kernel
def gemm_minimal_traced_kernel(
    A: cute.Tensor,
    B: cute.Tensor,
    C: cute.Tensor,
    trace_buffer: cute.Tensor,
    n: Int32,
    row_stride: Int32,
):
    tidx = cute.arch.thread_idx()[0]
    bidx = cute.arch.block_idx()[0]

    # Warp 0 traces
    warp_id = tidx // 32
    should_trace = (tidx % 32) == 0

    start_time = Int32(0)
    if should_trace:
        start_time = read_globaltimer_lo()

    # Minimal compute (not a real GEMM, just testing trace)
    if tidx < n:
        C[tidx] = A[tidx] * B[tidx]

    if should_trace:
        end_time = read_globaltimer_lo()
        sm_id = read_smid()

        lane_offset = (bidx * 4 + warp_id) * row_stride
        header_ptr = trace_buffer.iterator + lane_offset
        store_global_wt_v2_u32(header_ptr, sm_id, Int32(lane_offset + 16))

        event_ptr = trace_buffer.iterator + lane_offset + 8
        store_global_wt_v2_u32(event_ptr, start_time, end_time)


@cute.jit
def launch_gemm_minimal(A, B, C, trace, n, row_stride):
    # Grid/block must be compile-time constants
    gemm_minimal_traced_kernel(A, B, C, trace, n, row_stride).launch(
        grid=[1, 1, 1], block=[128, 1, 1]
    )


def test_gemm_trace():
    print("Running GEMM trace test...")
    # Suppress the DSL warning about dynamic block size
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*Dynamic variable in block size.*"
    )
    device = "cuda"
    n = 128
    A = torch.randn(n, device=device)
    B = torch.randn(n, device=device)
    C = torch.zeros(n, device=device)

    T_Comp, CTA = create_gemm_trace_types()

    num_warps = 4
    grid = 1

    builder = StaticTraceBuilder(
        num_lanes=num_warps,
        trace_types=[T_Comp] * num_warps,
        max_events_per_lane=1,
        grid_dims=(grid, 1, 1),
    )

    trace_buffer = torch.zeros(
        grid * num_warps * builder.row_stride_bytes,
        device=device,
        dtype=torch.uint8,
    )

    # Launch
    row_stride = Int32(builder.row_stride_bytes)
    n_arg = Int32(n)
    cute.compile(
        launch_gemm_minimal,
        A,
        B,
        C,
        trace_buffer,
        n_arg,
        row_stride,
    )(A, B, C, trace_buffer, n_arg, row_stride)

    torch.cuda.synchronize()

    # Write
    builder._buffer = trace_buffer.cpu().numpy()
    writer = TraceWriter("gemm_minimal")
    writer.set_block_type(CTA)
    writer.add_tensor(builder)

    output_path = Path("gemm_minimal.nanotrace")
    writer.write(str(output_path), compress=False)
    print(f"✓ Minimal GEMM trace written to {output_path}")


if __name__ == "__main__":
    test_gemm_trace()
