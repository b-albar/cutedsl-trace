#!/usr/bin/env python3
"""Benchmark measuring tracing overhead on GEMM kernel."""

import sys
import time
import warnings
import statistics
from pathlib import Path

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute import Int32

sys.path.insert(0, str(Path(__file__).parent.parent))

from cutedsl_trace import TraceType, DynamicTraceBuilder, reset_all_counters
from cutedsl_trace.config import set_tracing_enabled
from cutedsl_trace.device import (
    begin_lane_dynamic_raw,
    start,
    end_event_dynamic_raw_0,
    finish_lane_dynamic_raw,
)

# Tile sizes
TILE_M, TILE_N, TILE_K = 64, 64, 16


@cute.kernel
def gemm_raw(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, K: Int32):
    """Raw GEMM kernel without tracing."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    smem_a = smem.allocate_tensor(mA.element_type, cute.make_layout((TILE_M, TILE_K)))
    smem_b = smem.allocate_tensor(mB.element_type, cute.make_layout((TILE_K, TILE_N)))

    gA = cute.local_tile(mA, (TILE_M, TILE_K), (bidx, None))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (bidy, None))
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (bidx, bidy))

    accum = cute.make_fragment_like(cute.local_tile(mC, (1, TILE_N // 2), (0, 0)))
    accum.fill(0.0)

    for k in range(K // TILE_K):
        for i in range(0, TILE_M * TILE_K, 128):
            idx = i + tidx
            if idx < TILE_M * TILE_K:
                smem_a[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K, k]
        for i in range(0, TILE_K * TILE_N, 128):
            idx = i + tidx
            if idx < TILE_K * TILE_N:
                smem_b[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N, k]
        cute.arch.sync_threads()

        for i in range(TILE_N // 2):
            row, col = tidx // 2, (tidx % 2) * (TILE_N // 2) + i
            for kk in range(TILE_K):
                accum[0, i] += smem_a[row, kk] * smem_b[kk, col]
        cute.arch.sync_threads()

    for i in range(TILE_N // 2):
        row, col = tidx // 2, (tidx % 2) * (TILE_N // 2) + i
        gC[row, col] = accum[0, i]


@cute.kernel
def gemm_traced(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    trace_buffer: cute.Tensor,
    row_stride: Int32,
    num_lanes: Int32,
    K: Int32,
    f_load: Int32,
    f_mma: Int32,
    f_store: Int32,
):
    """GEMM kernel with tracing."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_id, lane_in_warp = tidx // 32, tidx % 32
    linear_block = bidx * cute.arch.grid_dim()[1] + bidy

    lane = begin_lane_dynamic_raw(
        num_lanes, row_stride, Int32(linear_block), Int32(warp_id), lane_in_warp == 0
    )

    smem = cutlass.utils.SmemAllocator()
    smem_a = smem.allocate_tensor(mA.element_type, cute.make_layout((TILE_M, TILE_K)))
    smem_b = smem.allocate_tensor(mB.element_type, cute.make_layout((TILE_K, TILE_N)))

    gA = cute.local_tile(mA, (TILE_M, TILE_K), (bidx, None))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (bidy, None))
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (bidx, bidy))

    accum = cute.make_fragment_like(cute.local_tile(mC, (1, TILE_N // 2), (0, 0)))
    accum.fill(0.0)

    for k in range(K // TILE_K):
        s = start()
        for i in range(0, TILE_M * TILE_K, 128):
            idx = i + tidx
            if idx < TILE_M * TILE_K:
                smem_a[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K, k]
        for i in range(0, TILE_K * TILE_N, 128):
            idx = i + tidx
            if idx < TILE_K * TILE_N:
                smem_b[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N, k]
        lane = end_event_dynamic_raw_0(s, trace_buffer, row_stride, lane, f_load)
        cute.arch.sync_threads()

        s = start()
        for i in range(TILE_N // 2):
            row, col = tidx // 2, (tidx % 2) * (TILE_N // 2) + i
            for kk in range(TILE_K):
                accum[0, i] += smem_a[row, kk] * smem_b[kk, col]
        lane = end_event_dynamic_raw_0(s, trace_buffer, row_stride, lane, f_mma)
        cute.arch.sync_threads()

    s = start()
    for i in range(TILE_N // 2):
        row, col = tidx // 2, (tidx % 2) * (TILE_N // 2) + i
        gC[row, col] = accum[0, i]
    lane = end_event_dynamic_raw_0(s, trace_buffer, row_stride, lane, f_store)
    finish_lane_dynamic_raw(trace_buffer, lane)


def benchmark(kernel_fn, args, warmup=10, iters=100):
    """Run benchmark and return time in microseconds."""
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        kernel_fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def run_benchmark(num_trials=3):
    """Run the GEMM tracing overhead benchmark."""
    warnings.filterwarnings("ignore", category=UserWarning)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("\n" + "=" * 70)
    print("GEMM Tracing Overhead Benchmark")
    print("=" * 70)

    # Only use 1024x1024 for compute-bound measurement
    sizes = [(1024, 1024, 256)]

    for M, N, K in sizes:
        print(f"\nMatrix: {M}×{N}×{K}, Trials: {num_trials}")
        print("-" * 50)

        grid_x, grid_y = M // TILE_M, N // TILE_N
        num_blocks, num_warps = grid_x * grid_y, 4

        # Allocate tensors for this size
        A = torch.randn((M, K), device="cuda", dtype=torch.float32)
        B = torch.randn((K, N), device="cuda", dtype=torch.float32)
        C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

        # Compile raw kernel
        @cute.jit
        def launch_raw(A, B, C, K, gx, gy):
            gemm_raw(A, B, C, K).launch(grid=[gx, gy, 1], block=[128, 1, 1])

        raw_kernel = cute.compile(
            launch_raw, A, B, C, Int32(K), Int32(grid_x), Int32(grid_y)
        )

        # Setup tracing
        reset_all_counters()
        T_Load = TraceType("Load", "Load", "")
        T_MMA = TraceType("MMA", "MMA", "")
        T_Store = TraceType("Store", "Store", "")

        builder = DynamicTraceBuilder(num_warps, 50, (num_blocks, 1, 1))
        trace_buf = torch.zeros(
            num_blocks * num_warps * builder.row_stride_bytes,
            device="cuda",
            dtype=torch.uint8,
        )

        # Compile traced kernels
        traced_kernels = {}
        for enabled in [False, True]:
            set_tracing_enabled(enabled)

            @cute.jit
            def launch_traced(A, B, C, buf, rs, nl, K, fl, fm, fs, gx, gy):
                gemm_traced(A, B, C, buf, rs, nl, K, fl, fm, fs).launch(
                    grid=[gx, gy, 1], block=[128, 1, 1]
                )

            traced_kernels[enabled] = cute.compile(
                launch_traced,
                A,
                B,
                C,
                trace_buf,
                Int32(builder.row_stride_bytes),
                Int32(num_warps),
                Int32(K),
                Int32(T_Load.id),
                Int32(T_MMA.id),
                Int32(T_Store.id),
                Int32(grid_x),
                Int32(grid_y),
            )

        # Run trials
        raw_times, disabled_times, enabled_times = [], [], []

        raw_args = (A, B, C, Int32(K), Int32(grid_x), Int32(grid_y))
        traced_args = (
            A,
            B,
            C,
            trace_buf,
            Int32(builder.row_stride_bytes),
            Int32(num_warps),
            Int32(K),
            Int32(T_Load.id),
            Int32(T_MMA.id),
            Int32(T_Store.id),
            Int32(grid_x),
            Int32(grid_y),
        )

        for _ in range(num_trials):
            raw_times.append(benchmark(raw_kernel, raw_args))
            disabled_times.append(benchmark(traced_kernels[False], traced_args))
            enabled_times.append(benchmark(traced_kernels[True], traced_args))

        # Results
        raw_mean = statistics.mean(raw_times)
        dis_mean = statistics.mean(disabled_times)
        en_mean = statistics.mean(enabled_times)

        print(f"  Raw (no tracing):     {raw_mean:7.2f}µs  (baseline)")
        print(
            f"  Traced (disabled):    {dis_mean:7.2f}µs  ({(dis_mean / raw_mean - 1) * 100:+.1f}%)"
        )
        print(
            f"  Traced (enabled):     {en_mean:7.2f}µs  ({(en_mean / raw_mean - 1) * 100:+.1f}%)"
        )

        # Check if disabled is near raw
        dis_overhead = abs((dis_mean / raw_mean - 1) * 100)
        print()
        if dis_overhead < 5:
            print("✓ Disabled tracing achieves near-raw performance")
        else:
            print(f"⚠ Disabled tracing has {dis_overhead:.1f}% overhead")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmark()
