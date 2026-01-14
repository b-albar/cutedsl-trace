#!/usr/bin/env python3
"""
Benchmarks for cutedsl-trace overhead measurement.

Measures the overhead of trace instrumentation on CuteDSL kernels.
Requires CUDA and cutlass.
"""

import time
import statistics

import torch
import cutlass.cute as cute
from cutlass.cute import Int32

from cutedsl_trace.device import read_globaltimer_lo, read_smid


# =============================================================================
# Benchmarks
# =============================================================================


def benchmark_trace_overhead():
    """Benchmark the overhead of trace timer reads."""

    # Kernel without tracing
    @cute.jit
    def vector_add_baseline(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        n: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bdim = cute.arch.block_dim()[0]

        idx = bidx * bdim + tidx
        if idx < n:
            a_val = a_ptr[idx]
            b_val = b_ptr[idx]
            c_ptr[idx] = a_val + b_val

    # Kernel with timer reads (simulated trace overhead)
    @cute.jit
    def vector_add_with_timer(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        n: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bdim = cute.arch.block_dim()[0]

        idx = bidx * bdim + tidx

        # Trace overhead: read timer at start
        _ = read_globaltimer_lo()

        if idx < n:
            a_val = a_ptr[idx]
            b_val = b_ptr[idx]
            c_ptr[idx] = a_val + b_val

        # Trace overhead: read timer at end + SM ID
        _ = read_globaltimer_lo()
        _ = read_smid()

    sizes = [1024, 16384, 262144, 1048576, 4194304]
    results = []

    print("\n" + "=" * 70)
    print("cutedsl-trace Overhead Benchmark")
    print("=" * 70)
    print(
        f"{'Size':>12} | {'Baseline (µs)':>15} | {'With Timer (µs)':>15} | {'Overhead':>10}"
    )
    print("-" * 70)

    for n in sizes:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = torch.zeros(n, device="cuda", dtype=torch.float32)

        a_ptr = cute.from_dlpack(a)
        b_ptr = cute.from_dlpack(b)
        c_ptr = cute.from_dlpack(c)

        grid = (n + 255) // 256
        block = 256

        # Warmup baseline
        for _ in range(10):
            vector_add_baseline(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()

        # Benchmark baseline
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            vector_add_baseline(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / n_iters * 1e6

        # Warmup traced
        for _ in range(10):
            vector_add_with_timer(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()

        # Benchmark traced
        start = time.perf_counter()
        for _ in range(n_iters):
            vector_add_with_timer(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()
        traced_time = (time.perf_counter() - start) / n_iters * 1e6

        overhead = (traced_time / baseline_time - 1) * 100
        overhead_str = f"{overhead:+.1f}%" if overhead > 0 else f"{overhead:.1f}%"

        print(
            f"{n:>12,} | {baseline_time:>15.2f} | {traced_time:>15.2f} | {overhead_str:>10}"
        )

        results.append(
            {
                "size": n,
                "baseline_us": baseline_time,
                "traced_us": traced_time,
                "overhead_pct": overhead,
            }
        )

    print("-" * 70)
    avg_overhead = statistics.mean(r["overhead_pct"] for r in results)
    print(f"Average overhead: {avg_overhead:+.2f}%")
    print("=" * 70)

    return results


def benchmark_memory_store_overhead():
    """Benchmark the overhead of trace memory stores."""

    # Kernel with timer reads + global store (full trace overhead simulation)
    @cute.jit
    def vector_add_with_store(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        trace_ptr: cute.Pointer,
        n: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bdim = cute.arch.block_dim()[0]

        idx = bidx * bdim + tidx

        start = read_globaltimer_lo()

        if idx < n:
            a_val = a_ptr[idx]
            b_val = b_ptr[idx]
            c_ptr[idx] = a_val + b_val

        end = read_globaltimer_lo()

        # Simulate trace store (2 uint32s per event)
        trace_ptr[idx * 2] = start
        trace_ptr[idx * 2 + 1] = end

    # Baseline (no tracing)
    @cute.jit
    def vector_add_baseline(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        n: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        bidx = cute.arch.block_idx()[0]
        bdim = cute.arch.block_dim()[0]

        idx = bidx * bdim + tidx
        if idx < n:
            a_val = a_ptr[idx]
            b_val = b_ptr[idx]
            c_ptr[idx] = a_val + b_val

    sizes = [1024, 16384, 262144, 1048576]
    results = []

    print("\n" + "=" * 70)
    print("cutedsl-trace Memory Store Overhead Benchmark")
    print("=" * 70)
    print(
        f"{'Size':>12} | {'Baseline (µs)':>15} | {'With Store (µs)':>15} | {'Overhead':>10}"
    )
    print("-" * 70)

    for n in sizes:
        a = torch.randn(n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, device="cuda", dtype=torch.float32)
        c = torch.zeros(n, device="cuda", dtype=torch.float32)
        trace = torch.zeros(n * 2, device="cuda", dtype=torch.int32)

        a_ptr = cute.from_dlpack(a)
        b_ptr = cute.from_dlpack(b)
        c_ptr = cute.from_dlpack(c)
        trace_ptr = cute.from_dlpack(trace)

        grid = (n + 255) // 256
        block = 256

        # Warmup
        for _ in range(10):
            vector_add_baseline(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
            vector_add_with_store(
                a_ptr, b_ptr, c_ptr, trace_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()

        # Benchmark baseline
        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            vector_add_baseline(
                a_ptr, b_ptr, c_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / n_iters * 1e6

        # Benchmark with store
        start = time.perf_counter()
        for _ in range(n_iters):
            vector_add_with_store(
                a_ptr, b_ptr, c_ptr, trace_ptr, Int32(n), grid=(grid,), block=(block,)
            )
        torch.cuda.synchronize()
        traced_time = (time.perf_counter() - start) / n_iters * 1e6

        overhead = (traced_time / baseline_time - 1) * 100
        overhead_str = f"{overhead:+.1f}%" if overhead > 0 else f"{overhead:.1f}%"

        print(
            f"{n:>12,} | {baseline_time:>15.2f} | {traced_time:>15.2f} | {overhead_str:>10}"
        )

        results.append(
            {
                "size": n,
                "baseline_us": baseline_time,
                "traced_us": traced_time,
                "overhead_pct": overhead,
            }
        )

    print("-" * 70)
    avg_overhead = statistics.mean(r["overhead_pct"] for r in results)
    print(f"Average overhead (with memory store): {avg_overhead:+.2f}%")
    print("=" * 70)

    return results


# =============================================================================
# Main
# =============================================================================


def run_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("cutedsl-trace Benchmarks")
    print("=" * 70)

    benchmark_trace_overhead()
    benchmark_memory_store_overhead()


if __name__ == "__main__":
    run_benchmarks()
