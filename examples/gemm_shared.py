#!/usr/bin/env python3
"""
Real GEMM kernel with shared memory and highly granular hierarchical tracing.
This version traces every significant step of the kernel for detailed visualization.
"""

import sys
import warnings
from pathlib import Path
import torch

# Ensure we can import from the source
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cutlass
import cutlass.cute as cute
from cutlass.cute import Int32

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


def create_gemm_trace_types():
    """Create all trace types for detailed GEMM tracing."""
    reset_all_counters()

    # Memory operations - more granular
    T_LoadA = TraceType("LoadA", "Load A", "Load Matrix A tile to SMEM")
    T_LoadB = TraceType("LoadB", "Load B", "Load Matrix B tile to SMEM")

    # Synchronization
    T_SyncLoad = TraceType("SyncLoad", "Sync (Load)", "Barrier after loading tiles")
    T_SyncMMA = TraceType("SyncMMA", "Sync (MMA)", "Barrier after MMA computation")

    # Computation - broken down
    T_MMA = TraceType("MMA", "MMA", "Matrix Multiply-Accumulate")

    # Store phase
    T_Store = TraceType("Store", "Store", "Store result to global memory")

    # Block type
    CTABlock = BlockType("CTA", "CTA {blockLinear}", "Block {blockLinear}")

    return T_LoadA, T_LoadB, T_SyncLoad, T_SyncMMA, T_MMA, T_Store, CTABlock


@cute.kernel
def gemm_shared_traced_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    trace_buffer: cute.Tensor,
    row_stride_bytes: Int32,
    num_lanes: Int32,
    M: Int32,
    N: Int32,
    K: Int32,
    f_load_a: Int32,
    f_load_b: Int32,
    f_sync_load: Int32,
    f_sync_mma: Int32,
    f_mma: Int32,
    f_store: Int32,
):
    """GEMM kernel with granular tracing at every major step."""
    # Tile sizes
    TILE_M = 64
    TILE_N = 64
    TILE_K = 16

    # Thread/Block IDs
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    warp_id = tidx // 32
    lane_in_warp = tidx % 32

    # Tracing setup (warp-level: first lane of each warp traces)
    should_trace = lane_in_warp == 0

    grid_dim_y = cute.arch.grid_dim()[1]
    linear_block_id = bidx * grid_dim_y + bidy

    # Initialize tracing lane
    lane = begin_lane_dynamic_raw(
        num_lanes,
        row_stride_bytes,
        Int32(linear_block_id),
        Int32(warp_id),
        should_trace,
    )

    # Shared memory allocation
    smem = cutlass.utils.SmemAllocator()
    smem_a = smem.allocate_tensor(mA.element_type, cute.make_layout((TILE_M, TILE_K)))
    smem_b = smem.allocate_tensor(mB.element_type, cute.make_layout((TILE_K, TILE_N)))

    # Global memory tiles
    gA = cute.local_tile(mA, (TILE_M, TILE_K), (bidx, None))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (bidy, None))
    gC = cute.local_tile(mC, (TILE_M, TILE_N), (bidx, bidy))

    num_k_tiles = K // TILE_K

    # Accumulator (in registers)
    accum_template = cute.local_tile(mC, (1, TILE_N // 2), (0, 0))
    accum = cute.make_fragment_like(accum_template)
    accum.fill(0.0)

    # ========== MAIN K-LOOP ==========
    for k in range(num_k_tiles):
        # === LOAD PHASE: Matrix A ===
        s_load_a = start()
        for i in range(0, TILE_M * TILE_K, 128):
            idx = i + tidx
            if idx < TILE_M * TILE_K:
                smem_a[idx // TILE_K, idx % TILE_K] = gA[idx // TILE_K, idx % TILE_K, k]
        lane = end_event_dynamic_raw_0(
            s_load_a, trace_buffer, row_stride_bytes, lane, f_load_a
        )

        # === LOAD PHASE: Matrix B ===
        s_load_b = start()
        for i in range(0, TILE_K * TILE_N, 128):
            idx = i + tidx
            if idx < TILE_K * TILE_N:
                smem_b[idx // TILE_N, idx % TILE_N] = gB[idx // TILE_N, idx % TILE_N, k]
        lane = end_event_dynamic_raw_0(
            s_load_b, trace_buffer, row_stride_bytes, lane, f_load_b
        )

        # === SYNC: After loads ===
        s_sync_load = start()
        cute.arch.sync_threads()
        lane = end_event_dynamic_raw_0(
            s_sync_load, trace_buffer, row_stride_bytes, lane, f_sync_load
        )

        # === COMPUTE PHASE: MMA ===
        s_mma = start()
        for i in range(TILE_N // 2):
            row = tidx // 2
            col = (tidx % 2) * (TILE_N // 2) + i
            for kk in range(TILE_K):
                accum[0, i] += smem_a[row, kk] * smem_b[kk, col]
        lane = end_event_dynamic_raw_0(
            s_mma, trace_buffer, row_stride_bytes, lane, f_mma
        )

        # === SYNC: After MMA ===
        s_sync_mma = start()
        cute.arch.sync_threads()
        lane = end_event_dynamic_raw_0(
            s_sync_mma, trace_buffer, row_stride_bytes, lane, f_sync_mma
        )

    # ========== STORE PHASE ==========
    s_store = start()
    for i in range(TILE_N // 2):
        row = tidx // 2
        col = (tidx % 2) * (TILE_N // 2) + i
        gC[row, col] = accum[0, i]
    lane = end_event_dynamic_raw_0(
        s_store, trace_buffer, row_stride_bytes, lane, f_store
    )

    # Finalize tracing
    finish_lane_dynamic_raw(trace_buffer, lane)


@cute.jit
def launch_gemm_jit(
    A,
    B,
    C,
    trace_buffer,
    row_stride,
    num_lanes,
    M,
    N,
    K,
    f_load_a,
    f_load_b,
    f_sync_load,
    f_sync_mma,
    f_mma,
    f_store,
):
    """JIT wrapper for the GEMM kernel."""
    grid = [4, 4, 1]  # 4x4 = 16 blocks for 256x256 matrices with 64x64 tiles
    block = [128, 1, 1]
    gemm_shared_traced_kernel(
        A,
        B,
        C,
        trace_buffer,
        row_stride,
        num_lanes,
        M,
        N,
        K,
        f_load_a,
        f_load_b,
        f_sync_load,
        f_sync_mma,
        f_mma,
        f_store,
    ).launch(grid=grid, block=block)


_compiled_kernel_cache = {}


def run_gemm_shared_example():
    """Run the granularly-traced GEMM example."""
    print("Running Shared-Memory GEMM with granular tracing...")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Check for CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available. Cannot run example.")
        return

    device = "cuda"

    # Matrix dimensions - larger to create more blocks for interesting SM utilization
    M, N, K = 256, 256, 64

    # Create input matrices
    A = torch.randn((M, K), device=device, dtype=torch.float32)
    B = torch.randn((K, N), device=device, dtype=torch.float32)
    C = torch.zeros((M, N), device=device, dtype=torch.float32)

    # Create trace types
    T_LoadA, T_LoadB, T_SyncLoad, T_SyncMMA, T_MMA, T_Store, CTA = (
        create_gemm_trace_types()
    )
    all_trace_types = [T_LoadA, T_LoadB, T_SyncLoad, T_SyncMMA, T_MMA, T_Store]

    # 4 warps per block (128 threads / 32)
    num_warps = 4
    warp_tracks, _ = create_hierarchical_tracks(num_warps)

    # Grid configuration - 4x4 = 16 blocks for better SM utilization visibility
    grid = (M // 64, N // 64, 1)  # 4x4 grid with 256x256 matrices
    num_blocks = grid[0] * grid[1]
    print(f"  Grid: {grid[0]}x{grid[1]} = {num_blocks} blocks")

    # Calculate events per lane:
    # Per K-tile: 1 KIter + 1 LoadA + 1 LoadB + 1 SyncLoad + 1 MMA + 1 SyncMMA = 6 events
    # K tiles = 64/16 = 4
    # Total per lane = 4*6 + 1 Store = 25 events + some margin
    max_events = 50

    builder = DynamicTraceBuilder(
        num_lanes=num_warps,
        max_events_per_lane=max_events,
        grid_dims=(num_blocks, 1, 1),
    )

    for i, track in enumerate(warp_tracks):
        builder.set_track_type(track, lane=i)

    # Allocate trace buffer
    trace_buffer = torch.zeros(
        num_blocks * num_warps * builder.row_stride_bytes,
        device=device,
        dtype=torch.uint8,
    )

    # Prepare arguments
    row_stride = Int32(builder.row_stride_bytes)
    nl = Int32(builder.num_lanes)
    M_arg = Int32(M)
    N_arg = Int32(N)
    K_arg = Int32(K)

    # Trace format IDs
    f_la = Int32(T_LoadA.id)
    f_lb = Int32(T_LoadB.id)
    f_sl = Int32(T_SyncLoad.id)
    f_sm = Int32(T_SyncMMA.id)
    f_mma = Int32(T_MMA.id)
    f_st = Int32(T_Store.id)

    # Compile if needed
    compile_key = "gemm_shared_traced_granular"
    if compile_key not in _compiled_kernel_cache:
        print("  Compiling kernel (first run)...")
        _compiled_kernel_cache[compile_key] = cute.compile(
            launch_gemm_jit,
            A,
            B,
            C,
            trace_buffer,
            row_stride,
            nl,
            M_arg,
            N_arg,
            K_arg,
            f_la,
            f_lb,
            f_sl,
            f_sm,
            f_mma,
            f_st,
        )

    # Run the kernel
    print("  Running kernel...")
    _compiled_kernel_cache[compile_key](
        A,
        B,
        C,
        trace_buffer,
        row_stride,
        nl,
        M_arg,
        N_arg,
        K_arg,
        f_la,
        f_lb,
        f_sl,
        f_sm,
        f_mma,
        f_st,
    )

    torch.cuda.synchronize()

    # Write trace
    print("  Writing trace file...")
    builder._buffer = trace_buffer.cpu().numpy()
    writer = TraceWriter("gemm_shared_kernel")
    writer.set_block_type(CTA)
    writer.add_tensor(builder)

    # Register all trace types
    for t in all_trace_types:
        writer.register_trace_type(t)

    output_path = Path("gemm_shared.nanotrace")
    writer.write(str(output_path), compress=False)

    # Verify correctness
    C_ref = torch.mm(A, B)
    max_diff = (C - C_ref).abs().max().item()
    print(f"  Max difference from reference: {max_diff:.6e}")

    print(f"✓ GEMM trace written to {output_path}")
    print(f"  Trace types: {', '.join(t.name for t in all_trace_types)}")


if __name__ == "__main__":
    run_gemm_shared_example()
