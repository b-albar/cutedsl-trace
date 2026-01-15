"""
Device-side API for cutedsl-trace.

This module provides the device-side tracing functions that are called from
within CuteDSL kernels. These functions use inline PTX assembly to capture
timestamps and write trace data efficiently.

Usage Pattern:
    lane = begin_lane(trace_handle, block_id, warp_id, should_trace)

    for i in range(...):
        s = start()
        # ... work ...
        end_event(s, trace_handle, lane, format_id)

    finish_lane(trace_handle, lane)
"""

from cutlass import cute, const_expr
from cutlass.cute import Int32, Boolean
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .config import is_tracing_disabled
from .types import TraceTensorHandle, LaneContext, DynamicLaneContext


# =============================================================================
# Inline Assembly Utilities
# =============================================================================


@dsl_user_op
def read_globaltimer_lo(*, loc=None, ip=None) -> Int32:
    """Read the low 32-bit portion of the global timer.

    The GPU global timer has 32ns resolution. This function captures
    the lower 32 bits, which is sufficient for kernel traces up to
    ~4.3 seconds.

    Returns:
        Int32: Current global timer value (lower 32 bits)
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %globaltimer_lo;",
            "=r",
            has_side_effects=True,  # Prevent optimization of timer reads
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def read_smid(*, loc=None, ip=None) -> Int32:
    """Read the SM ID of the current thread.

    Returns:
        Int32: SM ID (0-65535)
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=True,  # Prevent optimization of SM ID reads
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def store_global_wt_v2_u32(
    ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store 2 uint32 values using write-through policy.

    Uses the .wt (write-through) modifier to bypass L1 cache and
    minimize pollution for write-once trace data.

    Args:
        ptr: Pointer to write location (must be 8-byte aligned)
        v0, v1: Values to store
    """
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_int, v0.ir_value(loc=loc, ip=ip), v1.ir_value(loc=loc, ip=ip)],
        "st.global.wt.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_wt_v4_u32(
    ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store 4 uint32 values using write-through policy.

    Args:
        ptr: Pointer to write location (must be 16-byte aligned)
        v0-v3: Values to store
    """
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            ptr_int,
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
        ],
        "st.global.wt.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_global_wt_v8_u32(
    ptr: cute.Pointer,
    v0: Int32,
    v1: Int32,
    v2: Int32,
    v3: Int32,
    v4: Int32,
    v5: Int32,
    v6: Int32,
    v7: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store 8 uint32 values using write-through policy.

    Args:
        ptr: Pointer to write location (must be 32-byte aligned)
        v0-v7: Values to store
    """
    ptr_int = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            ptr_int,
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
            v4.ir_value(loc=loc, ip=ip),
            v5.ir_value(loc=loc, ip=ip),
            v6.ir_value(loc=loc, ip=ip),
            v7.ir_value(loc=loc, ip=ip),
        ],
        "st.global.wt.v8.u32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# High-level Device API
# =============================================================================


@cute.jit
def start() -> Int32:
    """Capture the current global timer value as a start timestamp."""
    if const_expr(is_tracing_disabled()):
        return Int32(0)
    return read_globaltimer_lo()


@cute.jit
def start_zero() -> Int32:
    """Return a zero-initialized start token."""
    return Int32(0)


@cute.jit
def begin_lane(
    handle: TraceTensorHandle,
    block_id: Int32,
    lane_index: Int32,
    enabled: Boolean = True,
) -> LaneContext:
    """Initialize a lane context for tracing."""
    base_offset_bytes = (
        block_id * handle.num_lanes * handle.row_stride_bytes
        + lane_index * handle.row_stride_bytes
    )
    write_offset_bytes = base_offset_bytes + handle.max_event_width_bytes

    return LaneContext(
        base_offset_bytes=int(base_offset_bytes),
        write_offset_bytes=int(write_offset_bytes),
        max_event_width=handle.max_event_width,
        enabled=bool(enabled) if not handle.disabled else False,
    )


@cute.jit
def begin_lane_dynamic(
    handle: TraceTensorHandle,
    block_id: Int32,
    lane_index: Int32,
    enabled: Boolean = True,
) -> DynamicLaneContext:
    """Initialize a dynamic lane context for tracing."""
    base_offset_bytes = (
        block_id * handle.num_lanes * handle.row_stride_bytes
        + lane_index * handle.row_stride_bytes
    )
    write_offset_bytes = base_offset_bytes + DynamicLaneContext.EVENT_WIDTH_BYTES

    return DynamicLaneContext(
        base_offset_bytes=base_offset_bytes,
        write_offset_bytes=write_offset_bytes,
        enabled=enabled if not handle.disabled else False,
    )


@cute.jit
def begin_lane_dynamic_raw(
    num_lanes: Int32,
    row_stride_bytes: Int32,
    block_id: Int32,
    lane_index: Int32,
    enabled: Boolean = True,
) -> DynamicLaneContext:
    """Raw version of begin_lane_dynamic that takes fields directly."""
    base_offset_bytes = (
        block_id * num_lanes * row_stride_bytes + lane_index * row_stride_bytes
    )
    write_offset_bytes = base_offset_bytes + DynamicLaneContext.EVENT_WIDTH_BYTES

    actual_enabled = enabled
    if const_expr(is_tracing_disabled()):
        actual_enabled = False

    return DynamicLaneContext(
        base_offset_bytes=base_offset_bytes,
        write_offset_bytes=write_offset_bytes,
        enabled=actual_enabled,
    )


# -------------------------------------------------------------------------
# end_event variants
# -------------------------------------------------------------------------


@cute.jit
def end_event_0(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v2_u32(ptr, start_time, end_time)
        lane.advance()
    return lane


@cute.jit
def end_event_1(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v4_u32(ptr, start_time, end_time, p0, Int32(0))
        lane.advance()
    return lane


@cute.jit
def end_event_2(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
    p1: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v4_u32(ptr, start_time, end_time, p0, p1)
        lane.advance()
    return lane


@cute.jit
def end_event_3(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
    p1: Int32,
    p2: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, p0, p1, p2, Int32(0), Int32(0), Int32(0)
            )
        lane.advance()
    return lane


@cute.jit
def end_event_4(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, p0, p1, p2, p3, Int32(0), Int32(0)
            )
        lane.advance()
    return lane


@cute.jit
def end_event_5(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    p4: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, p0, p1, p2, p3, p4, Int32(0)
            )
        lane.advance()
    return lane


@cute.jit
def end_event_6(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    p4: Int32,
    p5: Int32,
) -> LaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(ptr, start_time, end_time, p0, p1, p2, p3, p4, p5)
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_raw_0(
    start_time: Int32,
    buffer: cute.Tensor,
    row_stride_bytes: Int32,
    lane: DynamicLaneContext,
    format_id: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = buffer.iterator + lane.write_offset_bytes
            store_global_wt_v4_u32(ptr, start_time, end_time, format_id, Int32(0))
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_0(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v4_u32(ptr, start_time, end_time, format_id, Int32(0))
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_1(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    p0: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v4_u32(ptr, start_time, end_time, format_id, p0)
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_2(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    p0: Int32,
    p1: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr,
                start_time,
                end_time,
                format_id,
                p0,
                p1,
                Int32(0),
                Int32(0),
                Int32(0),
            )
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_3(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    p0: Int32,
    p1: Int32,
    p2: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, format_id, p0, p1, p2, Int32(0), Int32(0)
            )
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_4(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, format_id, p0, p1, p2, p3, Int32(0)
            )
        lane.advance()
    return lane


@cute.jit
def end_event_dynamic_5(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    p0: Int32,
    p1: Int32,
    p2: Int32,
    p3: Int32,
    p4: Int32,
) -> DynamicLaneContext:
    if const_expr(is_tracing_disabled()):
        return lane
    if lane.enabled:
        end_time = read_globaltimer_lo()
        max_offset = lane.base_offset_bytes + handle.row_stride_bytes
        if lane.write_offset_bytes < max_offset:
            ptr = handle.buffer.iterator + lane.write_offset_bytes
            store_global_wt_v8_u32(
                ptr, start_time, end_time, format_id, p0, p1, p2, p3, p4
            )
        lane.advance()
    return lane


def end_event(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: LaneContext,
    format_id: int,
    *params: Int32,
) -> LaneContext:
    n = len(params)
    if n == 0:
        return end_event_0(start_time, handle, lane, format_id)
    if n == 1:
        return end_event_1(start_time, handle, lane, format_id, params[0])
    if n == 2:
        return end_event_2(start_time, handle, lane, format_id, params[0], params[1])
    if n == 3:
        return end_event_3(
            start_time, handle, lane, format_id, params[0], params[1], params[2]
        )
    if n == 4:
        return end_event_4(
            start_time,
            handle,
            lane,
            format_id,
            params[0],
            params[1],
            params[2],
            params[3],
        )
    if n == 5:
        return end_event_5(
            start_time,
            handle,
            lane,
            format_id,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
    if n == 6:
        return end_event_6(
            start_time,
            handle,
            lane,
            format_id,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
        )
    return lane


def end_event_dynamic(
    start_time: Int32,
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
    format_id: Int32,
    *params: Int32,
) -> DynamicLaneContext:
    n = len(params)
    if n == 0:
        return end_event_dynamic_0(start_time, handle, lane, format_id)
    if n == 1:
        return end_event_dynamic_1(start_time, handle, lane, format_id, params[0])
    if n == 2:
        return end_event_dynamic_2(
            start_time, handle, lane, format_id, params[0], params[1]
        )
    if n == 3:
        return end_event_dynamic_3(
            start_time, handle, lane, format_id, params[0], params[1], params[2]
        )
    if n == 4:
        return end_event_dynamic_4(
            start_time,
            handle,
            lane,
            format_id,
            params[0],
            params[1],
            params[2],
            params[3],
        )
    if n == 5:
        return end_event_dynamic_5(
            start_time,
            handle,
            lane,
            format_id,
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
        )
    return lane


@cute.jit
def finish_lane(
    handle: TraceTensorHandle,
    lane: LaneContext,
) -> None:
    if const_expr(is_tracing_disabled()):
        return
    if lane.enabled:
        sm_id = read_smid()
        ptr = handle.buffer.iterator + lane.base_offset_bytes
        store_global_wt_v2_u32(ptr, sm_id, Int32(lane.write_offset_bytes))


@cute.jit
def finish_lane_dynamic_raw(buffer: cute.Tensor, lane: DynamicLaneContext) -> None:
    if const_expr(is_tracing_disabled()):
        return
    if lane.enabled:
        sm_id = read_smid()
        ptr = buffer.iterator + lane.base_offset_bytes
        store_global_wt_v2_u32(ptr, sm_id, Int32(lane.write_offset_bytes))


@cute.jit
def finish_lane_dynamic(
    handle: TraceTensorHandle,
    lane: DynamicLaneContext,
) -> None:
    if const_expr(is_tracing_disabled()):
        return
    if lane.enabled:
        sm_id = read_smid()
        ptr = handle.buffer.iterator + lane.base_offset_bytes
        store_global_wt_v2_u32(ptr, sm_id, Int32(lane.write_offset_bytes))
