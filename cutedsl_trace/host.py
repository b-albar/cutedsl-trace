"""
Host-side API for cutedsl-trace.

This module provides the host-side trace tensor builders and file writer.
These are used to set up tracing before kernel launch and to write
the resulting .nanotrace file after execution.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, BinaryIO

import numpy as np
import torch

from .types import (
    TraceType,
    BlockType,
    TrackType,
    TrackLevel,
    FormatDescriptor,
    TraceTensorHandle,
    LaneType,
)
from .config import is_tracing_disabled


# =============================================================================
# Trace Builders
# =============================================================================


@dataclass
class StaticTraceBuilder:
    """Build a static trace tensor for GPU tracing.

    Static tensors have a fixed trace type per lane, which means the format_id
    doesn't need to be written with each event (saving space and bandwidth).

    The event width is determined at construction time based on the maximum
    parameter count across all trace types assigned to lanes.

    Attributes:
        num_lanes: Number of lanes per block
        trace_types: List of TraceType, one per lane
        max_events_per_lane: Maximum events per lane (excluding header)
        grid_dims: Grid dimensions (x, y, z)
        cluster_dims: Cluster dimensions (0, 0, 0) if not using clusters
    """

    num_lanes: int
    trace_types: list[TraceType]
    max_events_per_lane: int
    grid_dims: tuple[int, int, int]
    cluster_dims: tuple[int, int, int] = (0, 0, 0)

    # Internal state
    _buffer: Any = field(init=False, default=None)  # GPU array
    _host_buffer: np.ndarray | None = field(init=False, default=None)
    _track_types: list[TrackType | None] = field(init=False, default_factory=list)
    max_event_width: int = field(init=False, default=2)
    row_stride_bytes: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if len(self.trace_types) != self.num_lanes:
            raise ValueError(
                f"Expected {self.num_lanes} trace types, got {len(self.trace_types)}"
            )

        # Compute max event width from trace types
        self.max_event_width = max(t.event_width for t in self.trace_types)

        # Verify all trace types are STATIC
        for i, tt in enumerate(self.trace_types):
            if tt.lane_type != LaneType.STATIC:
                raise ValueError(
                    f"Lane {i}: trace type '{tt.name}' must be STATIC for StaticTraceBuilder"
                )

        # Compute row stride: header (1 slot) + max_events slots
        # Each slot is max_event_width * 4 bytes
        self.row_stride_bytes = (
            (self.max_events_per_lane + 1) * self.max_event_width * 4
        )

        # Validate buffer size won't overflow uint32
        total_blocks = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
        total_bytes = total_blocks * self.num_lanes * self.row_stride_bytes

        if total_bytes > 0xFFFFFFFF:
            raise ValueError(
                f"Trace buffer size ({total_bytes} bytes) exceeds uint32 max. "
                f"Reduce grid size, lanes, or events per lane."
            )

        # Initialize track types to None
        self._track_types = [None] * self.num_lanes

        # Allocate GPU buffer
        if is_tracing_disabled():
            # Minimal allocation when disabled
            self._buffer = np.zeros(1, dtype=np.uint8)
        else:
            self._allocate_buffer(total_bytes)

    def _allocate_buffer(self, total_bytes: int) -> None:
        """Allocate GPU buffer for trace data."""
        try:
            import torch

            self._buffer = torch.zeros(total_bytes, dtype=torch.uint8, device="cuda")
        except (ImportError, RuntimeError):
            # Fallback to numpy (for testing without GPU)
            self._buffer = np.zeros(total_bytes, dtype=np.uint8)

    def get_handle(self) -> TraceTensorHandle:
        """Get handle to pass to kernel.

        Returns:
            TraceTensorHandle: Handle containing buffer pointer and metadata
        """
        return TraceTensorHandle(
            buffer=self._buffer,
            row_stride_bytes=self.row_stride_bytes,
            num_lanes=self.num_lanes,
            max_event_width=self.max_event_width,
            disabled=is_tracing_disabled(),
        )

    def set_track_type(self, track_type: TrackType, lane: int | None = None) -> None:
        """Set track type for lanes.

        Args:
            track_type: The track type to use
            lane: Specific lane index, or None for all lanes
        """
        if lane is None:
            self._track_types = [track_type] * self.num_lanes
        else:
            if lane < 0 or lane >= self.num_lanes:
                raise ValueError(f"Lane {lane} out of range [0, {self.num_lanes})")
            self._track_types[lane] = track_type

    def reset(self) -> None:
        """Reset the trace buffer to zeros.

        Call this after warmup runs and before the traced run to ensure
        clean trace data without cold-start artifacts.
        """
        if hasattr(self._buffer, "fill_"):
            self._buffer.fill_(0)
        else:
            self._buffer[:] = 0

    def copy_to_host(self) -> np.ndarray:
        """Copy data back to host."""
        if hasattr(self._buffer, "cpu"):
            self._host_buffer = self._buffer.cpu().detach().numpy()
        else:
            self._host_buffer = np.asarray(self._buffer).copy()

        return self._host_buffer


@dataclass
class DynamicTraceBuilder:
    """Build a dynamic trace tensor for GPU tracing.

    Dynamic tensors allow different event types within the same lane.
    The format_id is written with each event, so event_width is always 8.

    Attributes:
        num_lanes: Number of lanes per block
        max_events_per_lane: Maximum events per lane (excluding header)
        grid_dims: Grid dimensions (x, y, z)
        cluster_dims: Cluster dimensions (0, 0, 0) if not using clusters
    """

    num_lanes: int
    max_events_per_lane: int
    grid_dims: tuple[int, int, int]
    cluster_dims: tuple[int, int, int] = (0, 0, 0)

    # Internal state
    _buffer: Any = field(init=False, default=None)
    _host_buffer: np.ndarray | None = field(init=False, default=None)
    _track_types: list[TrackType | None] = field(init=False, default_factory=list)

    # Dynamic lanes always use event_width=8
    EVENT_WIDTH: int = 8

    def __post_init__(self) -> None:
        # Compute row stride
        self.row_stride_bytes = (self.max_events_per_lane + 1) * self.EVENT_WIDTH * 4

        # Validate buffer size
        total_blocks = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
        total_bytes = total_blocks * self.num_lanes * self.row_stride_bytes

        if total_bytes > 0xFFFFFFFF:
            raise ValueError("Trace buffer size exceeds uint32 max")

        # Initialize track types
        self._track_types = [None] * self.num_lanes

        # Allocate buffer
        if is_tracing_disabled():
            self._buffer = np.zeros(1, dtype=np.uint8)
        else:
            self._allocate_buffer(total_bytes)

    def _allocate_buffer(self, total_bytes: int) -> None:
        """Allocate GPU buffer."""
        self._buffer = torch.zeros(total_bytes, dtype=torch.uint8, device="cuda")

    def get_handle(self) -> TraceTensorHandle:
        """Get handle to pass to kernel."""
        return TraceTensorHandle(
            buffer=self._buffer,
            row_stride_bytes=self.row_stride_bytes,
            num_lanes=self.num_lanes,
            max_event_width=self.EVENT_WIDTH,
            disabled=is_tracing_disabled(),
        )

    def set_track_type(self, track_type: TrackType, lane: int | None = None) -> None:
        """Set track type for lanes."""
        if lane is None:
            self._track_types = [track_type] * self.num_lanes
        else:
            self._track_types[lane] = track_type

    def reset(self) -> None:
        """Reset the trace buffer to zeros."""
        if hasattr(self._buffer, "fill_"):
            self._buffer.fill_(0)
        else:
            self._buffer[:] = 0

    def copy_to_host(self) -> np.ndarray:
        """Copy GPU buffer to host memory."""
        if hasattr(self._buffer, "cpu"):
            self._host_buffer = self._buffer.cpu().detach().numpy()
        else:
            self._host_buffer = np.asarray(self._buffer).copy()

        return self._host_buffer


def create_hierarchical_tracks(
    warps_per_block: int,
    threads_per_warp: int = 32,
    sample_threads: int | None = None,
) -> tuple[list[TrackType], list[TrackType]]:
    """Create hierarchical warp and thread track types.

    This helper creates track types for hierarchical visualization where
    warps can be expanded to show individual thread traces.

    Args:
        warps_per_block: Number of warps per block
        threads_per_warp: Threads per warp (default 32)
        sample_threads: If set, only create thread tracks for every Nth thread
                        to reduce overhead. None means all threads.

    Returns:
        Tuple of (warp_tracks, thread_tracks) where:
        - warp_tracks: List of TrackType for warps (len = warps_per_block)
        - thread_tracks: List of TrackType for threads under each warp
                         (len depends on sample_threads)

    Example:
        warp_tracks, thread_tracks = create_hierarchical_tracks(8, sample_threads=8)
        # Creates 8 warp tracks, each with 4 sampled thread tracks (32/8)
    """
    warp_tracks = []
    thread_tracks = []

    for warp_id in range(warps_per_block):
        warp_track = TrackType(
            name=f"Warp{warp_id}",
            label_string=f"Warp {warp_id}",
            tooltip_string=f"Warp {warp_id}",
            level=TrackLevel.WARP,
            parent_lane=None,  # Warps are top-level under blocks
        )
        warp_tracks.append(warp_track)

        # Create thread tracks under this warp
        thread_step = sample_threads if sample_threads else 1
        for thread_offset in range(0, threads_per_warp, thread_step):
            thread_track = TrackType(
                name=f"Thread{warp_id}_{thread_offset}",
                label_string=f"Thread {thread_offset}",
                tooltip_string=f"Warp {warp_id}, Thread {thread_offset}",
                level=TrackLevel.THREAD,
                parent_lane=warp_id,  # Parent is the warp's lane index
            )
            thread_tracks.append(thread_track)

    return warp_tracks, thread_tracks


# =============================================================================
# Event Parsing
# =============================================================================


@dataclass
class ParsedEvent:
    """A parsed trace event."""

    time_offset_ns: int  # Nanoseconds since trace start
    duration_ns: int
    format_id: int
    params: list[int]


@dataclass
class ParsedTrack:
    """A parsed event track (lane within a block)."""

    block_id: int
    lane_id: int
    sm_id: int
    track_type_id: int
    level: int  # TrackLevel value
    parent_lane: int | None
    events: list[ParsedEvent]


@dataclass
class ParsedBlock:
    """A parsed block descriptor."""

    block_id: int
    cluster_id: int
    sm_id: int
    format_id: int


def parse_events_from_buffer(
    buffer: np.ndarray,
    builder: StaticTraceBuilder | DynamicTraceBuilder,
    lane_offset: int = 0,
) -> tuple[list[ParsedBlock], list[ParsedTrack], int]:
    """Parse events from a trace buffer.

    Args:
        buffer: Host-side trace buffer
        builder: The builder that created the buffer
        lane_offset: Offset to add to lane IDs (for multi-tensor stacking)

    Returns:
        Tuple of (block_descriptors, tracks, min_time)
    """
    blocks: list[ParsedBlock] = []
    tracks: list[ParsedTrack] = []
    min_time = 0xFFFFFFFF
    max_time = 0

    total_blocks = builder.grid_dims[0] * builder.grid_dims[1] * builder.grid_dims[2]
    num_lanes = builder.num_lanes
    row_stride_bytes = builder.row_stride_bytes

    if isinstance(builder, StaticTraceBuilder):
        max_event_width = builder.max_event_width
    else:
        max_event_width = DynamicTraceBuilder.EVENT_WIDTH

    event_width_bytes = max_event_width * 4

    # Parse each block
    for block_id in range(total_blocks):
        block_base = block_id * num_lanes * row_stride_bytes
        block_sm_id = None

        # Parse each lane in the block
        for lane_id in range(num_lanes):
            lane_base = block_base + lane_id * row_stride_bytes

            # Read header: [sm_id, write_offset_bytes, ...]
            header_data = buffer[lane_base : lane_base + 8]
            sm_id, write_offset_bytes = struct.unpack("<II", header_data)

            # Skip empty lanes
            if write_offset_bytes == 0:
                continue

            if block_sm_id is None:
                block_sm_id = sm_id

            # Compute event count from write_offset_bytes
            events_start = lane_base + event_width_bytes
            bytes_written = write_offset_bytes - (lane_base + event_width_bytes)

            if bytes_written <= 0:
                continue

            event_count = bytes_written // event_width_bytes

            # Parse events
            events: list[ParsedEvent] = []

            for i in range(event_count):
                event_offset = events_start + i * event_width_bytes
                event_data = buffer[event_offset : event_offset + event_width_bytes]

                if max_event_width == 2:
                    start_time, end_time = struct.unpack("<II", event_data)
                    params = []
                    format_id = (
                        builder.trace_types[lane_id].id
                        if isinstance(builder, StaticTraceBuilder)
                        else 0
                    )
                elif max_event_width == 4:
                    start_time, end_time, p0, p1 = struct.unpack("<IIII", event_data)
                    params = [p0, p1]
                    format_id = (
                        builder.trace_types[lane_id].id
                        if isinstance(builder, StaticTraceBuilder)
                        else 0
                    )
                else:  # max_event_width == 8
                    values = struct.unpack("<IIIIIIII", event_data)
                    start_time, end_time = values[0], values[1]
                    if isinstance(builder, DynamicTraceBuilder):
                        format_id = values[2]
                        params = list(values[3:])
                    else:
                        format_id = builder.trace_types[lane_id].id
                        params = list(values[2:])

                # Track min/max times
                min_time = min(min_time, start_time)
                max_time = max(max_time, end_time)

                # Compute duration (clamp to minimum 32ns for timer resolution)
                duration = max(end_time - start_time, 32)

                events.append(
                    ParsedEvent(
                        time_offset_ns=start_time,  # Will be normalized later
                        duration_ns=duration,
                        format_id=format_id,
                        params=params[: builder.trace_types[lane_id].param_count]
                        if isinstance(builder, StaticTraceBuilder)
                        else params,
                    )
                )

            # Get track type
            track_type = builder._track_types[lane_id]
            track_type_id = track_type.id if track_type else 0
            level = track_type.level.value if track_type else TrackLevel.WARP.value
            parent_lane_val = track_type.parent_lane if track_type else None

            # Adjust parent_lane if it's relative to the start of this tensor
            if parent_lane_val is not None:
                parent_lane_val += lane_offset

            tracks.append(
                ParsedTrack(
                    block_id=block_id,
                    lane_id=lane_id + lane_offset,
                    sm_id=sm_id,
                    track_type_id=track_type_id,
                    level=level,
                    parent_lane=parent_lane_val,
                    events=events,
                )
            )

        # Create block descriptor
        if block_sm_id is not None:
            blocks.append(
                ParsedBlock(
                    block_id=block_id,
                    cluster_id=0,  # TODO: compute from cluster dims
                    sm_id=block_sm_id,
                    format_id=0,  # Set by writer
                )
            )

    return blocks, tracks, min_time


# =============================================================================
# Trace Writer
# =============================================================================


class TraceWriter:
    """Write trace data to a .nanotrace file.

    This class collects trace tensors and writes them to the nanotrace
    binary format, which can be visualized with the nanotrace WebGPU visualizer.

    Usage:
        writer = TraceWriter("my_kernel")
        writer.set_block_type(my_block_type)
        writer.add_tensor(trace_tensor)
        writer.write("output.nanotrace")
    """

    MAGIC = b"nanotrace\x00"
    FORMAT_VERSION = 1

    def __init__(self, kernel_name: str) -> None:
        """Create trace writer.

        Args:
            kernel_name: Name of the kernel being traced
        """
        self.kernel_name = kernel_name
        self.block_type: BlockType | None = None
        self.format_descriptors: dict[int, FormatDescriptor] = {}  # id -> descriptor
        self.tensors: list[StaticTraceBuilder | DynamicTraceBuilder] = []
        self._grid_dims: tuple[int, int, int] | None = None
        self._cluster_dims: tuple[int, int, int] = (0, 0, 0)

    def set_block_type(self, block_type: BlockType) -> None:
        """Set the block format descriptor.

        Must be called before add_tensor().

        Args:
            block_type: Block type for block labels
        """
        self.block_type = block_type
        self._register_descriptor(block_type.descriptor)

    def register_trace_type(self, trace_type: TraceType) -> None:
        """Register a trace type for dynamic tensors.

        Static tensor trace types are auto-registered by add_tensor().
        Dynamic tensor trace types must be registered manually.

        Args:
            trace_type: Trace type to register
        """
        self._register_descriptor(trace_type.descriptor)

    def _register_descriptor(self, desc: FormatDescriptor) -> None:
        """Register a format descriptor."""
        if desc.id in self.format_descriptors:
            existing = self.format_descriptors[desc.id]
            if existing != desc:
                raise ValueError(
                    f"Format descriptor ID {desc.id} already registered with "
                    f"different definition"
                )
        else:
            self.format_descriptors[desc.id] = desc

    def add_tensor(self, builder: StaticTraceBuilder | DynamicTraceBuilder) -> None:
        """Add a trace tensor to the writer.

        All tensors must have the same grid dimensions.
        Static tensor trace types are auto-registered.

        Args:
            builder: Trace tensor builder
        """
        # Check grid dimensions match
        if self._grid_dims is None:
            self._grid_dims = builder.grid_dims
            self._cluster_dims = builder.cluster_dims
        elif self._grid_dims != builder.grid_dims:
            raise ValueError(
                f"Grid dimensions mismatch: expected {self._grid_dims}, "
                f"got {builder.grid_dims}"
            )

        # Auto-register trace types from static tensor
        if isinstance(builder, StaticTraceBuilder):
            for tt in builder.trace_types:
                self._register_descriptor(tt.descriptor)

        # Auto-register track types
        for track_type in builder._track_types:
            if track_type is not None:
                self._register_descriptor(track_type.descriptor)

        self.tensors.append(builder)

    def _collect_and_parse(
        self,
    ) -> tuple[list[ParsedBlock], list[ParsedTrack]]:
        """Copy device buffers to host, parse events, and normalize times.

        Returns:
            Tuple of (blocks, tracks) with normalized event times.

        Raises:
            ValueError: If no tensors added or block type not set.
        """
        if not self.tensors:
            raise ValueError("No tensors added to writer")

        if self.block_type is None:
            raise ValueError("Block type not set. Call set_block_type() first.")

        all_blocks: list[ParsedBlock] = []
        all_tracks: list[ParsedTrack] = []
        lane_offset = 0
        min_time = 0xFFFFFFFF

        for tensor in self.tensors:
            buffer = tensor.copy_to_host()
            blocks, tracks, tensor_min_time = parse_events_from_buffer(
                buffer, tensor, lane_offset
            )
            all_blocks.extend(blocks)
            all_tracks.extend(tracks)
            min_time = min(min_time, tensor_min_time)
            lane_offset += tensor.num_lanes

        # Normalize event times to start at 0 and truncate parameters
        for track in all_tracks:
            for event in track.events:
                event.time_offset_ns -= min_time

                # Truncate parameters to match the format descriptor's count
                fmt_desc = self.format_descriptors.get(event.format_id)
                if fmt_desc is not None:
                    event.params = event.params[: fmt_desc.param_count]

        # Set block format ID
        for block in all_blocks:
            block.format_id = self.block_type.id

        return all_blocks, all_tracks

    def write(self, filename: str, compress: bool = True) -> None:
        """Write the trace file.

        Args:
            filename: Output filename (should end with .nanotrace)
            compress: Whether to use deflate compression (default True)
        """
        if is_tracing_disabled():
            print(
                f"Tracing is disabled (CUTEDSL_TRACE_DISABLED). Skipping write to {filename}"
            )
            return

        all_blocks, all_tracks = self._collect_and_parse()

        # Build format descriptor mapping (original ID -> file index)
        sorted_descriptors = sorted(
            self.format_descriptors.values(), key=lambda d: d.id
        )
        id_to_file_index = {d.id: i for i, d in enumerate(sorted_descriptors)}

        # Write the file
        with open(filename, "wb") as f:
            self._write_header(f, compress)

            if compress:
                # Write compressed payload
                import io

                payload = io.BytesIO()
                self._write_payload(
                    payload,
                    sorted_descriptors,
                    all_blocks,
                    all_tracks,
                    id_to_file_index,
                )
                compressed = zlib.compress(payload.getvalue(), level=6)
                f.write(compressed)
            else:
                self._write_payload(
                    f, sorted_descriptors, all_blocks, all_tracks, id_to_file_index
                )

    def write_perfetto(self, filename: str) -> None:
        """Write trace data as Chrome JSON Trace Event format for Perfetto.

        Exports the trace as a JSON file compatible with Perfetto UI
        (https://ui.perfetto.dev) and Chrome's chrome://tracing.

        Mapping:
            - Kernel → process (pid=0)
            - Each (block, lane) pair → thread (unique tid)
            - Events → complete events (ph="X")

        Args:
            filename: Output filename (typically .json)
        """
        if is_tracing_disabled():
            print(
                f"Tracing is disabled (CUTEDSL_TRACE_DISABLED). Skipping write to {filename}"
            )
            return

        all_blocks, all_tracks = self._collect_and_parse()

        # Build block_id -> sm_id lookup
        block_sm_ids: dict[int, int] = {}
        for block in all_blocks:
            block_sm_ids[block.block_id] = block.sm_id

        # Build format_id -> label lookup
        format_labels: dict[int, str] = {}
        for desc in self.format_descriptors.values():
            format_labels[desc.id] = desc.label_string

        trace_events: list[dict] = []

        # Process metadata event
        trace_events.append(
            {
                "ph": "M",
                "pid": 0,
                "tid": 0,
                "name": "process_name",
                "args": {"name": self.kernel_name},
            }
        )

        # Assign unique tid per (block_id, lane_id) and emit events
        tid_counter = 0
        for track in all_tracks:
            tid = tid_counter
            tid_counter += 1

            # Resolve track label
            track_fmt = self.format_descriptors.get(track.track_type_id)
            if track_fmt:
                track_label = track_fmt.label_string.replace(
                    "{lane}", str(track.lane_id)
                )
            else:
                track_label = f"Lane {track.lane_id}"

            thread_name = f"Block {track.block_id} / {track_label}"

            # Thread metadata
            trace_events.append(
                {
                    "ph": "M",
                    "pid": 0,
                    "tid": tid,
                    "name": "thread_name",
                    "args": {"name": thread_name},
                }
            )

            sm_id = block_sm_ids.get(track.block_id, 0)

            for event in track.events:
                # Resolve event name from format descriptor
                event_label = format_labels.get(event.format_id, f"Event {event.format_id}")

                event_args: dict[str, Any] = {
                    "block_id": track.block_id,
                    "lane_id": track.lane_id,
                    "sm_id": sm_id,
                }
                if event.params:
                    for i, p in enumerate(event.params):
                        event_args[f"param{i}"] = p

                trace_events.append(
                    {
                        "name": event_label,
                        "ph": "X",
                        "ts": event.time_offset_ns / 1000.0,
                        "dur": event.duration_ns / 1000.0,
                        "pid": 0,
                        "tid": tid,
                        "cat": "gpu",
                        "args": event_args,
                    }
                )

        with open(filename, "w") as f:
            json.dump(trace_events, f)

    def _write_header(self, f: BinaryIO, compress: bool) -> None:
        """Write file header (uncompressed)."""
        f.write(self.MAGIC)
        f.write(struct.pack("<B", self.FORMAT_VERSION))
        f.write(struct.pack("<B", 1 if compress else 0))

    def _write_payload(
        self,
        f: BinaryIO,
        descriptors: list[FormatDescriptor],
        blocks: list[ParsedBlock],
        tracks: list[ParsedTrack],
        id_to_file_index: dict[int, int],
    ) -> None:
        """Write trace payload."""
        # Write kernel name
        name_bytes = self.kernel_name.encode("utf-8")
        f.write(struct.pack("<H", len(name_bytes)))
        f.write(name_bytes)

        # Write grid dimensions
        f.write(struct.pack("<III", *self._grid_dims))

        # Write cluster dimensions
        f.write(struct.pack("<III", *self._cluster_dims))

        # Write counts
        total_events = sum(len(t.events) for t in tracks)
        f.write(struct.pack("<I", len(descriptors)))  # Format descriptor count
        f.write(struct.pack("<I", len(blocks)))  # Block descriptor count
        f.write(struct.pack("<I", len(tracks)))  # Track count
        f.write(struct.pack("<Q", total_events))  # Total event count

        # Write format descriptors
        for desc in descriptors:
            label_bytes = desc.label_string.encode("utf-8")
            tooltip_bytes = desc.tooltip_string.encode("utf-8")
            f.write(struct.pack("<H", len(label_bytes)))
            f.write(label_bytes)
            f.write(struct.pack("<H", len(tooltip_bytes)))
            f.write(tooltip_bytes)
            f.write(struct.pack("<B", desc.param_count))

        # Write block descriptors
        for block in blocks:
            f.write(struct.pack("<I", block.block_id))
            f.write(struct.pack("<I", block.cluster_id))
            f.write(struct.pack("<H", block.sm_id))
            f.write(struct.pack("<H", id_to_file_index[block.format_id]))

        # Write tracks
        for track in tracks:
            # Find block descriptor index
            block_desc_idx = next(
                i for i, b in enumerate(blocks) if b.block_id == track.block_id
            )
            f.write(struct.pack("<I", block_desc_idx))
            f.write(struct.pack("<H", id_to_file_index[track.track_type_id]))
            f.write(struct.pack("<I", track.lane_id))

            # Write track format parameters (count = placeholder count of track's format descriptor)
            track_fmt_desc = self.format_descriptors.get(track.track_type_id)
            if track_fmt_desc:
                # We don't currently support track-level params in the ParsedTrack object,
                # so we just write zeros if needed by the format.
                for _ in range(track_fmt_desc.param_count):
                    f.write(struct.pack("<I", 0))

            # Write event count
            f.write(struct.pack("<I", len(track.events)))

            # Write events
            for event in track.events:
                f.write(struct.pack("<I", event.time_offset_ns))
                f.write(struct.pack("<I", event.duration_ns))
                f.write(struct.pack("<H", id_to_file_index[event.format_id]))

                # Write parameters
                for param in event.params:
                    f.write(struct.pack("<I", param))
