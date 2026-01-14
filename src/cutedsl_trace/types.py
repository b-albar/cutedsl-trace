"""
Core type definitions for cutedsl-trace.

This module defines the data structures used for trace types, lane contexts,
and tensor handles.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Any


# Global counter for all descriptor IDs to ensure uniqueness across types
_global_descriptor_counter: int = 0


def _next_descriptor_id() -> int:
    """Get the next unique descriptor ID."""
    global _global_descriptor_counter
    result = _global_descriptor_counter
    _global_descriptor_counter += 1
    return result


def reset_all_counters() -> None:
    """Reset the global descriptor counter. Useful for testing."""
    global _global_descriptor_counter
    _global_descriptor_counter = 0


class LaneType(Enum):
    """Lane type enumeration."""

    STATIC = 0  # Fixed format per lane (format_id not written per event)
    DYNAMIC = 1  # Format_id written per event


class TrackLevel(Enum):
    """Track hierarchy level.

    Defines the granularity level of a track for hierarchical visualization.
    This enables the visualizer to group and expand tracks appropriately.
    """

    BLOCK = 0  # Block/CTA level (top level)
    SM = 1  # Streaming Multiprocessor level
    WARP = 2  # Warp level (default)
    THREAD = 3  # Individual thread level (most detailed)


@dataclass
class FormatDescriptor:
    """Format descriptor for trace events, blocks, or tracks."""

    label_string: str
    tooltip_string: str
    id: int
    param_count: int = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FormatDescriptor):
            return NotImplemented
        return (
            self.id == other.id
            and self.param_count == other.param_count
            and self.label_string == other.label_string
            and self.tooltip_string == other.tooltip_string
        )


@dataclass
class TraceType:
    """Define a trace event type.

    Trace types define the format of events recorded during kernel execution.
    Each trace type has a unique ID, format strings for display, and a parameter count.

    Attributes:
        name: Human-readable name for the trace type
        label_string: Short format string for labels (e.g., "Load {0}")
        tooltip_string: Full format string for tooltips (e.g., "Load from address {0}")
        param_count: Number of parameters (0-6)
        lane_type: Whether this is for STATIC or DYNAMIC lanes
    """

    name: str
    label_string: str
    tooltip_string: str
    param_count: int = 0
    lane_type: LaneType = LaneType.STATIC

    # Auto-assigned ID using global counter
    _id: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        if self._id == -1:
            self._id = _next_descriptor_id()

    @property
    def id(self) -> int:
        """Get the unique ID for this trace type."""
        return self._id

    @property
    def event_width(self) -> int:
        """Compute event width in uint32s based on param count.

        - 0 params → 2 (start, end)
        - 1-2 params → 4 (start, end, p0, p1 or padding)
        - 3-6 params → 8 (start, end, p0-p5 or padding)
        """
        if self.param_count == 0:
            return 2
        elif self.param_count <= 2:
            return 4
        else:
            return 8

    @property
    def descriptor(self) -> FormatDescriptor:
        """Get the format descriptor for this trace type."""
        return FormatDescriptor(
            label_string=self.label_string,
            tooltip_string=self.tooltip_string,
            id=self._id,
            param_count=self.param_count,
        )

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the global counter. Useful for testing."""
        reset_all_counters()


@dataclass
class BlockType:
    """Define a block descriptor type.

    Block types define the format of block descriptors in the trace file.
    Block format strings support special placeholders like {blockLinear}, {blockX}, etc.

    Attributes:
        name: Human-readable name for the block type
        label_string: Short format string for labels
        tooltip_string: Full format string for tooltips
    """

    name: str
    label_string: str
    tooltip_string: str

    _id: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        if self._id == -1:
            self._id = _next_descriptor_id()

    @property
    def id(self) -> int:
        """Get the unique ID for this block type."""
        return self._id

    @property
    def descriptor(self) -> FormatDescriptor:
        """Get the format descriptor for this block type."""
        return FormatDescriptor(
            label_string=self.label_string,
            tooltip_string=self.tooltip_string,
            id=self._id,
            param_count=0,  # Block types don't have numbered params
        )

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the global counter. Useful for testing."""
        reset_all_counters()


@dataclass
class TrackType:
    """Define a track/sublane type.

    Track types define the format of track (sublane) descriptors in the trace file.
    Track format strings support the special placeholder {lane}.

    Attributes:
        name: Human-readable name for the track type
        label_string: Short format string for labels
        tooltip_string: Full format string for tooltips
        param_count: Number of parameters (usually 0)
        level: Hierarchy level (BLOCK, SM, WARP, THREAD)
        parent_lane: Optional parent lane index for nested tracks
    """

    name: str
    label_string: str
    tooltip_string: str
    param_count: int = 0
    level: TrackLevel = TrackLevel.WARP
    parent_lane: int | None = None  # Index of parent lane, None for top-level

    _id: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        if self._id == -1:
            self._id = _next_descriptor_id()

    @property
    def id(self) -> int:
        """Get the unique ID for this track type."""
        return self._id

    @property
    def descriptor(self) -> FormatDescriptor:
        """Get the format descriptor for this track type."""
        return FormatDescriptor(
            label_string=self.label_string,
            tooltip_string=self.tooltip_string,
            id=self._id,
            param_count=self.param_count,
        )

    @classmethod
    def reset_counter(cls) -> None:
        """Reset the global counter. Useful for testing."""
        reset_all_counters()


@dataclass
class TraceTensorHandle:
    """Handle to a trace tensor, passed to kernels.

    This is the device-side handle that provides access to the trace buffer.
    It contains the buffer pointer and stride information needed for
    computing write offsets.

    Attributes:
        buffer: GPU buffer pointer (interpreted as uint8*)
        row_stride_bytes: Bytes per lane row
        num_lanes: Number of lanes per block
        max_event_width: Maximum event width in uint32s (2, 4, or 8)
    """

    buffer: Any  # GPU array/pointer
    row_stride_bytes: int
    num_lanes: int
    max_event_width: int
    disabled: bool = False

    @property
    def max_event_width_bytes(self) -> int:
        """Get the maximum event width in bytes."""
        return self.max_event_width * 4


@dataclass
class LaneContext:
    """Lane context for tracing.

    Maintains the state needed for tracing within a single lane.
    Created by begin_lane() and used by end_event() and finish_lane().

    Attributes:
        base_offset_bytes: Base offset of this lane in the buffer
        write_offset_bytes: Current write offset (advances with each event)
        max_event_width: Maximum event width in uint32s
        enabled: Whether tracing is enabled for this lane
    """

    base_offset_bytes: int
    write_offset_bytes: int
    max_event_width: int
    enabled: bool = True

    @property
    def max_event_width_bytes(self) -> int:
        """Get the maximum event width in bytes."""
        return self.max_event_width * 4

    def advance(self) -> None:
        """Advance the write offset by one event slot."""
        self.write_offset_bytes += self.max_event_width_bytes


# For dynamic tracing support
@dataclass
class DynamicLaneContext:
    """Lane context for dynamic (mixed-type) tracing.

    Similar to LaneContext but for dynamic lanes where different
    event types can be recorded in the same lane.

    Dynamic lanes always use event_width=8 to accommodate format_id.
    """

    base_offset_bytes: int
    write_offset_bytes: int
    enabled: bool = True

    EVENT_WIDTH: ClassVar[int] = 8  # Dynamic lanes always use width 8
    EVENT_WIDTH_BYTES: ClassVar[int] = 32  # 8 * 4 bytes

    def advance(self) -> None:
        """Advance the write offset by one event slot."""
        self.write_offset_bytes += self.EVENT_WIDTH_BYTES
