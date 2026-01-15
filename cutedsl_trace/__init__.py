"""
cutedsl-trace: Nanotrace-compatible GPU kernel tracing for CuteDSL.

This library provides nanosecond-resolution GPU kernel tracing for CuteDSL kernels,
producing .nanotrace files compatible with the nanotrace visualizer.
"""

from .types import (
    TraceType,
    BlockType,
    TrackType,
    TrackLevel,
    LaneType,
    FormatDescriptor,
    TraceTensorHandle,
    LaneContext,
    reset_all_counters,
)
from .device import (
    start,
    start_zero,
    begin_lane,
    end_event,
    finish_lane,
    # Inline assembly utilities
    read_globaltimer_lo,
    read_smid,
)
from .host import (
    StaticTraceBuilder,
    DynamicTraceBuilder,
    TraceWriter,
    create_hierarchical_tracks,
)
from .config import is_tracing_disabled, set_tracing_enabled

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "is_tracing_disabled",
    "set_tracing_enabled",
    # Types
    "TraceType",
    "BlockType",
    "TrackType",
    "TrackLevel",
    "LaneType",
    "FormatDescriptor",
    "TraceTensorHandle",
    "LaneContext",
    "reset_all_counters",
    # Device API
    "start",
    "start_zero",
    "begin_lane",
    "end_event",
    "finish_lane",
    "read_globaltimer_lo",
    "read_smid",
    # Host API
    "StaticTraceBuilder",
    "DynamicTraceBuilder",
    "TraceWriter",
    "create_hierarchical_tracks",
]
