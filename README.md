# cutedsl-trace

Nanotrace-compatible GPU kernel tracing for CuteDSL.

## Overview

`cutedsl-trace` provides nanosecond-resolution GPU kernel tracing for CuteDSL kernels, producing `.nanotrace` files compatible with the [nanotrace visualizer](https://aikitoria.github.io/nanotrace).

This library reimplements the [nanotrace-cuda](https://github.com/aikitoria/nanotrace) functionality for Python/CuteDSL kernels.

## Features

- **Low overhead**: Uses inline PTX assembly with vectorized stores
- **32ns resolution**: Captures GPU global timer timestamps
- **Nanotrace compatible**: Outputs `.nanotrace` files for the WebGPU visualizer
- **CuteDSL native**: Integrates seamlessly with `@cute.jit` kernels

## Installation

```bash
pip install cutedsl-trace
```

Or install from source:

```bash
git clone https://github.com/b-albar/cutedsl-trace.git
cd cutedsl-trace
pip install -e .
```

## Quick Start

```python
import cutlass
from cutlass import cute
from cutedsl_trace import (
    TraceType, BlockType, TrackType,
    StaticTraceBuilder, TraceWriter,
    start, end, begin_lane, finish_lane
)

# Define trace types
TraceKernel = TraceType(
    name="TraceKernel",
    label_string="Kernel",
    tooltip_string="Kernel execution",
    param_count=0,
)

SimpleBlock = BlockType(
    name="SimpleBlock",
    label_string="Block {blockLinear}",
    tooltip_string="Block {blockLinear} on SM",
)

WarpTrack = TrackType(
    name="WarpTrack",
    label_string="Warp {lane}",
    tooltip_string="Warp {lane}",
)

# Create trace tensor
trace = StaticTraceBuilder(
    num_lanes=8,
    trace_types=[TraceKernel] * 8,
    max_events_per_lane=100,
    grid_dims=(16, 1, 1),
)
trace.set_track_type(WarpTrack)

# Instrument your kernel with start()/end() calls
# ...

# Write trace file
writer = TraceWriter("my_kernel")
writer.set_block_type(SimpleBlock)
writer.add_tensor(trace)
writer.write("trace.nanotrace")
```

## Visualization

Open the trace file at [aikitoria.github.io/nanotrace](https://aikitoria.github.io/nanotrace) or use the bundled visualizer:

```bash
cd visualizer
python -m http.server 8080
# Open http://localhost:8080
```

### Features

- **Collapse/Expand**: Click CTAs to expand/collapse their tracks
- **Named Tracks**: Custom names like "Producer", "Consumer", "Math"
- **Zoom/Pan**: Mouse wheel to zoom, drag to pan
- **Tooltips**: Hover over events for details

### Tracing Granularity

Traces can be captured at different granularities:

| Level | Lanes | Use Case |
|-------|-------|----------|
| **Warp** | 8-32 per block | Standard profiling (1 lane per warp) |
| **Thread** | 256+ per block | Detailed debugging (1 lane per thread sample) |
| **Phase** | 2-4 per block | Pipeline analysis (producer/consumer/math) |

See `tests/test_kernel.py` for examples of each granularity.

## Roadmap

- [ ] **Hierarchical expansion**: Expand warps to see individual thread traces
- [ ] **SM-level aggregation**: Collapsed view showing SM activity
- [ ] **Timeline filtering**: Filter by trace type or time range

## License

MIT License - see LICENSE file for details.
