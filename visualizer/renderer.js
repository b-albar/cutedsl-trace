/**
 * High-performance timeline renderer
 *
 * Designed to handle millions of events efficiently through:
 * - Viewport culling: Only render visible events
 * - Level-of-detail: Aggregate events when zoomed out
 * - Spatial indexing: Fast viewport queries using interval trees
 * - Canvas batching: Minimize draw calls with path batching
 */

class TimelineRenderer {
    constructor(canvas, rulerCanvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d', { alpha: false });
        this.rulerCanvas = rulerCanvas;
        this.rulerCtx = rulerCanvas.getContext('2d');

        // Trace data
        this.trace = null;
        this.tracks = [];

        // View state
        this.viewState = {
            offsetX: 0,        // Horizontal scroll (in pixels)
            offsetY: 0,        // Vertical scroll (in pixels)
            scale: 1,          // Pixels per nanosecond
            minScale: 1e-12,   // Min zoom (very zoomed out)
            maxScale: 1000,    // Max zoom (1 pixel = 1 picosecond)
        };

        // Layout
        this.layout = {
            trackHeight: 28,
            trackGap: 2,
            eventPadding: 1,
            minEventWidth: 2,     // Minimum pixels to render event individually
            aggregateThreshold: 1 // Events smaller than this get aggregated
        };

        // Colors (from CSS variables, cached)
        this.colors = {
            background: '#0d1117',
            trackBg: '#161b22',
            trackBgAlt: '#1c2128',
            border: '#30363d',
            text: '#8b949e',
            textMuted: '#6e7681',
            ruler: '#21262d',
            rulerText: '#8b949e',
            events: [
                '#7c3aed', '#2563eb', '#0891b2', '#059669',
                '#ca8a04', '#dc2626', '#db2777', '#9333ea'
            ]
        };

        // Performance: Spatial index for events (per track)
        this.eventIndex = null;

        // Hover state
        this.hoveredEvent = null;
        this.hoveredTrack = null;

        // Label formatter function
        this.labelFormatter = null;

        // Animation frame ID
        this.rafId = null;
        this.needsRender = false;

        // Stats
        this.stats = {
            eventsRendered: 0,
            eventsSkipped: 0,
            renderTime: 0
        };
    }

    /**
     * Set the function used to format event labels
     */
    setLabelFormatter(formatter) {
        this.labelFormatter = formatter;
    }

    /**
     * Get a consistent color for an event type label
     * Colors are dynamically assigned when the trace is loaded
     */
    getColorForEventType(label) {
        if (!this.eventTypeColors) {
            this.eventTypeColors = new Map();
            this.nextColorIndex = 0;
        }

        if (this.eventTypeColors.has(label)) {
            return this.eventTypeColors.get(label);
        }

        // Assign next color from the generated palette
        const color = this.generateDistinctColor(this.nextColorIndex);
        this.eventTypeColors.set(label, color);
        this.nextColorIndex++;
        return color;
    }

    /**
     * Generate visually distinct colors using golden ratio distribution
     * This ensures colors are spread out across the hue spectrum
     */
    generateDistinctColor(index) {
        // Golden ratio conjugate for optimal hue distribution
        const goldenRatio = 0.618033988749895;

        // Start with a base hue offset for aesthetics
        const baseHue = 0.65; // Start around purple/blue

        // Calculate hue using golden ratio for maximum visual separation
        const hue = (baseHue + index * goldenRatio) % 1.0;

        // Use high saturation and medium-high lightness for vibrant, visible colors
        const saturation = 0.70 + (index % 3) * 0.1; // Vary saturation slightly (70-90%)
        const lightness = 0.50 + (index % 2) * 0.08; // Vary lightness slightly (50-58%)

        return this.hslToHex(hue * 360, saturation * 100, lightness * 100);
    }

    /**
     * Convert HSL to hex color string
     */
    hslToHex(h, s, l) {
        s /= 100;
        l /= 100;

        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = l - c / 2;

        let r, g, b;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }

        const toHex = (n) => {
            const hex = Math.round((n + m) * 255).toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        };

        return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }

    /**
     * Reset event type colors (called when loading a new trace)
     */
    resetEventTypeColors() {
        this.eventTypeColors = new Map();
        this.nextColorIndex = 0;
    }

    /**
     * Set trace data and build spatial index
     */
    setTrace(trace) {
        this.trace = trace;
        this.tracks = trace.tracks;

        // Reset color assignments for new trace
        this.resetEventTypeColors();

        // Initialize visible tracks to all tracks
        this.visibleTrackIndices = Array.from({ length: this.tracks.length }, (_, i) => i);

        // Build spatial index for efficient viewport queries
        console.time('Building spatial index');
        this.buildSpatialIndex();
        console.timeEnd('Building spatial index');

        // Calculate initial scale to fit trace
        this.fitToView();

        this.requestRender();
    }

    /**
     * Set which tracks are visible (for collapse/expand)
     */
    setVisibleTracks(indices) {
        this.visibleTrackIndices = indices;
        this.requestRender();
    }

    /**
     * Get the absolute Y position for a track (based on its row in visible tracks)
     * @param {number} trackIdx - Original track index
     * @returns {number} Absolute Y position in pixels
     */
    getTrackAbsY(trackIdx) {
        // Find the row number for this track in the visible tracks list
        const row = this.visibleTrackIndices ? this.visibleTrackIndices.indexOf(trackIdx) : trackIdx;
        if (row === -1) return 0; // Track is not visible
        return row * (this.layout.trackHeight + this.layout.trackGap);
    }

    /**
     * Get the total content height based on visible tracks
     * @returns {number} Total height in pixels
     */
    get totalHeight() {
        const numTracks = this.visibleTrackIndices ? this.visibleTrackIndices.length : (this.tracks ? this.tracks.length : 0);
        return numTracks * (this.layout.trackHeight + this.layout.trackGap);
    }

    /**
     * Build spatial index for fast event lookups
     * Uses a simplified approach: sort events by start time per track
     */
    buildSpatialIndex() {
        this.eventIndex = new Map();

        for (let i = 0; i < this.tracks.length; i++) {
            const track = this.tracks[i];

            // Pre-compute event bounds and create sorted index
            const indexedEvents = track.events.map((event, idx) => ({
                idx,
                start: event.timeOffset,
                end: event.timeOffset + event.duration,
                event
            }));

            // Sort by start time for binary search
            indexedEvents.sort((a, b) => a.start - b.start);

            // Create a separate list for drawing: larger duration first
            // This ensures nested events (start earlier/same but end later) are drawn behind
            // smaller events that they contain.
            const drawOrdered = [...indexedEvents].sort((a, b) => (b.end - b.start) - (a.end - a.start));

            this.eventIndex.set(i, {
                sorted: indexedEvents,
                drawOrdered: drawOrdered,
                // Pre-compute aggregation buckets at different scales
                aggregated: this.precomputeAggregation(indexedEvents)
            });
        }
    }

    /**
     * Pre-compute aggregated event data at different zoom levels
     * This enables O(1) rendering when zoomed out
     */
    precomputeAggregation(events) {
        if (events.length === 0) return [];

        const buckets = [];

        // Create buckets at exponentially increasing time scales
        // Scale 0: 1µs buckets, Scale 1: 10µs, Scale 2: 100µs, etc.
        const scales = [1000, 10000, 100000, 1000000, 10000000, 100000000];

        for (const bucketSize of scales) {
            const bucket = new Map();

            for (const e of events) {
                const key = Math.floor(e.start / bucketSize);
                if (!bucket.has(key)) {
                    bucket.set(key, {
                        start: e.start,
                        end: e.end,
                        count: 0,
                        totalDuration: 0,
                        formatCounts: new Map()
                    });
                }
                const b = bucket.get(key);
                b.count++;
                b.totalDuration += (e.end - e.start);

                // Track actual start/end extent within this bucket to avoid coloring empty space
                b.start = Math.min(b.start, e.start);
                b.end = Math.max(b.end, e.end);

                const fId = e.event.formatId;
                b.formatCounts.set(fId, (b.formatCounts.get(fId) || 0) + 1);
            }

            for (const b of bucket.values()) {
                // Find dominant format ID for more accurate coloring when zoomed out
                let maxCount = -1;
                let dominantId = 0;
                for (const [id, count] of b.formatCounts) {
                    if (count > maxCount) {
                        maxCount = count;
                        dominantId = id;
                    }
                }
                b.dominantFormatId = dominantId;
                delete b.formatCounts; // Cleanup
            }

            buckets.push({
                bucketSize,
                buckets: Array.from(bucket.values())
            });
        }

        return buckets;
    }

    /**
     * Find events visible in a time range using binary search
     */
    findEventsInRange(trackIdx, startTime, endTime) {
        const index = this.eventIndex.get(trackIdx);
        if (!index) return [];

        const sorted = index.sorted;
        if (sorted.length === 0) return [];

        // Binary search for first event that could be visible
        let left = 0;
        let right = sorted.length - 1;

        while (left < right) {
            const mid = Math.floor((left + right) / 2);
            if (sorted[mid].end < startTime) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        // Collect events in range
        const result = [];
        for (let i = left; i < sorted.length && sorted[i].start <= endTime; i++) {
            if (sorted[i].end >= startTime) {
                result.push(sorted[i]);
            }
        }

        return result;
    }

    /**
     * Get aggregated events for a time range at appropriate scale
     */
    getAggregatedEvents(trackIdx, startTime, endTime, pixelsPerNs) {
        const index = this.eventIndex.get(trackIdx);
        if (!index || index.aggregated.length === 0) return [];

        // Find appropriate aggregation level based on zoom
        // We want buckets that are at least a few pixels wide
        const targetBucketPixels = 4;
        const targetBucketNs = targetBucketPixels / pixelsPerNs;

        // Find the smallest bucket size that's still reasonable
        let bucketData = index.aggregated[index.aggregated.length - 1];
        for (const bd of index.aggregated) {
            if (bd.bucketSize >= targetBucketNs * 0.5) {
                bucketData = bd;
                break;
            }
        }

        // Filter buckets to range
        return bucketData.buckets.filter(b => b.end >= startTime && b.start <= endTime);
    }

    /**
     * Fit the trace to the current view
     */
    fitToView() {
        if (!this.trace) return;

        const duration = this.trace.timeRange.duration;
        const width = (this.cssWidth || this.canvas.width) - 40; // Padding

        this.viewState.scale = width / duration;
        this.viewState.offsetX = 0;
        this.viewState.offsetY = 0;

        // Clamp scale
        this.viewState.scale = Math.max(
            this.viewState.minScale,
            Math.min(this.viewState.maxScale, this.viewState.scale)
        );

        // Store initial scale for zoom level calculation
        this.initialScale = this.viewState.scale;
    }

    /**
     * Convert time (ns) to pixel X coordinate
     */
    timeToX(time) {
        return (time - this.trace.timeRange.min) * this.viewState.scale - this.viewState.offsetX;
    }

    /**
     * Convert pixel X to time (ns)
     */
    xToTime(x) {
        return (x + this.viewState.offsetX) / this.viewState.scale + this.trace.timeRange.min;
    }

    /**
     * Get track Y coordinate
     */
    trackToY(trackIdx) {
        return trackIdx * (this.layout.trackHeight + this.layout.trackGap) - this.viewState.offsetY;
    }

    /**
     * Request a render on the next animation frame
     */
    requestRender() {
        if (!this.needsRender) {
            this.needsRender = true;
            this.rafId = requestAnimationFrame(() => {
                this.needsRender = false;
                this.render();
            });
        }
    }

    /**
     * Main render function
     */
    render() {
        if (!this.trace) return;

        const startTime = performance.now();

        this.stats.eventsRendered = 0;
        this.stats.eventsSkipped = 0;

        const ctx = this.ctx;
        // Use CSS dimensions (not buffer dimensions which are DPR-scaled)
        const width = this.cssWidth || this.canvas.width;
        const height = this.cssHeight || this.canvas.height;

        // Clear
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);

        // Determine visible time range
        const visibleStartTime = this.xToTime(0);
        const visibleEndTime = this.xToTime(width);
        const pixelsPerNs = this.viewState.scale;

        // Determine if we should use aggregation
        // Use raw events if each would be at least minEventWidth pixels
        const useAggregation = pixelsPerNs < this.layout.aggregateThreshold / 1000;

        // Determine visible tracks (based on visibility and viewport)
        const numVisibleTracks = this.visibleTrackIndices ? this.visibleTrackIndices.length : this.tracks.length;
        const firstVisibleRow = Math.max(0, Math.floor(this.viewState.offsetY / (this.layout.trackHeight + this.layout.trackGap)));
        const lastVisibleRow = Math.min(
            numVisibleTracks - 1,
            Math.ceil((this.viewState.offsetY + height) / (this.layout.trackHeight + this.layout.trackGap))
        );

        // Render tracks (using visibleTrackIndices if available)
        for (let row = firstVisibleRow; row <= lastVisibleRow; row++) {
            const trackIdx = this.visibleTrackIndices ? this.visibleTrackIndices[row] : row;
            if (trackIdx === undefined) continue;
            this.renderTrackAtRow(ctx, trackIdx, row, visibleStartTime, visibleEndTime, pixelsPerNs, useAggregation);
        }

        // Render ruler
        this.renderRuler();

        // Render hover highlight if any
        if (this.hoveredEvent && this.hoveredTrack !== null) {
            this.renderHoverHighlight(ctx);
        }

        this.stats.renderTime = performance.now() - startTime;
    }

    /**
     * Render a single track at a specific row position
     * @param {number} trackIdx - Original track index in trace data
     * @param {number} row - Visual row number for positioning
     */
    renderTrackAtRow(ctx, trackIdx, row, visibleStartTime, visibleEndTime, pixelsPerNs, useAggregation) {
        const track = this.tracks[trackIdx];
        const y = row * (this.layout.trackHeight + this.layout.trackGap) - this.viewState.offsetY;
        const h = this.layout.trackHeight;

        // Skip if not visible
        if (y + h < 0 || y > this.canvas.height) return;

        // Draw track background
        ctx.fillStyle = row % 2 === 0 ? this.colors.trackBg : this.colors.trackBgAlt;
        ctx.fillRect(0, y, this.canvas.width, h);

        // Get events to render
        let eventsToRender;
        if (useAggregation) {
            eventsToRender = this.getAggregatedEvents(trackIdx, visibleStartTime, visibleEndTime, pixelsPerNs);
            this.renderAggregatedEvents(ctx, eventsToRender, y, h, pixelsPerNs, track);
        } else {
            eventsToRender = this.findEventsInRange(trackIdx, visibleStartTime, visibleEndTime);
            this.renderEvents(ctx, eventsToRender, y, h, pixelsPerNs, track);
        }
    }

    /**
     * Render individual events
     */
    renderEvents(ctx, events, y, h, pixelsPerNs, track) {
        const padding = this.layout.eventPadding;
        const minWidth = this.layout.minEventWidth;
        const canvasWidth = this.cssWidth || this.canvas.width;

        // Sort by duration descending to handle nested events correctly (larger ones drawn first)
        const sortedEvents = [...events].sort((a, b) => (b.end - b.start) - (a.end - a.start));

        // Batch events by color for fewer draw calls
        const batches = new Map();

        for (const e of sortedEvents) {
            const x = this.timeToX(e.start);
            let w = e.event.duration * pixelsPerNs;

            // Ensure minimum width for visibility
            if (w < minWidth) {
                w = minWidth;
            }

            // Skip if off-screen
            if (x + w < 0 || x > canvasWidth) {
                this.stats.eventsSkipped++;
                continue;
            }

            // Color by event type label for consistency (same type = same color)
            const eventLabel = e.event.format?.label || 'Event';
            const color = this.getColorForEventType(eventLabel);

            if (!batches.has(color)) {
                batches.set(color, []);
            }
            batches.get(color).push({ x, y: y + padding, w, h: h - padding * 2, event: e });

            this.stats.eventsRendered++;
        }

        // Draw batches
        for (const [color, rects] of batches) {
            ctx.fillStyle = color;
            ctx.beginPath();
            for (const r of rects) {
                ctx.rect(r.x, r.y, r.w, r.h);
            }
            ctx.fill();
        }

        // Draw labels
        if (this.labelFormatter) {
            ctx.fillStyle = '#000000'; // Black text for better visibility on vibrant colors
            ctx.font = '600 11px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif';
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';

            for (const [color, rects] of batches) {
                for (const r of rects) {
                    // Only draw labels if there's enough horizontal space (at least 20px)
                    if (r.w > 20) {
                        let label = this.labelFormatter(
                            r.event.event.format?.label || 'Event',
                            {
                                params: r.event.event.params,
                                blockId: track.blockId,
                                laneId: track.laneId
                            }
                        );

                        // Add duration
                        const durationStr = r.event.event.duration.toLocaleString('en-US') + ' ns';
                        label = `${label} (${durationStr})`;

                        const textWidth = ctx.measureText(label).width;

                        // Sticky left logic: stick to screen left edge if event extends off-screen
                        let textX = Math.max(r.x, 0) + 4;

                        // Constrain right: prevent text from sliding off the right side of the event
                        textX = Math.min(textX, r.x + r.w - textWidth - 4);

                        // Safety: never draw to the left of the event start
                        textX = Math.max(textX, r.x + 4);

                        // Calculate available width from textX to end of event (minus padding)
                        const availableWidth = Math.max(0, r.x + r.w - textX - 4);

                        // Vertical bias: shift detail events (short) down and container events (long) up
                        const duration = r.event.event.duration;
                        const isLong = duration > 5000;
                        const vOffset = isLong ? -2 : 2;

                        ctx.fillText(label, textX, r.y + r.h / 2 + vOffset, availableWidth);
                    }
                }
            }
        }
    }

    /**
     * Render aggregated events (when zoomed out)
     */
    renderAggregatedEvents(ctx, buckets, y, h, pixelsPerNs, track) {
        const padding = this.layout.eventPadding;

        for (const bucket of buckets) {
            const x = this.timeToX(bucket.start);
            const w = (bucket.end - bucket.start) * pixelsPerNs;

            if (x + w < 0 || x > this.canvas.width) continue;

            // Use dominant format ID color for consistency
            const format = this.trace.formatDescriptors[bucket.dominantFormatId];
            const color = this.getColorForEventType(format?.label || 'Event');

            // Color by density (more events = more saturated)
            const density = Math.min(bucket.count / 10, 1);

            // Draw density bar
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.3 + density * 0.7;
            ctx.fillRect(x, y + padding, Math.max(w, 1), h - padding * 2);
            ctx.globalAlpha = 1;

            this.stats.eventsRendered++;
        }
    }

    /**
     * Render hover highlight
     */
    renderHoverHighlight(ctx) {
        const e = this.hoveredEvent;
        const trackIdx = this.hoveredTrack;

        const x = this.timeToX(e.start);
        const w = Math.max(e.event.duration * this.viewState.scale, 2);
        const y = this.trackToY(trackIdx);
        const h = this.layout.trackHeight;

        // Draw highlight border
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y, w + 2, h);
    }

    /**
     * Render time ruler
     */
    renderRuler() {
        const ctx = this.rulerCtx;
        const width = this.rulerCanvas.width;
        const height = this.rulerCanvas.height;

        ctx.fillStyle = this.colors.ruler;
        ctx.fillRect(0, 0, width, height);

        // Determine nice tick intervals
        const visibleDuration = width / this.viewState.scale;
        const targetTicks = 10;
        const rawInterval = visibleDuration / targetTicks;

        // Round to nice numbers
        const magnitude = Math.pow(10, Math.floor(Math.log10(rawInterval)));
        let interval;
        if (rawInterval / magnitude < 2) {
            interval = magnitude;
        } else if (rawInterval / magnitude < 5) {
            interval = 2 * magnitude;
        } else {
            interval = 5 * magnitude;
        }

        // Draw ticks
        ctx.fillStyle = this.colors.rulerText;
        ctx.font = '10px -apple-system, sans-serif';
        ctx.textAlign = 'center';

        const visibleStartTime = this.xToTime(0);
        const visibleEndTime = this.xToTime(width);
        const firstTick = Math.ceil(visibleStartTime / interval) * interval;

        // Draw border
        ctx.strokeStyle = this.colors.border;
        ctx.beginPath();
        ctx.moveTo(0, height - 0.5);
        ctx.lineTo(width, height - 0.5);
        ctx.stroke();

        for (let t = firstTick; t <= visibleEndTime; t += interval) {
            const x = this.timeToX(t);

            // Draw tick
            ctx.fillStyle = this.colors.border;
            ctx.fillRect(x, height - 8, 1, 8);

            // Draw label
            ctx.fillStyle = this.colors.rulerText;
            const label = this.formatTime(t - this.trace.timeRange.min);
            ctx.fillText(label, x, height - 12);
        }
    }

    /**
     * Format time in appropriate units
     */
    formatTime(ns) {
        if (ns >= 1e9) {
            return (ns / 1e9).toFixed(2) + 's';
        } else if (ns >= 1e6) {
            return (ns / 1e6).toFixed(2) + 'ms';
        } else if (ns >= 1e3) {
            return (ns / 1e3).toFixed(2) + 'µs';
        } else {
            return ns.toFixed(0) + 'ns';
        }
    }

    /**
     * Handle zoom
     */
    zoom(factor, centerX) {
        const centerTime = this.xToTime(centerX);

        this.viewState.scale *= factor;
        this.viewState.scale = Math.max(
            this.viewState.minScale,
            Math.min(this.viewState.maxScale, this.viewState.scale)
        );

        // Adjust offset to keep center point in place
        const newCenterX = (centerTime - this.trace.timeRange.min) * this.viewState.scale;
        this.viewState.offsetX = newCenterX - centerX;

        // Clamp offset
        this.viewState.offsetX = Math.max(0, this.viewState.offsetX);

        this.requestRender();
    }

    /**
     * Handle pan
     */
    pan(dx, dy) {
        this.viewState.offsetX -= dx;
        this.viewState.offsetY -= dy;

        // Clamp
        this.viewState.offsetX = Math.max(0, this.viewState.offsetX);
        this.viewState.offsetY = Math.max(0, this.viewState.offsetY);

        const numTracks = this.visibleTrackIndices ? this.visibleTrackIndices.length : this.tracks.length;
        const maxY = numTracks * (this.layout.trackHeight + this.layout.trackGap) - this.canvas.height;
        this.viewState.offsetY = Math.min(Math.max(0, maxY), this.viewState.offsetY);

        this.requestRender();
    }

    /**
     * Find event at pixel coordinates
     */
    hitTest(x, y) {
        if (!this.trace) return null;

        // Find row (visual position)
        const row = Math.floor((y + this.viewState.offsetY) / (this.layout.trackHeight + this.layout.trackGap));
        const numTracks = this.visibleTrackIndices ? this.visibleTrackIndices.length : this.tracks.length;
        if (row < 0 || row >= numTracks) return null;

        // Get actual track index from row
        const trackIdx = this.visibleTrackIndices ? this.visibleTrackIndices[row] : row;
        if (trackIdx === undefined) return null;

        // Find time
        const time = this.xToTime(x);

        // Find events at this time
        const events = this.findEventsInRange(trackIdx, time, time);

        // Return the most specific one (minimum duration) containing the time
        // This ensures detail events are prioritized over container events like loop iterations
        let bestMatch = null;
        let minDuration = Infinity;

        for (const e of events) {
            if (e.start <= time && e.end >= time) {
                const duration = e.end - e.start;
                if (duration < minDuration) {
                    minDuration = duration;
                    bestMatch = e;
                }
            }
        }

        return bestMatch ? { trackIdx, row, event: bestMatch } : null;
    }

    /**
     * Update canvas size
     */
    resize(width, height) {
        // Handle device pixel ratio for crisp rendering
        const dpr = window.devicePixelRatio || 1;

        // Store CSS dimensions for layout calculations
        this.cssWidth = width;
        this.cssHeight = height;

        this.canvas.width = width * dpr;
        this.canvas.height = height * dpr;
        this.canvas.style.width = width + 'px';
        this.canvas.style.height = height + 'px';

        // Reset transform before scaling (scale is cumulative otherwise)
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.scale(dpr, dpr);

        // Ruler canvas
        this.rulerCanvas.width = width * dpr;
        this.rulerCanvas.height = 32 * dpr;
        this.rulerCanvas.style.width = width + 'px';
        this.rulerCanvas.style.height = '32px';
        this.rulerCtx.setTransform(1, 0, 0, 1, 0, 0);
        this.rulerCtx.scale(dpr, dpr);

        this.requestRender();
    }

    /**
     * Clean up
     */
    destroy() {
        if (this.rafId) {
            cancelAnimationFrame(this.rafId);
        }
    }
}

// Export
window.TimelineRenderer = TimelineRenderer;
