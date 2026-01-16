/**
 * cutedsl-trace Visualizer Application
 * Main application logic and event handling
 */

class TraceVisualizerApp {
    constructor() {
        // DOM elements
        this.fileInput = document.getElementById('file-input');
        this.timelineContainer = document.getElementById('timeline-container');
        this.canvasContainer = document.getElementById('canvas-container');
        this.timelineCanvas = document.getElementById('timeline-canvas');
        this.rulerCanvas = document.getElementById('ruler-canvas');

        this.treeContainer = document.getElementById('tree-container');
        this.emptyState = document.getElementById('empty-state');
        this.tooltip = document.getElementById('tooltip');
        this.tooltipHeader = document.getElementById('tooltip-header');
        this.tooltipBody = document.getElementById('tooltip-body');
        this.statusText = document.getElementById('status-text');
        this.traceInfo = document.getElementById('trace-info');
        this.cursorTime = document.getElementById('cursor-time');

        // Info overlay elements
        this.infoOverlay = document.getElementById('info-overlay');
        this.infoKernel = document.getElementById('info-kernel');
        this.infoDuration = document.getElementById('info-duration');
        this.infoGrid = document.getElementById('info-grid');
        this.infoCluster = document.getElementById('info-cluster');
        this.infoSms = document.getElementById('info-sms');
        this.infoBlocks = document.getElementById('info-blocks');
        this.infoEvents = document.getElementById('info-events');

        // Initialize parser and renderer
        this.parser = new NanotraceParser();
        this.renderer = new TimelineRenderer(this.timelineCanvas, this.rulerCanvas);
        this.renderer.setLabelFormatter(this.parser.formatLabel.bind(this.parser));

        // State
        this.trace = null;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.collapsedSMs = new Set();     // Track which SMs are collapsed
        this.collapsedBlocks = new Set();  // Track which blocks are collapsed
        this.collapsedWarps = new Set();   // Track which warps are collapsed ("blockId-laneId")
        this.structure = null;             // Initialized hierarchy

        // Bind event handlers
        this.bindEvents();

        // Initial resize
        this.handleResize();
    }

    /**
     * Bind all event handlers
     */
    bindEvents() {
        // File input
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        document.body.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.timelineContainer.classList.add('drag-over');
        });

        document.body.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.timelineContainer.classList.remove('drag-over');
        });

        document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.timelineContainer.classList.remove('drag-over');

            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].name.endsWith('.nanotrace')) {
                this.loadFile(files[0]);
            }
        });

        // Zoom buttons
        document.getElementById('zoom-in').addEventListener('click', () => {
            this.renderer.zoom(1.5, this.timelineCanvas.width / 2);
        });

        document.getElementById('zoom-out').addEventListener('click', () => {
            this.renderer.zoom(0.67, this.timelineCanvas.width / 2);
        });

        document.getElementById('zoom-fit').addEventListener('click', () => {
            this.renderer.fitToView();
            this.renderer.requestRender();
        });

        // Canvas interactions
        this.timelineCanvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.timelineCanvas.addEventListener('mouseleave', () => this.handleMouseLeave());
        this.timelineCanvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        window.addEventListener('mouseup', () => this.handleMouseUp());
        this.timelineCanvas.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });

        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Sync scroll from sidebar to timeline
        this.treeContainer.addEventListener('scroll', () => {
            if (this.renderer.viewState.offsetY !== this.treeContainer.scrollTop) {
                this.renderer.viewState.offsetY = this.treeContainer.scrollTop;
                this.renderer.requestRender();
            }
        });
    }

    /**
     * Handle file selection
     */
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.loadFile(file);
        }
    }

    /**
     * Load a trace file
     */
    async loadFile(file) {
        this.setStatus('Loading...', true);

        try {
            const buffer = await file.arrayBuffer();

            this.setStatus('Parsing...', true);
            const trace = await this.parser.parse(buffer);

            this.trace = trace;

            // Sort tracks by SM -> Block -> Lane
            this.trace.tracks.sort((a, b) => {
                const smDiff = (a.block?.smId || 0) - (b.block?.smId || 0);
                if (smDiff !== 0) return smDiff;
                const blockDiff = a.blockId - b.blockId;
                if (blockDiff !== 0) return blockDiff;
                return a.laneId - b.laneId;
            });

            // Build hierarchy structure
            this.buildHierarchy();

            // Reset collapse states
            this.collapsedSMs.clear();
            this.collapsedBlocks.clear();
            this.collapsedWarps.clear();

            // Collapse all warps by default to keep threads hidden
            // Find warp keys from structure
            for (const sm of this.structure.sms) {
                for (const block of sm.blocks) {
                    for (const warp of block.warps) {
                        this.collapsedWarps.add(warp.key);
                    }
                }
            }

            this.setStatus('Indexing...', true);
            this.renderer.setTrace(trace);

            // Update UI
            this.emptyState.classList.add('hidden');

            console.time('Building tree');
            this.buildTree();
            console.timeEnd('Building tree');

            this.updateVisibleRows();
            this.updateTraceInfo();

            this.setStatus('Ready');

        } catch (error) {
            console.error('Error loading trace:', error);
            this.setStatus(`Error: ${error.message}`);
        }
    }

    /**
     * Build the hierarchical structure of the trace
     */
    buildHierarchy() {
        const smMap = new Map();

        // Pass 1: Group tracks by SM/Block
        for (let i = 0; i < this.trace.tracks.length; i++) {
            const track = this.trace.tracks[i];
            const smId = track.block?.smId ?? 0;
            const blockId = track.blockId;

            if (!smMap.has(smId)) smMap.set(smId, { id: smId, blocks: new Map(), count: 0 });
            const smData = smMap.get(smId);

            if (!smData.blocks.has(blockId)) {
                smData.blocks.set(blockId, {
                    id: blockId,
                    block: track.block,
                    tracks: [],
                    warps: new Map() // laneId -> { track, trackIdx, threads: [] }
                });
                smData.count++; // Count blocks
            }
            const blockData = smData.blocks.get(blockId);
            blockData.tracks.push({ track, index: i });
        }

        // Pass 2: Organize Blocks into Warps
        const sortedSmIds = Array.from(smMap.keys()).sort((a, b) => a - b);
        const smList = [];

        for (const smId of sortedSmIds) {
            const sm = smMap.get(smId);
            const sortedBlockIds = Array.from(sm.blocks.keys()).sort((a, b) => a - b);
            const blockList = [];

            for (const blockId of sortedBlockIds) {
                const block = sm.blocks.get(blockId);
                const warpMap = new Map();

                // Find warp tracks (headers)
                for (const item of block.tracks) {
                    const track = item.track;
                    // Assuming level <= 2 is warp/lane header
                    if (track.level <= 2) {
                        warpMap.set(track.laneId, {
                            key: `${blockId}-${track.laneId}`,
                            laneId: track.laneId,
                            track: item.track,
                            trackIdx: item.index,
                            eventCount: item.track.events.length,
                            threads: []
                        });
                    }
                }

                // Associate threads
                for (const item of block.tracks) {
                    const track = item.track;
                    // Assuming level == 3 is thread
                    if (track.level === 3) {
                        const parentLane = track.parentLane !== null ? track.parentLane : track.laneId;
                        let warp = warpMap.get(parentLane);

                        // If no parent warp found, maybe treat as its own warp?
                        // For now assuming well-formed trace
                        if (warp) {
                            warp.threads.push({
                                track: item.track,
                                trackIdx: item.index,
                                laneId: track.laneId,
                                eventCount: item.track.events.length
                            });
                        }
                    }
                }

                // Prepare label
                const label = this.parser.formatLabel(
                    block.block?.format?.label || `Block ${blockId}`,
                    { blockId }
                );

                blockList.push({
                    id: blockId,
                    label: label,
                    trackCount: block.tracks.length,
                    warps: Array.from(warpMap.values()).sort((a, b) => a.laneId - b.laneId)
                });
            }

            smList.push({
                id: smId,
                blocks: blockList
            });
        }

        this.structure = { sms: smList };
    }

    /**
     * Update visible rows and notify renderer
     */
    updateVisibleRows() {
        if (!this.structure) return;

        const rows = [];

        for (const sm of this.structure.sms) {
            // SM Header
            rows.push({ type: 'header', subtype: 'sm', label: `SM ${sm.id} (${sm.blocks.length} blocks)` });

            if (this.collapsedSMs.has(sm.id)) continue;

            for (const block of sm.blocks) {
                // Block Header
                rows.push({ type: 'header', subtype: 'block', label: `${block.label}` });

                if (this.collapsedBlocks.has(block.id)) continue;

                for (const warp of block.warps) {
                    // Warp Track (Level 2)
                    rows.push({ type: 'track', trackIdx: warp.trackIdx });

                    if (this.collapsedWarps.has(warp.key)) continue;

                    // Thread Tracks (Level 3)
                    for (const thread of warp.threads) {
                        rows.push({ type: 'track', trackIdx: thread.trackIdx });
                    }
                }
            }
        }

        this.visibleRows = rows;
        this.renderer.setRows(rows);
    }

    /**
     * Scroll to a specific track
     */
    scrollToTrack(trackIdx) {
        const y = this.renderer.getTrackAbsY(trackIdx);
        this.renderer.viewState.offsetY = y;
        this.canvasContainer.scrollTop = y;
        this.renderer.requestRender();
    }

    /**
     * Build tree view in sidebar - SM → Block → Warp hierarchy
     */
    buildTree() {
        // Clear container
        this.treeContainer.innerHTML = '';
        if (!this.structure) return;

        for (const sm of this.structure.sms) {
            const smNode = document.createElement('div');
            smNode.className = 'tree-node tree-sm';
            if (this.collapsedSMs.has(sm.id)) smNode.classList.add('collapsed');

            smNode.innerHTML = `
                <div class="tree-node-header tree-sm-header">
                    <span class="tree-node-icon">${this.collapsedSMs.has(sm.id) ? '▶' : '▼'}</span>
                    <span class="tree-node-label">SM ${sm.id}</span>
                    <span class="tree-node-count">${sm.blocks.length} blocks</span>
                </div>
                <div class="tree-node-children"></div>
            `;

            const smHeader = smNode.querySelector('.tree-node-header');
            const smChildren = smNode.querySelector('.tree-node-children');

            smHeader.addEventListener('click', () => {
                const isCollapsed = smNode.classList.toggle('collapsed');
                smHeader.querySelector('.tree-node-icon').textContent = isCollapsed ? '▶' : '▼';

                if (isCollapsed) this.collapsedSMs.add(sm.id);
                else this.collapsedSMs.delete(sm.id);

                this.updateVisibleRows();
            });

            for (const block of sm.blocks) {
                const blockNode = document.createElement('div');
                blockNode.className = 'tree-node tree-block';
                if (this.collapsedBlocks.has(block.id)) blockNode.classList.add('collapsed');

                blockNode.innerHTML = `
                    <div class="tree-node-header tree-block-header">
                        <span class="tree-node-icon">${this.collapsedBlocks.has(block.id) ? '▶' : '▼'}</span>
                        <span class="tree-node-label">${block.label}</span>
                        <span class="tree-node-count">${block.trackCount}</span>
                    </div>
                    <div class="tree-node-children"></div>
                `;

                const blockHeader = blockNode.querySelector('.tree-node-header');
                const blockChildren = blockNode.querySelector('.tree-node-children');

                blockHeader.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isCollapsed = blockNode.classList.toggle('collapsed');
                    blockHeader.querySelector('.tree-node-icon').textContent = isCollapsed ? '▶' : '▼';

                    if (isCollapsed) this.collapsedBlocks.add(block.id);
                    else this.collapsedBlocks.delete(block.id);

                    this.updateVisibleRows();
                });

                for (const warp of block.warps) {
                    const warpNode = document.createElement('div');
                    warpNode.className = 'tree-node tree-warp';
                    if (this.collapsedWarps.has(warp.key) && warp.threads.length > 0) warpNode.classList.add('collapsed');

                    const warpLabel = this.parser.formatLabel(
                        warp.track.trackFormat?.label || `Warp ${warp.laneId}`,
                        { laneId: warp.laneId }
                    );

                    // If threads exist, it's collapsible
                    const hasThreads = warp.threads.length > 0;
                    const icon = hasThreads ? (this.collapsedWarps.has(warp.key) ? '▶' : '▼') : '◆';

                    warpNode.innerHTML = `
                        <div class="tree-node-header">
                            <span class="tree-node-icon">${icon}</span>
                            <span class="tree-node-label">${warpLabel}</span>
                            <span class="tree-node-count">${warp.eventCount}</span>
                        </div>
                        <div class="tree-node-children"></div>
                    `;

                    const warpHeader = warpNode.querySelector('.tree-node-header');

                    if (hasThreads) {
                        warpHeader.addEventListener('click', (e) => {
                            e.stopPropagation();
                            const collapsed = warpNode.classList.toggle('collapsed');
                            warpHeader.querySelector('.tree-node-icon').textContent = collapsed ? '▶' : '▼';

                            if (collapsed) this.collapsedWarps.add(warp.key);
                            else this.collapsedWarps.delete(warp.key);

                            this.updateVisibleRows();
                        });

                        const threadsContainer = warpNode.querySelector('.tree-node-children');
                        for (const thread of warp.threads) {
                            const threadNode = document.createElement('div');
                            threadNode.className = 'tree-node tree-thread';

                            const threadLabel = this.parser.formatLabel(
                                thread.track.trackFormat?.label || `Thread ${thread.laneId}`,
                                { laneId: thread.laneId }
                            );

                            threadNode.innerHTML = `
                                <div class="tree-node-header">
                                    <span class="tree-node-icon">◆</span>
                                    <span class="tree-node-label">${threadLabel}</span>
                                    <span class="tree-node-count">${thread.eventCount}</span>
                                </div>
                            `;
                            threadNode.addEventListener('click', (e) => {
                                e.stopPropagation();
                                this.scrollToTrack(thread.trackIdx);
                            });
                            threadsContainer.appendChild(threadNode);
                        }
                    } else {
                        // Click to scroll to track?
                        warpHeader.addEventListener('click', (e) => {
                            e.stopPropagation();
                            this.scrollToTrack(warp.trackIdx);
                        });
                    }

                    blockChildren.appendChild(warpNode);
                }

                smChildren.appendChild(blockNode);
            }

            this.treeContainer.appendChild(smNode);
        }
    }

    /**
     * Update trace info in status bar
     */
    updateTraceInfo() {
        const t = this.trace;
        const duration = this.renderer.formatTime(t.timeRange.duration);
        this.traceInfo.textContent = `${t.kernelName} | ${t.tracks.length} tracks | ${t.totalEventCount} events | ${duration}`;

        // Show and update info overlay
        this.updateInfoOverlay();
    }

    /**
     * Update the info overlay panel
     */
    updateInfoOverlay() {
        if (!this.trace) {
            this.infoOverlay.style.display = 'none';
            return;
        }

        this.infoOverlay.style.display = 'block';
        const t = this.trace;

        // Kernel name
        this.infoKernel.textContent = t.kernelName;

        // Duration
        this.infoDuration.textContent = this.renderer.formatTime(t.timeRange.duration);

        // Grid dimensions
        this.infoGrid.textContent = `(${t.gridDims.x}, ${t.gridDims.y}, ${t.gridDims.z})`;

        // Cluster dimensions
        this.infoCluster.textContent = `(${t.clusterDims.x}, ${t.clusterDims.y}, ${t.clusterDims.z})`;

        // Count unique SMs
        const smSet = new Set();
        for (const block of t.blockDescriptors) {
            smSet.add(block.smId);
        }
        this.infoSms.textContent = smSet.size.toLocaleString();

        // Blocks count
        this.infoBlocks.textContent = t.blockDescriptors.length.toLocaleString();

        // Events count
        this.infoEvents.textContent = t.totalEventCount.toLocaleString();
    }

    /**
     * Handle mouse wheel for zoom/scroll
     */
    handleWheel(e) {
        e.preventDefault();

        let zoomFactor = 1;
        let mouseX = 0;

        if (e.ctrlKey || e.metaKey) {
            // Zoom
            zoomFactor = e.deltaY < 0 ? 1.2 : 0.83;
            const rect = this.timelineCanvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;

            this.renderer.zoom(zoomFactor, mouseX);

            // Fix: Update tooltip/highlight immediately after zoom to prevent coordinate desync
            this.handleMouseMove(e);
        } else {
            // Pan
            this.renderer.pan(-e.deltaX, -e.deltaY);
            // Sync scroll positions
            this.syncScrollPositions();
        }
    }

    /**
     * Synchronize scroll positions of both containers
     */
    syncScrollPositions() {
        const y = this.renderer.viewState.offsetY;
        this.canvasContainer.scrollTop = y;
        this.treeContainer.scrollTop = y;
    }

    /**
     * Handle mouse down
     */
    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvasContainer.style.cursor = 'grabbing';
    }

    /**
     * Handle mouse move
     */
    handleMouseMove(e) {
        const rect = this.timelineCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this.isDragging) {
            // Pan
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;
            this.renderer.pan(dx, dy);
            this.syncScrollPositions();
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        } else {
            // Hit test for hover
            const hit = this.renderer.hitTest(x, y);

            if (hit) {
                this.renderer.hoveredEvent = hit.event || null;
                this.renderer.hoveredTrack = hit.trackIdx || null;
                this.renderer.hoveredRowIdx = hit.rowIndex !== undefined ? hit.rowIndex : null;

                if (hit.event) {
                    this.showTooltip(e.clientX, e.clientY, hit);
                } else {
                    this.hideTooltip();
                }
            } else {
                this.renderer.hoveredEvent = null;
                this.renderer.hoveredTrack = null;
                this.renderer.hoveredRowIdx = null;
                this.hideTooltip();
            }

            this.renderer.requestRender();
            this.syncTreeHover();

            // Update cursor time
            if (this.trace) {
                const time = this.renderer.xToTime(x);
                this.cursorTime.textContent = this.renderer.formatTime(time - this.trace.timeRange.min);
            }
        }
    }

    /**
     * Synchronize hover state with the tree view
     */
    syncTreeHover() {
        const rowIdx = this.renderer.hoveredRowIdx;
        const headers = this.treeContainer.querySelectorAll('.tree-node-header');

        // Remove existing highlights
        headers.forEach(h => h.classList.remove('hovered'));

        if (rowIdx !== null && rowIdx !== undefined) {
            // This is a bit expensive, but find the header matching the physical index
            // In a production app, we would cache this mapping.
            if (headers[rowIdx]) {
                headers[rowIdx].classList.add('hovered');
            }
        }
    }

    /**
     * Handle mouse up
     */
    handleMouseUp(e) {
        this.isDragging = false;
        this.canvasContainer.style.cursor = '';
    }

    /**
     * Handle mouse leave
     */
    handleMouseLeave(e) {
        this.isDragging = false;
        this.canvasContainer.style.cursor = '';
        this.hideTooltip();
        this.renderer.hoveredEvent = null;
        this.renderer.hoveredTrack = null;
        this.renderer.requestRender();
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyDown(e) {
        switch (e.key) {
            case '+':
            case '=':
                this.renderer.zoom(1.5, this.timelineCanvas.width / 2);
                break;
            case '-':
            case '_':
                this.renderer.zoom(0.67, this.timelineCanvas.width / 2);
                break;
            case 'f':
            case 'F':
                this.renderer.fitToView();
                this.renderer.requestRender();
                break;
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        const container = this.canvasContainer;
        const width = container.clientWidth;
        const height = container.clientHeight;

        this.renderer.resize(width, height);

        // Also resize ruler
        const rulerWidth = this.timelineContainer.clientWidth;
        this.rulerCanvas.style.width = rulerWidth + 'px';
    }

    /**
     * Show tooltip for an event
     */
    showTooltip(x, y, hit) {
        const event = hit.event.event;
        const track = this.trace.tracks[hit.trackIdx];

        // Format header
        const label = this.parser.formatLabel(
            event.format?.label || 'Event',
            { params: event.params }
        );
        this.tooltipHeader.textContent = label;

        // Calculate grid coordinates
        const blockId = track.blockId;
        const dims = this.trace.gridDims;
        const bx = blockId % dims.x;
        const by = Math.floor((blockId % (dims.x * dims.y)) / dims.x);
        const bz = Math.floor(blockId / (dims.x * dims.y));

        // Calculate cluster coordinates
        const clusterId = track.block?.clusterId ?? 0;
        const g = this.trace.gridDims;
        const c = this.trace.clusterDims;
        // Number of clusters in each dimension
        const ncx = Math.max(1, g.x / c.x);
        const ncy = Math.max(1, g.y / c.y);

        const cx = clusterId % ncx;
        const cy = Math.floor((clusterId % (ncx * ncy)) / ncx);
        const cz = Math.floor(clusterId / (ncx * ncy));

        // Format body
        let body = `
            <div class="tooltip-row">
                <span class="tooltip-label">Duration:</span>
                <span class="tooltip-value">${this.renderer.formatTime(event.duration)}</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Start:</span>
                <span class="tooltip-value">${this.renderer.formatTime(hit.event.start - this.trace.timeRange.min)}</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Grid:</span>
                <span class="tooltip-value">(${bx}, ${by}, ${bz})</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Cluster:</span>
                <span class="tooltip-value">(${cx}, ${cy}, ${cz})</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Lane:</span>
                <span class="tooltip-value">${track.laneId}</span>
            </div>
        `;

        this.tooltipBody.innerHTML = body;

        // Position tooltip
        this.tooltip.style.display = 'block';
        const tooltipRect = this.tooltip.getBoundingClientRect();

        // Keep tooltip on screen
        let tooltipX = x + 16;
        let tooltipY = y + 16;

        if (tooltipX + tooltipRect.width > window.innerWidth - 8) {
            tooltipX = x - tooltipRect.width - 16;
        }
        if (tooltipY + tooltipRect.height > window.innerHeight - 8) {
            tooltipY = y - tooltipRect.height - 16;
        }

        this.tooltip.style.left = tooltipX + 'px';
        this.tooltip.style.top = tooltipY + 'px';
    }

    /**
     * Hide tooltip
     */
    hideTooltip() {
        this.tooltip.style.display = 'none';
    }

    /**
     * Set status bar text
     */
    setStatus(text, loading = false) {
        this.statusText.textContent = text;
        this.statusText.classList.toggle('loading', loading);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TraceVisualizerApp();
});
