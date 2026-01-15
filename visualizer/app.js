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
        this.collapsedBlocks = new Set();  // Track which blocks are collapsed
        this.collapsedWarps = new Set();   // Track which warps are collapsed ("blockId-laneId")
        this.visibleTrackIndices = [];     // Indices of visible tracks

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

            // Reset collapse states
            this.collapsedBlocks.clear();
            this.collapsedWarps.clear();

            // Collapse all warps by default to keep threads hidden
            for (const track of trace.tracks) {
                if (track.level === 2) { // WARP
                    this.collapsedWarps.add(`${track.blockId}-${track.laneId}`);
                }
            }

            this.setStatus('Indexing...', true);
            this.renderer.setTrace(trace);

            // Update UI
            this.emptyState.classList.add('hidden');
            this.updateVisibleTrackIndices();
            console.time('Building tree');
            this.buildTree();
            console.timeEnd('Building tree');
            this.updateTraceInfo();

            this.setStatus('Ready');

        } catch (error) {
            console.error('Error loading trace:', error);
            this.setStatus(`Error: ${error.message}`);
        }
    }

    /**
     * Update visible track indices based on collapsed blocks/warps
     */
    updateVisibleTrackIndices() {
        // Build list of visible track indices
        this.visibleTrackIndices = [];
        for (let i = 0; i < this.trace.tracks.length; i++) {
            const track = this.trace.tracks[i];

            // Hide if block is collapsed
            if (this.collapsedBlocks.has(track.blockId)) {
                continue;
            }

            // Hide if it's a thread and its warp is collapsed
            if (track.level === 3 && track.parentLane !== null) {
                if (this.collapsedWarps.has(`${track.blockId}-${track.parentLane}`)) {
                    continue;
                }
            }

            this.visibleTrackIndices.push(i);
        }

        // Update renderer with visible tracks
        this.renderer.setVisibleTracks(this.visibleTrackIndices);
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
     * Update visible tracks after collapse/expand
     */
    updateVisibleTracks() {
        this.updateVisibleTrackIndices();
        this.renderer.requestRender();
    }

    /**
     * Build tree view in sidebar - SM → Block → Warp hierarchy
     */
    buildTree() {
        console.log(`Building tree for ${this.trace.tracks.length} tracks...`);
        this.treeContainer.innerHTML = '';

        // Group tracks by SM, then by Block
        const smMap = new Map(); // smId -> { blocks: Map(blockId -> { block, tracks }) }

        for (let i = 0; i < this.trace.tracks.length; i++) {
            const track = this.trace.tracks[i];
            const smId = track.block?.smId ?? 0;
            const blockId = track.blockId;

            if (!smMap.has(smId)) {
                smMap.set(smId, { blocks: new Map() });
            }

            const smData = smMap.get(smId);
            if (!smData.blocks.has(blockId)) {
                smData.blocks.set(blockId, {
                    block: track.block,
                    tracks: []
                });
            }
            smData.blocks.get(blockId).tracks.push({ track, index: i });
        }

        // Sort SMs by ID
        const sortedSmIds = Array.from(smMap.keys()).sort((a, b) => a - b);

        // Build tree nodes for each SM
        for (const smId of sortedSmIds) {
            const smData = smMap.get(smId);

            const smNode = document.createElement('div');
            smNode.className = 'tree-node tree-sm';

            smNode.innerHTML = `
                <div class="tree-node-header tree-sm-header">
                    <span class="tree-node-icon">▼</span>
                    <span class="tree-node-label">SM ${smId}</span>
                    <span class="tree-node-count">${smData.blocks.size} blocks</span>
                </div>
                <div class="tree-node-children"></div>
            `;

            const smHeader = smNode.querySelector('.tree-node-header');
            const smChildren = smNode.querySelector('.tree-node-children');

            smHeader.addEventListener('click', () => {
                const isCollapsed = smNode.classList.toggle('collapsed');
                smHeader.querySelector('.tree-node-icon').textContent = isCollapsed ? '▶' : '▼';
            });

            // Sort blocks by ID
            const sortedBlockIds = Array.from(smData.blocks.keys()).sort((a, b) => a - b);

            // Add blocks within this SM
            for (const blockId of sortedBlockIds) {
                const blockData = smData.blocks.get(blockId);

                const blockNode = document.createElement('div');
                blockNode.className = 'tree-node tree-block';

                const blockLabel = this.parser.formatLabel(
                    blockData.block?.format?.label || `Block ${blockId}`,
                    { blockId }
                );

                blockNode.innerHTML = `
                    <div class="tree-node-header tree-block-header">
                        <span class="tree-node-icon">▼</span>
                        <span class="tree-node-label">${blockLabel}</span>
                        <span class="tree-node-count">${blockData.tracks.length}</span>
                    </div>
                    <div class="tree-node-children"></div>
                `;

                const blockHeader = blockNode.querySelector('.tree-node-header');
                const blockChildren = blockNode.querySelector('.tree-node-children');

                blockHeader.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isCollapsed = blockNode.classList.toggle('collapsed');
                    blockHeader.querySelector('.tree-node-icon').textContent = isCollapsed ? '▶' : '▼';

                    if (isCollapsed) {
                        this.collapsedBlocks.add(blockId);
                    } else {
                        this.collapsedBlocks.delete(blockId);
                    }
                    this.updateVisibleTracks();
                });

                // Group tracks into warps and threads
                const warpMap = new Map();

                for (const item of blockData.tracks) {
                    const track = item.track;
                    if (track.level <= 2) {
                        warpMap.set(track.laneId, { ...item, threads: [] });
                    }
                }

                for (const item of blockData.tracks) {
                    const track = item.track;
                    if (track.level === 3 && track.parentLane !== null) {
                        const warp = warpMap.get(track.parentLane);
                        if (warp) {
                            warp.threads.push(item);
                        }
                    }
                }

                // Add warps within this block
                for (const warpData of warpMap.values()) {
                    const { track, index, threads } = warpData;
                    const warpNode = document.createElement('div');
                    warpNode.className = 'tree-node tree-warp';

                    if (threads.length > 0) {
                        const warpKey = `${blockId}-${track.laneId}`;
                        const isCollapsed = this.collapsedWarps.has(warpKey);
                        if (isCollapsed) warpNode.classList.add('collapsed');

                        const warpLabel = this.parser.formatLabel(
                            track.trackFormat?.label || `Warp ${track.laneId}`,
                            { laneId: track.laneId }
                        );

                        warpNode.innerHTML = `
                            <div class="tree-node-header">
                                <span class="tree-node-icon">${isCollapsed ? '▶' : '▼'}</span>
                                <span class="tree-node-label">${warpLabel}</span>
                                <span class="tree-node-count">${track.events.length}</span>
                            </div>
                            <div class="tree-node-children"></div>
                        `;

                        const warpHeader = warpNode.querySelector('.tree-node-header');
                        const warpIcon = warpHeader.querySelector('.tree-node-icon');

                        warpHeader.addEventListener('click', (e) => {
                            e.stopPropagation();
                            const collapsed = warpNode.classList.toggle('collapsed');
                            warpIcon.textContent = collapsed ? '▶' : '▼';

                            if (collapsed) {
                                this.collapsedWarps.add(warpKey);
                            } else {
                                this.collapsedWarps.delete(warpKey);
                            }
                            this.updateVisibleTracks();
                        });

                        const threadsContainer = warpNode.querySelector('.tree-node-children');
                        for (const threadItem of threads) {
                            const threadNode = document.createElement('div');
                            threadNode.className = 'tree-node tree-thread';
                            const threadLabel = this.parser.formatLabel(
                                threadItem.track.trackFormat?.label || `Thread ${threadItem.track.laneId}`,
                                { laneId: threadItem.track.laneId }
                            );
                            threadNode.innerHTML = `
                                <div class="tree-node-header">
                                    <span class="tree-node-icon">◆</span>
                                    <span class="tree-node-label">${threadLabel}</span>
                                    <span class="tree-node-count">${threadItem.track.events.length}</span>
                                </div>
                            `;
                            threadNode.addEventListener('click', () => this.scrollToTrack(threadItem.index));
                            threadsContainer.appendChild(threadNode);
                        }
                    } else {
                        const warpLabel = this.parser.formatLabel(
                            track.trackFormat?.label || `Lane ${track.laneId}`,
                            { laneId: track.laneId }
                        );

                        warpNode.innerHTML = `
                            <div class="tree-node-header">
                                <span class="tree-node-icon">◆</span>
                                <span class="tree-node-label">${warpLabel}</span>
                                <span class="tree-node-count">${track.events.length}</span>
                            </div>
                        `;
                        warpNode.querySelector('.tree-node-header').addEventListener('click', () => {
                            this.scrollToTrack(index);
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
            this.canvasContainer.scrollTop = this.renderer.viewState.offsetY;
        }
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
            this.canvasContainer.scrollTop = this.renderer.viewState.offsetY;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
        } else {
            // Hit test for hover
            const hit = this.renderer.hitTest(x, y);

            if (hit) {
                this.renderer.hoveredEvent = hit.event;
                this.renderer.hoveredTrack = hit.trackIdx;
                this.showTooltip(e.clientX, e.clientY, hit);
            } else {
                this.renderer.hoveredEvent = null;
                this.renderer.hoveredTrack = null;
                this.hideTooltip();
            }

            this.renderer.requestRender();

            // Update cursor time
            if (this.trace) {
                const time = this.renderer.xToTime(x);
                this.cursorTime.textContent = this.renderer.formatTime(time - this.trace.timeRange.min);
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
                <span class="tooltip-label">Block:</span>
                <span class="tooltip-value">${track.blockId}</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Cluster:</span>
                <span class="tooltip-value">${track.block?.clusterId ?? '0'}</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">Lane:</span>
                <span class="tooltip-value">${track.laneId}</span>
            </div>
            <div class="tooltip-row">
                <span class="tooltip-label">SM:</span>
                <span class="tooltip-value">${track.block?.smId ?? '?'}</span>
            </div>
        `;

        // Add parameters if any
        if (event.params && event.params.length > 0) {
            body += `<div class="tooltip-row" style="margin-top: 8px; border-top: 1px solid #30363d; padding-top: 8px;">
                <span class="tooltip-label">Parameters:</span>
            </div>`;
            for (let i = 0; i < event.params.length; i++) {
                body += `<div class="tooltip-row">
                    <span class="tooltip-label">{${i}}:</span>
                    <span class="tooltip-value">${event.params[i]} (0x${event.params[i].toString(16)})</span>
                </div>`;
            }
        }

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
