/**
 * nanotrace file parser
 * Parses the binary .nanotrace format into JavaScript objects
 */

class NanotraceParser {
    constructor() {
        this.MAGIC = 'nanotrace\0';
    }

    /**
     * Parse a .nanotrace file
     * @param {ArrayBuffer} buffer - The file contents
     * @returns {Object} Parsed trace data
     */
    parse(buffer) {
        const view = new DataView(buffer);
        const decoder = new TextDecoder('utf-8');
        let offset = 0;

        // Read and verify magic
        const magic = decoder.decode(new Uint8Array(buffer, 0, 10));
        if (magic !== this.MAGIC) {
            throw new Error(`Invalid magic: expected "${this.MAGIC}", got "${magic}"`);
        }
        offset = 10;

        // Read version and compression flag
        const version = view.getUint8(offset++);
        const compressed = view.getUint8(offset++) !== 0;

        console.log(`Nanotrace version ${version}, compressed: ${compressed}`);

        // If compressed, decompress the rest
        let payload;
        if (compressed) {
            const compressedData = new Uint8Array(buffer, offset);
            payload = this.decompress(compressedData);
        } else {
            payload = new Uint8Array(buffer, offset);
        }

        return this.parsePayload(payload);
    }

    /**
     * Decompress deflate data using pako
     */
    decompress(data) {
        // Use pako for decompression if available
        if (typeof pako !== 'undefined') {
            return pako.inflate(data);
        }
        // Fallback: try DecompressionStream (modern browsers)
        throw new Error('Compressed traces require pako library. Include pako.min.js or use uncompressed traces.');
    }

    /**
     * Parse the payload (after decompression if needed)
     */
    parsePayload(data) {
        const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        const decoder = new TextDecoder('utf-8');
        let offset = 0;

        // Helper functions
        const readUint8 = () => view.getUint8(offset++);
        const readUint16 = () => { const v = view.getUint16(offset, true); offset += 2; return v; };
        const readUint32 = () => { const v = view.getUint32(offset, true); offset += 4; return v; };
        const readInt32 = () => { const v = view.getInt32(offset, true); offset += 4; return v; };
        const readUint64 = () => {
            const low = view.getUint32(offset, true);
            const high = view.getUint32(offset + 4, true);
            offset += 8;
            return low + high * 0x100000000;
        };
        const readString = () => {
            const len = readUint16();
            const str = decoder.decode(new Uint8Array(data.buffer, data.byteOffset + offset, len));
            offset += len;
            return str;
        };

        // Read kernel name
        const kernelName = readString();

        // Read grid dimensions
        const gridDims = {
            x: readUint32(),
            y: readUint32(),
            z: readUint32()
        };

        // Read cluster dimensions
        const clusterDims = {
            x: readUint32(),
            y: readUint32(),
            z: readUint32()
        };

        // Read counts
        const formatDescriptorCount = readUint32();
        const blockDescriptorCount = readUint32();
        const trackCount = readUint32();
        const totalEventCount = readUint64();

        console.log(`Kernel: ${kernelName}`);
        console.log(`Grid: ${gridDims.x}x${gridDims.y}x${gridDims.z}`);
        console.log(`Formats: ${formatDescriptorCount}, Blocks: ${blockDescriptorCount}, Tracks: ${trackCount}, Events: ${totalEventCount}`);

        // Read format descriptors
        const formatDescriptors = [];
        for (let i = 0; i < formatDescriptorCount; i++) {
            const label = readString();
            const tooltip = readString();
            const paramCount = readUint8();
            formatDescriptors.push({ id: i, label, tooltip, paramCount });
        }

        // Read block descriptors
        const blockDescriptors = [];
        for (let i = 0; i < blockDescriptorCount; i++) {
            const blockId = readUint32();
            const clusterId = readUint32();
            const smId = readUint16();
            const formatId = readUint16();
            blockDescriptors.push({
                blockId,
                clusterId,
                smId,
                formatId,
                format: formatDescriptors[formatId]
            });
        }

        // Read tracks and events
        const tracks = [];
        for (let i = 0; i < trackCount; i++) {
            const blockDescIdx = readUint32();
            const trackFormatId = readUint16();
            const laneId = readUint32();

            // Read track format parameters (count = placeholder count of track's format descriptor)
            const trackFormat = formatDescriptors[trackFormatId];
            if (!trackFormat) {
                console.error(`Track ${i}: Invalid track format ID ${trackFormatId}`);
                throw new Error(`Sync lost at Track ${i}: invalid track format`);
            }
            const trackParams = [];
            for (let k = 0; k < trackFormat.paramCount; k++) {
                trackParams.push(readUint32());
            }

            const eventCount = readUint32();

            const block = blockDescriptors[blockDescIdx];

            if (!block) {
                console.error(`Track ${i}: Invalid block descriptor index ${blockDescIdx}`);
            }

            // Heuristic for hierarchy
            let level = 2; // Default: Warp
            let parentLane = null;

            // Simple heuristic: if label contains "Thread", treat as thread (level 3)
            if (trackFormat && trackFormat.label && trackFormat.label.toLowerCase().includes('thread')) {
                level = 3;
                // If it's a thread, parent is the warp.
                // We assume simplistic mapping: parentLane = floor(laneId / 32)
                parentLane = Math.floor(laneId / 32);
            }

            const events = [];
            for (let j = 0; j < eventCount; j++) {
                const timeOffset = readUint32();
                const duration = readUint32();
                const eventFormatId = readUint16();

                const eventFormat = formatDescriptors[eventFormatId];
                const params = [];
                for (let k = 0; k < eventFormat.paramCount; k++) {
                    params.push(readUint32());
                }

                events.push({
                    timeOffset,
                    duration,
                    formatId: eventFormatId,
                    format: eventFormat,
                    params
                });
            }

            tracks.push({
                blockId: block.blockId,
                blockDescIdx,
                block,
                laneId,
                level,
                parentLane: parentLane === -1 ? null : parentLane,
                trackFormatId,
                trackFormat,
                params: trackParams,
                events
            });
        }

        // Compute time range
        let minTime = Infinity;
        let maxTime = 0;
        for (const track of tracks) {
            for (const event of track.events) {
                minTime = Math.min(minTime, event.timeOffset);
                maxTime = Math.max(maxTime, event.timeOffset + event.duration);
            }
        }

        return {
            kernelName,
            gridDims,
            clusterDims,
            formatDescriptors,
            blockDescriptors,
            tracks,
            totalEventCount,
            timeRange: {
                min: minTime,
                max: maxTime,
                duration: maxTime - minTime
            }
        };
    }

    /**
     * Format a label string with placeholders
     */
    formatLabel(template, context) {
        return template
            .replace(/{blockLinear}/g, context.blockId ?? '')
            .replace(/{blockX}/g, context.blockX ?? '')
            .replace(/{blockY}/g, context.blockY ?? '')
            .replace(/{blockZ}/g, context.blockZ ?? '')
            .replace(/{lane}/g, context.laneId ?? '')
            .replace(/{(\d+)}/g, (match, idx) => {
                const i = parseInt(idx);
                return context.params && context.params[i] !== undefined ? context.params[i] : match;
            });
    }
}

// Export for use
window.NanotraceParser = NanotraceParser;
