#!/usr/bin/env python3
"""
Nanotrace file verification tool.

Parses a .nanotrace file and verifies its structure is valid,
reporting any issues found. Useful for debugging parsing errors.
"""

import argparse
import struct
import sys
from pathlib import Path


def read_uint8(data: bytes, offset: int) -> tuple[int, int]:
    return data[offset], offset + 1


def read_uint16(data: bytes, offset: int) -> tuple[int, int]:
    val = struct.unpack("<H", data[offset : offset + 2])[0]
    return val, offset + 2


def read_uint32(data: bytes, offset: int) -> tuple[int, int]:
    val = struct.unpack("<I", data[offset : offset + 4])[0]
    return val, offset + 4


def read_int32(data: bytes, offset: int) -> tuple[int, int]:
    val = struct.unpack("<i", data[offset : offset + 4])[0]
    return val, offset + 4


def read_uint64(data: bytes, offset: int) -> tuple[int, int]:
    val = struct.unpack("<Q", data[offset : offset + 8])[0]
    return val, offset + 8


def read_string(data: bytes, offset: int) -> tuple[str, int]:
    length, offset = read_uint16(data, offset)
    string = data[offset : offset + length].decode("utf-8")
    return string, offset + length


def verify_nanotrace(filepath: str, verbose: bool = False) -> bool:
    """Verify a nanotrace file and report any issues.

    Returns True if the file is valid, False otherwise.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return False

    data = filepath.read_bytes()
    file_size = len(data)
    print(f"File: {filepath}")
    print(f"Size: {file_size} bytes")
    print()

    offset = 0
    errors = []
    warnings = []

    # --- Header ---
    print("=== Header ===")

    # Magic (10 bytes)
    magic = data[0:10]
    if magic != b"nanotrace\x00":
        errors.append(f"Invalid magic: {magic!r}")
        print(f"  Magic: INVALID ({magic!r})")
    else:
        print("  Magic: OK")
    offset = 10

    # Version (1 byte)
    version, offset = read_uint8(data, offset)
    print(f"  Version: {version}")
    if version != 1:
        warnings.append(f"Unknown version: {version}")

    # Compression (1 byte)
    compressed, offset = read_uint8(data, offset)
    print(f"  Compressed: {'Yes' if compressed else 'No'}")

    if compressed:
        errors.append("Compressed files not supported by this tool")
        print("\nError: Cannot verify compressed files.")
        return False

    payload_start = offset
    print(f"  Payload starts at: {payload_start}")
    print()

    # --- Payload ---
    print("=== Payload ===")

    # Kernel name
    kernel_name, offset = read_string(data, offset)
    print(f"  Kernel name: {kernel_name}")

    # Grid dimensions
    grid_x, offset = read_uint32(data, offset)
    grid_y, offset = read_uint32(data, offset)
    grid_z, offset = read_uint32(data, offset)
    print(f"  Grid: {grid_x} x {grid_y} x {grid_z}")

    # Cluster dimensions
    cluster_x, offset = read_uint32(data, offset)
    cluster_y, offset = read_uint32(data, offset)
    cluster_z, offset = read_uint32(data, offset)
    print(f"  Cluster: {cluster_x} x {cluster_y} x {cluster_z}")

    # Counts
    format_count, offset = read_uint32(data, offset)
    block_count, offset = read_uint32(data, offset)
    track_count, offset = read_uint32(data, offset)
    total_events, offset = read_uint64(data, offset)
    print(f"  Format descriptors: {format_count}")
    print(f"  Block descriptors: {block_count}")
    print(f"  Tracks: {track_count}")
    print(f"  Total events: {total_events}")
    print()

    # --- Format Descriptors ---
    print("=== Format Descriptors ===")
    formats = []
    for i in range(format_count):
        label, offset = read_string(data, offset)
        tooltip, offset = read_string(data, offset)
        param_count, offset = read_uint8(data, offset)
        formats.append(
            {
                "id": i,
                "label": label,
                "tooltip": tooltip,
                "param_count": param_count,
            }
        )
        if verbose:
            print(f'  [{i}] label="{label}", tooltip="{tooltip}", params={param_count}')

    if not verbose:
        print(f"  {format_count} formats parsed (use -v to show details)")
    print()

    # --- Block Descriptors ---
    print("=== Block Descriptors ===")
    blocks = []
    for i in range(block_count):
        block_id, offset = read_uint32(data, offset)
        cluster_id, offset = read_uint32(data, offset)
        sm_id, offset = read_uint16(data, offset)
        format_id, offset = read_uint16(data, offset)

        if format_id >= format_count:
            errors.append(
                f"Block {i}: Invalid format_id {format_id} (max={format_count - 1})"
            )

        blocks.append(
            {
                "block_id": block_id,
                "cluster_id": cluster_id,
                "sm_id": sm_id,
                "format_id": format_id,
            }
        )
        if verbose:
            print(
                f"  [{i}] block_id={block_id}, cluster={cluster_id}, sm={sm_id}, format={format_id}"
            )

    if not verbose:
        print(f"  {block_count} blocks parsed (use -v to show details)")
    print()

    # --- Tracks ---
    print("=== Tracks ===")
    tracks_parsed = 0
    events_parsed = 0

    for i in range(track_count):
        track_start = offset

        # Track header: block_desc_idx (4), track_format_id (2), lane_id (4), params (variable), event_count (4)
        # Minimum is 14 bytes (without params)
        if offset + 14 > file_size:
            errors.append(f"Track {i}: Premature EOF reading header at offset {offset}")
            break

        block_desc_idx, offset = read_uint32(data, offset)
        track_format_id, offset = read_uint16(data, offset)
        lane_id, offset = read_uint32(data, offset)

        if block_desc_idx >= block_count:
            errors.append(
                f"Track {i} @{track_start}: Invalid block_desc_idx {block_desc_idx} (max={block_count - 1})"
            )

        if track_format_id >= format_count:
            errors.append(
                f"Track {i} @{track_start}: Invalid track_format_id {track_format_id} (max={format_count - 1})"
            )
            break  # Can't continue without knowing param count

        # Read track format parameters
        track_param_count = formats[track_format_id]["param_count"]
        for _ in range(track_param_count):
            _, offset = read_uint32(data, offset)

        event_count, offset = read_uint32(data, offset)

        # Parse events
        for j in range(event_count):
            event_start = offset

            if offset + 10 > file_size:
                errors.append(f"Track {i}, Event {j}: Premature EOF at offset {offset}")
                break

            time_offset, offset = read_uint32(data, offset)
            duration, offset = read_uint32(data, offset)
            event_format_id, offset = read_uint16(data, offset)

            if event_format_id >= format_count:
                errors.append(
                    f"Track {i}, Event {j} @{event_start}: Invalid event_format_id {event_format_id} (max={format_count - 1})"
                )
                # Cannot continue - don't know how many params to skip
                print(f"\n  SYNC LOST at Track {i}, Event {j}")
                print(f"  Event format ID {event_format_id} is invalid")
                print(
                    f"  Bytes at offset {event_start}: {data[event_start : event_start + 20].hex()}"
                )
                break

            # Read params based on format's param_count
            param_count = formats[event_format_id]["param_count"]
            for k in range(param_count):
                if offset + 4 > file_size:
                    errors.append(
                        f"Track {i}, Event {j}: Premature EOF reading param {k}"
                    )
                    break
                _, offset = read_uint32(data, offset)

            events_parsed += 1
        else:
            # Only increment if we didn't break out of the event loop
            tracks_parsed += 1
            continue
        break  # Break out of track loop if we broke out of event loop

    if not verbose:
        print(f"  Parsed {tracks_parsed}/{track_count} tracks")
        print(f"  Parsed {events_parsed}/{total_events} events")
    print()

    # Final position check
    if offset != file_size and tracks_parsed == track_count:
        warnings.append(f"Extra {file_size - offset} bytes at end of file")

    # --- Summary ---
    print("=== Summary ===")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for err in errors:
            print(f"    - {err}")
    else:
        print("  No errors found.")

    if warnings:
        print(f"  WARNINGS: {len(warnings)}")
        for warn in warnings:
            print(f"    - {warn}")

    print()

    if errors:
        print("❌ File is INVALID")
        return False
    else:
        print("✓ File appears valid")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Verify the structure of a .nanotrace file"
    )
    parser.add_argument("file", help="Path to the .nanotrace file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    success = verify_nanotrace(args.file, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
