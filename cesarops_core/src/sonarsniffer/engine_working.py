#!/usr/bin/env python3
"""
Working RSD parser that doesn't hang - replacement for engine_classic_varstruct.py
Based on signature-aware approach from your documentation
"""

import mmap
import struct
from dataclasses import dataclass
from typing import Iterator, Optional, Dict, Any, List
from .core_shared import find_magic, _mapunit_to_deg


@dataclass
class RSDRecord:
    ofs: int = 0
    channel_id: int = 0
    seq: int = 0
    time_ms: int = 0
    lat: float = 0.0
    lon: float = 0.0
    depth_m: float = 0.0
    sample_cnt: int = 0
    sonar_ofs: int = 0
    sonar_size: int = 0
    beam_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    heave_m: float = 0.0
    tx_ofs_m: float = 0.0
    rx_ofs_m: float = 0.0
    color_id: int = 0
    extras: Dict[str, Any] = None

    def __post_init__(self):
        if self.extras is None:
            self.extras = {}


RECORD_MAGIC = 0xB7E9DA86


def parse_rsd_records_classic(
    rsd_path: str, start_ofs: int = 0x5000, limit_records: Optional[int] = None
) -> Iterator[RSDRecord]:
    """
    Replacement parser that doesn't hang.
    Uses header scanning and heuristic coordinate extraction.
    """

    with open(rsd_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:

            # Find record headers
            magic_bytes = struct.pack("<I", RECORD_MAGIC)
            headers = []

            current_pos = start_ofs
            while current_pos < len(mm) - 16:
                magic_pos = find_magic(mm, magic_bytes, current_pos, len(mm))
                if magic_pos is None:
                    break
                headers.append(magic_pos)
                current_pos = magic_pos + 4

                # Limit header search for performance
                if len(headers) > 20000:
                    break

            if not headers:
                return

            # Process each record
            record_count = 0
            for i, header_pos in enumerate(headers):
                if limit_records and record_count >= limit_records:
                    break

                # Determine record size
                if i + 1 < len(headers):
                    record_size = headers[i + 1] - header_pos
                else:
                    record_size = min(len(mm) - header_pos, 0x10000)

                if record_size < 16 or record_size > 0x100000:
                    continue

                try:
                    record = parse_record_heuristic(
                        mm, header_pos, record_size, record_count
                    )
                    if record:
                        yield record
                        record_count += 1
                except Exception:
                    continue


def parse_record_heuristic(
    mm: mmap.mmap, header_pos: int, record_size: int, seq: int
) -> Optional[RSDRecord]:
    """Parse a single record using heuristic approach with coordinate clustering"""

    record = RSDRecord(ofs=header_pos, seq=seq)

    # Skip the 4-byte magic header
    data_start = header_pos + 4
    data_end = header_pos + record_size

    if data_end > len(mm):
        return None

    # Skip A1/B2 pad region (around +160 to +200)
    search_start = max(data_start + 8, data_start + 200)

    # Collect all potential coordinates from this record
    potential_coords = []

    for offset in range(search_start, data_end - 8, 4):
        try:
            val1 = struct.unpack("<i", mm[offset : offset + 4])[0]
            val2 = struct.unpack("<i", mm[offset + 4 : offset + 8])[0]

            lat = _mapunit_to_deg(val1)
            lon = _mapunit_to_deg(val2)

            # Check for reasonable North American coordinates (Great Lakes region)
            if (40.0 <= lat <= 55.0) and (-100.0 <= lon <= -80.0):
                potential_coords.append((lat, lon, offset))

        except Exception:
            continue

    lat_found = False
    best_offset = -1

    if potential_coords:
        # Cluster coordinates: find the densest cluster within 0.001° (tight threshold)
        # This filters out scattered false positives and finds the real survey area
        best_cluster = []

        for i, (lat_i, lon_i, ofs_i) in enumerate(potential_coords):
            # Count how many other coordinates are within 0.001° of this one
            cluster = [(lat_i, lon_i, ofs_i)]

            for lat_j, lon_j, ofs_j in potential_coords[i + 1 :]:
                # Use tight clustering threshold (0.001°) to identify real survey tracks
                if abs(lat_i - lat_j) < 0.001 and abs(lon_i - lon_j) < 0.001:
                    cluster.append((lat_j, lon_j, ofs_j))

            # Keep the cluster with most points (most likely real survey data)
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        if best_cluster:
            # Use the mean of the cluster as the record coordinates
            mean_lat = sum(c[0] for c in best_cluster) / len(best_cluster)
            mean_lon = sum(c[1] for c in best_cluster) / len(best_cluster)
            best_offset = best_cluster[0][2]  # Use first offset for depth search

            record.lat = mean_lat
            record.lon = mean_lon
            lat_found = True

    if lat_found:
        # Look for depth nearby the coordinates
        depth_search_start = max(data_start, best_offset - 24)
        depth_search_end = min(data_end - 4, best_offset + 32)

        for depth_offset in range(depth_search_start, depth_search_end, 4):
            try:
                depth_val = struct.unpack("<i", mm[depth_offset : depth_offset + 4])[0]
                # Check if it's reasonable depth in mm (0-500m for lakes)
                if 0 <= depth_val <= 500000:
                    record.depth_m = depth_val / 1000.0
                    break
            except Exception:
                continue

    # Try to find channel_id and other basic fields
    for offset in range(data_start, min(data_start + 64, data_end - 4), 4):
        try:
            val = struct.unpack("<I", mm[offset : offset + 4])[0]
            if 0 <= val <= 255:  # Reasonable channel ID
                if record.channel_id == 0:  # Only set if not already set
                    record.channel_id = val
                    break
        except Exception:
            continue

    # Set sequence number as time placeholder
    record.time_ms = seq * 1000  # Approximate timing

    # Only return record if we found coordinates
    return record if lat_found else None


def parse_rsd(rsd_path: str, csv_out: str, limit_rows: Optional[int] = None):
    """Simple CSV export function matching the original interface"""

    records = []

    for record in parse_rsd_records_classic(rsd_path, limit_records=limit_rows):
        records.append(record)

    # Write CSV
    with open(csv_out, "w", newline="") as csvf:
        csvf.write(
            "ofs,channel_id,seq,time_ms,lat,lon,depth_m,sample_cnt,sonar_ofs,sonar_size,beam_deg,pitch_deg,roll_deg,heave_m,tx_ofs_m,rx_ofs_m,color_id,extras_json\n"
        )

        for record in records:
            import json

            extras_json = json.dumps(record.extras) if record.extras else "{}"

            csvf.write(
                f'{record.ofs},{record.channel_id},{record.seq},{record.time_ms},{record.lat:.8f},{record.lon:.8f},{record.depth_m:.3f},{record.sample_cnt},{record.sonar_ofs},{record.sonar_size},{record.beam_deg:.2f},{record.pitch_deg:.2f},{record.roll_deg:.2f},{record.heave_m:.3f},{record.tx_ofs_m:.3f},{record.rx_ofs_m:.3f},{record.color_id},"{extras_json}"\n'
            )

    print(f"Wrote {len(records)} records to {csv_out}")
    return csv_out


if __name__ == "__main__":
    # Test the parser
    csv_out = parse_rsd("126SV-UHD2-GT54.RSD", "test_records.csv", limit_rows=50)

    # Show some sample results
    with open(csv_out, "r") as f:
        lines = f.readlines()
        print(f"\nSample results ({len(lines)-1} records):")
        for i, line in enumerate(lines[:6]):
            print(f"Line {i}: {line.strip()}")
