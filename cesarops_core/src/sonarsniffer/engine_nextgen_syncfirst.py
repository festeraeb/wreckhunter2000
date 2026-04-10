#!/usr/bin/env python3
# engine_nextgen_syncfirst.py — tolerant (warn CRC) + stall watchdog

import os, mmap, struct, csv
from dataclasses import dataclass
from typing import Optional, Iterable, Callable, Tuple, Dict, Any
from .core_shared import (
    MAGIC_REC_HDR,
    MAGIC_REC_TRL,
    _parse_varstruct,
    _mapunit_to_deg,
    _read_varint_from,
    find_magic,
    set_progress_hook,
    _emit,
)


@dataclass
class RSDRecord:
    """Record data from a Garmin RSD file."""

    ofs: int  # File offset where record starts
    channel_id: int  # Channel/transducer identifier
    seq: int  # Sequence number
    time_ms: int  # Timestamp in milliseconds
    lat: float  # Latitude in degrees
    lon: float  # Longitude in degrees
    depth_m: float  # Water depth in meters
    sample_cnt: int  # Number of samples in sonar data
    sonar_ofs: int  # Offset to sonar sample data
    sonar_size: int  # Size of sonar sample data
    beam_deg: float  # Beam angle in degrees
    pitch_deg: float  # Pitch angle in degrees
    roll_deg: float  # Roll angle in degrees
    heave_m: Optional[float]  # Heave in meters
    tx_ofs_m: Optional[float]  # Transmitter offset in meters
    rx_ofs_m: Optional[float]  # Receiver offset in meters
    color_id: Optional[int]  # Color scheme ID
    extras: Dict[str, Any]  # Additional decoded fields


def parse_rsd_records_nextgen(
    path: str, limit_records: int = 0, progress: Callable[[float, str], None] = None
) -> Iterable[RSDRecord]:
    if progress:
        set_progress_hook(progress)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        limit = size
        mbytes = MAGIC_REC_HDR.to_bytes(4, "little")

        _emit(0.0, "Scanning file for sonar data...")
        j = find_magic(mm, mbytes, 0, limit)
        if j < 0:
            _emit(0.0, "No sonar data found in file")
            mm.close()
            return
        pos = j - 1
        _emit(5.0, f"Found sonar data, analyzing structure...")

        count = 0
        last_magic = None
        stuck_hits = 0
        MAX_STUCK = 2

        while pos + 12 < limit:
            k = find_magic(mm, mbytes, max(pos, 0), min(limit, pos + 16 * 1024 * 1024))
            if k < 0:
                break
            pos_magic = k
            # Don't emit technical header messages, just update progress

            if pos_magic == last_magic:
                stuck_hits += 1
                if stuck_hits > MAX_STUCK:
                    pos = pos_magic + 8
                    # Skip corrupt data quietly, don't show as error
                    last_magic = None
                    stuck_hits = 0
                    continue
            else:
                last_magic = pos_magic
                stuck_hits = 0

            # Read header as varstruct
            hdr_block = None
            for back in range(1, 65):
                try:
                    start = pos_magic - back
                    if start < 0:
                        break
                    hdr, body_start = _parse_varstruct(
                        mm, start, limit, crc_mode="warn"
                    )
                    if (
                        struct.unpack("<I", hdr.get(0, b"\x00" * 4)[:4])[0]
                        == MAGIC_REC_HDR
                    ):
                        hdr_block = (hdr, start, body_start)
                        break
                except Exception:
                    pass
            if not hdr_block:
                pos = pos_magic + 4
                # Skip unparseable data quietly
                continue

            hdr, hdr_start, body_start = hdr_block
            seq = struct.unpack("<I", hdr.get(2, b"\x00" * 4)[:4])[0]
            time_ms = struct.unpack("<I", hdr.get(5, b"\x00" * 4)[:4])[0]
            data_sz = struct.unpack("<H", (hdr.get(4, b"\x00\x00")[:2] or b"\x00\x00"))[
                0
            ]

            lat = lon = depth = beam_deg = None
            sample = None
            ch = None
            # Read body as varstruct, but handle failures gracefully
            try:
                body, body_end = _parse_varstruct(
                    mm, body_start, limit, crc_mode="warn"
                )
                used = max(0, body_end - body_start)

                if 0 in body:
                    ch = int.from_bytes(body[0][:4].ljust(4, b"\x00"), "little")
                if 9 in body and len(body[9]) >= 4:
                    lat_raw = int.from_bytes(body[9][:4], "little", signed=True)
                    lat = _mapunit_to_deg(lat_raw)
                if 10 in body and len(body[10]) >= 4:
                    lon_raw = int.from_bytes(body[10][:4], "little", signed=True)
                    lon = _mapunit_to_deg(lon_raw)
                if 1 in body:
                    try:
                        v, _ = _read_varint_from(
                            mm[body_start : body_start + len(body[1])], 0, len(body[1])
                        )
                        depth = v / 1000.0
                    except Exception:
                        pass
                if 7 in body:
                    sample = int.from_bytes(body[7][:4].ljust(4, b"\x00"), "little")
                if 11 in body and len(body[11]) >= 4:
                    beam_deg = struct.unpack("<f", body[11][:4])[0]
            except Exception as e:
                # Body parsing failed, use default values and assume fixed size
                used = 32  # Assume standard metadata size
                ch = seq  # Use seq as channel ID fallback
                lat = lon = depth = beam_deg = None
                sample = 0

            sonar_ofs = body_start + used
            sonar_len = max(0, data_sz - used) if data_sz > 0 else 0

            hop = None
            trailer_pos = body_start + data_sz
            if trailer_pos + 12 <= limit:
                try:
                    tr_magic, chunk_size, tr_crc = struct.unpack(
                        ">III", mm[trailer_pos : trailer_pos + 12]
                    )
                    if tr_magic == MAGIC_REC_TRL and chunk_size > 0:
                        hop = chunk_size
                except Exception:
                    pass

            yield RSDRecord(
                ofs=hdr_start,
                channel_id=ch if ch is not None else 0,
                seq=seq,
                time_ms=time_ms,
                lat=lat if lat is not None else 0.0,
                lon=lon if lon is not None else 0.0,
                depth_m=depth if depth is not None else 0.0,
                sample_cnt=sample if sample is not None else 0,
                sonar_ofs=sonar_ofs if sonar_len > 0 else 0,
                sonar_size=sonar_len if sonar_len > 0 else 0,
                beam_deg=(
                    beam_deg if beam_deg is not None else 0.0
                ),  # Extracted from field 11
                pitch_deg=0.0,  # Not in original data
                roll_deg=0.0,  # Not in original data
                heave_m=None,  # Not in original data
                tx_ofs_m=None,  # Not in original data
                rx_ofs_m=None,  # Not in original data
                color_id=None,  # Not in original data
                extras={},  # Not in original data
            )
            count += 1
            if count % 100 == 0:  # Update progress more frequently
                progress_pct = min(
                    95.0, (trailer_pos / limit) * 100.0
                )  # Cap at 95% until done
                _emit(progress_pct, f"Processing sonar records... ({count} found)")
            if limit_records and count >= limit_records:
                break

            if hop:
                pos = hdr_start + hop
            else:
                nxt = find_magic(
                    mm,
                    mbytes,
                    min(limit, trailer_pos + 2),
                    min(limit, hdr_start + 4 * 1024 * 1024),
                )
                if nxt < 0:
                    break
                pos = nxt - 1

        _emit(100.0, f"✓ Parsing complete! Found {count} sonar records")
        mm.close()


def parse_rsd(
    rsd_path: str, out_dir: str, max_records: Optional[int] = None
) -> Tuple[int, str, str]:
    """Parse RSD file and write records to CSV.
    Returns (record_count, csv_path, log_path).
    """
    # Setup output paths
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(rsd_path))[0]
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    log_path = os.path.join(out_dir, f"{base_name}.log")

    # CSV header - must match what GUI expects
    header = [
        "ofs",
        "channel_id",
        "seq",
        "time_ms",
        "lat",
        "lon",
        "depth_m",
        "sample_cnt",
        "sonar_ofs",
        "sonar_size",
        "beam_deg",
        "pitch_deg",
        "roll_deg",
        "heave_m",
        "tx_ofs_m",
        "rx_ofs_m",
        "color_id",
        "extras_json",
    ]

    record_count = 0

    # Parse records and write to CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        try:
            for record in parse_rsd_records_nextgen(
                rsd_path, limit_records=max_records
            ):
                writer.writerow(
                    [
                        record.ofs,
                        record.channel_id,
                        record.seq,
                        record.time_ms,
                        record.lat,
                        record.lon,
                        record.depth_m,
                        record.sample_cnt,
                        record.sonar_ofs,
                        record.sonar_size,
                        record.beam_deg,
                        record.pitch_deg,
                        record.roll_deg,
                        record.heave_m,
                        record.tx_ofs_m,
                        record.rx_ofs_m,
                        record.color_id,
                        "{}",  # Empty JSON for extras
                    ]
                )
                record_count += 1

        except Exception as e:
            # Write error to log
            with open(log_path, "w") as log_file:
                log_file.write(f"Parse error: {str(e)}\n")
            raise

    return record_count, csv_path, log_path
