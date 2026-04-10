#!/usr/bin/env python3
"""
SonarSniffer Parser Module

Unified interface for parsing various sonar data formats (RSD, SON, XTF)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Handle both package and direct imports
try:
    from .engine_nextgen_syncfirst import parse_rsd_records_nextgen, RSDRecord
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from engine_nextgen_syncfirst import parse_rsd_records_nextgen, RSDRecord


class SonarParser:
    """
    Unified sonar data parser supporting multiple formats
    """

    def __init__(self):
        self.supported_formats = [".rsd", ".son", ".xtf"]

    def parse_file(
        self, file_path: str, max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Parse a sonar data file

        Args:
            file_path: Path to the sonar data file
            max_records: Maximum number of records to parse (None for all)

        Returns:
            Dictionary containing parsed data and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()

        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}"
            )

        if file_ext == ".rsd":
            return self._parse_rsd_file(str(file_path), max_records)
        elif file_ext == ".son":
            return self._parse_son_file(str(file_path), max_records)
        elif file_ext == ".xtf":
            return self._parse_xtf_file(str(file_path), max_records)
        else:
            raise ValueError(f"No parser available for format: {file_ext}")

    def _parse_rsd_file(
        self, file_path: str, max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """Parse Garmin RSD format file"""
        records = list(
            parse_rsd_records_nextgen(file_path, limit_records=max_records or 0)
        )

        # Extract metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "format": "RSD",
            "total_records_in_file": len(records),  # All records from file
            "file_size": os.path.getsize(file_path),
        }

        # Calculate bounds if records exist
        if records:
            import math

            # Filter for valid coordinates: not 0, not NaN, within valid ranges
            # Valid latitude: -90 to 90, Valid longitude: -180 to 180
            valid_records = [
                r
                for r in records
                if (
                    r.lat != 0.0
                    and not math.isnan(r.lat)
                    and -90 <= r.lat <= 90
                    and r.lon != 0.0
                    and not math.isnan(r.lon)
                    and -180 <= r.lon <= 180
                )
            ]

            # Count valid records for display
            metadata["record_count"] = len(valid_records)

            lats = [r.lat for r in valid_records]
            lons = [r.lon for r in valid_records]

            if lats and lons:
                metadata.update(
                    {
                        "bounds": {
                            "north": max(lats),
                            "south": min(lats),
                            "east": max(lons),
                            "west": min(lons),
                        },
                        "center_lat": sum(lats) / len(lats),
                        "center_lon": sum(lons) / len(lons),
                    }
                )

            depths = [r.depth_m for r in valid_records if r.depth_m > 0]
            if depths:
                metadata.update(
                    {
                        "depth_range": f"{min(depths):.1f}m - {max(depths):.1f}m",
                        "min_depth": min(depths),
                        "max_depth": max(depths),
                    }
                )

        # Convert records to dictionaries for easier processing
        data_records = []
        for record in records:
            data_records.append(
                {
                    "ofs": record.ofs,
                    "channel_id": record.channel_id,
                    "seq": record.seq,
                    "time_ms": record.time_ms,
                    "lat": record.lat,
                    "lon": record.lon,
                    "depth_m": record.depth_m,
                    "sample_cnt": record.sample_cnt,
                    "sonar_ofs": record.sonar_ofs,
                    "sonar_size": record.sonar_size,
                    "beam_deg": record.beam_deg,
                    "pitch_deg": record.pitch_deg,
                    "roll_deg": record.roll_deg,
                    "heave_m": record.heave_m,
                    "tx_ofs_m": record.tx_ofs_m,
                    "rx_ofs_m": record.rx_ofs_m,
                    "color_id": record.color_id,
                    "extras": record.extras,
                }
            )

        return {
            "metadata": metadata,
            "records": data_records,
        }

    def _parse_son_file(
        self, file_path: str, max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """Parse SON format file - placeholder implementation"""
        # TODO: Implement SON parser
        raise NotImplementedError("SON format parsing not yet implemented")

    def _parse_xtf_file(
        self, file_path: str, max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """Parse XTF format file - placeholder implementation"""
        # TODO: Implement XTF parser
        raise NotImplementedError("XTF format parsing not yet implemented")

    def get_supported_formats(self) -> list:
        """Get list of supported file formats"""
        return self.supported_formats.copy()
