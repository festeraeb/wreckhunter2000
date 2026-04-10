"""
Incremental Loading Module for CESAROPS
Memory-efficient streaming and chunked data processing patterns
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Iterator, Tuple, List, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)


class StreamingDataLoader:
    """Load large NetCDF files in chunks to minimize memory"""

    def __init__(self, file_path: str, chunk_size: int = 100):
        """
        Args:
            file_path: Path to NetCDF file
            chunk_size: Number of time steps per chunk
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        self.ds = None

    def __enter__(self):
        """Context manager entry"""
        self.ds = xr.open_dataset(self.file_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.ds:
            self.ds.close()

    def iter_time_chunks(self, var_name: str) -> Iterator[np.ndarray]:
        """
        Iterate through time dimension in chunks

        Yields:
            Chunk of data with shape (chunk_size, lat, lon)
        """
        if self.ds is None:
            raise RuntimeError("Use with context manager")

        var = self.ds[var_name]
        time_dim = var.shape[0]

        for i in range(0, time_dim, self.chunk_size):
            end = min(i + self.chunk_size, time_dim)
            chunk = var.isel(time=slice(i, end)).values
            yield chunk

    def iter_spatial_chunks(
        self, var_name: str, lat_chunk: int = 64, lon_chunk: int = 64
    ) -> Iterator[Tuple[np.ndarray, dict]]:
        """
        Iterate through spatial dimensions in chunks

        Yields:
            Tuple of (data_chunk, bounds_dict)
        """
        if self.ds is None:
            raise RuntimeError("Use with context manager")

        var = self.ds[var_name]
        lats = self.ds.coords.get("lat", np.arange(var.shape[1]))
        lons = self.ds.coords.get("lon", np.arange(var.shape[2]))

        lat_dim = len(lats)
        lon_dim = len(lons)

        for lat_i in range(0, lat_dim, lat_chunk):
            for lon_j in range(0, lon_dim, lon_chunk):
                lat_end = min(lat_i + lat_chunk, lat_dim)
                lon_end = min(lon_j + lon_chunk, lon_dim)

                chunk = var.isel(
                    lat=slice(lat_i, lat_end), lon=slice(lon_j, lon_end)
                ).values

                bounds = {
                    "lat_start": float(lats[lat_i]),
                    "lat_end": float(lats[lat_end - 1]),
                    "lon_start": float(lons[lon_j]),
                    "lon_end": float(lons[lon_end - 1]),
                    "lat_indices": (lat_i, lat_end),
                    "lon_indices": (lon_j, lon_end),
                }

                yield chunk, bounds


class SonarDataStreamer:
    """Stream large sonar data files without loading all in memory"""

    def __init__(self, file_path: str, buffer_size: int = 1024 * 1024):
        """
        Args:
            file_path: Path to sonar data file
            buffer_size: Read buffer size in bytes
        """
        self.file_path = Path(file_path)
        self.buffer_size = buffer_size

    def stream_binary_chunks(self) -> Iterator[bytes]:
        """
        Stream binary file in chunks

        Yields:
            Buffer of binary data
        """
        try:
            with open(self.file_path, "rb") as f:
                while True:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break
                    yield chunk
        except Exception as e:
            logger.error(f"Failed to stream {self.file_path}: {e}")

    def iter_lines(self, encoding: str = "utf-8") -> Iterator[str]:
        """
        Stream text file line by line

        Yields:
            Decoded line
        """
        buffer = ""
        try:
            for chunk in self.stream_binary_chunks():
                buffer += chunk.decode(encoding, errors="ignore")
                lines = buffer.split("\n")
                buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    yield line

            if buffer:
                yield buffer
        except Exception as e:
            logger.error(f"Line iteration failed: {e}")


class ChunkedDataProcessor:
    """Process data in chunks with optional caching"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: Optional cache directory for processed chunks
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    def process_chunks(
        self,
        data_source: Iterator[np.ndarray],
        processor_func,
        cache_key: Optional[str] = None,
    ) -> Iterator[Any]:
        """
        Process data chunks with optional caching

        Args:
            data_source: Iterator yielding data chunks
            processor_func: Function to process each chunk
            cache_key: Optional key for caching results

        Yields:
            Processed chunk results
        """
        for i, chunk in enumerate(data_source):
            # Check cache
            if cache_key and self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}_chunk_{i}.npy"
                if cache_file.exists():
                    try:
                        result = np.load(cache_file)
                        logger.debug(f"Loaded chunk {i} from cache")
                        yield result
                        continue
                    except Exception as e:
                        logger.warning(f"Cache load failed: {e}")

            # Process chunk
            try:
                result = processor_func(chunk)

                # Save to cache
                if cache_key and self.cache_dir:
                    cache_file = self.cache_dir / f"{cache_key}_chunk_{i}.npy"
                    np.save(cache_file, result)

                yield result

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")

    def aggregate_chunks(self, chunks: Iterator[np.ndarray]) -> np.ndarray:
        """
        Concatenate chunks along first dimension

        Args:
            chunks: Iterator of arrays to concatenate

        Returns:
            Aggregated array
        """
        chunk_list = list(chunks)
        if not chunk_list:
            return np.array([])
        return np.concatenate(chunk_list, axis=0)


class ProgressiveDataLoader:
    """Load data progressively with progress tracking"""

    def __init__(self, data_source: Iterator[Any]):
        """
        Args:
            data_source: Iterator of data items
        """
        self.data_source = data_source
        self.loaded_count = 0
        self.total_size = 0

    def iter_with_progress(self) -> Iterator[Tuple[Any, dict]]:
        """
        Iterate data with progress information

        Yields:
            Tuple of (data_item, progress_dict)
        """
        for item in self.data_source:
            self.loaded_count += 1

            # Estimate size
            if isinstance(item, np.ndarray):
                item_size = item.nbytes
            elif isinstance(item, dict):
                item_size = len(json.dumps(item))
            else:
                item_size = 0

            self.total_size += item_size

            progress = {
                "items_loaded": self.loaded_count,
                "total_memory_mb": self.total_size / (1024 * 1024),
                "timestamp": np.datetime64("now"),
            }

            yield item, progress


class MemoryEfficientAnalyzer:
    """Analyze large datasets without loading all data"""

    def __init__(self):
        self.stats = {
            "min": np.inf,
            "max": -np.inf,
            "sum": 0.0,
            "count": 0,
            "mean": 0.0,
        }

    def update_statistics(self, chunk: np.ndarray) -> dict:
        """
        Update running statistics with new chunk

        Args:
            chunk: Data chunk

        Returns:
            Current statistics
        """
        self.stats["min"] = min(self.stats["min"], np.min(chunk))
        self.stats["max"] = max(self.stats["max"], np.max(chunk))

        chunk_sum = np.sum(chunk)
        chunk_count = chunk.size

        # Update running mean
        old_mean = self.stats["mean"]
        old_count = self.stats["count"]

        self.stats["count"] += chunk_count
        self.stats["sum"] += chunk_sum
        self.stats["mean"] = self.stats["sum"] / self.stats["count"]

        return dict(self.stats)

    def get_statistics(self) -> dict:
        """Get current statistics"""
        return dict(self.stats)


def load_large_netcdf_progressively(
    file_path: str, var_name: str, chunk_size: int = 100
) -> Iterator[Tuple[np.ndarray, dict]]:
    """
    Convenience function to load NetCDF file progressively

    Args:
        file_path: Path to NetCDF file
        var_name: Variable name to load
        chunk_size: Chunks size

    Yields:
        Tuple of (data_chunk, metadata)
    """
    with StreamingDataLoader(file_path, chunk_size) as loader:
        for i, chunk in enumerate(loader.iter_time_chunks(var_name)):
            metadata = {
                "chunk_index": i,
                "chunk_shape": chunk.shape,
                "timestamp": np.datetime64("now"),
            }
            yield chunk, metadata


def process_sonar_incrementally(file_path: str, processor_func) -> Iterator[Any]:
    """
    Convenience function to process sonar file incrementally

    Args:
        file_path: Path to sonar file
        processor_func: Function to process each line/record

    Yields:
        Processed results
    """
    streamer = SonarDataStreamer(file_path)
    processor = ChunkedDataProcessor()

    for result in processor.process_chunks(
        streamer.stream_binary_chunks(), processor_func
    ):
        yield result
