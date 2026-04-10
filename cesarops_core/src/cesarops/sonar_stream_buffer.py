"""
Sonar Stream Buffer Module (Phase 12.1)

High-performance circular buffer for real-time sonar data streaming.
Provides thread-safe queuing, buffering, and performance monitoring.

Author: CESARops Development
Date: January 27, 2026
Version: 1.0.0
"""

import queue
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import numpy as np


@dataclass
class SonarPing:
    """Container for single sonar ping data."""
    ping_id: int
    timestamp: float
    frequency: float  # Hz
    intensity: np.ndarray  # Array of intensity values
    range_resolution: float  # meters
    num_beams: int = 256
    
    def __post_init__(self):
        """Validate ping data."""
        if not isinstance(self.intensity, np.ndarray):
            self.intensity = np.array(self.intensity)
        if self.intensity.ndim != 1:
            raise ValueError(f"Intensity must be 1D array, got shape {self.intensity.shape}")
        if len(self.intensity) != self.num_beams:
            raise ValueError(f"Expected {self.num_beams} beams, got {len(self.intensity)}")


@dataclass
class StreamStatistics:
    """Statistics for stream performance."""
    total_pings: int = 0
    total_frames: int = 0
    buffer_utilization: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    current_throughput: float = 0.0  # pings per second
    pings_dropped: int = 0
    uptime_seconds: float = 0.0
    fps: float = 0.0


class SonarStreamBuffer:
    """
    High-performance circular buffer for sonar data streaming.
    
    Features:
    - Thread-safe queue management
    - Configurable circular buffer capacity
    - Performance statistics
    - Latency tracking
    - Overflow handling with circular overwrite
    
    Usage:
        buffer = SonarStreamBuffer(capacity=10000)
        buffer.push(ping)
        data = buffer.pop(count=100)
    """
    
    def __init__(self, capacity: int = 10000, max_ping_age_seconds: float = 5.0):
        """
        Initialize sonar stream buffer.
        
        Args:
            capacity: Maximum number of pings to buffer
            max_ping_age_seconds: Maximum age of ping before discard
        """
        self.capacity = capacity
        self.max_ping_age = max_ping_age_seconds
        
        # Thread-safe queue for incoming data
        self._input_queue = queue.Queue(maxsize=capacity)
        
        # Circular buffer storage
        self._buffer = deque(maxlen=capacity)
        
        # Timing
        self._start_time = time.time()
        self._last_pop_time = self._start_time
        
        # Performance tracking
        self._latencies = deque(maxlen=100)  # Last 100 latencies
        self._push_times = deque(maxlen=1000)  # Last 1000 push times
        self._ping_count = 0
        self._frame_count = 0
        self._pings_dropped = 0
        
        # Thread safety
        self._lock = threading.RLock()
        self._worker_thread = None
        self._running = False
    
    def push(self, ping: SonarPing) -> bool:
        """
        Push ping to buffer (thread-safe).
        
        Args:
            ping: SonarPing object to buffer
            
        Returns:
            True if successfully added, False if buffer full
        """
        if not isinstance(ping, SonarPing):
            raise TypeError(f"Expected SonarPing, got {type(ping)}")
        
        try:
            self._input_queue.put(ping, block=False)
            return True
        except queue.Full:
            with self._lock:
                self._pings_dropped += 1
            return False
    
    def pop(self, count: int = 1) -> List[SonarPing]:
        """
        Retrieve data from buffer (FIFO).
        
        Args:
            count: Number of pings to retrieve
            
        Returns:
            List of SonarPing objects (may be < count if buffer depleted)
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        
        with self._lock:
            # Process any queued items
            while not self._input_queue.empty():
                try:
                    ping = self._input_queue.get_nowait()
                    self._buffer.append(ping)
                    
                    # Record latency (ensure non-negative)
                    latency = (time.time() - ping.timestamp) * 1000
                    if latency >= 0:  # Only record non-negative latencies
                        self._latencies.append(latency)
                    
                    self._ping_count += 1
                except queue.Empty:
                    break
            
            # Return requested pings
            result = []
            for _ in range(min(count, len(self._buffer))):
                result.append(self._buffer.popleft())
            
            self._last_pop_time = time.time()
            return result
    
    def peek(self, count: int = 1, skip_old: bool = True) -> List[SonarPing]:
        """
        View buffer contents without removing.
        
        Args:
            count: Number of pings to view
            skip_old: Skip pings older than max_ping_age
            
        Returns:
            List of SonarPing objects
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        
        with self._lock:
            # Process queued items
            while not self._input_queue.empty():
                try:
                    ping = self._input_queue.get_nowait()
                    self._buffer.append(ping)
                except queue.Empty:
                    break
            
            current_time = time.time()
            result = []
            
            for ping in list(self._buffer):
                if skip_old:
                    age = current_time - ping.timestamp
                    if age > self.max_ping_age:
                        continue
                
                result.append(ping)
                if len(result) >= count:
                    break
            
            return result
    
    def get_latest(self, count: int = 1) -> List[SonarPing]:
        """
        Get latest N pings from buffer.
        
        Args:
            count: Number of recent pings to retrieve
            
        Returns:
            List of most recent SonarPing objects
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        
        with self._lock:
            # Process queued items
            while not self._input_queue.empty():
                try:
                    ping = self._input_queue.get_nowait()
                    self._buffer.append(ping)
                except queue.Empty:
                    break
            
            # Get from right (most recent)
            buffer_list = list(self._buffer)
            return buffer_list[-count:] if buffer_list else []
    
    def clear(self) -> int:
        """
        Clear buffer contents.
        
        Returns:
            Number of pings cleared
        """
        with self._lock:
            # First process queued items
            while not self._input_queue.empty():
                try:
                    ping = self._input_queue.get_nowait()
                    self._buffer.append(ping)
                except queue.Empty:
                    break
            
            count = len(self._buffer)
            self._buffer.clear()
            
            return count
    
    def get_stats(self) -> StreamStatistics:
        """
        Get buffer performance statistics.
        
        Returns:
            StreamStatistics object with current metrics
        """
        with self._lock:
            current_time = time.time()
            uptime = current_time - self._start_time
            
            # Calculate FPS
            fps = self._ping_count / uptime if uptime > 0 else 0.0
            
            # Calculate averages
            avg_latency = np.mean(list(self._latencies)) if self._latencies else 0.0
            max_latency = np.max(list(self._latencies)) if self._latencies else 0.0
            
            # Calculate throughput (pings per second)
            if len(self._push_times) > 1:
                time_span = self._push_times[-1] - self._push_times[0]
                throughput = len(self._push_times) / time_span if time_span > 0 else 0.0
            else:
                throughput = 0.0
            
            # Buffer utilization
            utilization = len(self._buffer) / self.capacity * 100 if self.capacity > 0 else 0.0
            
            return StreamStatistics(
                total_pings=self._ping_count,
                total_frames=self._frame_count,
                buffer_utilization=utilization,
                avg_latency_ms=avg_latency,
                max_latency_ms=max_latency,
                current_throughput=throughput,
                pings_dropped=self._pings_dropped,
                uptime_seconds=uptime,
                fps=fps
            )
    
    def get_buffer_size(self) -> int:
        """Get current number of pings in buffer."""
        with self._lock:
            return len(self._buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        with self._lock:
            return len(self._buffer) >= self.capacity
    
    def mark_frame(self) -> None:
        """Mark end of frame for statistics."""
        with self._lock:
            self._frame_count += 1
            self._push_times.append(time.time())
    
    def reset_statistics(self) -> None:
        """Reset all statistics (keeps buffer contents)."""
        with self._lock:
            self._start_time = time.time()
            self._ping_count = 0
            self._frame_count = 0
            self._pings_dropped = 0
            self._latencies.clear()
            self._push_times.clear()


class DataStreamManager:
    """
    Manages multiple sonar data streams.
    
    Features:
    - Register multiple data sources
    - Synchronized processing
    - Frame aggregation
    - Performance monitoring across sources
    
    Usage:
        manager = DataStreamManager()
        manager.register_source("sonar1", buffer1)
        manager.register_source("sonar2", buffer2)
        frame = manager.get_latest_frame()
    """
    
    def __init__(self):
        """Initialize stream manager."""
        self._sources: Dict[str, SonarStreamBuffer] = {}
        self._lock = threading.RLock()
        self._frame_id = 0
        self._last_frame_time = time.time()
    
    def register_source(self, source_name: str, buffer: SonarStreamBuffer) -> None:
        """
        Register a data source.
        
        Args:
            source_name: Unique identifier for source
            buffer: SonarStreamBuffer instance
        """
        if not isinstance(buffer, SonarStreamBuffer):
            raise TypeError(f"Expected SonarStreamBuffer, got {type(buffer)}")
        
        with self._lock:
            self._sources[source_name] = buffer
    
    def process_stream(self, source_name: str, data: List[SonarPing]) -> None:
        """
        Process incoming data from a source.
        
        Args:
            source_name: Source identifier
            data: List of SonarPing objects
        """
        if source_name not in self._sources:
            raise KeyError(f"Unknown source: {source_name}")
        
        buffer = self._sources[source_name]
        for ping in data:
            buffer.push(ping)
    
    def get_latest_frame(self, pings_per_source: int = 1) -> Dict[str, List[SonarPing]]:
        """
        Get combined latest data from all sources.
        
        Args:
            pings_per_source: Number of pings from each source
            
        Returns:
            Dict mapping source_name to list of SonarPing objects
        """
        with self._lock:
            frame = {}
            for source_name, buffer in self._sources.items():
                frame[source_name] = buffer.pop(count=pings_per_source)
            
            self._frame_id += 1
            self._last_frame_time = time.time()
            
            return frame
    
    def get_source_stats(self, source_name: Optional[str] = None) -> Dict[str, StreamStatistics]:
        """
        Get statistics from sources.
        
        Args:
            source_name: Specific source, or None for all
            
        Returns:
            Dict mapping source names to StreamStatistics
        """
        with self._lock:
            if source_name:
                if source_name not in self._sources:
                    raise KeyError(f"Unknown source: {source_name}")
                return {source_name: self._sources[source_name].get_stats()}
            else:
                return {name: buf.get_stats() for name, buf in self._sources.items()}
    
    def list_sources(self) -> List[str]:
        """Get list of registered sources."""
        with self._lock:
            return list(self._sources.keys())


class StreamProcessor:
    """
    Process sonar data streams using Phase 11 modules.
    
    Features:
    - Batch processing support
    - Phase 11 module integration
    - Incremental updates
    - Error resilience
    
    Usage:
        processor = StreamProcessor(detector, mapper)
        results = processor.process_ping(ping)
    """
    
    def __init__(self, anomaly_detector=None, bathymetry_mapper=None):
        """
        Initialize processor with Phase 11 modules.
        
        Args:
            anomaly_detector: SonarAnomalyDetector instance (optional)
            bathymetry_mapper: BathymetryMapper instance (optional)
        """
        self.anomaly_detector = anomaly_detector
        self.bathymetry_mapper = bathymetry_mapper
        
        self._lock = threading.RLock()
        self._processed_count = 0
        self._error_count = 0
    
    def process_ping(self, ping: SonarPing) -> Dict[str, Any]:
        """
        Process single sonar ping.
        
        Args:
            ping: SonarPing object
            
        Returns:
            Dict with analysis results
        """
        with self._lock:
            try:
                results = {
                    'ping_id': ping.ping_id,
                    'timestamp': ping.timestamp,
                    'anomalies': None,
                    'bathymetry': None,
                    'errors': []
                }
                
                # Anomaly detection
                if self.anomaly_detector is not None:
                    try:
                        anomalies = self.anomaly_detector.detect_anomalies(
                            ping.intensity.reshape(1, -1)
                        )
                        results['anomalies'] = anomalies
                    except Exception as e:
                        results['errors'].append(f"Anomaly detection: {str(e)}")
                
                self._processed_count += 1
                return results
                
            except Exception as e:
                self._error_count += 1
                return {
                    'ping_id': ping.ping_id,
                    'timestamp': ping.timestamp,
                    'error': str(e)
                }
    
    def process_frame(self, pings: List[SonarPing]) -> Dict[str, Any]:
        """
        Process frame of multiple pings.
        
        Args:
            pings: List of SonarPing objects
            
        Returns:
            Dict with aggregated analysis results
        """
        with self._lock:
            results = {
                'frame_id': len(pings),
                'timestamp': time.time(),
                'pings_processed': len(pings),
                'ping_results': [],
                'errors': []
            }
            
            for ping in pings:
                result = self.process_ping(ping)
                results['ping_results'].append(result)
            
            return results
    
    def get_processed_data(self) -> Dict[str, int]:
        """Get processor statistics."""
        with self._lock:
            return {
                'total_processed': self._processed_count,
                'total_errors': self._error_count
            }


class PerformanceMonitor:
    """
    Track system performance metrics.
    
    Features:
    - Latency tracking
    - Throughput measurement
    - FPS calculation
    - Historical statistics
    
    Usage:
        monitor = PerformanceMonitor()
        monitor.record_latency(12.5)
        metrics = monitor.get_metrics()
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of measurements to keep
        """
        self.history_size = history_size
        self._latencies = deque(maxlen=history_size)
        self._throughputs = deque(maxlen=history_size)
        self._frame_times = deque(maxlen=history_size)
        self._start_time = time.time()
        self._lock = threading.RLock()
    
    def record_latency(self, latency_ms: float) -> None:
        """
        Record latency measurement.
        
        Args:
            latency_ms: Latency in milliseconds
        """
        if latency_ms < 0:
            raise ValueError(f"Latency must be non-negative, got {latency_ms}")
        
        with self._lock:
            self._latencies.append(latency_ms)
    
    def record_throughput(self, count: int, duration_seconds: float) -> None:
        """
        Record throughput measurement.
        
        Args:
            count: Number of items processed
            duration_seconds: Time elapsed
        """
        if duration_seconds <= 0:
            raise ValueError(f"Duration must be positive, got {duration_seconds}")
        
        with self._lock:
            throughput = count / duration_seconds
            self._throughputs.append(throughput)
    
    def record_frame_time(self, frame_time_ms: float) -> None:
        """
        Record frame rendering time.
        
        Args:
            frame_time_ms: Frame time in milliseconds
        """
        if frame_time_ms < 0:
            raise ValueError(f"Frame time must be non-negative, got {frame_time_ms}")
        
        with self._lock:
            self._frame_times.append(frame_time_ms)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Return performance statistics.
        
        Returns:
            Dict with latency, throughput, FPS metrics
        """
        with self._lock:
            metrics = {
                'latency_avg_ms': 0.0,
                'latency_max_ms': 0.0,
                'latency_min_ms': 0.0,
                'throughput_avg': 0.0,
                'fps': 0.0,
                'uptime_seconds': time.time() - self._start_time
            }
            
            if self._latencies:
                metrics['latency_avg_ms'] = float(np.mean(list(self._latencies)))
                metrics['latency_max_ms'] = float(np.max(list(self._latencies)))
                metrics['latency_min_ms'] = float(np.min(list(self._latencies)))
            
            if self._throughputs:
                metrics['throughput_avg'] = float(np.mean(list(self._throughputs)))
            
            if self._frame_times:
                metrics['fps'] = 1000.0 / float(np.mean(list(self._frame_times)))
            
            return metrics
    
    def reset(self) -> None:
        """Reset all measurements."""
        with self._lock:
            self._latencies.clear()
            self._throughputs.clear()
            self._frame_times.clear()
            self._start_time = time.time()
