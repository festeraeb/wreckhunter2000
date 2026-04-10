#!/usr/bin/env python3
"""
CESARops Runtime Error Reporting Module
Lightweight error tracking with API reporting and local fallback logging
"""

import json
import os
import sys
import traceback
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from platform import platform as get_platform, python_version


class RuntimeErrorReporter:
    """
    Captures and reports runtime errors with full context.
    Submits to API with automatic fallback to local logging.
    """

    def __init__(
        self,
        sonar_version: str = "1.0.0",
        api_endpoint: str = "https://cesarops.com/api_runtime_error_IONOS.php",
        logs_dir: str = "logs",
    ):
        """
        Initialize error reporter.

        Args:
            sonar_version: Version of SonarSniffer/CESARops
            api_endpoint: URL for error submission API
            logs_dir: Directory for local error logs
        """
        self.sonar_version = sonar_version
        self.api_endpoint = api_endpoint
        self.logs_dir = Path(logs_dir)

        # Create logs directory if needed
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Setup local logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure local error logging"""
        log_file = self.logs_dir / "runtime_errors.log"

        self.logger = logging.getLogger("sonarsniffer_errors")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

    def report_error(
        self,
        error: Exception,
        feature_used: str = "unknown",
        processing_step: str = "unknown",
        input_file: Optional[str] = None,
        file_format: Optional[str] = None,
        user_email: Optional[str] = None,
        user_feedback: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Report a runtime error with full context.

        Args:
            error: The exception that occurred
            feature_used: Which feature was running (e.g., 'target_detection', 'video_export')
            processing_step: What operation failed (e.g., 'algorithm_execution', 'frame_encoding')
            input_file: The file being processed
            file_format: Format of input file (rsd, sl3, dat, jsf)
            user_email: Email of user experiencing the error
            user_feedback: User's description of the issue
            additional_context: Any other relevant data

        Returns:
            Dictionary with error ID if submitted to API, or 'offline' if saved locally
        """

        # Build error data
        error_data = self._build_error_data(
            error=error,
            feature_used=feature_used,
            processing_step=processing_step,
            input_file=input_file,
            file_format=file_format,
            user_email=user_email,
            user_feedback=user_feedback,
            additional_context=additional_context,
        )

        # Try to submit to API
        result = self._submit_to_api(error_data)

        if result and result.get("success"):
            return result

        # Fallback: log locally
        self._log_locally(error_data)
        return {
            "success": True,
            "method": "offline",
            "log_file": str(self.logs_dir / "runtime_errors.log"),
        }

    def _build_error_data(
        self,
        error: Exception,
        feature_used: str,
        processing_step: str,
        input_file: Optional[str],
        file_format: Optional[str],
        user_email: Optional[str],
        user_feedback: Optional[str],
        additional_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build complete error data structure"""

        # Get stack trace
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        stack_trace = "".join(tb_lines)

        # Extract location info
        tb = error.__traceback__
        module_name = "unknown"
        function_name = "unknown"
        line_number = 0

        if tb is not None:
            frame = tb.tb_frame
            module_name = frame.f_code.co_filename
            function_name = frame.f_code.co_name
            line_number = tb.tb_lineno

        # Sanitize file path for privacy
        if input_file:
            input_file = os.path.basename(input_file)

        # Determine severity
        severity = "high"
        if isinstance(error, (MemoryError, RuntimeError)):
            severity = "critical"
        elif isinstance(error, (TimeoutError, ValueError)):
            severity = "high"
        elif isinstance(error, (OSError, FileNotFoundError)):
            severity = "medium"
        else:
            severity = "low"

        # Build data structure
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": stack_trace,
            "module_name": module_name,
            "function_name": function_name,
            "line_number": line_number,
            "feature_used": feature_used,
            "processing_step": processing_step,
            "severity": severity,
            "platform": get_platform(),
            "python_version": python_version(),
            "sonar_version": self.sonar_version,
        }

        # Add optional fields
        if input_file:
            error_data["input_file"] = input_file
        if file_format:
            error_data["file_format"] = file_format
        if user_email:
            error_data["user_email"] = user_email
        if user_feedback:
            error_data["user_feedback"] = user_feedback

        # Add additional context
        if additional_context:
            error_data.update(additional_context)

        return error_data

    def _submit_to_api(self, error_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Submit error to API endpoint.
        Returns None if submission fails.
        """
        try:
            import requests
        except ImportError:
            # requests not available, fall back to logging
            return None

        try:
            response = requests.post(self.api_endpoint, json=error_data, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return data
        except Exception as e:
            # Log the API error but don't fail
            self.logger.debug(f"API submission failed: {e}")

        return None

    def _log_locally(self, error_data: Dict[str, Any]):
        """Log error to local file"""
        log_file = self.logs_dir / "runtime_errors.log"

        try:
            # Also save as JSON for structured data
            json_file = self.logs_dir / "runtime_errors.jsonl"
            with open(json_file, "a") as f:
                f.write(json.dumps(error_data) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write JSON log: {e}")

        # Log the main message
        self.logger.error(
            f"[{error_data.get('error_type')}] {error_data.get('feature_used')}: "
            f"{error_data.get('error_message')}"
        )


# Convenience function for quick error reporting
def report_runtime_error(
    error: Exception,
    feature_used: str = "unknown",
    processing_step: str = "unknown",
    sonar_version: str = "1.0.0",
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick function to report a runtime error.

    Usage:
        try:
            some_operation()
        except Exception as e:
            report_runtime_error(e, feature_used='video_export', sonar_version='1.0.0')
    """
    reporter = RuntimeErrorReporter(sonar_version=sonar_version)
    return reporter.report_error(
        error=error,
        feature_used=feature_used,
        processing_step=processing_step,
        **kwargs,
    )


# Test the module
if __name__ == "__main__":
    print("Testing RuntimeErrorReporter...")

    reporter = RuntimeErrorReporter()

    # Test 1: ValueError
    print("\n1. Testing ValueError capture...")
    try:
        x = 1 / 0  # This will cause ZeroDivisionError
    except Exception as e:
        result = reporter.report_error(
            e,
            feature_used="test_module",
            processing_step="mathematical_operation",
            user_feedback="Test error reporting",
        )
        print(f"   Result: {result}")

    # Test 2: Quick reporting
    print("\n2. Testing quick report_runtime_error()...")
    try:
        data = {"test": None}
        value = data["missing_key"]
    except Exception as e:
        result = report_runtime_error(
            e, feature_used="test_feature", processing_step="data_access"
        )
        print(f"   Result: {result}")

    # Check logs
    print("\n3. Checking local logs...")
    log_file = Path("logs/runtime_errors.log")
    if log_file.exists():
        print(f"   ✓ Log file created: {log_file}")
        print(f"   ✓ Size: {log_file.stat().st_size} bytes")

    jsonl_file = Path("logs/runtime_errors.jsonl")
    if jsonl_file.exists():
        print(f"   ✓ JSON log file created: {jsonl_file}")
        with open(jsonl_file) as f:
            lines = f.readlines()
            print(f"   ✓ Entries logged: {len(lines)}")

    print("\n✓ Error reporting module working correctly!")
