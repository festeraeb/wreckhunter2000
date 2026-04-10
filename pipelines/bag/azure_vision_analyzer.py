"""
Azure AI Vision analyzer for BAG depth restoration images.

Uses Azure Computer Vision (free student tier) to detect patterns in
restored depth grid visualizations that may not be visible to the human eye.

All analysis is on BAG-derived depth images — no PDF dependency.
PDFs may be used as optional cross-reference elsewhere but are not
required or expected here.

Environment variables required:
  AZURE_VISION_ENDPOINT  — e.g. https://<region>.cognitiveservices.azure.com
  AZURE_VISION_KEY       — API key from Azure portal
"""

import os
import json
import logging
import time as _time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default endpoint (Wreckhunter 2000 Computer Vision F0 free tier) ─────────
DEFAULT_ENDPOINT = "https://wreckhunter2000.cognitiveservices.azure.com/"

# ── Free tier (F0) rate limits ────────────────────────────────────────────────
# Azure CV F0: 20 transactions per minute, 5,000 per month
RATE_LIMIT_PER_MINUTE = 20
RATE_LIMIT_PER_MONTH = 5000
_call_timestamps: list[float] = []   # timestamps of recent calls
_monthly_calls = 0                   # incremented per session (resets on restart)

# Lazy import to avoid hard dependency
_vision_client = None
_runtime_key: Optional[str] = None       # set via set_key() from the API
_runtime_endpoint: Optional[str] = None  # set via set_key() from the API


def set_key(key: str, endpoint: str | None = None) -> None:
    """Set the Azure Vision API key at runtime (from the GUI)."""
    global _vision_client, _runtime_key, _runtime_endpoint
    _runtime_key = key
    _runtime_endpoint = endpoint or DEFAULT_ENDPOINT
    _vision_client = None  # force re-creation with new creds
    # Also push into env so child processes inherit
    os.environ["AZURE_VISION_KEY"] = key
    os.environ["AZURE_VISION_ENDPOINT"] = _runtime_endpoint
    # Persist to .env file at repo root so it survives restarts
    _persist_key(key, _runtime_endpoint)


def _persist_key(key: str, endpoint: str) -> None:
    """Save key to .azure_vision_key in the repo root."""
    key_file = Path(__file__).resolve().parents[1] / ".azure_vision_key"
    key_file.write_text(json.dumps({"endpoint": endpoint, "key": key}), encoding="utf-8")
    logger.info("Azure Vision key saved to %s", key_file)


def _load_persisted_key() -> tuple[Optional[str], Optional[str]]:
    """Try to load a previously saved key."""
    key_file = Path(__file__).resolve().parents[1] / ".azure_vision_key"
    if key_file.exists():
        try:
            data = json.loads(key_file.read_text(encoding="utf-8"))
            return data.get("key"), data.get("endpoint")
        except Exception:
            pass
    return None, None


def _resolve_credentials() -> tuple[str, str]:
    """Resolve endpoint + key from runtime → env → persisted file."""
    endpoint = _runtime_endpoint or os.environ.get("AZURE_VISION_ENDPOINT")
    key = _runtime_key or os.environ.get("AZURE_VISION_KEY")
    if not key:
        pk, pe = _load_persisted_key()
        key = key or pk
        endpoint = endpoint or pe
    endpoint = endpoint or DEFAULT_ENDPOINT
    return endpoint, key or ""


def _check_rate_limit() -> None:
    """Enforce F0 free tier rate limits."""
    global _monthly_calls
    now = _time.time()
    # Prune timestamps older than 60 s
    cutoff = now - 60.0
    while _call_timestamps and _call_timestamps[0] < cutoff:
        _call_timestamps.pop(0)
    if len(_call_timestamps) >= RATE_LIMIT_PER_MINUTE:
        wait = _call_timestamps[0] + 60.0 - now
        raise RuntimeError(
            f"Azure Vision F0 rate limit: {RATE_LIMIT_PER_MINUTE} calls/min. "
            f"Wait {wait:.0f}s before retrying."
        )
    if _monthly_calls >= RATE_LIMIT_PER_MONTH:
        raise RuntimeError(
            f"Azure Vision F0 monthly limit reached ({RATE_LIMIT_PER_MONTH} calls). "
            "Upgrade the tier in Azure Portal or wait until next billing cycle."
        )


def _record_call() -> None:
    global _monthly_calls
    _call_timestamps.append(_time.time())
    _monthly_calls += 1


def _get_client():
    """Create or return a cached Azure Computer Vision client."""
    global _vision_client
    if _vision_client is not None:
        return _vision_client

    endpoint, key = _resolve_credentials()

    if not key:
        raise EnvironmentError(
            "Azure Vision API key not set. Enter it in the Restoration panel "
            "or set AZURE_VISION_KEY environment variable."
        )

    try:
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        from msrest.authentication import CognitiveServicesCredentials
    except ImportError:
        raise ImportError(
            "Install azure-cognitiveservices-vision-computervision and msrest: "
            "pip install azure-cognitiveservices-vision-computervision msrest"
        )

    _vision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))
    return _vision_client


def analyze_depth_image(
    image_path: str,
    features: list = None,
) -> dict:
    """Send a restored depth visualization image to Azure AI Vision.

    Parameters
    ----------
    image_path : str
        Path to a PNG/JPEG of the restored depth grid visualization.
    features : list, optional
        Azure vision features to request.  Defaults to a broad set.

    Returns
    -------
    dict with keys:
        description, tags, objects, colors, metadata, raw_response
    """
    client = _get_client()

    from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes

    if features is None:
        features = [
            VisualFeatureTypes.description,
            VisualFeatureTypes.tags,
            VisualFeatureTypes.objects,
            VisualFeatureTypes.color,
            VisualFeatureTypes.image_type,
        ]

    _check_rate_limit()
    with open(image_path, "rb") as f:
        result = client.analyze_image_in_stream(f, visual_features=features)
    _record_call()

    parsed = {
        "image_path": image_path,
        "description": (
            result.description.captions[0].text
            if result.description and result.description.captions
            else None
        ),
        "description_confidence": (
            result.description.captions[0].confidence
            if result.description and result.description.captions
            else None
        ),
        "tags": [
            {"name": t.name, "confidence": t.confidence}
            for t in (result.tags or [])
        ],
        "objects": [
            {
                "name": obj.object_property,
                "confidence": obj.confidence,
                "rectangle": {
                    "x": obj.rectangle.x,
                    "y": obj.rectangle.y,
                    "w": obj.rectangle.w,
                    "h": obj.rectangle.h,
                },
            }
            for obj in (result.objects or [])
        ],
        "dominant_colors": (
            result.color.dominant_colors if result.color else []
        ),
        "is_bw": result.color.is_bw_img if result.color else None,
    }

    return parsed


def analyze_restoration_set(
    output_dir: str,
    bag_stem: str,
) -> dict:
    """Analyze all restoration result PNGs for a given BAG file.

    Looks for files matching {bag_stem}_*.png in output_dir.

    Returns
    -------
    dict mapping technique name → Azure vision analysis result
    """
    results = {}
    out_path = Path(output_dir)

    for png in sorted(out_path.glob(f"{bag_stem}_*.png")):
        technique = png.stem.replace(f"{bag_stem}_", "")
        logger.info(f"Analyzing {technique}: {png}")
        try:
            analysis = analyze_depth_image(str(png))
            results[technique] = analysis
        except EnvironmentError:
            logger.warning("Azure Vision not configured — skipping AI analysis")
            break
        except Exception as e:
            logger.error(f"Azure Vision failed for {png}: {e}")
            results[technique] = {"error": str(e)}

    return results


def check_available() -> dict:
    """Check if Azure AI Vision is configured and reachable."""
    endpoint, key = _resolve_credentials()

    status = {
        "configured": bool(key),
        "endpoint": endpoint,
        "sdk_installed": False,
        "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        "rate_limit_per_month": RATE_LIMIT_PER_MONTH,
        "calls_this_session": _monthly_calls,
        "calls_last_minute": len(_call_timestamps),
    }

    try:
        from azure.cognitiveservices.vision.computervision import ComputerVisionClient
        status["sdk_installed"] = True
    except ImportError:
        pass

    return status
