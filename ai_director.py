#!/usr/bin/env python3
"""
AI DIRECTOR — Qwen-Powered Tool Picker, Parameter Tuner & Data Interpreter

What it does:
1. Takes user requests (natural language)
2. Calls Qwen LLM to pick tools, set bounding boxes, tune parameters
3. Executes the tools
4. Sends results back to Qwen for interpretation & next-step recommendations

Usage:
    python ai_director.py --request "Search for Gilcher near Fox Islands"
    python ai_director.py --request "Find triple locks at Beaver Islands" --execute
    python ai_director.py --bbox 45.8,-84.6,46.0,-84.4 --tools thermal,optical --sensitivity 1.5
    python ai_director.py --interpret outputs/probes/*.json
"""

# Fix Windows console encoding (prevents UnicodeEncodeError on cp1252)
import sys, io
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import argparse
import json
import os
import re
import subprocess
import sys
import requests
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# ── API Key Loading ───────────────────────────────────────────────────────────

def load_env(path: Path) -> Dict[str, str]:
    """Parse a simple .env file into a dict."""
    env = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            env[key.strip()] = val.strip()
    return env

_dotenv = load_env(Path(__file__).parent / ".env")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", _dotenv.get("QWEN_API_KEY", ""))
QWEN_MODEL = os.environ.get("QWEN_MODEL", _dotenv.get("QWEN_MODEL", "qwen-plus"))
QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", _dotenv.get("QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1"))

# Gemini (Google AI Studio — free tier: gemini-2.5-flash)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", _dotenv.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", _dotenv.get("GEMINI_MODEL", "gemini-2.5-flash"))
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", _dotenv.get("GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai"))

# Anthropic Claude (sonnet/haiku — free tier via API key)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", _dotenv.get("ANTHROPIC_API_KEY", ""))
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", _dotenv.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", _dotenv.get("ANTHROPIC_BASE_URL",
    "https://api.anthropic.com/v1"))

# Groq Cloud (LPU — free tier: llama-3.3-70b-versatile)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", _dotenv.get("GROQ_API_KEY", ""))
GROQ_MODEL = os.environ.get("GROQ_MODEL", _dotenv.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", _dotenv.get("GROQ_BASE_URL",
    "https://api.groq.com/openai/v1"))

# ── API Key Setup Helper ─────────────────────────────────────────────────────

def check_and_prompt_api_keys() -> Dict[str, str]:
    """
    Check for required API keys and prompt user interactively if missing.
    Returns a dict of found/entered keys. Creates .env file if it doesn't exist.
    """
    env_path = Path(__file__).parent / ".env"
    keys_found = {}
    keys_needed = {}

    # Check existing keys
    if QWEN_API_KEY:
        keys_found['QWEN_API_KEY'] = QWEN_API_KEY[:20] + "..."
    else:
        keys_needed['QWEN_API_KEY'] = {
            'prompt': "Qwen (DashScope) API key — for AI Director parsing & result interpretation",
            'url': "https://dashscope.console.aliyun.com/",
        }

    earthdata = os.environ.get("EARTHDATA_TOKEN", _dotenv.get("EARTHDATA_TOKEN", ""))
    if earthdata:
        keys_found['EARTHDATA_TOKEN'] = earthdata[:20] + "..."
    else:
        keys_needed['EARTHDATA_TOKEN'] = {
            'prompt': "NASA Earthdata token — for SWOT, ICESat-2, and HLS downloads",
            'url': "https://urs.earthdata.nasa.gov/",
        }

    # Report status
    if keys_found:
        print("\n✓ API Keys Found:")
        for key, val in keys_found.items():
            print(f"  {key}: {val}")

    if keys_needed:
        print("\n⚠ Missing API Keys:")
        for key, info in keys_needed.items():
            print(f"  ✗ {key}: {info['prompt']}")
            print(f"    Register at: {info['url']}")

        # Interactive prompt
        print("\n──────────────────────────────────────────────────────────────")
        print("You can enter API keys now, or skip and use --no-llm mode.")
        print("Keys will be saved to .env for future use.")
        print("──────────────────────────────────────────────────────────────\n")

        new_keys = {}
        for key, info in keys_needed.items():
            while True:
                val = input(f"Enter {key} (or press Enter to skip): ").strip()
                if val:
                    new_keys[key] = val
                    break
                elif input("  Skip this key? (y/n): ").strip().lower() == 'n':
                    continue  # Ask again
                else:
                    break  # Skip

        if new_keys:
            # Load existing .env or create new one
            existing = {}
            if env_path.exists():
                existing = load_env(env_path)

            existing.update(new_keys)

            # Write .env file
            lines = []
            for k, v in existing.items():
                lines.append(f"{k}={v}")
            env_path.write_text('\n'.join(lines) + '\n')

            print(f"\n✓ Saved {len(new_keys)} key(s) to {env_path}")

            # Update current session via module reload
            import importlib
            if 'QWEN_API_KEY' in new_keys:
                # Directly update the module-level variable
                import sys
                mod = sys.modules[__name__]
                mod.QWEN_API_KEY = new_keys['QWEN_API_KEY']

            return new_keys

    return {}


# ── Bounding box presets ─────────────────────────────────────────────────────

BOUNDING_BOXES = {
    'fox_islands': {
        'name': 'Fox Islands',
        'lat_min': 45.80, 'lat_max': 46.00,
        'lon_min': -84.60, 'lon_max': -84.40,
        'targets': ['Gilcher'],
    },
    'beaver_islands': {
        'name': 'Beaver Islands',
        'lat_min': 45.60, 'lat_max': 45.80,
        'lon_min': -85.60, 'lon_max': -85.40,
        'targets': ['Parnell'],
    },
    'bridge_builder_x': {
        'name': 'Bridge Builder X Area',
        'lat_min': 45.70, 'lat_max': 45.80,
        'lon_min': -84.70, 'lon_max': -84.50,
        'targets': ['Bridge Builder X'],
    },
    'lake_michigan_south': {
        'name': 'Lake Michigan South (Andaste)',
        'lat_min': 42.30, 'lat_max': 43.20,
        'lon_min': -88.50, 'lon_max': -87.40,
        'targets': ['Andaste', 'Chicorah'],
    },
    'lake_michigan_north': {
        'name': 'Lake Michigan North',
        'lat_min': 43.20, 'lat_max': 45.00,
        'lon_min': -87.50, 'lon_max': -86.00,
        'targets': [],
    },
    'straits_of_mackinac': {
        'name': 'Straits of Mackinac',
        'lat_min': 45.80, 'lat_max': 46.10,
        'lon_min': -84.80, 'lon_max': -84.40,
        'targets': [],
    },
    'lake_erie': {
        'name': 'Lake Erie (full)',
        'lat_min': 41.30, 'lat_max': 42.50,
        'lon_min': -83.50, 'lon_max': -78.80,
        'targets': ['Anthony Wayne', 'Atlantic', 'Dean Richmond'],
    },
    'erie_central_basin': {
        'name': 'Erie Central Basin — Marquette & Bessemer No. 2',
        'lat_min': 41.80, 'lat_max': 42.50,
        'lon_min': -82.50, 'lon_max': -80.00,
        'targets': ['Marquette', 'Bessemer', 'Marquette and Bessemer', 'MB2', 'M&B2'],
    },
    'lake_huron': {
        'name': 'Lake Huron',
        'lat_min': 43.00, 'lat_max': 46.50,
        'lon_min': -84.00, 'lon_max': -79.50,
        'targets': [],
    },
    'lake_ontario': {
        'name': 'Lake Ontario',
        'lat_min': 43.20, 'lat_max': 44.20,
        'lon_min': -79.90, 'lon_max': -76.00,
        'targets': [],
    },
    'lake_superior': {
        'name': 'Lake Superior',
        'lat_min': 46.40, 'lat_max': 48.20,
        'lon_min': -92.00, 'lon_max': -84.40,
        'targets': [],
    },
}

# ── Tool metadata (sent to Qwen as reference) ────────────────────────────────

AVAILABLE_TOOLS = {
    'thermal': {
        'script': 'lake_michigan_scan.py',
        'description': 'Thermal cold-sink detection (Landsat B10/B11)',
        'default_threshold': 2.5,
        'best_for': ['steel masses', 'large vessels', 'engine blocks'],
        'resolution': '100m/pixel',
    },
    'optical': {
        'script': 'lake_michigan_scan.py',
        'description': 'Optical glint detection (Sentinel-2 B04/B08)',
        'default_threshold': 2.0,
        'best_for': ['aluminum', 'aircraft', 'surface debris'],
        'resolution': '10m/pixel',
    },
    'sar': {
        'script': 'lake_michigan_scan.py',
        'description': 'SAR VV/VH ratio (Sentinel-1)',
        'default_threshold': 2.0,
        'best_for': ['heavy steel', 'dense masses', 'submerged structures'],
        'resolution': '20m/pixel',
    },
    'triple_lock': {
        'script': 'triple_lock_fusion.py',
        'description': 'Multi-sensor fusion (thermal + optical + SAR)',
        'default_threshold': 2.5,
        'best_for': ['high confidence targets', 'verification'],
    },
    'vrt_slicer': {
        'script': 'cesarops-slicer (Rust)',
        'description': 'VRT Master Stack — multi-source GeoTIFF slicer with coordinate baking',
        'default_threshold': None,
        'best_for': ['multi-provider alignment', 'Sentinel+Landsat fusion'],
    },
    'swot': {
        'script': 'swot_ssh_extractor.py',
        'description': 'SWOT Ka-band sea surface height displacement',
        'default_threshold': 1.5,
        'best_for': ['large displacement', 'hull shapes'],
    },
    'atl13': {
        'script': 'swot_ssh_extractor.py',
        'description': 'ICESat-2 ATL13 bathymetry',
        'default_threshold': 2.0,
        'best_for': ['depth verification', 'seafloor mapping'],
    },
    # ── Mission-level tools (call cesarops_mission.py) ────────────────────
    'nauticuvs': {
        'script': 'cesarops_mission.py',
        'description': 'NauticUVs multi-scale Laplacian-of-Gaussian blob detector on B02+B10 — hull curvature energy proxy',
        'default_threshold': 3.5,
        'best_for': ['buried hull curvature', 'soft-bottom detection', 'LoG energy peaks'],
        'mission_pass': 'nauticuvs',
        'mission_params': {'nauticuvs': {'enabled': True, 'energy_threshold': 3.5, 'top_n': 50}},
    },
    'hydrocarbon': {
        'script': 'cesarops_mission.py',
        'description': 'HC pass — B11 SWIR dark anomaly + B04 Red bright confirm (oil sheen / fuel slick)',
        'default_threshold': -1.8,
        'best_for': ['oil slicks', 'fuel leaks', 'hydrocarbon seep over wreck site'],
        'mission_pass': 'hydrocarbon',
        'mission_params': {'hydrocarbon': {'enabled': True, 'swir_thresh': -1.8, 'red_thresh': 1.5}},
    },
    'stumpf': {
        'script': 'cesarops_mission.py',
        'description': 'Stumpf log-ratio B02/B03 bathymetric shallow anomaly — hard structures in clear water',
        'default_threshold': 2.0,
        'best_for': ['shallow hull outline', 'clear-water detection', 'structural bottom anomaly'],
        'mission_pass': 'stumpf',
        'mission_params': {'stumpf': {'enabled': True, 'threshold': 2.0}},
    },
    'swir_silt_erasure': {
        'script': 'cesarops_mission.py',
        'description': 'B11/B12 SWIR ratio — reveals ferrous metal beneath silt (M&B2 / Erie specialist)',
        'default_threshold': 2.5,
        'best_for': ['sub-silt wreck detection', 'silted-over steel hull', 'Lake Erie central basin'],
        'mission_pass': 'swir_silt_erasure',
        'mission_params': {'swir_silt_erasure': {'enabled': True, 'threshold': 2.5, 'top_n': 30}},
    },
    'mussel_clearspot': {
        'script': 'cesarops_mission.py',
        'description': 'Positive B02 anomaly in turbid Erie background — Dreissenid mussel colony on wreck',
        'default_threshold': 2.0,
        'best_for': ['Marquette & Bessemer No. 2', 'Erie mussel colonies', 'wreck ecosystem signal'],
        'mission_pass': 'mussel_clearspot',
        'mission_params': {'mussel_clearspot': {'enabled': True, 'threshold': 2.0, 'top_n': 30}},
    },
    'mission_triple_lock_erie': {
        'script': 'cesarops_mission.py',
        'description': 'Triple Lock Lake Erie — HC + Thermal + NauticUVs simultaneously',
        'default_threshold': None,
        'best_for': ['Lake Erie multi-sensor confirmation', 'oil leak detection', 'Erie wreck search'],
        'mission_preset': 'triple_lock_erie',
    },
    'mission_mb2': {
        'script': 'cesarops_mission.py',
        'description': 'Full M&B2 wreck hunt — all 7 passes on Erie central basin',
        'default_threshold': None,
        'best_for': ['Marquette and Bessemer No. 2', 'all-pass Erie central basin scan'],
        'mission_preset': 'mb2_wreck_hunt',
    },
    'mission_straits': {
        'script': 'cesarops_mission.py',
        'description': 'Triple Lock Straits of Mackinac — HC + Thermal + NauticUVs + Stumpf',
        'default_threshold': None,
        'best_for': ['Straits', 'Mackinac', 'Line 5 area', 'Line5 wreck'],
        'mission_preset': 'straits_triple_lock',
    },
}


# ── Qwen LLM client ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the AI Director for CESAROPS — a multi-sensor wreck detection system for the Great Lakes.

IMPORTANT: Your job is to TUNE existing tool parameters. You are NOT allowed to write new code,
create new scripts, or build new tools. Only adjust the knobs on existing tools:
bounding boxes, sensitivity values, z-score thresholds, tile sizes, delegate assignments,
resampling methods, and which tools to run. If you think new code is needed, say so
explicitly and WAIT for human approval before generating any code.

The user gives you a natural language request. You must respond with ONLY a JSON object
(no markdown, no explanation) in this exact schema:

{
  "bbox_name": "<one of the preset names below, or null for custom>",
  "bbox": [lat_min, lon_min, lat_max, lon_max],   // use null if bbox_name is set
  "tools": ["tool_id", ...],                        // list of tool IDs from AVAILABLE_TOOLS
  "sensitivity": <1.0-3.0>,                         // 1.0=aggressive, 2.0=balanced, 3.0=conservative
  "thresholds": {"thermal_zscore": 2.5, ...},       // optional per-sensor threshold overrides
  "reasoning": "<brief explanation of your choices>"
}

Preset bounding boxes:
"""

def _build_system_prompt() -> str:
    from llm_context_injector import build_knowledge_context, build_satellite_context
    prompt = SYSTEM_PROMPT
    for name, bbox in BOUNDING_BOXES.items():
        prompt += (
            f"  {name}: "
            f"[{bbox['lat_min']}, {bbox['lon_min']}, "
            f"{bbox['lat_max']}, {bbox['lon_max']}] "
            f"→ {bbox['name']}\n"
        )
    prompt += "\nAvailable tools:\n"
    for tid, tinfo in AVAILABLE_TOOLS.items():
        prompt += (
            f"  {tid}: {tinfo['description']} "
            f"(best for: {', '.join(tinfo['best_for'])})\n"
        )
    prompt += (
        "\nMission presets (use tool id 'mission_triple_lock_erie', 'mission_mb2', "
        "or 'mission_straits' to run a full multi-pass mission):\n"
        "  mission_triple_lock_erie — HC + Thermal + NauticUVs on full Lake Erie\n"
        "  mission_mb2             — All 7 passes on Erie central basin (M&B2 wreck hunt)\n"
        "  mission_straits         — HC + Thermal + NauticUVs + Stumpf on Straits\n"
        "\nSingle-pass mission tools (run one pass only on any bbox):\n"
        "  nauticuvs       — NauticUVs LoG blob on B02+B10 (hull curvature energy)\n"
        "  hydrocarbon     — B11 SWIR dark + B04 Red bright (oil slick / fuel leak)\n"
        "  stumpf          — B02/B03 log-ratio bathymetric shallow anomaly\n"
        "  swir_silt_erasure — B11/B12 ratio: ferrous metal under silt (Erie specialist)\n"
        "  mussel_clearspot  — Positive B02 bias: Dreissenid mussel colony on wreck\n"
    )
    # Inject knowledge base context
    prompt += "\n" + build_knowledge_context() + "\n"
    prompt += "\n" + build_satellite_context() + "\n"
    prompt += (
        "\nRules:\n"
        "- If the user mentions a known wreck name, pick the matching bbox.\n"
        "- If the user mentions Lake Erie, oil leak, or Line 5, default to 'lake_erie'.\n"
        "- If the user mentions Marquette, Bessemer, M&B2, or central basin, default to 'erie_central_basin'.\n"
        "- If unclear, default to 'lake_michigan_south'.\n"
        "- Always include at least one tool.\n"
        "- For multi-sensor confidence use mission tools (mission_triple_lock_erie etc.) not just 'triple_lock'.\n"
        "- Use 'nauticuvs' specifically when hull curvature, buried structure, or soft-bottom detection is requested.\n"
        "- Use sensitivity 2.0 unless the user says 'aggressive' (1.0) or 'conservative/strict' (3.0).\n"
        "- Use web search to check weather/cloud cover AND wind history for the scan area. "
        "Best scanning is 12-48 hours AFTER a storm with 20-40mph winds — silt plumes "
        "settle over wrecks creating visible signatures. Note wind history and conditions. "
        "SAR works in all weather.\n"
        "- Respond with ONLY valid JSON. No markdown. No extra text.\n"
    )
    return prompt


def call_qwen(messages: List[Dict]) -> str:
    """Send messages to Qwen via DashScope compatible API."""
    if not QWEN_API_KEY:
        raise RuntimeError(
            "QWEN_API_KEY not set. Add it to .env or export QWEN_API_KEY=sk-..."
        )

    url = f"{QWEN_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_gemini(messages: List[Dict]) -> str:
    """Send messages to Google Gemini via OpenAI-compatible API."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")

    url = f"{GEMINI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GEMINI_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_anthropic(messages: List[Dict]) -> str:
    """Send messages to Anthropic Claude via the Messages API."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY not set. Get one at https://console.anthropic.com/")

    url = f"{ANTHROPIC_BASE_URL}/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    # Convert OpenAI-style messages to Anthropic format
    system_text = ""
    user_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text += m["content"] + "\n"
        else:
            user_messages.append({"role": m["role"], "content": m["content"]})
    if not user_messages:
        user_messages = [{"role": "user", "content": "Hello"}]

    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "temperature": 0.3,
        "messages": user_messages,
    }
    if system_text.strip():
        body["system"] = system_text.strip()

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    # Anthropic returns content as a list of blocks
    blocks = data.get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def call_groq(messages: List[Dict]) -> str:
    """Send messages to Groq Cloud via OpenAI-compatible API (LPU fast inference)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Get one at https://console.groq.com/keys")

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ── Multi-provider dispatch ───────────────────────────────────────────────────

# Provider registry: name -> (key_var, call_function)
PROVIDER_REGISTRY = {
    "qwen":      lambda: (QWEN_API_KEY,      call_qwen),
    "gemini":    lambda: (GEMINI_API_KEY,     call_gemini),
    "anthropic": lambda: (ANTHROPIC_API_KEY,  call_anthropic),
    "groq":      lambda: (GROQ_API_KEY,       call_groq),
}

# Fallback order when primary provider fails
PROVIDER_FALLBACK_ORDER = ["qwen", "gemini", "groq", "anthropic"]


def call_llm(messages: List[Dict], preferred_provider: str = None) -> str:
    """
    Call an LLM provider with automatic fallback.
    Tries preferred_provider first, then falls through the fallback chain.
    """
    # Build try-order: preferred first, then remaining from fallback list
    order = []
    if preferred_provider and preferred_provider in PROVIDER_REGISTRY:
        order.append(preferred_provider)
    for p in PROVIDER_FALLBACK_ORDER:
        if p not in order:
            order.append(p)

    last_error = None
    for provider_name in order:
        key, call_fn = PROVIDER_REGISTRY[provider_name]()
        if not key:
            continue  # skip providers without API keys
        try:
            result = call_fn(messages)
            if result:
                if provider_name != order[0]:
                    print(f"  [LLM] Used fallback provider: {provider_name}")
                return result
        except Exception as e:
            last_error = e
            print(f"  [LLM] {provider_name} failed: {e}")
            continue

    if last_error:
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")
    raise RuntimeError(
        "No LLM provider configured. Set at least one API key in .env:\n"
        "  QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY"
    )


def parse_with_qwen(user_request: str, provider: str = None) -> Dict:
    """Use an LLM to parse a natural language request into tool config."""
    system = _build_system_prompt()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request},
    ]

    raw = call_llm(messages, preferred_provider=provider)

    # Extract JSON from response (strip markdown code blocks if present)
    raw = raw.strip()
    if raw.startswith("```"):
        # strip ```json or ```
        first_nl = raw.index("\n")
        raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    parsed = json.loads(raw)
    return parsed


def interpret_results_with_qwen(results: List[Dict], config: Dict, provider: str = None) -> str:
    """Send tool execution results to an LLM for interpretation and next steps."""
    # Build a concise summary of results
    summary_parts = []
    for r in results:
        tool = r.get('tool', 'unknown')
        status = "SUCCESS" if r.get('success') else f"FAILED: {r.get('error', r.get('stderr', '')[:200])}"
        summary_parts.append(f"{tool}: {status}")

        # Extract key lines from stdout
        if r.get('stdout'):
            for line in r['stdout'].split('\n'):
                if any(kw in line.lower() for kw in ['detection', 'anomalies', 'lock', 'complete', 'total', 'fused']):
                    summary_parts.append(f"  -> {line.strip()}")

    summary = "\n".join(summary_parts)

    messages = [
        {"role": "system", "content": (
            "You are a Great Lakes wreck detection analyst AI. "
            "Review the CESAROPS sensor probe results below and provide:\n"
            "1. A brief executive summary of what was found\n"
            "2. Key anomalies worth investigating\n"
            "3. Recommended next steps -- ONLY parameter adjustments on existing tools "
            "(e.g., 'raise thermal_zscore to 3.0', 'narrow bbox to ...', 'add SAR sensor'). "
            "Do NOT suggest writing new code or new tools without explicit human approval.\n"
            "4. Confidence assessment of any detected targets\n"
            "Be concise. Use bullet points."
        )},
        {"role": "user", "content": (
            f"Search config:\n"
            f"  Area: {config.get('bbox', {}).get('name', 'Unknown')}\n"
            f"  Tools: {config.get('tools', [])}\n"
            f"  Sensitivity: {config.get('sensitivity', 2.0)}\n\n"
            f"Results:\n{summary}"
        )},
    ]

    return call_llm(messages, preferred_provider=provider)


# ── AI Director class ────────────────────────────────────────────────────────

class AIDirector:
    """AI Director — multi-provider LLM-powered tool picker, executor, and interpreter."""

    def __init__(self, use_llm: bool = True, provider: str = None):
        self.results = []
        self.use_llm = use_llm
        self.provider = provider  # preferred LLM provider (None = auto-fallback)
        self.config = {
            'bbox': None,
            'tools': [],
            'sensitivity': 2.0,
            'thresholds': {},
        }

    def parse_request(self, request: str) -> Dict:
        """Parse natural language request into tool config."""
        if self.use_llm and any([QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY]):
            try:
                provider_label = self.provider or "auto"
                print(f"  [LLM] Calling provider '{provider_label}' for tool selection...")
                parsed = parse_with_qwen(request, provider=self.provider)
                reasoning = parsed.pop('reasoning', '')
                if reasoning:
                    print(f"  [LLM] Reasoning: {reasoning}")
                return parsed
            except Exception as e:
                print(f"  [LLM] All providers failed ({e}), falling back to keyword matching")

        # Keyword matching fallback
        request_lower = request.lower()

        # ── Bbox routing ───────────────────────────────────────────────────
        bbox_name = None
        # Explicit lake/area keywords first
        _BB_KEYWORDS = [
            (['lake erie', 'erie full', 'erie scan'],            'lake_erie'),
            (['central basin', 'mb2', 'm&b2', 'marquette bessemer',
              'marquette and bessemer', 'bessemer no. 2'],        'erie_central_basin'),
            (['straits', 'mackinac', 'line 5', 'line5'],         'straits_of_mackinac'),
            (['lake michigan south', 'andaste', 'chicorah'],      'lake_michigan_south'),
            (['lake michigan north', 'lake michigan'],            'lake_michigan_north'),
            (['fox islands', 'gilcher'],                          'fox_islands'),
            (['beaver islands', 'parnell'],                       'beaver_islands'),
            (['lake huron', 'huron'],                             'lake_huron'),
            (['lake ontario', 'ontario'],                         'lake_ontario'),
            (['lake superior', 'superior'],                       'lake_superior'),
        ]
        for kws, bname in _BB_KEYWORDS:
            if any(kw in request_lower for kw in kws):
                bbox_name = bname
                break
        # Fallback: scan BOUNDING_BOXES targets
        if not bbox_name:
            for name, bbox in BOUNDING_BOXES.items():
                if any(kw in request_lower for kw in name.lower().split('_')):
                    bbox_name = name
                    break
                if any(t.lower() in request_lower for t in bbox.get('targets', [])):
                    bbox_name = name
                    break

        tools = []
        keyword_map = {
            ('thermal', 'cold', 'heat', 'sink'):                           'thermal',
            ('optical', 'glint', 'aluminum', 'aircraft'):                  'optical',
            ('sar', 'vv', 'vh', 'radar'):                                  'sar',
            ('fusion', 'triple lock', 'triple_lock', 'verify'):            'triple_lock',
            ('swot', 'displacement', 'mass'):                              'swot',
            ('icesat', 'atl13', 'bathy', 'depth'):                         'atl13',
            ('vrt', 'stack', 'multi-source', 'multi source'):              'vrt_slicer',
            ('nauticuvs', 'nautic', 'log blob', 'laplacian', 'curvelet'): 'nauticuvs',
            ('hydrocarbon', 'hc pass', 'oil slick', 'fuel leak', 'swir'):  'hydrocarbon',
            ('stumpf', 'bathymetric', 'log ratio', 'shallow hull'):        'stumpf',
            ('swir silt', 'silt erasure', 'sub-silt', 'subsilt'):          'swir_silt_erasure',
            ('mussel', 'clearspot', 'dreissenid', 'clear spot'):           'mussel_clearspot',
            ('mission erie', 'triple lock erie', 'erie triple'):           'mission_triple_lock_erie',
            ('mission mb2', 'mb2 hunt', 'bessemer hunt'):                  'mission_mb2',
            ('mission straits', 'straits triple'):                         'mission_straits',
        }
        for kws, tool in keyword_map.items():
            if any(kw in request_lower for kw in kws):
                tools.append(tool)

        # ── Smart Erie defaults ────────────────────────────────────────────
        if bbox_name in ('lake_erie', 'erie_central_basin') and not tools:
            tools = ['hydrocarbon', 'thermal', 'nauticuvs']
        elif bbox_name == 'erie_central_basin' and not any(t in tools for t in
                ('swir_silt_erasure', 'mussel_clearspot', 'mission_mb2')):
            tools += ['swir_silt_erasure', 'mussel_clearspot']

        if not tools:
            tools = ['thermal', 'optical']

        sensitivity = 2.0
        if any(w in request_lower for w in ['conservative', 'strict', 'high confidence']):
            sensitivity = 3.0
        elif any(w in request_lower for w in ['aggressive', 'sensitive', 'all', 'wide net']):
            sensitivity = 1.0

        return {'bbox_name': bbox_name, 'tools': tools, 'sensitivity': sensitivity}

    # ── Config setters ───────────────────────────────────────────────────

    def set_bounding_box(self, bbox_name: str = None,
                         lat_min=None, lat_max=None, lon_min=None, lon_max=None):
        if bbox_name and bbox_name in BOUNDING_BOXES:
            self.config['bbox'] = BOUNDING_BOXES[bbox_name]
        elif all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
            self.config['bbox'] = {
                'name': 'Custom',
                'lat_min': lat_min, 'lat_max': lat_max,
                'lon_min': lon_min, 'lon_max': lon_max,
            }
        else:
            self.config['bbox'] = BOUNDING_BOXES['lake_michigan_south']
        print(f"✓ Area: {self.config['bbox']['name']}")

    def set_tools(self, tools: List[str]):
        valid = [t for t in tools if t in AVAILABLE_TOOLS]
        self.config['tools'] = valid
        print(f"✓ Tools: {', '.join(valid)}")

    def set_parameters(self, sensitivity: float = None, thresholds: Dict = None):
        if sensitivity is not None:
            self.config['sensitivity'] = sensitivity
        if thresholds:
            self.config['thresholds'] = thresholds
        print(f"✓ Sensitivity: {self.config['sensitivity']}")

    # ── Execution ──────────────────────────────────────────────────────────

    def run_tool(self, tool_name: str) -> Dict:
        tool = AVAILABLE_TOOLS[tool_name]
        script_path = Path(__file__).parent / tool['script']
        sensor_threshold = tool.get('default_threshold', 2.0)

        # ── Mission-level tools route through cesarops_mission.py ─────────
        if 'mission_preset' in tool or 'mission_pass' in tool:
            return self._run_mission_tool(tool_name, tool)

        if not script_path.exists():
            return {'tool': tool_name, 'success': False,
                    'error': f'Script not found: {script_path}'}

        # Build per-sensor output dir
        out_dir = Path(__file__).parent / 'outputs' / 'probes' / tool_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [sys.executable, str(script_path), '--sensor', tool_name,
               '--zscore', str(sensor_threshold),
               '--output', str(out_dir)]

        # Pass .env variables to subprocess so scan scripts can find data
        import copy as _copy
        child_env = _copy.copy(dict(os.environ))
        for k, v in _dotenv.items():
            child_env[k] = v

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600,
                env=child_env,
            )
            return {
                'tool': tool_name,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {'tool': tool_name, 'success': False,
                    'error': 'Timed out (10 min)'}
        except Exception as e:
            return {'tool': tool_name, 'success': False, 'error': str(e)}

    def _run_mission_tool(self, tool_name: str, tool: dict) -> dict:
        """Dispatch a tool that maps to cesarops_mission.py."""
        mission_script = Path(__file__).parent / 'cesarops_mission.py'
        if not mission_script.exists():
            return {'tool': tool_name, 'success': False, 'error': 'cesarops_mission.py not found'}

        bbox = self.config.get('bbox') or {}
        bbox_arr = [
            bbox.get('lat_min', 41.30), bbox.get('lon_min', -83.50),
            bbox.get('lat_max', 42.50), bbox.get('lon_max', -78.80),
        ]
        sensitivity = self.config.get('sensitivity', 2.0)
        # Map sensitivity → thresholds: 1.0 aggressive = lower sigmas, 3.0 = higher
        sigma_scale = sensitivity  # 2.0 = default

        if 'mission_preset' in tool:
            # Use a named built-in preset
            mission = {'preset': tool['mission_preset']}
            cmd = [sys.executable, str(mission_script),
                   '--preset', tool['mission_preset']]
        else:
            # Build a custom single-pass mission JSON
            pass_params = dict(tool.get('mission_params', {}))
            # Apply sensitivity scaling to thresholds
            for pname, pcfg in pass_params.items():
                if 'threshold' in pcfg:
                    pcfg['threshold'] = round(float(pcfg['threshold']) * (sigma_scale / 2.0), 2)
                if 'energy_threshold' in pcfg:
                    pcfg['energy_threshold'] = round(float(pcfg['energy_threshold']) * (sigma_scale / 2.0), 2)

            # All other passes disabled
            all_passes = {
                'standard':          {'enabled': False, 'threshold': 1.5},
                'hydrocarbon':       {'enabled': False, 'swir_thresh': -1.8, 'red_thresh': 1.5},
                'thermal':           {'enabled': False, 'threshold': 2.0},
                'stumpf':            {'enabled': False, 'threshold': 2.0},
                'nauticuvs':         {'enabled': False, 'energy_threshold': 3.5, 'top_n': 50},
                'swir_silt_erasure': {'enabled': False, 'threshold': 2.5, 'top_n': 30},
                'mussel_clearspot':  {'enabled': False, 'threshold': 2.0, 'top_n': 30},
            }
            all_passes.update(pass_params)

            data_dirs = ['downloads/erie', 'downloads/hls']
            if bbox_arr[0] > 44.0:  # Northern lakes
                data_dirs = ['downloads/michigan', 'downloads/straits', 'downloads/hls']

            output_tag = re.sub(r'[^\w\-]', '_',
                f"{tool_name}_{bbox.get('name','area').lower().replace(' ','_')}")

            mission_json = json.dumps({
                'name': f"{tool['description'][:60]}",
                'bbox': bbox_arr,
                'output_tag': output_tag,
                'data_dirs': data_dirs,
                'passes': all_passes,
                'sub_zones': [],
            })
            cmd = [sys.executable, str(mission_script), '--mission-json', mission_json]

        import copy as _copy
        child_env = _copy.copy(dict(os.environ))
        for k, v in _dotenv.items():
            child_env[k] = v

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1200, env=child_env,
            )
            return {
                'tool': tool_name,
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {'tool': tool_name, 'success': False, 'error': 'Timed out (20 min)'}
        except Exception as e:
            return {'tool': tool_name, 'success': False, 'error': str(e)}

    def execute(self) -> List[Dict]:
        print(f"\n{'='*70}")
        print(f"AI DIRECTOR — {len(self.config['tools'])} tools")
        print(f"  Area: {self.config['bbox']['name']}")
        print(f"  Sensitivity: {self.config['sensitivity']}")
        print(f"  Tools: {', '.join(self.config['tools'])}")
        print(f"{'='*70}")

        results = []
        for tool_name in self.config['tools']:
            print(f"\n[{len(results)+1}/{len(self.config['tools'])}] {tool_name}...")
            result = self.run_tool(tool_name)
            results.append(result)
            status = "✓" if result.get('success') else "✗"
            print(f"  {status} {tool_name}")

        self.results = results
        return results

    def summarize(self) -> str:
        lines = [f"\n{'='*70}", "EXECUTION SUMMARY", f"{'='*70}"]
        for r in self.results:
            tool = r.get('tool', '?')
            status = "✓ SUCCESS" if r.get('success') else f"✗ FAILED: {r.get('error', '')}"
            lines.append(f"\n  {tool}: {status}")
            if r.get('stdout'):
                for line in r['stdout'].split('\n'):
                    if any(kw in line.lower() for kw in ['detection', 'anomalies', 'lock', 'complete', 'total', 'fused', 'triple']):
                        lines.append(f"    {line.strip()}")
        return '\n'.join(lines)

    # ── Interpretation ─────────────────────────────────────────────────────

    def interpret(self) -> str:
        """Send results to LLM for interpretation."""
        has_any_key = any([QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY])
        if not has_any_key:
            print("  [LLM] No API keys set, skipping interpretation")
            return self.summarize()

        try:
            provider_label = self.provider or "auto"
            print(f"\n  [LLM] Sending results to '{provider_label}' for interpretation...")
            briefing = interpret_results_with_qwen(self.results, self.config, provider=self.provider)
            return briefing
        except Exception as e:
            print(f"  [LLM] Interpretation failed: {e}")
            return self.summarize()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='AI Director — Qwen-Powered CESAROPS Orchestrator')
    parser.add_argument('--request', '-r', type=str, help='Natural language request')
    parser.add_argument('--bbox', type=str, help='Bounding box: lat_min,lon_min,lat_max,lon_max or preset name')
    parser.add_argument('--tools', type=str, help='Comma-separated tool list')
    parser.add_argument('--sensitivity', type=float, default=2.0, help='1.0=aggressive, 3.0=conservative')
    parser.add_argument('--list-tools', '-l', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Step 1: Generate a plan (JSON) without running.')
    parser.add_argument('--execute-plan', action='store_true', help='Step 2: Execute a saved plan (No LLM timeout).')
    parser.add_argument('--execute', '-x', action='store_true')
    parser.add_argument('--interpret', '-i', type=str, nargs='*', help='Step 3: Interpret existing results.')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM, use keyword matching')
    parser.add_argument('--provider', '-p', type=str,
                        choices=['qwen', 'gemini', 'anthropic', 'groq'],
                        help='Preferred LLM provider (default: auto-fallback)')
    parser.add_argument('--output', '-o', type=str, help='Save results to JSON')
    parser.add_argument('--setup-keys', action='store_true', help='Interactively configure API keys')
    parser.add_argument('--data-sources', action='store_true', help='List available satellite data sources')
    args = parser.parse_args()

    # ── Setup API keys ─────────────────────────────────────────────────────
    if args.setup_keys:
        check_and_prompt_api_keys()
        return

    # ── List data sources ──────────────────────────────────────────────────
    if args.data_sources:
        src_path = Path(__file__).parent / "satellite_data_sources.json"
        if src_path.exists():
            data = json.loads(src_path.read_text())
            print(f"\n{'='*100}")
            print("SATELLITE DATA SOURCES")
            print(f"{'='*100}")
            for src_id, src in data['satellite_data_sources'].items():
                print(f"\n  {src_id}")
                print(f"    Name: {src['name']}")
                print(f"    {src['description']}")
                print(f"    URL: {src['url']}")
                print(f"    Auth: {'API key required' if src['requires_api_key'] else 'Open access'}")
                if 'env_key' in src:
                    print(f"    Env var: {src['env_key']}")
                if 'notes' in src:
                    print(f"    Notes: {src['notes']}")
            print(f"\n{'='*100}")
        else:
            print("satellite_data_sources.json not found")
        return

    if not any([QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY]):
        print("-- No LLM API keys set -- keyword matching only")
        print("  Run: python ai_director.py --setup-keys")
        print("  Or set in .env: QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY")

    # ── Interpret mode ─────────────────────────────────────────────────────
    if args.interpret is not None:
        results_data = []
        for fp in args.interpret:
            p = Path(fp)
            if p.exists():
                results_data.append(json.loads(p.read_text(encoding='utf-8')))
            else:
                print(f"  ⚠ Not found: {fp}")

        if results_data and any([QWEN_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY]):
            messages = [
                {"role": "system", "content": (
                    "You are a Great Lakes wreck detection analyst AI. "
                    "Review these CESAROPS sensor results and provide:\n"
                    "1. Executive summary\n"
                    "2. Key anomalies to investigate\n"
                    "3. Recommended next steps\n"
                    "Be concise, use bullet points."
                )},
                {"role": "user", "content": json.dumps(results_data, indent=2)[:4000]},
            ]
            print(call_llm(messages, preferred_provider=getattr(args, 'provider', None)))
        elif results_data:
            print("  No LLM API keys set, cannot interpret")
        return

    # ── List tools ─────────────────────────────────────────────────────────
    if args.list_tools:
        print(f"\n{'='*100}")
        print("AVAILABLE TOOLS")
        print(f"{'='*100}")
        for tid, t in AVAILABLE_TOOLS.items():
            print(f"\n  {tid}")
            print(f"    {t['description']}")
            print(f"    Best for: {', '.join(t['best_for'])}")
        print(f"\n{'='*100}")
        return

    # ── PLAN MODE (LLM takes its time to think) ────────────────────────────
    if args.plan and args.request:
        print(f"\nRequest: {args.request}")
        director = AIDirector(use_llm=not args.no_llm, provider=getattr(args, 'provider', None))
        plan = director.parse_request(args.request)
        
        plan_path = Path(__file__).parent / "outputs" / "current_plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
        
        print(f"\n✅ PLAN SAVED to: {plan_path}")
        print(json.dumps(plan, indent=2))
        return

    # ── EXECUTE PLAN MODE (No LLM involved, just pure execution) ───────────
    if args.execute_plan:
        plan_path = Path(__file__).parent / "outputs" / "current_plan.json"
        if not plan_path.exists():
            print(f"❌ No plan found at {plan_path}. Run --plan first.")
            return
        
        plan = json.loads(plan_path.read_text(encoding='utf-8'))
        print(f"\n🚀 EXECUTING PLAN...")
        
        director = AIDirector(use_llm=False)
        
        # Set up the director from the plan
        if plan.get('bbox_name'):
            director.set_bounding_box(plan['bbox_name'])
        elif plan.get('bbox'):
            b = plan['bbox']
            director.set_bounding_box(lat_min=b[0], lon_min=b[1], lat_max=b[2], lon_max=b[3])
        
        director.set_tools(plan.get('tools', []))
        director.set_parameters(sensitivity=plan.get('sensitivity', 2.0))
        
        # Run it
        results = director.execute()
        
        # Save results
        out_path = Path(__file__).parent / "outputs" / "probes" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'plan': plan,
            'results': results,
            'summary': director.summarize(),
        }, indent=2), encoding='utf-8')
        print(f"\n✅ RESULTS SAVED to: {out_path}")
        return

    # ── Run ────────────────────────────────────────────────────────────────
    director = AIDirector(use_llm=not args.no_llm, provider=getattr(args, 'provider', None))

    if args.request:
        print(f"\nRequest: {args.request}")
        parsed = director.parse_request(args.request)

        if parsed.get('bbox_name'):
            director.set_bounding_box(parsed['bbox_name'])
        elif parsed.get('bbox'):
            b = parsed['bbox']
            director.set_bounding_box(lat_min=b[0], lon_min=b[1], lat_max=b[2], lon_max=b[3])

        director.set_tools(parsed.get('tools', []))
        director.set_parameters(
            sensitivity=parsed.get('sensitivity', args.sensitivity),
            thresholds=parsed.get('thresholds'),
        )
    else:
        # Manual config
        if args.bbox:
            if args.bbox in BOUNDING_BOXES:
                director.set_bounding_box(args.bbox)
            else:
                try:
                    parts = [float(x) for x in args.bbox.split(',')]
                    director.set_bounding_box(lat_min=parts[0], lon_min=parts[1],
                                              lat_max=parts[2], lon_max=parts[3])
                except Exception:
                    print(f"Invalid bbox: {args.bbox}")
        if args.tools:
            director.set_tools(args.tools.split(','))
        director.set_parameters(sensitivity=args.sensitivity)

    if args.execute or args.request:
        results = director.execute()

        # ── Triple Lock Fusion ────────────────────────────────────────
        fusion_path = Path(__file__).parent / 'triple_lock_fusion.py'
        if fusion_path.exists() and len(results) >= 2:
            print(f"\n{'='*70}")
            print("TRIPLE LOCK FUSION")
            print(f"{'='*70}")
            # Find per-sensor output files
            sensor_files = {}
            probes_dir = Path(__file__).parent / 'outputs' / 'probes'
            for sensor in ['thermal', 'optical', 'sar']:
                sensor_dir = probes_dir / sensor
                if sensor_dir.exists():
                    jsons = sorted(sensor_dir.glob('*.json'))
                    if jsons:
                        sensor_files[sensor] = jsons[-1]

            if len(sensor_files) >= 2:
                fusion_cmd = [sys.executable, str(fusion_path)]
                for sensor, fpath in sensor_files.items():
                    fusion_cmd += [f'--{sensor}', str(fpath)]
                fusion_out = Path(__file__).parent / 'outputs' / 'probes' / 'latest_fusion.json'
                fusion_cmd += ['--radius', '1000', '--output', str(fusion_out)]

                try:
                    fusion_result = subprocess.run(
                        fusion_cmd, capture_output=True, text=True,
                        env=child_env, timeout=60
                    )
                    if fusion_result.returncode == 0:
                        print(fusion_result.stdout)
                        # Load and show fusion results
                        if fusion_out.exists():
                            fusion_data = json.loads(fusion_out.read_text(encoding='utf-8'))
                            clusters = fusion_data.get('clusters', [])
                            if clusters:
                                print(f"\n  Targets found: {len(clusters)}")
                                for i, c in enumerate(clusters[:10], 1):
                                    print(f"    [{i}] {c['lock_type']} | "
                                          f"lat={c['lat']:.4f} lon={c['lon']:.4f} | "
                                          f"z={c['max_zscore']:.1f} | "
                                          f"sensors={', '.join(c['sensors'])}")
                            else:
                                print("\n  No multi-sensor lock targets found.")
                    else:
                        print(f"  Fusion stderr: {fusion_result.stderr[:200]}")
                except Exception as e:
                    print(f"  Fusion failed: {e}")

        # Interpretation
        print(f"\n{'='*70}")
        print("QWEN INTERPRETATION")
        print(f"{'='*70}")
        briefing = director.interpret()
        print(briefing)

        # Save
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'config': director.config,
                'results': results,
                'summary': director.summarize(),
                'interpretation': briefing,
            }, indent=2), encoding='utf-8')
            print(f"\n✓ Saved: {out}")


if __name__ == '__main__':
    main()
