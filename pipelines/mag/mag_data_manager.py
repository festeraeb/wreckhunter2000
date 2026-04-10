#!/usr/bin/env python3
"""Magnetic Data Lake Manager — persistent storage with cloud sync.

Manages the magnetic_data/ directory as a permanent local data lake.
Downloads once, stores forever, with optional AWS S3 backup.

Usage:
  # Show cached data status
  python scripts/mag_data_manager.py status

  # Download all sources to persistent cache
  python scripts/mag_data_manager.py download

  # Sync to/from AWS S3 (requires configured credentials)
  python scripts/mag_data_manager.py sync-up
  python scripts/mag_data_manager.py sync-down

  # Check for new data versions
  python scripts/mag_data_manager.py check-updates
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("mag_data_manager")

REPO_ROOT = Path(__file__).resolve().parents[1]
MAG_DATA_DIR = REPO_ROOT / "magnetic_data"
MAG_DATA_RAW = MAG_DATA_DIR / "raw"
MAG_DATA_GRIDS = MAG_DATA_DIR / "grids"
MAG_DATA_META = MAG_DATA_DIR / "meta"

S3_BUCKET = "bagrecovery-mag-data"  # default bucket name — change as needed
S3_PREFIX = "magnetic_data/"


def data_status() -> dict:
    """Return status of all cached data files."""
    status = {"sources": {}, "total_size_mb": 0}

    if not MAG_DATA_RAW.exists():
        return status

    for source_dir in sorted(MAG_DATA_RAW.iterdir()):
        if not source_dir.is_dir():
            continue
        key = source_dir.name
        files = [f for f in source_dir.iterdir() if f.is_file() and not f.name.endswith(".partial")]
        if not files:
            continue

        total_bytes = sum(f.stat().st_size for f in files)
        meta_path = MAG_DATA_META / f"{key}.json"
        meta = {}
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError) as e:
                log.warning("Could not load metadata for %s: %s", key, e)

        status["sources"][key] = {
            "files": [f.name for f in files],
            "size_mb": round(total_bytes / (1024 * 1024), 1),
            "downloaded_at": meta.get("downloaded_at", "unknown"),
            "url": meta.get("url", "unknown"),
        }
        status["total_size_mb"] += total_bytes / (1024 * 1024)

    # Check grids
    if MAG_DATA_GRIDS.exists():
        tifs = list(MAG_DATA_GRIDS.glob("*.tif"))
        status["grids"] = [t.name for t in tifs]
        status["grids_size_mb"] = round(sum(t.stat().st_size for t in tifs) / (1024 * 1024), 1)

    status["total_size_mb"] = round(status["total_size_mb"], 1)
    return status


def check_updates() -> dict:
    """Check remote sources for newer data than what we have cached."""
    import requests

    results = {}
    from scripts.mag_data_pipeline import SOURCES

    for src in SOURCES:
        if not src.urls or src.source_type == "api":
            continue

        cached_dir = MAG_DATA_RAW / src.key
        has_cache = cached_dir.exists() and any(
            f.is_file() and f.stat().st_size > 1_000
            for f in cached_dir.iterdir()
        ) if cached_dir.exists() else False

        remote_info = {}
        for url in src.urls[:1]:  # Check first URL only
            try:
                r = requests.head(url, timeout=15, allow_redirects=True)
                remote_info = {
                    "status": r.status_code,
                    "size": r.headers.get("Content-Length", "unknown"),
                    "last_modified": r.headers.get("Last-Modified", "unknown"),
                }
            except Exception as e:
                remote_info = {"status": "error", "error": str(e)}

        results[src.key] = {
            "name": src.name,
            "cached": has_cache,
            "remote": remote_info,
        }

    return results


def sync_to_s3(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    """Upload local magnetic_data/ to AWS S3."""
    try:
        import boto3
    except ImportError:
        log.error("boto3 not installed. Run: pip install boto3")
        return False

    s3 = boto3.client("s3")
    uploaded = 0

    for local_path in MAG_DATA_DIR.rglob("*"):
        if not local_path.is_file():
            continue
        if local_path.name.endswith(".partial"):
            continue

        rel = local_path.relative_to(MAG_DATA_DIR)
        s3_key = prefix + str(rel).replace("\\", "/")

        # Check if already uploaded (by size)
        try:
            resp = s3.head_object(Bucket=bucket, Key=s3_key)
            if resp["ContentLength"] == local_path.stat().st_size:
                log.info("Already in S3: %s", s3_key)
                continue
        except Exception:
            pass  # Not in S3 yet

        log.info("Uploading %s -> s3://%s/%s", local_path, bucket, s3_key)
        s3.upload_file(str(local_path), bucket, s3_key)
        uploaded += 1

    log.info("Uploaded %d files to S3", uploaded)
    return True


def sync_from_s3(bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
    """Download magnetic_data/ from AWS S3 to local cache."""
    try:
        import boto3
    except ImportError:
        log.error("boto3 not installed. Run: pip install boto3")
        return False

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = s3_key[len(prefix):]
            local_path = MAG_DATA_DIR / rel_path

            if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Downloading s3://%s/%s -> %s", bucket, s3_key, local_path)
            s3.download_file(bucket, s3_key, str(local_path))
            downloaded += 1

    log.info("Downloaded %d files from S3", downloaded)
    return True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Magnetic Data Lake Manager")
    p.add_argument("command", choices=["status", "download", "sync-up", "sync-down", "check-updates"])
    p.add_argument("--bucket", default=S3_BUCKET, help="S3 bucket name")
    p.add_argument("--sources", default="usgs_namag,usgs_usmag,wdmam", help="Sources to download")
    args = p.parse_args()

    if args.command == "status":
        s = data_status()
        print(json.dumps(s, indent=2))

    elif args.command == "download":
        from scripts.mag_data_pipeline import stage_download
        sp = stage_download(
            data_dir=MAG_DATA_DIR,
            source_keys=args.sources.split(","),
            progress_callback=lambda msg: print(f"  {msg}"),
        )
        print(f"\n{sp.message}")
        print(json.dumps(sp.details, indent=2))

    elif args.command == "sync-up":
        sync_to_s3(bucket=args.bucket)

    elif args.command == "sync-down":
        sync_from_s3(bucket=args.bucket)

    elif args.command == "check-updates":
        results = check_updates()
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
