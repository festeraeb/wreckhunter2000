from __future__ import annotations

import time
from pathlib import Path
import sys

# Ensure pipeline source directories are importable
_repo_root = Path(__file__).resolve().parents[2]
for _sub in ("pipelines/bag", "pipelines/mag", "scripts"):
    _p = str(_repo_root / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _collect_pdf_files(paths: list, explicit_pdf_paths: list | None = None) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()

    def add_pdf(p: Path):
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        files.append(p)

    for raw in explicit_pdf_paths or []:
        p = Path(str(raw)).expanduser()
        if p.is_file() and p.suffix.lower() == ".pdf":
            add_pdf(p)
        elif p.is_dir():
            for pdf in p.rglob("*.pdf"):
                add_pdf(pdf)

    if files:
        return files

    for raw in paths:
        p = Path(str(raw)).expanduser()
        if p.is_file() and p.suffix.lower() == ".pdf":
            add_pdf(p)
            continue
        if p.is_dir():
            for pdf in p.rglob("*.pdf"):
                add_pdf(pdf)

    return files


def run_pdf_breaker_stage(paths: list, output_dir: str, cfg: dict) -> dict:
    if not cfg.get("run_pdf_breaker", True):
        return {"enabled": False, "status": "skipped", "reason": "disabled_by_config"}

    from bag_processor.pdf_redaction_breaker import PDFRedactionBreaker

    explicit = cfg.get("pdf_paths") if isinstance(cfg.get("pdf_paths"), list) else None
    pdf_files = _collect_pdf_files(paths, explicit)
    if not pdf_files:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no_pdf_files_found",
            "pdfs_analyzed": 0,
        }

    stage_dir = Path(output_dir) / "pipeline" / "pdf_breaker"
    stage_dir.mkdir(parents=True, exist_ok=True)

    breaker = PDFRedactionBreaker(
        targets=cfg.get("pdf_targets") if isinstance(cfg.get("pdf_targets"), list) else None,
        save_images=bool(cfg.get("pdf_save_images", False)),
        output_dir=str(stage_dir),
        skip_ocr=bool(cfg.get("pdf_skip_ocr", True)),
    )

    results = [breaker.analyze(str(p)) for p in pdf_files]
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = stage_dir / f"results_{ts}.json"
    csv_path = stage_dir / f"findings_{ts}.csv"
    breaker.write_json(results, str(json_path))
    breaker.write_csv(results, str(csv_path))

    total_findings = sum(len(r.get("findings", [])) for r in results)
    total_redactions = sum(len(r.get("redaction_zones", [])) for r in results)
    return {
        "enabled": True,
        "status": "completed",
        "pdfs_analyzed": len(results),
        "target_hits": total_findings,
        "redaction_zones": total_redactions,
        "output_json": str(json_path),
        "output_csv": str(csv_path),
    }
