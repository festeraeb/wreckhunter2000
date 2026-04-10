import zipfile
import json
import pathlib
import datetime
import re

root=pathlib.Path(r"C:\Users\thomf\programming\Bagrecovery")
new_dir=root/"magnetic_data"/"new data to digest"
raw_dir=root/"magnetic_data"/"raw"
meta_dir=root/"magnetic_data"/"meta"
raw_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

ingested=[]
skipped=[]
errors=[]

for p in sorted(new_dir.iterdir()):
    if not p.is_file():
        continue
    if p.suffix.lower() != ".zip":
        skipped.append({"name":p.name, "reason":"not a zip"})
        continue
    # build a safe slug
    slug=re.sub(r"[^A-Za-z0-9_-]","_", p.stem).lower()
    dest=raw_dir/slug
    if dest.exists() and any(dest.iterdir()):
        skipped.append({"name":p.name, "reason":"destination exists and not empty", "dest":str(dest)})
        continue
    try:
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, 'r') as z:
            z.extractall(dest)
        # move the zip into dest for record
        new_zip_path=dest/p.name
        p.rename(new_zip_path)
        # write metadata
        meta_file=meta_dir/f"{slug}.json"
        meta={
            "original_name":p.name,
            "ingested_at":datetime.datetime.utcnow().isoformat()+"Z",
            "source": "email-attachment",
            "dest": str(dest)
        }
        with open(meta_file, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
        ingested.append({"name":p.name, "dest":str(dest)})
    except Exception as e:
        errors.append({"name":p.name, "error":str(e)})

print(json.dumps({"ingested":ingested, "skipped":skipped, "errors":errors}, indent=2))
