import json, pathlib
root=pathlib.Path(r"C:\Users\thomf\programming\Bagrecovery")
meta_dir=root/"magnetic_data"/"meta"
meta_files=sorted([p for p in meta_dir.iterdir() if p.is_file() and p.suffix=='.json' and not p.name.startswith(('usgs_'))])
summary=[]
for p in meta_files:
    try:
        j=json.loads(p.read_text(encoding='utf-8'))
        summary.append({"meta_file":p.name, "original_name":j.get('original_name','(n/a)'), "ingested_at":j.get('ingested_at','(n/a)'), "dest":j.get('dest','(n/a)')})
    except Exception as e:
        summary.append({"meta_file":p.name, "error":str(e)})
print(json.dumps(summary, indent=2))
