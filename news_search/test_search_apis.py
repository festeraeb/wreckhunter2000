"""Quick test of ScienceBase and NCEI API formats."""
import requests
import json

# ScienceBase spatial filter format trials
print("=== ScienceBase Tests ===")

# Test 1: bbox parameter (not filter)
params = {
    "q": "aeromagnetic magnetic",
    "bbox": "-83.5,41.3,-78.8,42.9",
    "fields": "title,spatial,webLinks,files",
    "format": "json",
    "max": 10,
}
try:
    r = requests.get("https://www.sciencebase.gov/catalog/items",
                     params=params, timeout=30)
    print(f"  bbox param: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        items = d.get("items", [])
        print(f"  Items: {len(items)}")
        for it in items[:5]:
            t = it.get("title", "?")[:90]
            fls = [f.get("name", "") for f in it.get("files", [])[:2]]
            print(f"    - {t}")
            if fls:
                print(f"      files: {fls}")
    else:
        print(f"  {r.text[:300]}")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: no spatial, just keyword
params2 = {
    "q": "aeromagnetic magnetic Great Lakes Erie",
    "fields": "title,spatial,webLinks,files",
    "format": "json",
    "max": 10,
}
try:
    r = requests.get("https://www.sciencebase.gov/catalog/items",
                     params=params2, timeout=30)
    print(f"\n  keyword only: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        items = d.get("items", [])
        print(f"  Items: {len(items)}")
        for it in items[:5]:
            print(f"    - {it.get('title', '?')[:90]}")
except Exception as e:
    print(f"  ERROR: {e}")

# Deep Blue test
print("\n=== Deep Blue Test ===")
try:
    r = requests.get("https://deepblue.lib.umich.edu/data/catalog.json",
                     params={"q": "Great Lakes magnetic geophysical", "rows": 5},
                     timeout=30)
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        docs = d.get("response", {}).get("docs", d.get("data", []))
        print(f"  Docs: {len(docs)}")
        for doc in docs[:3]:
            title = doc.get("title_tesim", doc.get("title", ["?"]))
            if isinstance(title, list):
                title = title[0] if title else "?"
            print(f"    - {str(title)[:80]}")
except Exception as e:
    print(f"  ERROR: {e}")
