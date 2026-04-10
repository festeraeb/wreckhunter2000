"""Probe NRCan and DataCite APIs to find correct endpoints."""
import requests
import json

# 1. NRCan: Try the GDR ArcGIS REST directory
print("=== NRCan ArcGIS REST ===")
try:
    r = requests.get(
        "https://gdr.agg.nrcan.gc.ca/arcgis/rest/services",
        params={"f": "json"},
        timeout=30,
        headers={"User-Agent": "MagLakeHarvester/1.0"}
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        services = d.get("services", [])
        folders = d.get("folders", [])
        print(f"  Folders: {folders[:10]}")
        for s in services[:5]:
            print(f"  Service: {s.get('name')} ({s.get('type')})")
except Exception as e:
    print(f"  Error: {e}")

# 2. NRCan: Try the open data portal (CKAN-based)
print("\n=== NRCan Open Data (CKAN) ===")
try:
    r = requests.get(
        "https://open.canada.ca/data/api/3/action/package_search",
        params={"q": "aeromagnetic", "fq": "organization:nrcan-rncan", "rows": 5},
        timeout=30,
    )
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        results = d.get("result", {}).get("results", [])
        print(f"  Results: {len(results)}")
        for item in results[:3]:
            print(f"    - {item.get('title', '?')[:80]}")
            resources = item.get("resources", [])
            for res in resources[:2]:
                print(f"      format={res.get('format')} url={res.get('url','')[:60]}")
except Exception as e:
    print(f"  Error: {e}")

# 3. DataCite: Try without client-id filter
print("\n=== DataCite (no client filter) ===")
for q in ["AI4Shipwrecks", "magnetometer shipwreck Great Lakes"]:
    try:
        r = requests.get(
            "https://api.datacite.org/dois",
            params={"query": q, "page[size]": "5"},
            timeout=30,
        )
        print(f"  '{q}': status={r.status_code}")
        if r.status_code == 200:
            d = r.json()
            items = d.get("data", [])
            print(f"  Found: {len(items)}")
            for item in items[:3]:
                attrs = item.get("attributes", {})
                titles = attrs.get("titles", [{}])
                title = titles[0].get("title", "?") if titles else "?"
                doi = attrs.get("doi", "")
                publisher = attrs.get("publisher", "")
                print(f"    - {title[:70]}")
                print(f"      doi={doi} pub={publisher}")
    except Exception as e:
        print(f"  Error: {e}")

# 4. DataCite: Try with umich or deep-blue client patterns
print("\n=== DataCite (UMich client variants) ===")
for cid in ["umich", "umich.deep", "umich.lib", "datacite.umich"]:
    try:
        r = requests.get(
            "https://api.datacite.org/dois",
            params={"query": "magnetometer", "client-id": cid, "page[size]": "3"},
            timeout=15,
        )
        d = r.json()
        count = len(d.get("data", []))
        print(f"  client-id={cid}: {count} results")
    except Exception as e:
        print(f"  client-id={cid}: Error {e}")
