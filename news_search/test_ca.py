import urllib.parse, urllib.request, json, ssl

# Disable SSL verification for testing (some corporate proxies cause issues)
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def try_url(label, url):
    print(f"\n=== {label} ===")
    print(f"URL: {url}")
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            print(f"Status: {resp.status}")
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Keys: {list(data.keys())}")
            total = data.get("totalItems", data.get("total", data.get("count", "?")))
            print(f"Total hits: {total}")
            items = data.get("items", data.get("results", []))
            for item in items[:3]:
                if isinstance(item, dict):
                    print(f"  - {item.get('date', '?')}: {item.get('title', '?')[:80]}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# 1. Original CA API
try_url("CA API (original)", 
    "https://chroniclingamerica.loc.gov/search/pages/results/?andtext=shipwreck+lake&format=json&rows=3")

# 2. LOC general search  
try_url("LOC search API",
    "https://www.loc.gov/search/?q=shipwreck+lake+michigan&fo=json&c=5&fa=partof:chronicling+america")

# 3. LOC collections API
try_url("LOC collections API",
    "https://www.loc.gov/collections/chronicling-america/?q=shipwreck&fo=json&c=5")

# 4. LOC newspaper titles
try_url("CA newspaper titles",
    "https://chroniclingamerica.loc.gov/newspapers.json")

# 5. Simple page search without date filter
try_url("CA simple search",
    "https://chroniclingamerica.loc.gov/search/pages/results?andtext=shipwreck&format=json&rows=3")
