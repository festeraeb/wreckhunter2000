"""Probe magnetic data source URLs to find what's actually available."""
import requests
import re

# Browse NRCan GDR
print("=== NRCan GDR directory ===")
r = requests.get('https://gdr.agg.nrcan.gc.ca/pub/gdr/GRD/', timeout=15)
links = re.findall(r'href="([^"]*)"', r.text)
for link in links:
    print(f"  {link}")

print("\n=== NRCan geophysical data root ===")
try:
    r2 = requests.get('https://gdr.agg.nrcan.gc.ca/', timeout=15)
    links2 = re.findall(r'href="([^"]*)"', r2.text)
    for link in links2:
        if any(k in link.lower() for k in ('mag', 'grid', 'aero', 'download')):
            print(f"  {link}")
except Exception as e:
    print(f"  Error: {e}")

# Check for NOAA NCEI alternative endpoints
print("\n=== NOAA NCEI mag data ===")
try:
    urls_to_check = [
        'https://www.ncei.noaa.gov/products/earth-magnetic-model-anomaly-grid',
        'https://data.noaa.gov/dataset/dataset/emag2-earth-magnetic-anomaly-grid-2-arc-minute-resolution-version-3',
        'https://www.ncei.noaa.gov/maps/grid-extract/',
    ]
    for url in urls_to_check:
        r = requests.head(url, timeout=10, allow_redirects=True)
        print(f"  {r.status_code} {url}")
except Exception as e:
    print(f"  Error: {e}")

# USGS confirmed working - verify sizes
print("\n=== USGS (confirmed working) ===")
for url in [
    'https://mrdata.usgs.gov/magnetic/NAmag_origmrg.zip',
    'https://mrdata.usgs.gov/magnetic/USmag_origmrg.zip',
    'https://mrdata.usgs.gov/magnetic/NAmag_hp500.zip',
]:
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
        cl = r.headers.get('Content-Length', '?')
        if cl != '?':
            cl = f"{int(cl)/1024/1024:.1f}MB"
        print(f"  {r.status_code} {cl:>10s}  {url.split('/')[-1]}")
    except Exception as e:
        print(f"  ERR {e}")

# WDMAM - works but let's confirm file size
print("\n=== WDMAM ===")
for url in [
    'https://wdmam.org/WDMAM2_v2_XYZ.zip',
    'https://wdmam.org/download/WDMAM2_xyz.zip',
]:
    try:
        r = requests.get(url, timeout=20, stream=True)
        chunk = r.raw.read(10)
        r.close()
        magic = "gzip" if chunk[:2] == b'\x1f\x8b' else ("zip" if chunk[:2] == b'PK' else "unknown")
        cl = r.headers.get('Content-Length', '?')
        if cl != '?':
            cl = f"{int(cl)/1024/1024:.1f}MB"
        print(f"  {r.status_code} {cl:>10s} {magic:>6s}  {url.split('/')[-1]}")
    except Exception as e:
        print(f"  ERR {e}")
