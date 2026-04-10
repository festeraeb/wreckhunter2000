import json, math, sys

def hav(lat1, lon1, lat2, lon2):
    R = 6371000
    p = math.pi/180
    a = math.sin((lat2-lat1)*p/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin((lon2-lon1)*p/2)**2
    return 2*R*math.asin(math.sqrt(a))

optical = json.load(open('wreck_hunting_ml/runs/sentinel_optical/optical_all_concepts.json'))
shadow  = [r for r in optical if r['concept']=='shadow_roughness']
zebra   = [r for r in optical if r['concept']=='zebra_clarity']
sediment= [r for r in optical if r['concept']=='sediment_plume']

# --- Q1: shadow vs zebra overlap ---
print("=== Q1: shadow(25) vs zebra(25) within 2km ===")
sz_hits = []
for s in shadow:
    for z in zebra:
        d = hav(s['lat'], s['lon'], z['lat'], z['lon'])
        if d < 2000:
            sz_hits.append((round(s['lat'],4), round(s['lon'],4), round(z['lat'],4), round(z['lon'],4), int(d)))
print("shadow-zebra pairs within 2km:", len(sz_hits))
for m in sorted(sz_hits, key=lambda x: x[4])[:15]:
    print("  shadow(%s,%s) <-> zebra(%s,%s)  %dm" % m)

# --- Q2: sediment overlap ---
print()
print("=== Q2: sediment(3) overlap with shadow/zebra within 5km ===")
for s3 in sediment:
    lat3, lon3 = s3['lat'], s3['lon']
    print("  Sediment: %.4f,%.4f score=%.1f" % (lat3, lon3, s3['score']))
    for s in shadow:
        d = hav(lat3, lon3, s['lat'], s['lon'])
        if d < 5000:
            print("    -> shadow %.4f,%.4f  %dm" % (s['lat'], s['lon'], int(d)))
    for z in zebra:
        d = hav(lat3, lon3, z['lat'], z['lon'])
        if d < 5000:
            print("    -> zebra  %.4f,%.4f  %dm" % (z['lat'], z['lon'], int(d)))

# --- Q3: optical vs known wrecks ---
print()
print("=== Q3: optical(53) vs known wrecks (erie_known_wrecks_all.csv) within 5km ===")
import csv
known = []
with open('erie_known_wrecks_all.csv') as f:
    for row in csv.DictReader(f):
        try:
            known.append((row['name'], float(row['lat']), float(row['lon'])))
        except:
            pass

hits_kw = []
for o in optical:
    best_d, best_name = 999999, ''
    for name, klat, klon in known:
        d = hav(o['lat'], o['lon'], klat, klon)
        if d < best_d:
            best_d, best_name = d, name
    if best_d < 5000:
        hits_kw.append((o['concept'], round(o['lat'],4), round(o['lon'],4), best_name, int(best_d)))

print("Optical within 5km of a known wreck:", len(hits_kw))
for h in sorted(hits_kw, key=lambda x: x[4]):
    print("  %-22s (%.4f,%.4f) -> %-30s %dm" % h)

# --- Q4: optical vs 281 mag candidates ---
print()
print("=== Q4: optical(53) vs mag unknowns(281) within 10km ===")
dr = json.load(open('wreck_hunting_ml/models/discovery_report_v2.json'))
# candidates are in top5_unknowns but we need all 281 -- check if full list exists
# top5_unknowns only has 5; check awois_results for structure
print("Keys in report:", list(dr.keys()))
print("awois_results[0] keys:", list(dr['awois_results'][0].keys()) if dr['awois_results'] else "empty")
print("top5_unknowns[0] keys:", list(dr['top5_unknowns'][0].keys()) if dr['top5_unknowns'] else "empty")
