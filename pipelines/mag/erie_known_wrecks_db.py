#!/usr/bin/env python3
"""
Comprehensive Lake Erie Known Wreck Database
=============================================
Builds a training-ready CSV of ALL known wreck positions from:
  1. Niagara Divers Association (NDA) — precise GPS mooring coords (DD-MM.MMM)
  2. Wikipedia "List of shipwrecks in the Great Lakes" — Lake Erie section
  3. ShipwreckWorld additions not in above sources

Purpose: positive training targets for magnetic anomaly detection.
At this stage we just need "something is here" — type discrimination comes later
once we can reliably detect targets in both satellite and aero-mag data.

Sources scraped 2026-03-15.

Usage:
    python scripts/erie_known_wrecks_db.py
    python scripts/erie_known_wrecks_db.py --output erie_known_wrecks_all.csv
    python scripts/erie_known_wrecks_db.py --compare   # compare with existing DB
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class WreckRecord:
    name: str
    lat: float
    lon: float  # negative for West
    vessel_type: str = ""
    date_lost: str = ""
    source: str = ""
    coord_quality: str = ""  # "gps" | "chart" | "approx"
    notes: str = ""


# ── Coordinate parsers ──────────────────────────────────────────────────────

def parse_ddmm(lat_str: str, lon_str: str) -> tuple[float, float]:
    """Parse NDA DD-MM.MMM format: '42-36.601' / '79-29.842' → decimal degrees.
    Longitude is always West (negative) for Lake Erie."""
    def _parse(s: str) -> float:
        # Handle edge case like '42-39.2.88' → '42-39.288' (extra period)
        parts = s.strip().split("-")
        deg = int(parts[0])
        # Fix double-decimal: '39.2.88' → '39.288'
        min_str = parts[1].replace(".", "", 1) if parts[1].count(".") > 1 else parts[1]
        # Re-insert decimal at correct position
        if parts[1].count(".") > 1:
            # Original was like '2.88' → should be '2.88' but with '39.' prefix
            # Actually '39.2.88' means 39.288 minutes
            raw = parts[1]
            digits = raw.replace(".", "")
            # First decimal was after position 2 (e.g., '39.288')
            min_str = digits[0:2] + "." + digits[2:]
        minutes = float(min_str)
        return deg + minutes / 60.0

    lat = _parse(lat_str)
    lon = -_parse(lon_str)  # West = negative
    return lat, lon


def parse_dms(dms_str: str) -> float:
    """Parse DMS variants: '42°39′N', '42°05′N', '41°31.00′N', '42°33′0″N'."""
    import re
    # Remove direction suffix, we'll handle sign separately
    dms_str = dms_str.strip()

    # Try DD°MM′SS″ format
    m = re.match(r"(\d+)°(\d+)[′'](\d+(?:\.\d+)?)[″\"]?", dms_str)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0

    # Try DD°MM.MM′ format
    m = re.match(r"(\d+)°(\d+(?:\.\d+)?)[′']", dms_str)
    if m:
        return int(m.group(1)) + float(m.group(2)) / 60.0

    # Try DD°MM format (no minutes symbol)
    m = re.match(r"(\d+)°(\d+)", dms_str)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 60.0

    raise ValueError(f"Cannot parse DMS: {dms_str!r}")


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 1: Niagara Divers Association — GPS mooring coordinates
# https://www.niagaradivers.com/moor/locations.html (scraped 2026-03-15)
# Format: DD-MM.MMM / DD-MM.MMM (lat/lon)
# These are precise GPS positions on the actual wrecks.
# ═══════════════════════════════════════════════════════════════════════════

NDA_RAW = [
    # (name, lat_ddmm, lon_ddmm, mooring_note)
    ("Acme",               "42-36.601", "79-29.842", "block off Port Side"),
    ("Atlantic",           "42-30.620", "80-05.086", "rope & chain on Walking Beam"),
    ("Boland",             "42-22.794", "79-43.929", "rope & chain on Prop Shaft"),
    ("Betty Hedger",       "42-25.110", "79-36.528", "block off Bow"),
    ("Brunswick",          "42-35.510", "79-24.527", "rope & chain on Boiler"),
    ("Carlingford",        "42-39.288", "79-28.597", "blocks off Starboard Bow & Stern"),
    ("CB Benson",          "42-46.262", "79-14.609", "blocks off Bow & Stern"),
    ("Cracker",            "42-33.485", "79-51.649", ""),
    ("Dean Richmond",      "42-17.421", "79-55.859", "rope & chain to Prop Shaft"),
    ("Dupuis #10",         "42-49.095", "79-13.300", ""),
    ("Finch",              "42-50.965", "78-59.029", "rope & chain to wreck"),
    ("George Finney",      "42-40.087", "79-36.250", "block off Bow"),
    ("Indiana",            "42-17.819", "79-59.907", "rope & chain on Windlass"),
    ("Niagara",            "42-44.310", "79-36.285", "block off Starboard Side"),
    ("O.W. Cheney",        "42-50.251", "79-00.477", "rope & chain on Boiler"),
    ("Oneida/Arches",      "42-27.476", "80-01.021", "rope & chain on Arch"),
    ("Oxford",             "42-28.855", "79-51.843", "rope & chain on Railing"),
    ("Passaic",            "42-28.760", "79-27.782", "rope & chain on Steam Engine"),
    ("Persian",            "42-33.781", "79-54.696", "rope & chain on Boiler"),
    ("Raleigh",            "42-51.926", "79-09.254", "chain on Boiler"),
    ("Smith",              "42-28.486", "79-59.061", "rope & chain on Forward Wench"),
    ("St. James",          "42-27.014", "80-07.331", "rope to Ships Anchor on Bow"),
    ("Stern Castle",       "42-30.294", "80-02.379", ""),
    ("Stonewreck",         "42-40.076", "79-23.780", "block off Starboard Side"),
    ("Tonawanda",          "42-50.399", "78-58.932", "rope & chain on Boiler"),
    ("Tradewind",          "42-25.516", "80-12.056", "rope & chain on Windlass"),
    ("Washington Irving",  "42-32.371", "79-27.636", "rope & chain to Ships Anchor on Bow"),
]


def build_nda_wrecks() -> list[WreckRecord]:
    """Parse NDA mooring GPS locations."""
    wrecks = []
    for name, lat_s, lon_s, note in NDA_RAW:
        lat, lon = parse_ddmm(lat_s, lon_s)
        wrecks.append(WreckRecord(
            name=name, lat=lat, lon=lon,
            source="NDA_GPS",
            coord_quality="gps",
            notes=note,
        ))
    return wrecks


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 2: Wikipedia "List of shipwrecks in the Great Lakes" — Lake Erie
# https://en.wikipedia.org/wiki/List_of_shipwrecks_in_the_Great_Lakes#Lake_Erie
# Scraped 2026-03-15. Coordinates from Wikipedia's interactive map links.
# Format: DMS (degrees, minutes, optional seconds)
# Quality: "chart" — from historical records, generally ±0.5-2 nm accuracy
# ═══════════════════════════════════════════════════════════════════════════

# (name, lat_deg, lat_min, lon_deg, lon_min, vessel_type, date_lost, notes)
# Stored as decimal degrees directly (parsed from DMS during scrape)
WIKIPEDIA_ERIE = [
    WreckRecord("17 Fathom Wreck",     42.650, -80.050, "unknown",        "", "Wikipedia", "chart", "105ft depth, silt bottom"),
    WreckRecord("Admiral",             41.633, -81.900, "tug",            "1942-12-02", "Wikipedia", "chart", "Towing Cleveco, gale, 14 lost"),
    WreckRecord("Adventure",           41.633, -82.683, "sand dredge",    "1903-10-07", "Wikipedia", "chart", "Fire off Kelley's Island"),
    WreckRecord("Algeria",             41.517, -81.700, "schooner-barge", "1906-05-05", "Wikipedia", "chart", "Broke apart in storm"),
    WreckRecord("Alva B.",             41.500, -82.017, "tug",            "1917-11-01", "Wikipedia", "chart", "Ran aground Avon Point"),
    WreckRecord("America",             41.817, -82.633, "steamer",        "1854-04-05", "Wikipedia", "chart", "Sidewheel, ran aground Pelee Island"),
    WreckRecord("Anthony Wayne",       41.517, -82.383, "steamer",        "1850-04-28", "Wikipedia", "chart", "Oldest steamboat wreck in GL"),
    WreckRecord("Atlantic",            42.500, -80.083, "steamer",        "1852-08-20", "Wikipedia", "chart", "Paddlewheel, rammed, 5th-worst disaster"),
    WreckRecord("Arches (Oneida)",     42.450, -80.017, "steamer",        "1852-11-11", "Wikipedia", "chart", "Package freighter, storm off Long Point"),
    WreckRecord("Argo",                41.633, -82.500, "tank barge",     "1937-10-20", "Wikipedia", "chart", "Pollution risk, crude + benzole cargo"),
    WreckRecord("Bay Coal Schooner",   41.550, -81.933, "schooner",       "1874",       "Wikipedia", "chart", "Possibly Industry"),
    WreckRecord("Bow Cabin",           41.933, -82.233, "unknown",        "",           "Wikipedia", "chart", ""),
    WreckRecord("Brown Brothers",      42.617, -80.000, "unknown",        "1959-10-28", "Wikipedia", "chart", "Sank off Long Point"),
    WreckRecord("Brunswick",           42.583, -79.400, "steamer",        "1881-11-12", "Wikipedia", "chart", "Collision with Carlingford"),
    WreckRecord("Canobie",             42.167, -80.000, "steamer",        "1921-11-01", "Wikipedia", "chart", "Storm near Erie"),
    WreckRecord("Carlingford",         42.650, -79.467, "schooner",       "1881-11-12", "Wikipedia", "chart", "Collision with Brunswick"),
    WreckRecord("Cascade",             41.467, -82.183, "tug",            "1904-01-24", "Wikipedia", "chart", "Encountered ice"),
    WreckRecord("C.B. Benson",         42.767, -79.233, "schooner",       "1893-10-14", "Wikipedia", "chart", "Massive gale, route to Detroit"),
    WreckRecord("C.B. Lockwood",       41.933, -81.383, "unknown",        "1902-10-13", "Wikipedia", "chart", "Sunk below lake bottom"),
    WreckRecord("Cecil J.",            42.750, -80.217, "tugboat",        "1944-05-27", "Wikipedia", "chart", "Scuttled after fire"),
    WreckRecord("Charles H. Davis",    41.500, -81.717, "steamer",        "1903-06-13", "Wikipedia", "chart", "Sprung leak near Cleveland"),
    WreckRecord("Charles Foster",      42.167, -80.250, "bulk barge",     "1900-12-09", "Wikipedia", "chart", "Gale near Erie"),
    WreckRecord("Cleveco",             41.783, -81.600, "barge",          "1942-12-03", "Wikipedia", "chart", "Towed by Admiral, both foundered"),
    WreckRecord("Craftsman",           41.517, -82.000, "barge",          "1958-06-03", "Wikipedia", "chart", "Foundered off Avon Point"),
    WreckRecord("Crete",               42.167, -80.000, "unknown",        "",           "Wikipedia", "chart", ""),
    WreckRecord("Dean Richmond",       42.283, -79.917, "steamer",        "1893-10-14", "Wikipedia", "chart", "Sank near Erie"),
    WreckRecord("Duke Luedtke",        41.683, -81.950, "tug",            "1993-09-21", "Wikipedia", "chart", "Capsized, sprung leak"),
    WreckRecord("Dundee",              41.683, -81.833, "schooner-barge", "1900-11-09", "Wikipedia", "chart", "Foundered in gale under tow"),
    WreckRecord("Dunkirk Schooner",    42.550, -79.600, "schooner",       "",           "Wikipedia", "chart", "Early unidentified, off Dunkirk NY"),
    WreckRecord("Eldorado",            42.167, -80.000, "unknown",        "1880-11-20", "Wikipedia", "chart", "Off Erie harbor mouth"),
    WreckRecord("Erieau Quarry Stone",  42.250, -81.900, "unknown",       "",           "Wikipedia", "chart", ""),
    WreckRecord("F.A. Meyer",          41.917, -82.033, "bulk carrier",   "1909-12-18", "Wikipedia", "chart", "Ice cut hull"),
    WreckRecord("Fanny L. Jones",      41.500, -81.717, "schooner",       "1890-08-10", "Wikipedia", "chart", "Storm near Cleveland"),
    WreckRecord("Frank E. Vigor",      41.950, -81.950, "bulk carrier",   "1944-04-27", "Wikipedia", "chart", "Collision off Pt. Pelee"),
    WreckRecord("George Dunbar",       41.667, -82.550, "bulk carrier",   "1902-06-29", "Wikipedia", "chart", "Sank off Kelleys Island"),
    WreckRecord("H.A. Barr",           42.150, -81.383, "barge",          "1902-08-24", "Wikipedia", "chart", "Sank off Port Stanley"),
    WreckRecord("Hickory Stick",       41.533, -82.100, "derrick barge",  "1958-11-29", "Wikipedia", "chart", "Broke apart in storm"),
    WreckRecord("Indiana",             42.283, -79.983, "steamer",        "1848-12-05", "Wikipedia", "chart", "Ran aground, burned off Conneaut"),
    WreckRecord("Ivanhoe",             41.550, -82.033, "schooner",       "1855-04-10", "Wikipedia", "chart", "Collision with Arab"),
    WreckRecord("James B. Colgate",    42.083, -81.733, "whaleback",      "1916-10-20", "Wikipedia", "chart", "25 killed, 1 survivor, located 1991"),
    WreckRecord("Jay Gould",           41.850, -82.400, "bulk carrier",   "1918-07-18", "Wikipedia", "chart", "Storm near Pt Pelee"),
    WreckRecord("J.G. McGrath",        42.667, -79.383, "unknown",        "1878-10-28", "Wikipedia", "chart", "Foundered off Long Point"),
    WreckRecord("J.J. Boland Jr.",     42.367, -79.717, "bulk carrier",   "1932-10-05", "Wikipedia", "chart", "Hatches open near Westfield"),
    WreckRecord("John Pridgeon Jr.",   41.583, -81.967, "lumber carrier", "1908-09-18", "Wikipedia", "chart", "Sprung leak, storm off Cleveland"),
    WreckRecord("Little Wissahickon",  41.900, -81.933, "unknown",        "1896-07-10", "Wikipedia", "chart", "Off Rondeau Point"),
    WreckRecord("Lycoming",            42.250, -81.883, "steamer",        "1910-10-21", "Wikipedia", "chart", "Burned at dock Morpeth"),
    WreckRecord("Mabel Wilson",        41.500, -81.717, "schooner",       "1906-05-26", "Wikipedia", "chart", "Ran aground, towline snapped"),
    WreckRecord("Marshall F. Butters", 41.717, -82.283, "lumber carrier", "1916-10-10", "Wikipedia", "chart", "Same storm as Colgate"),
    WreckRecord("Mecosta",             41.517, -81.883, "bulk carrier",   "1922-10-29", "Wikipedia", "chart", "Foundered under tow near Rocky River"),
    WreckRecord("Merida",              42.217, -81.333, "steamer",        "1916-10-16", "Wikipedia", "chart", "Ward Line, same storm as Colgate"),
    WreckRecord("Morning Star",        41.600, -82.200, "steamer",        "1868-06-06", "Wikipedia", "chart", "Paddle steamer, collision near Vermilion"),
    WreckRecord("North Carolina",      41.717, -81.367, "tug",            "1968-12-09", "Wikipedia", "chart", "Sank off Mentor, unknown cause"),
    WreckRecord("Northern Indiana",    41.883, -82.500, "steamer",        "1856-07-17", "Wikipedia", "chart", "Fire near Pt Pelee, 56 lost"),
    WreckRecord("Oxford",              42.467, -79.850, "unknown",        "1856-05-30", "Wikipedia", "chart", "Collision off Long Point"),
    WreckRecord("Pascal P. Pratt",     42.550, -80.083, "unknown",        "1908",       "Wikipedia", "chart", "Ran aground Long Point"),
    WreckRecord("Passaic",             42.467, -79.450, "steamer",        "1891-11-01", "Wikipedia", "chart", "Off Dunkirk"),
    WreckRecord("Penelope",            41.517, -82.033, "tug",            "1909-12-19", "Wikipedia", "chart", "Fire, grounded, burned"),
    WreckRecord("Philip D. Armour",    42.117, -80.167, "barge",          "1915-11-13", "Wikipedia", "chart", "Foundered off Erie, towline broke"),
    WreckRecord("Philip Minch",        41.683, -82.500, "bulk carrier",   "1904-11-20", "Wikipedia", "chart", "Burned near Pelee Island"),
    WreckRecord("Queen of the West",   41.833, -82.383, "bulk carrier",   "1903-08-08", "Wikipedia", "chart", "Sprung leak"),
    WreckRecord("Robert",              42.250, -81.810, "tug",            "1982-09-26", "Wikipedia", "chart", "Collision off Chatham-Kent"),
    WreckRecord("S.F. Gale",           41.733, -81.867, "schooner",       "1876-11-28", "Wikipedia", "chart", "Foundered off Cleveland"),
    WreckRecord("S.K. Martin",         42.233, -79.933, "bulk carrier",   "1912-10-12", "Wikipedia", "chart", "Boiler exploded off Erie"),
    WreckRecord("Sand Merchant",       41.567, -82.950, "sandsucker",     "1936-10-17", "Wikipedia", "chart", "Storm off Cleveland"),
    WreckRecord("Sarah E. Sheldon",    41.483, -82.100, "bulk carrier",   "1905-10-20", "Wikipedia", "chart", "Struck reef off Lorain"),
    WreckRecord("St. James",           42.450, -80.117, "unknown",        "1870-10",    "Wikipedia", "chart", "Off Long Point, discovered 1984"),
    WreckRecord("Success",             41.517, -82.900, "barquentine",    "1946-07-04", "Wikipedia", "chart", "Burned, ran aground Port Clinton"),
    WreckRecord("Sultan",              41.600, -81.617, "unknown",        "1864-09-24", "Wikipedia", "chart", "Storm off Cleveland"),
    WreckRecord("T-8",                 42.583, -80.017, "unknown",        "",           "Wikipedia", "chart", ""),
    WreckRecord("Tasmania",            41.783, -82.483, "bulk carrier",   "1905-10-20", "Wikipedia", "chart", "Off Pt Pelee"),
    WreckRecord("Tire Reef",           42.683, -80.133, "unknown",        "",           "Wikipedia", "chart", ""),
    WreckRecord("Trade Wind",          42.417, -80.200, "schooner",       "1854-11-30", "Wikipedia", "chart", "Collision off Long Point"),
    WreckRecord("Two Fannies",         41.550, -81.917, "bark",           "1890-08-10", "Wikipedia", "chart", "Leak in heavy seas"),
    WreckRecord("Unknown (42.13/-81.62)", 42.133, -81.617, "unknown",    "",           "Wikipedia", "chart", ""),
    WreckRecord("Valentine",           41.917, -81.900, "schooner",       "1877-10-10", "Wikipedia", "chart", "Foundered in storm"),
    WreckRecord("Washington Irving",   42.533, -79.450, "steamer",        "1860-07-07", "Wikipedia", "chart", "Off Dunkirk NY"),
    WreckRecord("Wilma",               42.700, -80.033, "fishing vessel", "1936-04-14", "Wikipedia", "chart", "Off Port Dover"),
]


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE 3: ShipwreckWorld additions not covered by Wikipedia or NDA
# Only entries that have NO match within 5km in the other two sources
# ═══════════════════════════════════════════════════════════════════════════

SHIPWRECKWORLD_EXTRA = [
    # Colgate position from confirmed candidate #103 + Wikipedia
    WreckRecord("Colgate",         42.083, -81.733, "whaleback",  "1916-10-20", "ShipwreckWorld+Wikipedia", "chart",
                "308ft, steel hull. Wikipedia: 42°05′N 81°44′W. Confirmed target #103 at 42.173,-81.740"),
    WreckRecord("H.G. Cleveland",  42.300, -80.500, "schooner",   "1899-08",    "ShipwreckWorld", "approx",
                "Wikipedia has no coords. 3-mast, stone cargo, off Lorain"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Merge + deduplicate
# ═══════════════════════════════════════════════════════════════════════════

def merge_and_deduplicate(
    nda: list[WreckRecord],
    wiki: list[WreckRecord],
    extra: list[WreckRecord],
    dedup_radius_m: float = 3000.0,
) -> list[WreckRecord]:
    """Merge all sources, preferring NDA GPS positions when there's a match."""
    all_wrecks: list[WreckRecord] = []
    used_wiki_indices = set()

    # Step 1: Start with NDA (highest quality GPS coords)
    for nda_w in nda:
        all_wrecks.append(nda_w)

        # Find matching Wikipedia entry (same name or within dedup_radius)
        for i, wiki_w in enumerate(wiki):
            if i in used_wiki_indices:
                continue
            name_match = (
                nda_w.name.lower().replace(".", "").strip()
                in wiki_w.name.lower().replace(".", "").strip()
                or wiki_w.name.lower().replace(".", "").strip()
                in nda_w.name.lower().replace(".", "").strip()
            )
            dist = haversine_m(nda_w.lat, nda_w.lon, wiki_w.lat, wiki_w.lon)
            if name_match or dist < dedup_radius_m:
                used_wiki_indices.add(i)
                # Merge Wikipedia metadata into NDA record
                if wiki_w.vessel_type and not nda_w.vessel_type:
                    nda_w.vessel_type = wiki_w.vessel_type
                if wiki_w.date_lost and not nda_w.date_lost:
                    nda_w.date_lost = wiki_w.date_lost
                if wiki_w.notes:
                    nda_w.notes = (nda_w.notes + "; " + wiki_w.notes).strip("; ")
                nda_w.source = "NDA_GPS+Wikipedia"
                break

    # Step 2: Add remaining Wikipedia entries (not matched to NDA)
    for i, wiki_w in enumerate(wiki):
        if i not in used_wiki_indices:
            all_wrecks.append(wiki_w)

    # Step 3: Add extras (check they don't duplicate)
    for ext in extra:
        duplicate = False
        for existing in all_wrecks:
            dist = haversine_m(ext.lat, ext.lon, existing.lat, existing.lon)
            name_match = ext.name.lower() in existing.name.lower() or existing.name.lower() in ext.name.lower()
            if name_match or dist < dedup_radius_m:
                duplicate = True
                # If extra has better metadata, merge
                if ext.notes and ext.notes not in (existing.notes or ""):
                    existing.notes = (existing.notes + "; " + ext.notes).strip("; ")
                break
        if not duplicate:
            all_wrecks.append(ext)

    return all_wrecks


# ═══════════════════════════════════════════════════════════════════════════
# Compare with existing discriminator DB
# ═══════════════════════════════════════════════════════════════════════════

def compare_with_existing(new_wrecks: list[WreckRecord]):
    """Compare new positions against existing erie_wellhead_discriminator.py data."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    try:
        from erie_wellhead_discriminator import get_all_known_wrecks
    except ImportError:
        print("Cannot import erie_wellhead_discriminator — skipping comparison")
        return

    old_wrecks = get_all_known_wrecks()

    print("\n" + "=" * 100)
    print("POSITION COMPARISON: New DB vs Existing erie_wellhead_discriminator.py")
    print("=" * 100)

    print(f"\n  Old DB: {len(old_wrecks)} wrecks")
    print(f"  New DB: {len(new_wrecks)} wrecks")

    # For each old wreck, find nearest new wreck by name or position
    print(f"\n  {'Old Name':<25} {'Old Pos':>25} {'New Pos':>25} {'Offset':>10} {'Source':>10}")
    print("  " + "-" * 100)

    significant_offsets = 0
    for old in old_wrecks:
        best_match = None
        best_dist = float("inf")
        # Try name match first
        for new in new_wrecks:
            if old.name.lower().strip() in new.name.lower().strip() or new.name.lower().strip() in old.name.lower().strip():
                d = haversine_m(old.lat, old.lon, new.lat, new.lon)
                if d < best_dist:
                    best_dist = d
                    best_match = new
        # Fall back to distance match
        if best_match is None:
            for new in new_wrecks:
                d = haversine_m(old.lat, old.lon, new.lat, new.lon)
                if d < best_dist:
                    best_dist = d
                    best_match = new

        if best_match:
            flag = "  ◄ MOVED" if best_dist > 2000 else ""
            if best_dist > 2000:
                significant_offsets += 1
            print(f"  {old.name:<25} ({old.lat:8.4f},{old.lon:9.4f})  "
                  f"({best_match.lat:8.4f},{best_match.lon:9.4f})  "
                  f"{best_dist:8,.0f}m  {best_match.source:<10}{flag}")

    print(f"\n  Positions with >2km offset: {significant_offsets}")
    print(f"  New wrecks not in old DB:   {len(new_wrecks) - len(old_wrecks)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def build_database(output_csv: str = "erie_known_wrecks_all.csv", compare: bool = False) -> list[WreckRecord]:
    nda = build_nda_wrecks()
    all_wrecks = merge_and_deduplicate(nda, WIKIPEDIA_ERIE, SHIPWRECKWORLD_EXTRA)

    # Sort by longitude (west to east)
    all_wrecks.sort(key=lambda w: w.lon)

    # Stats
    n_gps = sum(1 for w in all_wrecks if w.coord_quality == "gps")
    n_chart = sum(1 for w in all_wrecks if w.coord_quality == "chart")
    n_approx = sum(1 for w in all_wrecks if w.coord_quality == "approx")

    print(f"Lake Erie Known Wreck Database")
    print(f"  Total wrecks:       {len(all_wrecks)}")
    print(f"  GPS-quality coords: {n_gps} (NDA mooring positions)")
    print(f"  Chart-quality:      {n_chart} (Wikipedia/historical)")
    print(f"  Approximate:        {n_approx} (ShipwreckWorld estimates)")
    print()

    # Basin breakdown
    western = [w for w in all_wrecks if w.lon < -82.0]
    central = [w for w in all_wrecks if -82.0 <= w.lon < -80.0]
    eastern = [w for w in all_wrecks if w.lon >= -80.0]
    print(f"  Western Basin (lon < -82°):  {len(western)}")
    print(f"  Central Basin (-82° to -80°): {len(central)}")
    print(f"  Eastern Basin (lon > -80°):  {len(eastern)}")
    print()

    # Deduplicated vessel types
    types = {}
    for w in all_wrecks:
        t = w.vessel_type or "unknown"
        types[t] = types.get(t, 0) + 1
    print(f"  Vessel types:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t:<20} {c:>3}")

    # Save
    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["name", "lat", "lon", "vessel_type", "date_lost", "source", "coord_quality", "notes"]
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for w in all_wrecks:
                writer.writerow(asdict(w))
        print(f"\n  Saved to: {out}")

    if compare:
        compare_with_existing(all_wrecks)

    return all_wrecks


def main():
    parser = argparse.ArgumentParser(description="Build comprehensive Lake Erie wreck database")
    parser.add_argument("--output", "-o", default="erie_known_wrecks_all.csv",
                        help="Output CSV path")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with existing erie_wellhead_discriminator.py positions")
    args = parser.parse_args()
    build_database(output_csv=args.output, compare=args.compare)


if __name__ == "__main__":
    main()
