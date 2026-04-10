"""
chronicling_america_search.py
Searches the Library of Congress Chronicling America API (free, no auth)
for newspaper articles about Great Lakes shipwrecks.

Usage:
    python scripts/chronicling_america_search.py           # search all wrecks
    python scripts/chronicling_america_search.py --limit 50  # first 50 only
    python scripts/chronicling_america_search.py --name "EDMUND FITZGERALD"  # one wreck
"""
import sqlite3
import json
import time
import re
import sys
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

DB_PATH = "db/wrecks.db"
# LOC Collections API (working endpoint)
LOC_BASE = "https://www.loc.gov/collections/chronicling-america/"
# Rate limit: be polite, ~1 req/sec
DELAY = 1.0
MAX_ARTICLES_PER_WRECK = 5


def search_chronicling_america(query, date_start=None, date_end=None, max_results=5):
    """Search LOC Chronicling America collection and return article metadata."""
    params = {
        "q": query,
        "fo": "json",
        "c": str(max_results),
        "sp": "1",
    }
    if date_start and date_end:
        # LOC date facet format: YYYY
        params["dates"] = f"{date_start}/{min(date_end, 1963)}"

    url = f"{LOC_BASE}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (CesarOps-WreckDB/1.0)",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
        print(f"    API error: {e}")
        return []

    articles = []
    for item in data.get("results", []):
        if not isinstance(item, dict):
            continue

        # Build article record
        title = item.get("title", "Untitled")
        date = item.get("date", "")
        item_url = item.get("url", item.get("id", ""))
        if item_url and not item_url.startswith("http"):
            item_url = f"https://www.loc.gov{item_url}"

        # Get description/snippet
        description = item.get("description", [])
        snippet = ""
        if isinstance(description, list) and description:
            snippet = " ".join(str(d) for d in description[:2])[:500]
        elif isinstance(description, str):
            snippet = description[:500]

        article = {
            "title": title[:200],
            "date": date,
            "url": item_url,
            "source": "Chronicling America (Library of Congress)",
            "snippet": snippet,
        }
        articles.append(article)

    return articles


def enrich_wrecks(limit=None, name_filter=None):
    """Search CA for each wreck and store results in news_articles column."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if name_filter:
        c.execute(
            "SELECT id, name, date FROM features WHERE UPPER(name) LIKE UPPER(?) AND name IS NOT NULL",
            (f"%{name_filter}%",)
        )
    else:
        # Only search wrecks that don't have articles yet
        c.execute(
            "SELECT id, name, date FROM features WHERE name IS NOT NULL AND (news_articles IS NULL OR news_articles = '[]') ORDER BY id"
        )

    rows = c.fetchall()
    if limit:
        rows = rows[:limit]

    print(f"Searching Chronicling America for {len(rows)} wrecks...")
    enriched = 0
    skipped = 0

    for i, (row_id, name, date_str) in enumerate(rows):
        # Clean name
        clean_name = re.sub(r'[*#()]', '', name).strip()
        if not clean_name or len(clean_name) < 3:
            skipped += 1
            continue

        # Extract year range from date
        year = None
        if date_str:
            m = re.search(r'(\d{4})', str(date_str))
            if m:
                year = int(m.group(1))

        # Skip if wreck is too recent for CA (post-1963)
        if year and year > 1963:
            skipped += 1
            continue

        # Build query
        query = f'"{clean_name}" shipwreck'
        date_start = max(year - 2, 1789) if year else None
        date_end = min(year + 5, 1963) if year else 1963

        print(f"  [{i+1}/{len(rows)}] {clean_name} ({year or '?'})...", end=" ")

        articles = search_chronicling_america(query, date_start, date_end, MAX_ARTICLES_PER_WRECK)

        # If no results with shipwreck keyword, try just the name
        if not articles and year:
            articles = search_chronicling_america(f'"{clean_name}"', date_start, date_end, 3)

        if articles:
            c.execute(
                "UPDATE features SET news_articles=? WHERE id=?",
                (json.dumps(articles), row_id)
            )
            enriched += 1
            print(f"found {len(articles)} articles")
        else:
            # Store empty array so we don't re-search
            c.execute(
                "UPDATE features SET news_articles='[]' WHERE id=?",
                (row_id,)
            )
            print("no articles")

        # Commit every 50 wrecks
        if (i + 1) % 50 == 0:
            conn.commit()
            print(f"  --- Committed {i+1} wrecks, {enriched} enriched ---")

        time.sleep(DELAY)

    conn.commit()
    conn.close()
    print(f"\nDone! Enriched {enriched} wrecks, skipped {skipped}")
    return enriched


if __name__ == "__main__":
    limit = None
    name = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        elif args[i] == "--name" and i + 1 < len(args):
            name = args[i + 1]
            i += 2
        else:
            i += 1

    enrich_wrecks(limit=limit, name_filter=name)
