"""
Pull GDELT TV 2.0 API — daily mention frequency for political keywords
on center-rated broadcast stations (AllSides / Ad Fontes Media ratings).

Neutral stations used:
  PBSNEWSHOUR  — PBS NewsHour       (center)
  BBCNEWS      — BBC News           (center)
  WCVB         — ABC Boston         (center)
  WBZ          — CBS Boston         (center)
  WBAL         — NBC Baltimore      (center)

For each station we query two aggregate terms:
  "right" — conservative/Republican political language
  "left"  — progressive/Democrat political language

Using timelinevol mode returns a full daily series per call,
so total API calls = 2 terms × 5 stations = 10 calls.

Outputs: out/gdelt_tv_ngrams.csv
Columns:  date, station, term, count
"""

import os
import time
import requests
import pandas as pd

START    = "20220427000000"
END      = "20230427235959"
OUT_FILE = "out/gdelt_tv_ngrams.csv"
TV_URL   = "https://api.gdeltproject.org/api/v2/tv/tv"

# Center-rated stations (AllSides / Ad Fontes Media)
# GDELT uses Internet Archive TV News Archive call-letter identifiers
# Verified working via API test: BBCNEWS, KQED, BLOOMBERG, CNBC, LINKTV
NEUTRAL_STATIONS = [
    "BBCNEWS",   # BBC News          — center (AllSides)
    "KQED",      # PBS San Francisco — center (AllSides)
    "BLOOMBERG", # Bloomberg TV      — center/business-focused, non-partisan on politics
    "CNBC",      # CNBC              — center on business/finance news
    "LINKTV",    # Link TV           — independent, non-partisan news
]

# Aggregate query terms — broad enough to capture political language
# without keyword-by-keyword calls
TERMS = {
    "right": (
        '"republican" OR "conservative" OR "trump" OR "gop" OR "maga"'
    ),
    "left": (
        '"democrat" OR "progressive" OR "biden" OR "liberal" OR "bernie"'
    ),
}


def fetch_timeline(term_label: str, query: str, station: str) -> pd.DataFrame:
    """One API call → full daily series for a query on a single station."""
    params = {
        "query":         f"({query}) station:{station}",
        "mode":          "timelinevol",
        "STARTDATETIME": START,
        "ENDDATETIME":   END,
        "format":        "json",
        "DATANORM":      "raw",
    }
    r = requests.get(TV_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = []
    for series in data.get("timeline", []):
        for entry in series.get("data", []):
            rows.append({
                "date":    pd.to_datetime(entry["date"]).date(),
                "station": station,
                "term":    term_label,
                "count":   entry.get("value", 0),
            })
    return pd.DataFrame(rows)


def pull_gdelt_tv():
    os.makedirs("out", exist_ok=True)

    # Resume: skip (station, term) pairs already collected
    already_done = set()
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE)
        already_done = set(zip(existing["station"], existing["term"]))
        print(f"Resuming — {len(already_done)} (station, term) pairs already collected.")

    header_written = os.path.exists(OUT_FILE)
    total = len(NEUTRAL_STATIONS) * len(TERMS)
    n = 0

    for station in NEUTRAL_STATIONS:
        for term_label, query in TERMS.items():
            n += 1
            if (station, term_label) in already_done:
                print(f"[{n}/{total}] {station} / {term_label} ... skipped.")
                continue

            print(f"[{n}/{total}] {station} / {term_label} ...", end=" ", flush=True)

            for attempt in range(3):
                try:
                    df = fetch_timeline(term_label, query, station)
                    if not df.empty:
                        df.to_csv(OUT_FILE, mode="a", index=False, header=not header_written)
                        header_written = True
                        print(f"{len(df)} rows saved.")
                    else:
                        print("no data returned.")
                    break
                except Exception as e:
                    wait = 30 * (attempt + 1)
                    print(f"ERROR: {e} — retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print("FAILED after 3 attempts, skipping.")

            time.sleep(10)   # polite

    print(f"\nDone. Output: {OUT_FILE}")

    # Quick summary
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"\nRows collected: {len(df):,}")
        print(df.groupby(["station", "term"])["count"].sum().to_string())


if __name__ == "__main__":
    pull_gdelt_tv()
