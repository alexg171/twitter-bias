"""
Pull GDELT Doc API v2 article volume timeline for a set of topics,
bracketing Elon Musk's Twitter acquisition (Oct 27, 2022).

Uses timelinevol mode: one API call per topic returns the full
date-series — 15 calls total instead of 855.

GDELT free API — no key needed.
Outputs: out/gdelt_coverage.csv
Columns: date, topic, article_count
"""

import os
import time
import requests
import pandas as pd

TOPICS = [
    # Political
    "Republican", "Democrat", "Biden", "Trump", "abortion",
    "gun control", "immigration", "climate change", "free speech", "censorship",
    # Neutral controls
    "NFL", "NBA", "Netflix", "Taylor Swift", "hurricane",
]

START = "20201027000000"
END   = "20241027235959"
OUT_FILE = "out/gdelt_coverage.csv"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_timeline(topic: str) -> pd.DataFrame:
    """One API call → full date-series of article volume for topic."""
    params = {
        "query":         f'"{topic}" sourcelang:english',
        "mode":          "timelinevol",
        "startdatetime": START,
        "enddatetime":   END,
        "format":        "json",
    }
    r = requests.get(GDELT_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    rows = []
    for entry in data.get("timeline", [{}])[0].get("data", []):
        rows.append({
            "date":          pd.to_datetime(entry["date"]),
            "topic":         topic,
            "article_count": entry["value"],
        })
    return pd.DataFrame(rows)


def pull_gdelt():
    os.makedirs("out", exist_ok=True)
    # Check which topics already have data so we can skip them
    already_done = set()
    if os.path.exists(OUT_FILE):
        existing = pd.read_csv(OUT_FILE)
        already_done = set(existing["topic"].unique())
        print(f"Skipping already-collected topics: {already_done}")

    header_written = os.path.exists(OUT_FILE)

    for i, topic in enumerate(TOPICS, 1):
        if topic in already_done:
            print(f"[{i}/{len(TOPICS)}] {topic} ... skipped (already collected).")
            continue
        print(f"[{i}/{len(TOPICS)}] {topic} ...", end=" ", flush=True)
        for attempt in range(3):
            try:
                df = fetch_timeline(topic)
                df.to_csv(OUT_FILE, mode="a", index=False,
                          header=not header_written)
                header_written = True
                print(f"{len(df)} rows saved.")
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                print(f"ERROR: {e} — retrying in {wait}s...")
                time.sleep(wait)
        else:
            print(f"FAILED after 3 attempts, skipping.")
        time.sleep(10)   # be polite

    print(f"\nDone. Output: {OUT_FILE}")


if __name__ == "__main__":
    pull_gdelt()
