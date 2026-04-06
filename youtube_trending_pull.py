"""
Pull YouTube trending video data from pre-scraped Kaggle dataset.
Source: rsrishav/youtube-trending-video-dataset (US region)

Setup:
  1. pip install kaggle pandas
  2. Get API token from kaggle.com/settings → Create New Token → places kaggle.json
  3. Run this script — it will download & process automatically

Outputs: out/youtube_trending.csv
Columns:  date, title, category_id, category_name, views, rank
"""

import os
import sys
import subprocess
import zipfile
import pandas as pd

START    = "2022-04-27"
END      = "2023-04-27"
OUT_FILE = "out/youtube_trending.csv"

KAGGLE_DATASET = "rsrishav/youtube-trending-video-dataset"
ZIP_FILE = "youtube-trending-video-dataset.zip"
US_FILE  = "US_youtube_trending_data.csv"

CATEGORY_NAMES = {
    1:  "Film & Animation",
    2:  "Autos & Vehicles",
    10: "Music",
    15: "Pets & Animals",
    17: "Sports",
    18: "Short Movies",
    19: "Travel & Events",
    20: "Gaming",
    21: "Videoblogging",
    22: "People & Blogs",
    23: "Comedy",
    24: "Entertainment",
    25: "News & Politics",
    26: "Howto & Style",
    27: "Education",
    28: "Science & Technology",
    29: "Nonprofits & Activism",
}

# Keep only apolitical/entertainment categories
NEUTRAL_CATEGORY_IDS = {1, 10, 15, 17, 20, 23, 24, 26}


def download_data():
    """Download dataset via kaggle CLI if zip or CSV not already present."""
    if os.path.exists(US_FILE) or os.path.exists(ZIP_FILE):
        return
    print("Downloading dataset from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("\nKaggle CLI download failed. To fix:")
        print("  1. pip install kaggle")
        print("  2. Go to kaggle.com → Settings → API → Create New Token")
        print("  3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print(f"  4. Re-run this script, or manually download:")
        print(f"     kaggle datasets download -d {KAGGLE_DATASET}")
        sys.exit(1)
    print(f"Download complete.")


def unzip_data():
    """Extract US trending file from zip."""
    if os.path.exists(US_FILE):
        return
    if not os.path.exists(ZIP_FILE):
        print(f"ERROR: {ZIP_FILE} not found. Run download first.")
        sys.exit(1)
    print(f"Unzipping {ZIP_FILE}...")
    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        # Find the US file regardless of internal path
        us_members = [m for m in z.namelist() if "US_youtube_trending_data" in m]
        if not us_members:
            print("ERROR: US_youtube_trending_data.csv not found in zip.")
            print(f"Files in zip: {z.namelist()}")
            sys.exit(1)
        z.extract(us_members[0])
        # If extracted to subdirectory, move to root
        if us_members[0] != US_FILE:
            os.rename(us_members[0], US_FILE)
    print("Unzip complete.")


def pull_youtube():
    os.makedirs("out", exist_ok=True)
    download_data()
    unzip_data()

    print(f"\nLoading {US_FILE}...")
    df = pd.read_csv(US_FILE, parse_dates=["trending_date"], dayfirst=False)
    print(f"  Raw rows: {len(df):,}")

    # Normalize column name (dataset has used both spellings)
    if "categoryId" in df.columns:
        df = df.rename(columns={"categoryId": "category_id"})
    if "view_count" in df.columns:
        df = df.rename(columns={"view_count": "views"})

    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").astype("Int64")
    df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce")
    df = df.dropna(subset=["trending_date", "category_id"])

    # Filter date window
    df = df[
        (df["trending_date"] >= START) &
        (df["trending_date"] <= END)
    ]
    print(f"  After date filter ({START} → {END}): {len(df):,} rows")

    # Keep only neutral categories
    df = df[df["category_id"].isin(NEUTRAL_CATEGORY_IDS)]
    print(f"  After neutral category filter: {len(df):,} rows")

    if df.empty:
        print("No data after filtering. Check date range and category IDs.")
        return

    # Add readable category name
    df["category_name"] = df["category_id"].map(CATEGORY_NAMES).fillna("Unknown")

    # Rank within each day by view count (descending)
    df = df.sort_values(["trending_date", "views"], ascending=[True, False])
    df["rank"] = df.groupby("trending_date").cumcount() + 1

    out = df[["trending_date", "title", "category_id", "category_name", "views", "rank"]].copy()
    out = out.rename(columns={"trending_date": "date"})
    out["date"] = out["date"].dt.date

    out.to_csv(OUT_FILE, index=False)
    print(f"\nSaved {len(out):,} rows to {OUT_FILE}")
    print(f"Date range: {out['date'].min()} to {out['date'].max()}")
    print("\nCategory breakdown:")
    print(out["category_name"].value_counts().to_string())


if __name__ == "__main__":
    pull_youtube()
