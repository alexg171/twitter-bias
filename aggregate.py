"""
Aggregates raw trending CSV into two analysis-ready datasets:

1. topic_day.csv  — one row per unique topic per day
   Columns: date, topic, label, appearances (hours in trending), avg_rank, total_volume

2. category_day.csv — one row per category (right/left/neutral) per day
   Columns: date, label, topic_count, total_appearances, avg_rank, share
   This is the DiD panel dataset.

Usage:
    python aggregate.py --input out/trending_XXXXXXXX.csv
"""

import argparse
import pandas as pd
from lexicon import classify_topic

TREATMENT_DATE = pd.Timestamp("2022-10-27")


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Volume"] = (
        df["Volume"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("K", "e3")
        .str.replace("M", "e6")
    )
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df = df[df["Topic"].notna() & (df["Topic"] != "-")]
    return df


def build_topic_day(df: pd.DataFrame) -> pd.DataFrame:
    """One row per topic per day with appearance count, avg rank, total volume."""
    agg = (
        df.groupby(["Date", "Topic"])
        .agg(
            appearances=("Hour", "count"),
            avg_rank=("Rank", "mean"),
            total_volume=("Volume", "sum"),
        )
        .reset_index()
    )
    agg["label"] = agg["Topic"].apply(classify_topic)
    agg["post"] = (agg["Date"] >= TREATMENT_DATE).astype(int)
    agg["right"] = (agg["label"] == "right").astype(int)
    agg["left"]  = (agg["label"] == "left").astype(int)
    return agg.sort_values(["Date", "avg_rank"])


def build_category_day(topic_day: pd.DataFrame) -> pd.DataFrame:
    """One row per category per day — the DiD panel."""
    daily_total = topic_day.groupby("Date")["appearances"].sum().rename("daily_total")

    agg = (
        topic_day.groupby(["Date", "label"])
        .agg(
            topic_count=("Topic", "nunique"),
            total_appearances=("appearances", "sum"),
            avg_rank=("avg_rank", "mean"),
        )
        .reset_index()
    )
    agg = agg.merge(daily_total, on="Date")
    agg["share"] = agg["total_appearances"] / agg["daily_total"]
    agg["post"]  = (agg["Date"] >= TREATMENT_DATE).astype(int)
    agg["right"] = (agg["label"] == "right").astype(int)
    agg["left"]  = (agg["label"] == "left").astype(int)
    # DiD interaction term
    agg["right_x_post"] = agg["right"] * agg["post"]
    agg["left_x_post"]  = agg["left"]  * agg["post"]
    return agg.sort_values(["Date", "label"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw trending CSV")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = load_and_clean(args.input)
    print(f"  {len(df):,} raw rows | {df['Date'].nunique()} days | {df['Topic'].nunique()} unique topics")

    topic_day = build_topic_day(df)
    topic_day.to_csv("out/topic_day.csv", index=False)
    print(f"\ntopic_day.csv: {len(topic_day):,} rows")
    print(topic_day["label"].value_counts().to_string())

    category_day = build_category_day(topic_day)
    category_day.to_csv("out/category_day.csv", index=False)
    print(f"\ncategory_day.csv: {len(category_day):,} rows")
    print(category_day.head(9).to_string(index=False))
