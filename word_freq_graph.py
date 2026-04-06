"""
Word frequency comparison: Twitter trending topics pre vs post Musk acquisition.

No ideological classification — just raw topic frequency.
"Frequency" = number of unique days a topic appeared in the trending list.

Treatment date: Oct 27, 2022
Pre  window: Apr 27, 2022 – Oct 26, 2022
Post window: Oct 27, 2022 – Apr 27, 2023

Output: out/figures/word_freq_pre_post.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

TREATMENT_DATE = pd.Timestamp("2022-10-27")
PRE_START      = pd.Timestamp("2022-04-27")
POST_END       = pd.Timestamp("2023-04-27")
DATA_FILE      = "out/twitter_1774593752.csv"
FIGURES_DIR    = "out/figures"
TOP_N          = 30

os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})


def load_and_dedup(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Loaded {len(df):,} rows from {path}")
    # One row per (Date, Topic) — ignore repeated hourly appearances
    df = df.drop_duplicates(subset=["Date", "Topic"])
    print(f"After dedup (unique date×topic pairs): {len(df):,}")
    return df


def topic_freq(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Days a topic appeared in the window (1 per day, regardless of hour/rank)."""
    mask = (df["Date"] >= start) & (df["Date"] < end)
    sub  = df[mask]
    # count distinct dates per topic
    freq = sub.groupby("Topic")["Date"].nunique().sort_values(ascending=False)
    return freq


def plot_side_by_side(pre_freq: pd.Series, post_freq: pd.Series, top_n: int = TOP_N):
    pre_top  = pre_freq.head(top_n)
    post_top = post_freq.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    for ax, freq, title, color in [
        (axes[0], pre_top,  f"Pre-Treatment  (Apr 27 – Oct 26, 2022)\nTop {top_n} topics by days trended", "#4e79a7"),
        (axes[1], post_top, f"Post-Treatment  (Oct 27, 2022 – Apr 27, 2023)\nTop {top_n} topics by days trended", "#e15759"),
    ]:
        bars = ax.barh(
            range(len(freq)),
            freq.values,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_yticks(range(len(freq)))
        ax.set_yticklabels(freq.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Days in Trending List")
        ax.set_title(title, pad=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        # value labels
        for bar, val in zip(bars, freq.values):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va="center", ha="left", fontsize=7.5
            )

    fig.suptitle(
        "US Twitter Trending Topics: Before vs. After Musk Acquisition (Oct 27, 2022)",
        fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out_path = f"{FIGURES_DIR}/word_freq_pre_post.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")
    return out_path


def summarize_overlap(pre_freq: pd.Series, post_freq: pd.Series, top_n: int = TOP_N):
    pre_set  = set(pre_freq.head(top_n).index)
    post_set = set(post_freq.head(top_n).index)
    both     = pre_set & post_set
    new_post = post_set - pre_set
    dropped  = pre_set - post_set

    print(f"\nTop-{top_n} overlap summary:")
    print(f"  Topics in BOTH windows  : {len(both)}")
    print(f"  NEW in post-treatment   : {len(new_post)}")
    print(f"  Dropped after treatment : {len(dropped)}")
    if new_post:
        print(f"\n  New post-treatment topics:")
        for t in sorted(new_post):
            print(f"    {t}  ({post_freq[t]:.0f} days)")
    if dropped:
        print(f"\n  Topics that left top-{top_n} after treatment:")
        for t in sorted(dropped):
            print(f"    {t}  ({pre_freq.get(t, 0):.0f} days pre)")


if __name__ == "__main__":
    df = load_and_dedup(DATA_FILE)

    print(f"\nDate range in file: {df['Date'].min().date()} to {df['Date'].max().date()}")

    pre_freq  = topic_freq(df, PRE_START,      TREATMENT_DATE)
    post_freq = topic_freq(df, TREATMENT_DATE, POST_END + pd.Timedelta(days=1))

    print(f"\nPre-treatment  unique topics: {len(pre_freq):,}")
    print(f"Post-treatment unique topics: {len(post_freq):,}")

    plot_side_by_side(pre_freq, post_freq)
    summarize_overlap(pre_freq, post_freq)

    print("\nDone.")
