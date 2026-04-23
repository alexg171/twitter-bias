"""
Generates per-category parallel trends and event study plots.
Compares Twitter trending trends (treated) vs Reddit activity (control) around Musk's Oct 27, 2022 acquisition.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from category_lexicon import classify_category, DISPLAY_GROUPS, CAT_COLORS
from category_subreddit_mapping import CAT_TO_SUBREDDIT

TREATMENT = pd.Timestamp("2022-10-27")
PRE_START = pd.Timestamp("2020-10-27")
POST_END = pd.Timestamp("2024-10-27")

os.makedirs("out/figures/parallel_trends", exist_ok=True)
os.makedirs("out/figures/event_study", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PLOT_CATS = [
    "wrestling", "combat_sports", "sports_nba", "sports_nfl",
    "sports_mlb", "sports_nhl", "sports_soccer", "sports_college",
    "sports_womens", "reality_tv", "entertainment",
    "taylor_swift", "fandom", "news_politics",
]


def build_twitter_panel() -> pd.DataFrame:
    """Load Twitter trending data and compute daily category shares."""
    print("Loading Twitter trends …")
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df["category"] = df["Topic"].apply(classify_category)

    df["category"] = df["category"].replace({
        "news_events": "news_politics",
        "politics": "news_politics",
    })

    daily_total = df.groupby("Date")["Topic"].count()
    daily_counts = df.groupby(["Date", "category"])["Topic"].count().unstack(fill_value=0)

    share = pd.DataFrame(index=daily_total.index)
    for cat in PLOT_CATS:
        share[cat] = daily_counts.get(cat, 0) / daily_total

    return share[(share.index >= PRE_START) & (share.index <= POST_END)]


def build_reddit_controls() -> tuple:
    """Load Reddit data and build matched category controls."""
    print("Loading Reddit controls …")
    rc = pd.read_csv("out/reddit_category.tsv", sep="\t", parse_dates=["date"])
    rc = rc[(rc["date"] >= PRE_START) & (rc["date"] <= POST_END)]

    full_idx = pd.date_range(PRE_START, POST_END, freq="D")
    generic = rc.groupby("date")["n_posts"].sum().reindex(full_idx, fill_value=np.nan).interpolate()

    cat_controls = {}
    for cat, subs in CAT_TO_SUBREDDIT.items():
        subset = rc[rc["subreddit"].isin(subs)]
        if len(subset) > 30:
            daily = subset.groupby("date")["n_posts"].sum().reindex(full_idx, fill_value=np.nan).interpolate()
            cat_controls[cat] = daily

    n_matched = len(cat_controls)
    n_missing = len(PLOT_CATS) - n_matched
    print(f"  Matched subreddits: {n_matched}/{len(PLOT_CATS)}")
    if n_missing > 0:
        print(f"  Fallback to generic: {n_missing}")

    return cat_controls, generic


def log_normalize(series: pd.Series, pre_mask: np.ndarray) -> pd.Series:
    """log(y + c) demeaned by pre-period log mean. Same scale for Twitter and Reddit."""
    const = 1e-4 if series.max() <= 1 else 1.0
    log_s = np.log(series.clip(lower=0) + const)
    log_pre_mean = log_s[pre_mask].mean()
    if np.isnan(log_pre_mean):
        return pd.Series(np.nan, index=series.index)
    return log_s - log_pre_mean


def plot_parallel_trends(tw_share: pd.DataFrame, cat_controls: dict, generic: pd.Series) -> None:
    """Plot parallel trends: log-deviation Twitter vs Reddit per category."""
    print("\nGenerating parallel trends plots …")
    all_dates = tw_share.index
    pre_mask  = np.array(all_dates < TREATMENT)

    for cat in PLOT_CATS:
        label = DISPLAY_GROUPS.get(cat, cat)
        tw_lnorm = log_normalize(tw_share[cat], pre_mask)
        tw_roll  = tw_lnorm.rolling(14, center=True, min_periods=5).mean()

        if cat in cat_controls:
            rd_series  = cat_controls[cat].reindex(all_dates)
            subs       = CAT_TO_SUBREDDIT.get(cat, [])
            ctrl_label = f"Reddit ({' + '.join(subs)})"
        else:
            rd_series  = generic.reindex(all_dates)
            ctrl_label = "Reddit (generic)"

        rd_lnorm = log_normalize(rd_series, pre_mask)
        rd_roll  = rd_lnorm.rolling(14, center=True, min_periods=5).mean()

        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(all_dates, tw_roll, color="#2166ac", lw=2, label="Twitter (treated)")
        ax.plot(all_dates, rd_roll, color="#d62728", lw=1.5, linestyle="--", label=ctrl_label)
        ax.axvline(TREATMENT, color="black", lw=1.0, linestyle=":", label="Treatment (Oct 27 2022)")
        ax.axhline(0, color="grey", lw=0.6, linestyle=":")
        ax.set_ylabel("log-deviation from pre-treatment mean", fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"out/figures/parallel_trends/{cat}.png", bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(PLOT_CATS)} plots.")


def run_event_study_bootstrap(tw_norm: np.ndarray, rd_norm: np.ndarray, all_dates: np.ndarray, n_boot: int = 500) -> tuple:
    """Run quarterly event study with bootstrap 95% CI."""
    tw_norm = np.asarray(tw_norm, dtype=float)
    rd_norm = np.asarray(rd_norm, dtype=float)

    def quarter_bin(d):
        return int(np.floor((pd.Timestamp(d) - TREATMENT).days / 91.25))

    q_bins = np.array([quarter_bin(d) for d in all_dates])
    raw_gap = tw_norm - rd_norm

    pre_mask = q_bins < 0
    pre_mean_gap = raw_gap[pre_mask].mean() if pre_mask.sum() > 0 else 0.0
    gap = raw_gap - pre_mean_gap

    gaps, ci_lo, ci_hi = [], [], []
    for q in range(-8, 9):
        mask = q_bins == q
        n = mask.sum()
        if n < 5:
            gaps.append(np.nan)
            ci_lo.append(np.nan)
            ci_hi.append(np.nan)
            continue

        g_obs = gap[mask].mean()
        tw_q = tw_norm[mask]
        rd_q_mean = rd_norm[mask].mean()
        boot = [np.random.choice(tw_q, size=n, replace=True).mean() - rd_q_mean - pre_mean_gap for _ in range(n_boot)]
        ci_lo.append(np.percentile(boot, 2.5))
        ci_hi.append(np.percentile(boot, 97.5))
        gaps.append(g_obs)

    return list(range(-8, 9)), np.array(gaps), np.array(ci_lo), np.array(ci_hi)


def plot_event_studies(tw_share: pd.DataFrame, cat_controls: dict, generic: pd.Series) -> None:
    """Plot event studies: quarterly Twitter-Reddit gaps with bootstrap CI."""
    print("\nGenerating event study plots …")
    all_dates = tw_share.index
    pre_mask = np.array(all_dates < TREATMENT)

    for cat in PLOT_CATS:
        label = DISPLAY_GROUPS.get(cat, cat)
        tw_norm   = log_normalize(tw_share[cat], pre_mask)
        rd_series = cat_controls[cat].reindex(all_dates) if cat in cat_controls else generic.reindex(all_dates)
        rd_norm   = log_normalize(rd_series, pre_mask)

        valid = (tw_norm.notna() & rd_norm.notna()).values
        quarters, gaps, ci_lo, ci_hi = run_event_study_bootstrap(
            tw_norm.values[valid], rd_norm.values[valid], all_dates[valid], n_boot=500
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        q_arr = np.array(quarters)
        valid_q = ~np.isnan(gaps)

        ax.axvspan(q_arr[0] - 0.5, -0.5, alpha=0.06, color="green", zorder=0)
        ax.axvspan(-0.5, q_arr[-1] + 0.5, alpha=0.06, color="red", zorder=0)
        ax.fill_between(q_arr[valid_q], ci_lo[valid_q], ci_hi[valid_q], alpha=0.20, color="#2166ac", label="95% CI", zorder=2)
        ax.plot(q_arr[valid_q], gaps[valid_q], color="#1a4e8a", lw=2, marker="o", ms=5, label="Twitter − Reddit gap", zorder=3)
        ax.axvline(-0.5, color="black", lw=1.3, linestyle="--", label="Treatment", zorder=4)
        ax.axhline(0, color="grey", lw=0.7, linestyle=":", zorder=1)

        ax.set_xticks(q_arr)
        ax.set_xticklabels([f"Q{q:+d}" if q != 0 else "Q0\n(treat)" for q in q_arr], fontsize=8)
        ax.set_xlabel("Quarters relative to treatment", fontsize=9)
        ax.set_ylabel("Twitter − Reddit gap (log-deviation units)", fontsize=9)
        ax.set_title(f"Event Study — {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"out/figures/event_study/{cat}.png", bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(PLOT_CATS)} plots.")


if __name__ == "__main__":
    tw_share = build_twitter_panel()
    cat_controls, generic = build_reddit_controls()
    plot_parallel_trends(tw_share, cat_controls, generic)
    plot_event_studies(tw_share, cat_controls, generic)
    print("\nDone.")

