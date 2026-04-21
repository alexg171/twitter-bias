"""
category_plots.py
Generates per-category parallel trends and event study plots.

Dropped categories (too noisy / confounded):
  - lgbtq_social  : event-driven spikes, no common rhythm with control
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from category_lexicon import classify_category, DISPLAY_GROUPS, CAT_COLORS

TREATMENT = pd.Timestamp("2022-10-27")
PRE_START  = pd.Timestamp("2020-10-27")
POST_END   = pd.Timestamp("2024-10-27")

os.makedirs("out/figures/parallel_trends", exist_ok=True)
os.makedirs("out/figures/event_study",    exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Categories to plot ────────────────────────────────────────────────────────
PLOT_CATS = [
    "wrestling", "combat_sports", "sports_nba", "sports_nfl",
    "sports_mlb", "sports_nhl", "sports_soccer", "sports_college",
    "sports_womens", "sports_other", "reality_tv", "entertainment",
    "taylor_swift", "fandom", "tech_gaming", "religious", "news_politics",
]

# ── Reddit subreddit map (matched controls) ───────────────────────────────────
CAT_TO_SUBREDDIT = {
    "wrestling":      ["SquaredCircle"],
    "combat_sports":  ["MMA"],
    "sports_nba":     ["nba"],
    "sports_nfl":     ["nfl"],
    "sports_nhl":     ["hockey"],
    "sports_soccer":  ["soccer"],
    "sports_college": ["CFB"],
    "sports_womens":  ["wnba", "NWSL"],
    "sports_other":   ["formula1", "golf"],
    "reality_tv":     ["BravoRealHousewives", "realtv"],
    "entertainment":  ["television", "movies", "Music"],
    "fandom":         ["anime", "kpop"],
    "tech_gaming":    ["gaming", "technology"],
    "religious":      ["Christianity", "religion"],
    "news_politics":  ["worldnews", "news", "politics", "PoliticalDiscussion",
                       "conservative", "republican", "democrats", "Liberal",
                       "progressive", "libertarian", "NeutralPolitics"],
}

# Merged Twitter category sources (for news_politics)
MERGED_SOURCES = {
    "news_politics": ["news_events", "politics"],
}


# ── 1. Load Twitter daily category shares ────────────────────────────────────

def build_twitter_panel() -> pd.DataFrame:
    print("Building Twitter panel …")
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df["category"] = df["Topic"].apply(classify_category)

    # Merge news_events + politics → news_politics
    df["category"] = df["category"].replace(
        {"news_events": "news_politics", "politics": "news_politics"}
    )

    all_cats = PLOT_CATS
    daily_total = df.groupby("Date")["Topic"].count()
    daily_counts = df.groupby(["Date", "category"])["Topic"].count().unstack(fill_value=0)

    share = pd.DataFrame(index=daily_total.index)
    for cat in all_cats:
        if cat in daily_counts.columns:
            share[cat] = daily_counts[cat] / daily_total
        else:
            share[cat] = 0.0

    share = share.loc[
        (share.index >= PRE_START) & (share.index <= POST_END)
    ]
    return share


# ── 2. Load Reddit category controls ─────────────────────────────────────────

def build_reddit_controls():
    print("Building Reddit controls …")
    cat_tsv = "out/reddit_category.tsv"

    rc = pd.read_csv(cat_tsv, sep="\t", parse_dates=["date"])
    rc = rc[(rc["date"] >= PRE_START) & (rc["date"] <= POST_END)]

    full_idx = pd.date_range(PRE_START, POST_END, freq="D")

    # Generic baseline: aggregate all subreddits in the file
    generic_raw = rc.groupby("date")["n_posts"].sum()
    generic = generic_raw.reindex(full_idx, fill_value=np.nan).interpolate()

    # Matched subreddit controls per category
    cat_controls = {}
    for cat, subs in CAT_TO_SUBREDDIT.items():
        subset = rc[rc["subreddit"].isin(subs)]
        if len(subset) > 30:
            daily = subset.groupby("date")["n_posts"].sum()
            daily = daily.reindex(full_idx, fill_value=np.nan).interpolate()
            cat_controls[cat] = daily

    n_matched  = len(cat_controls)
    n_fallback = len(PLOT_CATS) - n_matched
    print(f"  Matched subreddits: {n_matched}/{len(PLOT_CATS)} categories")
    print(f"  Fallback (generic): {max(n_fallback, 0)} categories")
    return cat_controls, generic


# ── 3. Normalize ─────────────────────────────────────────────────────────────

def normalize(series: pd.Series, pre_mask: np.ndarray) -> pd.Series:
    pre_mean = series[pre_mask].mean()
    if pre_mean == 0 or np.isnan(pre_mean):
        return pd.Series(np.nan, index=series.index)
    return (series - pre_mean) / pre_mean * 100


# ── 4. Parallel Trends plots ──────────────────────────────────────────────────

def plot_parallel_trends(tw_share, cat_controls, generic):
    print("\nGenerating parallel trends plots …")
    all_dates = tw_share.index
    pre_mask  = np.array(all_dates < TREATMENT)

    for cat in PLOT_CATS:
        label = DISPLAY_GROUPS.get(cat, cat)
        color = CAT_COLORS.get(cat, "#4e79a7")

        tw_norm  = normalize(tw_share[cat], pre_mask)
        tw_roll  = tw_norm.rolling(14, center=True, min_periods=5).mean()

        if cat in cat_controls:
            rd_series = cat_controls[cat]
            subs = CAT_TO_SUBREDDIT.get(cat, [])
            ctrl_label = f"Reddit control ({' + '.join(subs)})"
        else:
            rd_series  = generic
            ctrl_label = "Reddit control (generic political)"

        # Align index
        rd_series = rd_series.reindex(all_dates)
        rd_norm   = normalize(rd_series, pre_mask)
        rd_roll   = rd_norm.rolling(14, center=True, min_periods=5).mean()

        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(all_dates, tw_roll, color="#2166ac", lw=2,    label="Twitter (outcome)")
        ax.plot(all_dates, rd_roll, color="#d62728", lw=1.5,
                linestyle="--", label=ctrl_label)
        ax.axvline(TREATMENT, color="black", lw=1.0, linestyle=":", label="Treatment (Oct 27 2022)")
        ax.axhline(0, color="grey", lw=0.6, linestyle=":")

        ax.set_ylabel("% deviation from pre-treatment mean", fontsize=9)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=7, loc="upper left")
        fig.tight_layout()

        out = f"out/figures/parallel_trends/{cat}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(PLOT_CATS)} parallel trends plots.")


# ── 5. Event Study (quarterly bootstrap) ─────────────────────────────────────

def run_event_study_bootstrap(tw_norm, rd_norm, all_dates, n_boot=500):
    """
    Quarterly event study with bootstrap 95% CI.
    Normalizes so the PRE-PERIOD MEAN gap = 0 (not Q-1 specifically),
    which keeps pre-period quarters hovering around zero as expected.
    """
    tw_norm = np.asarray(tw_norm, dtype=float)
    rd_norm = np.asarray(rd_norm, dtype=float)

    def quarter_bin(d):
        delta = (pd.Timestamp(d) - TREATMENT).days
        return int(np.floor(delta / 91.25))

    q_bins  = np.array([quarter_bin(d) for d in all_dates])
    q_range = range(-8, 9)

    # Raw daily gap
    raw_gap = tw_norm - rd_norm

    # Normalize: subtract the mean of the raw gap in the pre-period
    # so that the average pre-period gap = 0
    pre_mask = q_bins < 0
    pre_mean_gap = raw_gap[pre_mask].mean() if pre_mask.sum() > 0 else 0.0
    gap = raw_gap - pre_mean_gap

    # Bootstrap CI per quarter: resample daily Twitter deviations within quarter
    gaps, ci_lo, ci_hi = [], [], []
    for q in q_range:
        mask = q_bins == q
        n    = mask.sum()
        if n < 5:
            gaps.append(np.nan); ci_lo.append(np.nan); ci_hi.append(np.nan)
            continue
        g_obs     = gap[mask].mean()
        tw_q      = tw_norm[mask]
        rd_q_mean = rd_norm[mask].mean()
        boot = [
            np.random.choice(tw_q, size=n, replace=True).mean() - rd_q_mean - pre_mean_gap
            for _ in range(n_boot)
        ]
        ci_lo.append(np.percentile(boot, 2.5))
        ci_hi.append(np.percentile(boot, 97.5))
        gaps.append(g_obs)

    return list(q_range), np.array(gaps), np.array(ci_lo), np.array(ci_hi)


def plot_event_studies(tw_share, cat_controls, generic):
    print("\nGenerating event study plots …")
    all_dates = tw_share.index
    pre_mask  = np.array(all_dates < TREATMENT)

    for cat in PLOT_CATS:
        label = DISPLAY_GROUPS.get(cat, cat)

        tw_norm = normalize(tw_share[cat], pre_mask)

        if cat in cat_controls:
            rd_series = cat_controls[cat].reindex(all_dates)
        else:
            rd_series = generic.reindex(all_dates)
        rd_norm = normalize(rd_series, pre_mask)

        # Drop NaN positions (use .values to avoid index alignment issues)
        valid   = (tw_norm.notna() & rd_norm.notna()).values
        tw_v    = tw_norm.values[valid]
        rd_v    = rd_norm.values[valid]
        dates_v = all_dates[valid]

        quarters, gaps, ci_lo, ci_hi = run_event_study_bootstrap(
            tw_v, rd_v, dates_v, n_boot=500
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        q_arr   = np.array(quarters)
        gaps    = np.array(gaps)
        ci_lo   = np.array(ci_lo)
        ci_hi   = np.array(ci_hi)
        valid_q = ~np.isnan(gaps)

        # Background shading: green=pre, pink=post
        ax.axvspan(q_arr[0] - 0.5, -0.5, alpha=0.06, color="green", zorder=0)
        ax.axvspan(-0.5, q_arr[-1] + 0.5, alpha=0.06, color="red",   zorder=0)

        # CI band and line — always blue so it reads clearly against backgrounds
        CI_COLOR   = "#2166ac"
        LINE_COLOR = "#1a4e8a"

        ax.fill_between(q_arr[valid_q], ci_lo[valid_q], ci_hi[valid_q],
                        alpha=0.20, color=CI_COLOR, label="95% CI (bootstrap)", zorder=2)
        ax.plot(q_arr[valid_q], gaps[valid_q],
                color=LINE_COLOR, lw=2, marker="o", ms=5,
                label="Twitter − Reddit gap", zorder=3)
        ax.axvline(-0.5, color="black", lw=1.3, linestyle="--",
                   label="Treatment (Oct 2022)", zorder=4)
        ax.axhline(0, color="grey", lw=0.7, linestyle=":", zorder=1)

        ax.set_xticks(q_arr)
        xlabels = [f"Q{q:+d}" if q != 0 else "Q0\n(treat)" for q in q_arr]
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("Quarters relative to treatment", fontsize=9)
        ax.set_ylabel("Twitter − Reddit gap\n(% dev from pre-mean)", fontsize=9)
        ax.set_title(f"Event Study — {label}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        out = f"out/figures/event_study/{cat}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close()

    print(f"  Saved {len(PLOT_CATS)} event study plots.")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw_share = build_twitter_panel()
    cat_controls, generic = build_reddit_controls()

    plot_parallel_trends(tw_share, cat_controls, generic)
    plot_event_studies(tw_share, cat_controls, generic)
    print("\nDone.")
