"""
DiD Analysis: Twitter Political Bias Before/After Musk Acquisition
Treatment date: October 27, 2022

Steps:
  1. Load trending archive data + Google Trends + GDELT
  2. Classify each trending topic (right / left / neutral)
  3. Aggregate to daily counts per category
  4. Run DiD: right vs left × post-acquisition
  5. Run event study (week-relative coefficients)
  6. Save all plots to out/figures/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf
from lexicon import classify_topic

warnings.filterwarnings("ignore")

TREATMENT_DATE = pd.Timestamp("2022-10-27")
FIGURES_DIR = "out/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------

def load_trending():
    """Load the most recent trending CSV (highest timestamp)."""
    import glob
    files = glob.glob("out/trending_*.csv")
    # exclude the 2019 test file, pick largest timestamp = most recent run
    files = [f for f in files if "1773834696" not in f]
    if not files:
        # fall back to any file
        files = glob.glob("out/trending_*.csv")
    latest = max(files, key=os.path.getmtime)
    print(f"Loading trending data from: {latest}")
    df = pd.read_csv(latest, parse_dates=["Date"])
    return df


def load_google_trends():
    path = "out/google_trends.csv"
    if not os.path.exists(path):
        print("Google Trends file not found — skipping.")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    # Pivot: one column per topic, indexed by week
    df = df.pivot(index="date", columns="topic", values="interest").reset_index()
    df.columns.name = None
    # Compute a simple left/right interest index
    right_cols = [c for c in df.columns if c in [
        "Republican","Trump","gun control","free speech","censorship"]]
    left_cols  = [c for c in df.columns if c in [
        "Democrat","Biden","abortion","immigration","climate change"]]
    if right_cols:
        df["gt_right"] = df[right_cols].mean(axis=1)
    if left_cols:
        df["gt_left"]  = df[left_cols].mean(axis=1)
    return df[["date"] + (["gt_right"] if right_cols else []) +
                         (["gt_left"]  if left_cols  else [])]


def load_gdelt():
    path = "out/gdelt_coverage.csv"
    if not os.path.exists(path):
        print("GDELT file not found — skipping.")
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.tz_localize(None)  # strip UTC timezone
    right_topics = ["Republican","Trump","gun control","free speech","censorship"]
    left_topics  = ["Democrat","Biden","abortion","immigration","climate change"]
    r = df[df["topic"].isin(right_topics)].groupby("date")["article_count"].mean().rename("gdelt_right")
    l = df[df["topic"].isin(left_topics)].groupby("date")["article_count"].mean().rename("gdelt_left")
    out = pd.concat([r, l], axis=1).reset_index()
    return out


# ---------------------------------------------------------------
# 2. CLASSIFY & AGGREGATE
# ---------------------------------------------------------------

def build_daily_panel(trending: pd.DataFrame) -> pd.DataFrame:
    """
    For each day, count how many right / left / neutral topics trended.
    Returns a long-format panel: (date, category, count, share).
    """
    trending["category"] = trending["Topic"].apply(classify_topic)

    daily = (
        trending.groupby(["Date", "category"])
        .size()
        .reset_index(name="count")
    )

    # total topics per day (for share calculation)
    total = trending.groupby("Date").size().reset_index(name="total")
    daily = daily.merge(total, on="Date")
    daily["share"] = daily["count"] / daily["total"]

    daily = daily.rename(columns={"Date": "date"})
    return daily


# ---------------------------------------------------------------
# 3. DiD REGRESSION
# ---------------------------------------------------------------

def run_did(panel: pd.DataFrame, gt=None, gdelt=None):
    """
    DiD: compare right vs left categories before/after Oct 27, 2022.
    Y = share of trending topics in category c on day t
    Model: share ~ right + post + right*post + controls + week_FE
    """
    # Keep only right and left (drop neutral for clean 2x2 DiD)
    df = panel[panel["category"].isin(["right", "left"])].copy()

    df["right"]   = (df["category"] == "right").astype(int)
    df["post"]    = (df["date"] >= TREATMENT_DATE).astype(int)
    df["did"]     = df["right"] * df["post"]
    df["week"]    = df["date"].dt.to_period("W").dt.start_time

    # Merge controls
    if gt is not None:
        # forward-fill weekly google trends to daily
        df = df.merge(gt.rename(columns={"date": "week"}), on="week", how="left")
    if gdelt is not None:
        df = df.merge(gdelt.rename(columns={"date": "week"}), on="week", how="left")

    df = df.fillna(0)

    # Base model
    formula = "share ~ right + post + did"

    # Add controls if available
    ctrl_cols = [c for c in ["gt_right","gt_left","gdelt_right","gdelt_left"]
                 if c in df.columns and df[c].std() > 0]
    if ctrl_cols:
        formula += " + " + " + ".join(ctrl_cols)

    model = smf.ols(formula, data=df).fit(
        cov_type="HC3"   # heteroskedasticity-robust SEs
    )
    print("\n" + "="*60)
    print("DiD RESULTS")
    print("="*60)
    print(model.summary2().tables[1].to_string())
    print(f"\nDiD coefficient (right×post): {model.params['did']:.4f}")
    print(f"p-value:                       {model.pvalues['did']:.4f}")

    return model, df


# ---------------------------------------------------------------
# 4. EVENT STUDY
# ---------------------------------------------------------------

def run_event_study(panel: pd.DataFrame):
    """
    Replace the single post dummy with week-relative-to-treatment dummies.
    Omit week -1 (baseline). Plot coefficients to check parallel pre-trends.
    Uses manual matrix construction to avoid patsy issues with negative column names.
    """
    import statsmodels.api as sm

    df = panel[panel["category"].isin(["right", "left"])].copy()
    df["right"]    = (df["category"] == "right").astype(int)
    df["week_num"] = ((df["date"] - TREATMENT_DATE).dt.days // 7).clip(-26, 26)
    df = df[df["week_num"] != -1].copy()   # omit week -1 (baseline)

    all_weeks = sorted(df["week_num"].unique())

    # Build design matrix manually
    X = pd.DataFrame({"const": 1, "right": df["right"].values}, index=df.index)
    for w in all_weeks:
        col = (df["week_num"] == w).astype(int)
        X[f"right_x_w{w}"] = df["right"].values * col.values

    model = sm.OLS(df["share"].values, X).fit(cov_type="HC3")

    interact_cols = [c for c in X.columns if c.startswith("right_x_w")]
    coefs = {int(c.replace("right_x_w", "")): model.params[i]
             for i, c in enumerate(X.columns) if c in interact_cols}
    cis   = {int(c.replace("right_x_w", "")): 1.96 * model.bse[i]
             for i, c in enumerate(X.columns) if c in interact_cols}

    weeks  = sorted(coefs.keys())
    betas  = [coefs[w] for w in weeks]
    errors = [cis[w]   for w in weeks]

    return weeks, betas, errors


# ---------------------------------------------------------------
# 5. PLOTS
# ---------------------------------------------------------------

def plot_time_series(panel: pd.DataFrame):
    """Daily share of right vs left topics in trending list."""
    df = panel[panel["category"].isin(["right","left"])].copy()
    pivot = df.pivot_table(index="date", columns="category", values="share", aggfunc="mean")
    pivot = pivot.rolling(7, min_periods=1).mean()   # 7-day smoothing

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pivot.index, pivot.get("right", []), color="#c0392b", label="Right-leaning topics", lw=1.8)
    ax.plot(pivot.index, pivot.get("left",  []), color="#2980b9", label="Left-leaning topics",  lw=1.8)
    ax.axvline(TREATMENT_DATE, color="black", linestyle="--", lw=1.2, label="Musk acquisition (Oct 27, 2022)")
    ax.set_title("Share of Political Topics in US Twitter Trending List (7-day rolling avg)")
    ax.set_ylabel("Share of Daily Trending Topics")
    ax.set_xlabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/timeseries.png")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/timeseries.png")


def plot_event_study(weeks, betas, errors):
    """Event study plot with 95% CI bands."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="black", lw=1.2, linestyle="--", label="Musk acquisition")

    pre  = [(w, b, e) for w, b, e in zip(weeks, betas, errors) if w < 0]
    post = [(w, b, e) for w, b, e in zip(weeks, betas, errors) if w >= 0]

    for subset, color in [(pre, "#2980b9"), (post, "#c0392b")]:
        if not subset:
            continue
        ws, bs, es = zip(*subset)
        ax.errorbar(ws, bs, yerr=es, fmt="o", color=color,
                    capsize=3, markersize=4, lw=1.2)

    ax.set_title("Event Study: Right vs Left Topic Share Relative to Musk Acquisition")
    ax.set_xlabel("Weeks Relative to Acquisition")
    ax.set_ylabel("Coefficient (right − left share difference)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/event_study.png")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/event_study.png")


def plot_did_bar(model):
    """Simple bar chart of DiD coefficient with CI."""
    coef = model.params["did"]
    ci   = 1.96 * model.bse["did"]

    fig, ax = plt.subplots(figsize=(4, 4))
    color = "#c0392b" if coef > 0 else "#2980b9"
    ax.bar(["Right × Post\n(DiD estimate)"], [coef], color=color,
           alpha=0.8, width=0.4)
    ax.errorbar(["Right × Post\n(DiD estimate)"], [coef],
                yerr=ci, fmt="none", color="black", capsize=6, lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Effect on Share of Trending Topics")
    ax.set_title("DiD Coefficient\n(right-leaning vs. left-leaning topics)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/did_bar.png")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/did_bar.png")


def plot_category_breakdown(panel: pd.DataFrame):
    """Stacked area: daily composition of trending list by category."""
    pivot = panel.pivot_table(
        index="date", columns="category", values="share", aggfunc="mean"
    ).fillna(0).rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    cats    = [c for c in ["right","left","neutral"] if c in pivot.columns]
    colors  = {"right": "#c0392b", "left": "#2980b9", "neutral": "#95a5a6"}

    ax.stackplot(pivot.index,
                 [pivot[c] for c in cats],
                 labels=cats,
                 colors=[colors[c] for c in cats],
                 alpha=0.75)
    ax.axvline(TREATMENT_DATE, color="black", linestyle="--", lw=1.2,
               label="Musk acquisition")
    ax.set_title("Trending Topic Composition by Political Category (7-day rolling avg)")
    ax.set_ylabel("Share of Daily Trending Topics")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/stacked_area.png")
    plt.close()
    print(f"Saved: {FIGURES_DIR}/stacked_area.png")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    trending = load_trending()
    gt       = load_google_trends()
    gdelt    = load_gdelt()

    print(f"Trending rows: {len(trending):,}  |  Date range: "
          f"{trending['Date'].min().date()} to {trending['Date'].max().date()}")

    print("\nClassifying topics...")
    panel = build_daily_panel(trending)
    print(panel.groupby("category")["count"].sum().to_string())

    print("\nGenerating plots...")
    plot_time_series(panel)
    plot_category_breakdown(panel)

    print("\nRunning DiD...")
    model, did_df = run_did(panel, gt=gt, gdelt=gdelt)
    plot_did_bar(model)

    print("\nRunning event study...")
    weeks, betas, errors = run_event_study(panel)
    plot_event_study(weeks, betas, errors)

    # Save regression table
    table = model.summary2().tables[1]
    table.to_csv("out/did_results.csv")
    print("\nSaved regression table to out/did_results.csv")
    print("\nAll done. Check out/figures/ for plots.")
