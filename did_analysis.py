"""
DiD Analysis: Did Musk's Twitter acquisition shift political trending content rightward?

Design
------
  Treated  : Twitter  (Musk acquisition Oct 27, 2022)
  Control  : Reddit political subreddits (Lai et al. 2024 ideology scores)
  Outcome  : right_share = right / (right + left) political content per day
             — Twitter:  keyword-classified trending topics (lexicon.py)
             — Reddit:   subreddit label from Lai et al. (right vs left only;
                         center subs excluded from denominator for comparability)
  Window   : Apr 27, 2022 – Apr 27, 2023  (±6 months)

Model
-----
  right_share ~ twitter + post + did + C(dow)
    twitter = 1 if Twitter, 0 if Reddit
    post    = 1 if date >= Oct 27, 2022
    did     = twitter × post   ← TREATMENT EFFECT

Outputs
-------
  out/figures/did_timeseries.png    — raw right_share, both platforms
  out/figures/did_event_study.png   — event study (week-relative coefficients)
  out/figures/did_placebo.png       — placebo DiD at 3 fake treatment dates
  out/figures/did_bar.png           — DiD coefficient with 95% CI
  out/did_results.csv               — full regression table
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from lexicon import classify_topic

# ── constants ────────────────────────────────────────────────────────────────
TREATMENT  = pd.Timestamp("2022-10-27")
PRE_START  = pd.Timestamp("2022-04-27")
POST_END   = pd.Timestamp("2023-04-27")
FIGURES    = "out/figures"
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


# ── 1. BUILD TWITTER DAILY PANEL ─────────────────────────────────────────────

def build_twitter_panel() -> pd.DataFrame:
    print("Loading Twitter data...")
    df = pd.read_csv("out/twitter_trending_4yr.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date", "Topic"])
    df = df[(df["Date"] >= PRE_START) & (df["Date"] <= POST_END)]
    df["label"] = df["Topic"].apply(classify_topic)

    pol = df[df["label"].isin(["right", "left"])].copy()
    daily = (
        pol.groupby([pol["Date"].dt.date, "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in daily:
            daily[col] = 0

    daily["total"]       = daily["right"] + daily["left"]
    daily                = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily["platform"]    = "twitter"

    print(f"  Twitter: {len(daily)} days, "
          f"avg {daily['total'].mean():.1f} pol topics/day, "
          f"right_share={daily['right_share'].mean():.3f}")
    return daily[["right_share", "platform"]].reset_index()


# ── 2. BUILD REDDIT DAILY PANEL ──────────────────────────────────────────────

def build_reddit_panel() -> pd.DataFrame:
    path = "out/reddit_trending.tsv"
    if not os.path.exists(path):
        print("Reddit data not found — skipping.")
        return pd.DataFrame()

    print("Loading Reddit data...")
    df = pd.read_csv(path, sep="\t", parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df[(df["date"] >= PRE_START.date()) & (df["date"] <= POST_END.date())]

    # right_share = right / (right + left) — exclude center for comparability
    pol = df[df["label"].isin(["right", "left"])].copy()
    daily = (
        pol.groupby(["date", "label"])
        .size()
        .unstack(fill_value=0)
        .rename_axis("date")
    )
    for col in ["right", "left"]:
        if col not in daily:
            daily[col] = 0

    daily["total"]       = daily["right"] + daily["left"]
    daily                = daily[daily["total"] >= 1]
    daily["right_share"] = daily["right"] / daily["total"]
    daily["platform"]    = "reddit"

    print(f"  Reddit:  {len(daily)} days, "
          f"avg {daily['total'].mean():.1f} pol posts/day, "
          f"right_share={daily['right_share'].mean():.3f}")
    return daily[["right_share", "platform"]].reset_index()


# ── 3. BUILD DiD PANEL ───────────────────────────────────────────────────────

def build_did_panel(tw: pd.DataFrame, rd: pd.DataFrame) -> pd.DataFrame:
    panel = pd.concat([tw, rd], ignore_index=True)
    panel["date"]    = pd.to_datetime(panel["date"])
    panel["twitter"] = (panel["platform"] == "twitter").astype(int)
    panel["post"]    = (panel["date"] >= TREATMENT).astype(int)
    panel["did"]     = panel["twitter"] * panel["post"]
    panel["dow"]     = panel["date"].dt.dayofweek   # 0=Mon … 6=Sun
    panel["week"]    = ((panel["date"] - TREATMENT).dt.days // 7).clip(-26, 26)
    return panel.dropna(subset=["right_share"])


# ── 4. DiD REGRESSION ────────────────────────────────────────────────────────

def run_did(panel: pd.DataFrame):
    model = smf.ols(
        "right_share ~ twitter + post + did + C(dow)",
        data=panel
    ).fit(cov_type="HC3")

    coef = model.params["did"]
    pval = model.pvalues["did"]
    ci   = model.conf_int().loc["did"]

    print("\n" + "="*55)
    print("DiD RESULTS")
    print("="*55)
    print(model.summary2().tables[1].to_string())
    print(f"\n  DiD coeff (twitter×post) : {coef:+.4f}")
    print(f"  95% CI                   : [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print(f"  p-value                  : {pval:.4f}")
    print(f"  Interpretation           : After Musk acquisition, Twitter right_share")
    print(f"                             shifted {coef*100:+.2f}pp relative to Reddit trend")

    table = model.summary2().tables[1]
    table.to_csv("out/did_results.csv")
    print("  Saved: out/did_results.csv")
    return model


# ── 5. EVENT STUDY ───────────────────────────────────────────────────────────

def run_event_study(panel: pd.DataFrame):
    """
    Interact twitter × week-relative dummies. Omit week -1 (baseline).
    Pre-treatment coefficients near 0 → parallel trends supported.
    """
    df = panel[panel["week"] != -1].copy()
    all_weeks = sorted(df["week"].unique())

    # Build design matrix manually (avoids patsy issues with negative col names)
    X = pd.DataFrame({
        "const":   1,
        "twitter": df["twitter"].values,
        "dow_1":   (df["dow"] == 1).astype(int).values,
        "dow_2":   (df["dow"] == 2).astype(int).values,
        "dow_3":   (df["dow"] == 3).astype(int).values,
        "dow_4":   (df["dow"] == 4).astype(int).values,
        "dow_5":   (df["dow"] == 5).astype(int).values,
        "dow_6":   (df["dow"] == 6).astype(int).values,
    }, index=df.index)

    for w in all_weeks:
        X[f"tw_w{w}"] = df["twitter"].values * (df["week"] == w).astype(int).values

    model = sm.OLS(df["right_share"].values, X).fit(cov_type="HC3")

    interact_cols = [c for c in X.columns if c.startswith("tw_w")]
    weeks, betas, cis = [], [], []
    for c in interact_cols:
        w = int(c.replace("tw_w", ""))
        idx = list(X.columns).index(c)
        weeks.append(w)
        betas.append(model.params[idx])
        cis.append(1.96 * model.bse[idx])

    # Sort by week
    order = np.argsort(weeks)
    weeks = [weeks[i] for i in order]
    betas = [betas[i] for i in order]
    cis   = [cis[i]   for i in order]

    return weeks, betas, cis


# ── 6. PLACEBO TEST ──────────────────────────────────────────────────────────

def run_placebo(panel: pd.DataFrame, real_coef: float) -> pd.DataFrame:
    """
    Run the same DiD at 3 fake treatment dates + the real one.
    A valid design shows only the real date yields a significant result.
    """
    fake_dates = [
        pd.Timestamp("2022-06-01"),
        pd.Timestamp("2022-08-01"),
        pd.Timestamp("2022-10-01"),
        TREATMENT,                      # real
    ]
    results = []
    for d in fake_dates:
        p = panel.copy()
        p["post_fake"] = (p["date"] >= d).astype(int)
        p["did_fake"]  = p["twitter"] * p["post_fake"]
        try:
            m = smf.ols("right_share ~ twitter + post_fake + did_fake + C(dow)",
                        data=p).fit(cov_type="HC3")
            results.append({
                "date":  d,
                "coef":  m.params["did_fake"],
                "ci":    1.96 * m.bse["did_fake"],
                "pval":  m.pvalues["did_fake"],
                "real":  (d == TREATMENT),
            })
        except Exception as e:
            print(f"  Placebo {d.date()} failed: {e}")

    return pd.DataFrame(results)


# ── 7. PLOTS ─────────────────────────────────────────────────────────────────

def plot_timeseries(panel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4))

    for platform, color, label in [
        ("twitter", "#c0392b", "Twitter (treated)"),
        ("reddit",  "#2980b9", "Reddit (control)"),
    ]:
        sub = panel[panel["platform"] == platform].set_index("date").sort_index()
        smooth = sub["right_share"].rolling(14, min_periods=3, center=True).mean()
        ax.plot(sub.index, sub["right_share"],
                color=color, lw=0.5, alpha=0.25)
        ax.plot(smooth.index, smooth,
                color=color, lw=2.2, label=label)

    ax.axvline(TREATMENT, color="black", lw=1.4, ls="--",
               label="Musk acquisition (Oct 27, 2022)")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.5)
    ax.set_ylabel("Right share of political content")
    ax.set_title("Right-Coded Share of Political Content: Twitter vs Reddit (14-day rolling avg)")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_timeseries.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_timeseries.png")


def plot_event_study(weeks, betas, cis):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Musk acquisition")
    ax.axvspan(-26, -1, alpha=0.04, color="blue")
    ax.axvspan(0,   26, alpha=0.04, color="red")

    pre  = [(w, b, e) for w, b, e in zip(weeks, betas, cis) if w < 0]
    post = [(w, b, e) for w, b, e in zip(weeks, betas, cis) if w >= 0]

    for subset, color in [(pre, "#2980b9"), (post, "#c0392b")]:
        if not subset:
            continue
        ws, bs, es = zip(*subset)
        ax.errorbar(ws, bs, yerr=es, fmt="o", color=color,
                    capsize=3, markersize=4, lw=1.2)

    ax.set_title("Event Study: Twitter vs Reddit Right Share (week relative to acquisition)")
    ax.set_xlabel("Weeks relative to Musk acquisition")
    ax.set_ylabel("Coeff: Twitter − Reddit right_share difference")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_event_study.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_event_study.png")


def plot_did_bar(model):
    coef = model.params["did"]
    ci   = 1.96 * model.bse["did"]
    pval = model.pvalues["did"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    color = "#c0392b" if coef > 0 else "#2980b9"
    ax.bar(["Twitter × Post\n(DiD estimate)"], [coef],
           color=color, alpha=0.85, width=0.45)
    ax.errorbar(["Twitter × Post\n(DiD estimate)"], [coef],
                yerr=ci, fmt="none", color="black", capsize=8, lw=2)
    ax.axhline(0, color="black", lw=0.8)
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else "n.s."
    ax.text(0, coef + (ci + 0.005) * np.sign(coef),
            f"{coef:+.3f} {sig}", ha="center", va="bottom" if coef > 0 else "top",
            fontsize=11, fontweight="bold")
    ax.set_ylabel("Effect on right_share")
    ax.set_title(f"DiD Coefficient\n(p = {pval:.3f})")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_bar.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_bar.png")


def plot_placebo(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))

    colors = ["#aaa", "#aaa", "#aaa", "#c0392b"]
    labels = [f"{r['date'].strftime('%b %d, %Y')}{' ← REAL' if r['real'] else ' (placebo)'}"
              for _, r in results.iterrows()]

    for i, (_, r) in enumerate(results.iterrows()):
        ax.bar(i, r["coef"], color=colors[i], alpha=0.85, width=0.55)
        ax.errorbar(i, r["coef"], yerr=r["ci"],
                    fmt="none", color="black", capsize=6, lw=1.5)
        sig = "***" if r["pval"] < 0.01 else "**" if r["pval"] < 0.05 \
              else "*" if r["pval"] < 0.1 else "n.s."
        ax.text(i, r["coef"] + r["ci"] * np.sign(r["coef"]) + 0.003,
                sig, ha="center", fontsize=11)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("DiD Coefficient")
    ax.set_title("Placebo Test: DiD Coefficient at Fake vs Real Treatment Dates")
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/did_placebo.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES}/did_placebo.png")


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw    = build_twitter_panel()
    rd    = build_reddit_panel()

    if rd.empty:
        print("\nERROR: Reddit data missing. Run reddit_trending_pull.py first.")
        sys.exit(1)

    panel = build_did_panel(tw, rd)

    tw_days = panel[panel["platform"] == "twitter"]["date"].nunique()
    rd_days = panel[panel["platform"] == "reddit"]["date"].nunique()
    print(f"\nDiD panel: {len(panel)} obs  |  Twitter: {tw_days} days  |  Reddit: {rd_days} days")

    pre_tw  = panel[(panel["platform"]=="twitter") & (panel["post"]==0)]["right_share"].mean()
    pre_rd  = panel[(panel["platform"]=="reddit")  & (panel["post"]==0)]["right_share"].mean()
    post_tw = panel[(panel["platform"]=="twitter") & (panel["post"]==1)]["right_share"].mean()
    post_rd = panel[(panel["platform"]=="reddit")  & (panel["post"]==1)]["right_share"].mean()
    print(f"\n  Raw 2×2 DiD table:")
    print(f"                  Pre       Post      Diff")
    fmt = lambda x: f"{x:.3f}" if not np.isnan(x) else "  n/a "
    print(f"  Twitter       {fmt(pre_tw)}     {fmt(post_tw)}     {fmt(post_tw-pre_tw)}")
    print(f"  Reddit        {fmt(pre_rd)}     {fmt(post_rd)}     {fmt(post_rd-pre_rd)}")
    raw_did = (post_tw - pre_tw) - (post_rd - pre_rd)
    print(f"  DiD (raw)                           {fmt(raw_did)}")

    print("\nPlotting time series...")
    plot_timeseries(panel)

    print("\nRunning DiD regression...")
    model = run_did(panel)
    plot_did_bar(model)

    print("\nRunning event study...")
    weeks, betas, cis = run_event_study(panel)
    plot_event_study(weeks, betas, cis)

    print("\nRunning placebo tests...")
    placebo_results = run_placebo(panel, model.params["did"])
    print(placebo_results[["date","coef","ci","pval"]].to_string(index=False))
    plot_placebo(placebo_results)

    print("\n" + "="*55)
    print("All done. Figures saved to out/figures/")
    print("Regression table saved to out/did_results.csv")
