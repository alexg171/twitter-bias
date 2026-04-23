# Algorithmic Amplification and Content Bias on Twitter/X
### Evidence from the Musk Acquisition — Panel Data & Discrete Choice Models, Spring 2026
**Alex Gamez — The University of Texas at El Paso**

---

## Research Question
Did Elon Musk's acquisition of Twitter (October 27, 2022) causally shift which content categories the platform algorithmically amplifies, relative to what organic user interest would have produced?

---

## Data

| File | Description |
|------|-------------|
| `out/twitter_trending_4yr.csv` | ~695,000 hourly Twitter trending snapshots, Oct 2020–Oct 2024 |
| `out/reddit_category.tsv` | Daily post counts for 25+ matched subreddits, Oct 2020–Oct 2024 |
| `out/unique_topics.csv` | Deduplicated topic list with auto-assigned categories |
| `out/top500_topics_4yr.csv` | Top 500 topics for manual review |

**Do not commit raw data files** — they are listed in `.gitignore`.

---

## Pipeline

Run everything with one command:
```
run_all.bat
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `twitter_unique.py` | `out/unique_topics.csv`, `out/twitter_category_counts.csv` |
| 2 | `category_analysis.py` | `out/figures/category_shift.png`, `out/figures/category_timeseries.png`, `out/category_counts.csv` |
| 3 | `category_did.py` | `out/category_did_results.csv`, `out/figures/category_did.png` |
| 4 | `category_plots.py` | `out/figures/parallel_trends/*.png`, `out/figures/event_study/*.png` |
| 5 | `category_demographics.py` | `out/figures/demographics/*.png` |

---

## Identification Strategy

**Difference-in-Differences (DiD), log-deviation model:**

```
log_dev(y_it) = β0 + β1·Twitter_i + β2·Post_t + β3·(Twitter_i × Post_t) + ε_it
```

- `log_dev(y)` = log(y + c) minus pre-period log mean — puts Twitter and Reddit on the same scale
- `Twitter_i = 1` for Twitter observations, 0 for Reddit
- `Post_t = 1` after October 27, 2022
- `β3` = causal effect in log-deviation units (≈ % for small values)
- HC3 heteroskedasticity-robust standard errors
- **Control:** matched Reddit subreddit(s) per category (e.g., r/SquaredCircle for Wrestling)

---

## File Reference

| File | Purpose |
|------|---------|
| `category_lexicon.py` | 18-category NLP keyword classifier (CamelCase-aware) |
| `twitter_unique.py` | Deduplicates raw trending data, assigns categories |
| `category_analysis.py` | Category composition bar charts and time series |
| `category_did.py` | Main DiD estimation, results table, coefficient plot |
| `category_plots.py` | Per-category parallel trends and quarterly event study |
| `category_demographics.py` | Audience demographic visualizations |
| `category_subreddit_mapping.py` | Shared subreddit ↔ category mapping (used by pull script) |
| `reddit_category_pull.py` | Arctic Shift API pull for matched subreddits (run once) |
| `run_all.bat` | Regenerates all analysis and figures end-to-end |
| `presentation.tex` | Beamer LaTeX slide deck |

---

## Categories (16 in DiD model)

`wrestling` · `combat_sports` · `sports_nba` · `sports_nfl` · `sports_mlb` · `sports_nhl` · `sports_soccer` · `sports_college` · `sports_womens` · `sports_other` · `reality_tv` · `entertainment` · `taylor_swift` · `fandom` · `tech_gaming` · `news_politics`

---

## Requirements

```
pip install pandas numpy matplotlib statsmodels
```
