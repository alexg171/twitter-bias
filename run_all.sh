#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_all.sh  —  Full regeneration of all analysis and figures
#
# Usage:
#   cd "path/to/Final Presentation"
#   bash run_all.sh
#
# Order:
#   1. Classify topics & update unique_topics.csv
#   2. Category composition charts (shift bars, timeseries)
#   3. Category DiD (main results table + bar chart)
#   4. Parallel trends + event study plots (per category)
#   5. Demographic visualizations
# ─────────────────────────────────────────────────────────────────────────────

set -e   # stop on any error

PYTHON="python"   # change to "python3" if needed on your machine
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "============================================================"
echo "  FULL ANALYSIS REGENERATION"
echo "  Working dir: $DIR"
echo "============================================================"

# ── Step 1: Classify topics ───────────────────────────────────────────────────
echo ""
echo "[ 1/5 ]  Classifying topics → unique_topics.csv"
echo "------------------------------------------------------------"
$PYTHON twitter_unique.py

# ── Step 2: Category composition ─────────────────────────────────────────────
echo ""
echo "[ 2/5 ]  Category composition charts"
echo "------------------------------------------------------------"
$PYTHON category_analysis.py

# ── Step 3: Category DiD ─────────────────────────────────────────────────────
echo ""
echo "[ 3/5 ]  Category DiD (main results)"
echo "------------------------------------------------------------"
$PYTHON category_did.py

# ── Step 4: Parallel trends + event study ────────────────────────────────────
echo ""
echo "[ 4/5 ]  Parallel trends + event study plots"
echo "         (this takes ~2–3 minutes for bootstrapping)"
echo "------------------------------------------------------------"
$PYTHON category_plots.py

# ── Step 5: Demographics ─────────────────────────────────────────────────────
echo ""
echo "[ 5/5 ]  Demographic visualizations"
echo "------------------------------------------------------------"
$PYTHON category_demographics.py

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL DONE"
echo ""
echo "  Key outputs:"
echo "    out/unique_topics.csv                  — classified topics"
echo "    out/category_did_results.csv           — DiD coefficients"
echo "    out/category_counts.csv                — pre/post counts"
echo "    out/figures/category_shift.png         — composition shift"
echo "    out/figures/category_did.png           — DiD bar chart"
echo "    out/figures/parallel_trends/*.png      — 17 parallel trends"
echo "    out/figures/event_study/*.png          — 17 event studies"
echo "    out/figures/demographics/summary_grid.png"
echo "    out/figures/demographics/bro_scatter.png"
echo "============================================================"
echo ""
