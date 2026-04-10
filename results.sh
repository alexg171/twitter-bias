# 0. reddit pull (if needed)
# python reddit_category_pull.py   # re-pull subreddit controls (fast now, ~30 sec)

# 1. Category composition
python .\category_analysis.py

# 2. Category DiD with matched subreddit controls
python .\category_did.py

# 3. Right/left DiD across 3 windows (Twitter vs Reddit)
python .\did_analysis.py

# 4. Single-platform ITS across 3 windows (Twitter only)
python .\multiwindow_analysis.py

# 5. Synthetic control (robustness check, known poor fit)
python .\synth_control.py

