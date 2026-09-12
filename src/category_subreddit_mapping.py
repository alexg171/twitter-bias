"""
Shared category ↔ subreddit mapping.
Single source of truth to avoid sync issues across analysis scripts.
"""

# Subreddit → Category mapping (used by reddit_category_pull.py)
SUBREDDIT_CATEGORY = {
    "SquaredCircle":      "wrestling",
    "nba":                "sports_nba",
    "nabadiscussion":       "sports_nba",
    "nfl":                "sports_nfl",
    "baseball":           "sports_mlb",
    "mlb":                "sports_mlb",
    "hockey":             "sports_nhl",
    "soccer":             "sports_soccer",
    "football":           "sports_soccer",
    "soccercirclejerk":       "sports_soccer",
    "CFB":                "sports_college",
    "CollegeBasketball":  "sports_college",
    "MMA":                "combat_sports",
    "UFC":                "combat_sports",
    "BravoRealHousewives": "reality_tv",
    "LoveIsBlindNetflix":        "reality_tv",
    "LoveIslandTV":       "reality_tv",
    "thebachelor":        "reality_tv",
    "survivor":           "reality_tv",
    "BigBrother":         "reality_tv",
    "Vanderpumprules":    "reality_tv",
    "MAFS_TV":            "reality_tv",
    "90DayFiance":        "reality_tv",
    "television":         "entertainment",
    "movies":             "entertainment",
    "Music":              "entertainment",
    "TaylorSwift":        "taylor_swift",
    "anime":              "fandom",
    "kpop":               "fandom",
    "gaming":             "tech_gaming",
    "lgbt":               "lgbtq_social",
    "worldnews":          "news_events",
    "news":               "news_events",
    "wnba":               "sports_womens",
    "NWSL":               "sports_womens",
    "Twitter":            "musk_twitter",
    "elonmusk":           "musk_twitter",
    "technology":         "musk_twitter",
    "formula1":           "sports_other",
    "tennis":             "sports_other",
    "golf":               "sports_other",
    "politics":           "politics",
    "PoliticalDiscussion": "politics",
    "conservative":       "politics",
    "republican":         "politics",
    "democrats":          "politics",
    "Liberal":            "politics",
    "progressive":        "politics",
    "libertarian":        "politics",
    "NeutralPolitics":    "politics",
}


def build_cat_to_subreddit():
    """Build reverse mapping: category → [subreddits]"""
    cat_to_sub = {}
    for sub, cat in SUBREDDIT_CATEGORY.items():
        if cat not in cat_to_sub:
            cat_to_sub[cat] = []
        cat_to_sub[cat].append(sub)
    return cat_to_sub


# Category → Subreddit mapping (used by category_did.py)
CAT_TO_SUBREDDIT = build_cat_to_subreddit()
