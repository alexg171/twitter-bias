"""
Political lexicon for classifying Twitter trending topics.
Sources:
  - Alex Gamez planning notes (planning.docx) — primary
  - Gentzkow & Shapiro (2010) congressional speech phrases
  - Common political hashtag knowledge

Labels: "right", "left", "neutral"
Usage:
    from lexicon import classify_topic
    label = classify_topic("MAGA")   # -> "right"
"""

import re

# ---------------------------------------------------------------
# RIGHT-LEANING keywords (conservative / MAGA / alt-right)
# ---------------------------------------------------------------
RIGHT_KEYWORDS = [
    # MAGA / Trump
    "maga", "trump", "makeamericagreatagain", "trumptrain", "trump2024",
    "trump2020", "trump2022", "djt", "americafirst", "stopthesteal", "fraudwasreal",
    # GOP / Republican party
    "republican", "gop", "rnc", "conservative", "tpusa",
    "turning point", "charlie kirk", "ben shapiro", "ted cruz",
    "desantis", "ron desantis", "marjorie taylor greene", "mtg",
    "matt gaetz", "jim jordan",
    # Clearly right-coded online figures
    "andrew tate", "sneako", "pearl davis", "adin ross",
    "jordan peterson", "incel", "redpill", "red pill",
    "alpha male", "soyboy", "soy boy", "cuck",
    # Anti-left / culture war rhetoric
    "anti-woke", "antiwoke", "woke mob",
    "dei bad", "dei hire", "groomer",
    "let's go brandon", "fjb", "deep state",
    "qanon", "great replacement",
    # Immigration (right framing)
    "build the wall", "illegal alien", "border invasion", "securetheborder",
    # Social / values
    "all lives matter", "blue lives matter", "backtheblue", "thinblueline",
    "pro life", "prolife", "anti abortion",
    "gun rights", "second amendment", "nra",
    # Energy / economy (right framing)
    "clean coal", "drill baby drill", "no green new deal", "anti esg",
    # Right-wing TV commentators
    "tucker carlson", "hannity", "sean hannity",
    "laura ingraham", "greg gutfeld", "gutfeld",
    "candace owens", "matt walsh", "steven crowder", "crowder",
    "dan bongino", "bongino", "glenn beck", "tomi lahren", "mark levin",
    # 2022 midterm Republicans + prominent figures
    "jd vance", "vance",
    "herschel walker",
    "dr oz", "mehmet oz",
    "kari lake",
    "lauren boebert", "boebert",
    "kevin mccarthy", "mccarthy",
    "rand paul", "tim scott", "ron johnson", "greg abbott",
    "blake masters",
    # Key events (right-coded)
    "speaker vote", "speaker of the house",
    "twitter files", "twitterfiles",
]

# ---------------------------------------------------------------
# LEFT-LEANING keywords (progressive / liberal / left)
# ---------------------------------------------------------------
LEFT_KEYWORDS = [
    # Progressive party / movement
    "progressive", "democrat", "dnc", "liberal",
    "bernie", "bernie sanders", "aoc", "alexandria ocasio cortez",
    "elizabeth warren", "ilhan omar", "rashida tlaib",
    "hasan piker", "sam seder", "majority report", "the young turks", "tyt",
    # Clearly left policy positions
    "medicare for all", "m4a", "green new deal", "gnd",
    "tax the rich", "taxtherich", "wealth tax",
    "student debt", "student loan forgiveness",
    "universal healthcare",
    "climate justice", "climate action",
    # Social justice movements
    "blacklivesmatter", "blm", "black lives matter",
    "stopaapihate", "stop aapi hate",
    "nodapl", "landback", "land back",
    "abolish ice", "defund the police", "acab",
    "the squad",
    # Reproductive rights
    "mybodymychoice", "prochoice", "pro choice",
    "abortion rights", "roevwade", "roe v wade",
    # LGBTQ+
    "lgbtq", "lgbt", "trans rights",
    "transrightsarehumanrights", "lovewins", "protecttranskids",
    "pride month",
    # Feminist
    "metoo", "me too", "feminist", "feminism",
    # Palestine (left framing)
    "freepalestine", "free palestine", "ceasefire", "gazagenocide",
    # Immigration (left framing)
    "dreamers", "daca", "familiesbelongtogether",
    # January 6th (left-coded — hearings, insurrection framing)
    "january 6", "january6", "january 6th", "january6th", "insurrection",
    # Voting / election (left framing)
    "vote blue", "voteblue", "democracy wins", "democracywins",
    # Democratic legislation
    "inflation reduction act", "build back better",
    # Anti-authoritarian (clearly coded)
    "notmypresident", "impeach", "antifascist",
    # Anti-Musk / Twitter chaos (left-coded post-acquisition)
    "rip twitter", "riptwitter", "block elon", "blockelon",
    "twitter layoffs", "twitterlayoffs",
    # Mainstream Democratic politicians
    "biden", "joe biden",
    "kamala", "kamala harris",
    "pelosi", "nancy pelosi",
    "schumer", "chuck schumer",
    "fetterman", "john fetterman",
    "warnock", "raphael warnock",
    "stacey abrams",
    "gavin newsom", "newsom",
    "pete buttigieg", "buttigieg",
    "mandela barnes", "katie hobbs", "val demings",
    # Left-leaning TV commentators
    "rachel maddow", "maddow",
    "joy reid", "don lemon",
    "brian stelter", "stelter",
    "chris hayes", "nicolle wallace", "joy behar",
]

# ---------------------------------------------------------------
# NEUTRAL / control topics (sports, entertainment, weather, etc.)
# ---------------------------------------------------------------
NEUTRAL_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "mls", "ncaa",
    "superbowl", "super bowl", "world series", "nba finals",
    "oscars", "grammys", "emmys", "golden globes",
    "netflix", "hulu", "disney plus", "hbo",
    "taylor swift", "beyonce", "drake", "bad bunny",
    "hurricane", "earthquake", "tornado", "wildfire",
    "storm", "blizzard", "heat wave",
    "iphone", "apple", "google", "microsoft",
    "worldcup", "world cup", "olympics", "fifa",
    "thanksgiving", "christmas", "halloween", "new year",
    "movie", "film", "album", "concert", "tour",
]

# ---------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------
# Tokenizer — two representations used together:
#   1. token_set : individual words after CamelCase splitting (whole-word matching)
#   2. flat      : full lowercased string, spaces only (multi-word phrase matching)
# ---------------------------------------------------------------

def _camel_split(text: str) -> list:
    """
    Split a CamelCase / PascalCase / ALL_CAPS string into individual word tokens.
    Also handles hashtags (#GoPackGo) and underscores.

    Examples:
        '#GoPackGo'      -> ['go', 'pack', 'go']
        '#TrumpIsGuilty' -> ['trump', 'is', 'guilty']
        'MAGA'           -> ['maga']
        '#SuzukaGP'      -> ['suzuka', 'gp']       # 'gp' != 'gop'  ✓
        'Timmy Trumpet'  -> ['timmy', 'trumpet']   # 'trumpet' != 'trump' ✓
        'Gore Magala'    -> ['gore', 'magala']      # 'magala' != 'maga'  ✓
        'DeSantis'       -> ['de', 'santis']
        '#GOPClownShow'  -> ['gop', 'clown', 'show']
    """
    # strip leading # or @
    text = re.sub(r'^[#@]+', '', text.strip())
    # insert space before: lowercase→Uppercase boundary
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # insert space before: run-of-caps → Cap+lower boundary  (e.g. "GOPClown" → "GOP Clown")
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    # split on whitespace, underscores, hyphens, digits boundaries
    tokens = re.split(r'[\s_\-]+', text)
    return [t.lower() for t in tokens if t]


def _prepare(topic: str):
    """
    Returns (token_set, flat_str) for a topic string.
      token_set : set of individual lowercase tokens (for whole-word keyword matching)
      flat_str  : full lowercased, stripped string with punctuation removed (for phrase matching)

    token_set also includes the fully-squashed form (spaces removed) so that compound
    proper names like 'DeSantis' → ['de','santis'] also produce 'desantis' as a token,
    allowing the single-word keyword 'desantis' to match.
    """
    token_set = set(_camel_split(topic))
    flat_str  = re.sub(r"[^a-z0-9 ]", "", topic.lower().strip())
    squashed  = flat_str.replace(" ", "")
    if squashed:
        token_set.add(squashed)
    return token_set, flat_str


def _count_hits(keywords: list, token_set: set, flat_str: str) -> int:
    """
    Score a keyword list against a topic.

    Single-word keywords → whole-word token match against token_set.
      Prevents 'gop' matching inside 'gopackgo'.

    Multi-word keywords  → ALL keyword tokens must be present in topic's token_set (AND-match).
      '#TuckerCarlson' → {tucker, carlson} ⊆ topic tokens  → match ✓
      'PJ Tucker'      → {pj, tucker}  ⊄ {tucker carlson}  → no match ✓
      '#HerschelWalker'→ {herschel, walker} ⊆ topic tokens → match ✓
    """
    hits = 0
    for kw in keywords:
        if " " in kw:                           # multi-word: AND-match on tokens
            kw_tokens = set(kw.lower().split())
            if kw_tokens.issubset(token_set):
                hits += 1
        else:                                   # single word: exact token match
            if kw in token_set:
                hits += 1
    return hits


def classify_topic(topic: str) -> str:
    """
    Returns 'right', 'left', or 'neutral'.
    Uses CamelCase-aware tokenization to avoid false substring matches.
    """
    token_set, flat_str = _prepare(topic)

    right_hits   = _count_hits(RIGHT_KEYWORDS,   token_set, flat_str)
    left_hits    = _count_hits(LEFT_KEYWORDS,    token_set, flat_str)
    neutral_hits = _count_hits(NEUTRAL_KEYWORDS, token_set, flat_str)

    if right_hits == 0 and left_hits == 0 and neutral_hits == 0:
        return "neutral"

    scores = {"right": right_hits, "left": left_hits, "neutral": neutral_hits}
    return max(scores, key=scores.get)


def classify_batch(topics: list) -> list:
    """Classify a list of topic strings. Returns list of labels."""
    return [classify_topic(t) for t in topics]


if __name__ == "__main__":
    # Smoke test — expected labels
    tests = [
        # True positives that should still work
        ("MAGA",                  "right"),
        ("Black Lives Matter",    "left"),
        ("Super Bowl",            "neutral"),
        ("Andrew Tate",           "right"),
        ("Free Palestine",        "left"),
        ("Taylor Swift",          "neutral"),
        ("gun rights",            "right"),
        ("Medicare for All",      "left"),
        ("#TrumpIsGuilty",        "right"),
        ("#GOPClownShow",         "right"),
        ("DeSantis",              "right"),
        # False positives that the old tokenizer got wrong
        ("#GoPackGo",             "neutral"),   # 'gop' inside camel token
        ("#LetsGoPens",           "neutral"),   # 'gop' inside camel token
        ("#SuzukaGP",             "neutral"),   # 'gp' != 'gop'
        ("Timmy Trumpet",         "neutral"),   # 'trumpet' != 'trump'
        ("Gore Magala",           "neutral"),   # 'magala' != 'maga'
        ("Ubisoft",               "neutral"),   # 'ubi' token ≠ 'ubi' keyword? actually will still match
        ("Hubie Brown",           "neutral"),   # 'ubi' inside 'hubie'
        ("Gophers",               "neutral"),   # 'gophers' != 'gop'
        ("#CarabaoCupFinal",      "neutral"),   # 'aoc' not a standalone token
        ("Suicide Squad",         "neutral"),   # 'squad' only, not 'the squad'
    ]
    print("Lexicon smoke test:")
    all_pass = True
    for topic, expected in tests:
        result = classify_topic(topic)
        status = "OK" if result == expected else f"FAIL (expected {expected})"
        if result != expected:
            all_pass = False
        print(f"  {topic:35s} -> {result:8s}  {status}")
    print(f"\n{'All tests passed.' if all_pass else 'Some tests FAILED — review above.'}")
