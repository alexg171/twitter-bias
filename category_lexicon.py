"""
Categorical topic classifier for Twitter trending topics.

Categories
----------
wrestling           WWE, AEW
sports_nba          NBA basketball
sports_nfl          NFL football
sports_mlb          MLB baseball
sports_nhl          NHL hockey
sports_soccer       International soccer / Premier League
sports_college      College sports (NCAA)
sports_womens       Women's sports (NWSL, WNBA, Caitlin Clark, etc.)
sports_other        NASCAR, golf, tennis, F1, Olympics, etc.
combat_sports       UFC, MMA, boxing
reality_tv          All reality TV — Housewives, dating shows, etc.
entertainment       Scripted TV, movies, music, pop culture
taylor_swift        Taylor Swift (volume justifies own bucket)
fandom              Anime, K-pop, gaming fandom
tech_gaming         Gaming hardware/titles, crypto, AI, tech
manosphere          Andrew Tate, Joe Rogan, red-pill adjacent
politics            Partisan political content
lgbtq_social        Pride, LGBTQ+, social justice, activism
religious           Religious content, prayer, faith
musk_twitter        Topics about Elon Musk and Twitter/X directly
holidays            Seasonal holidays and observances
true_crime          True crime, murder cases, podcasts
news_events         Breaking news, geopolitical events
social_filler       Day-of-week hashtags, good morning/night
other               Uncategorised
"""

import re
from typing import Tuple

# ── helpers ──────────────────────────────────────────────────────────────────

def _camel_split(text: str) -> list:
    text = re.sub(r'^[#@]+', '', text.strip())
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z]{2,})([A-Z][a-z])', r'\1 \2', text)
    tokens = re.split(r'[\s_\-]+', text)
    return [t.lower() for t in tokens if t]


def _prepare(topic: str) -> Tuple[set, str]:
    token_set = set(_camel_split(topic))
    flat = re.sub(r"[^a-z0-9 ]", "", topic.lower().strip())
    squashed = flat.replace(" ", "")
    if squashed:
        token_set.add(squashed)
    return token_set, flat


def _hits(keywords, token_set):
    for kw in keywords:
        if " " in kw:
            if set(kw.lower().split()).issubset(token_set):
                return True
        else:
            if kw in token_set:
                return True
    return False


# ── keyword lists ─────────────────────────────────────────────────────────────

WRESTLING = [
    "wwe", "aew", "wweraw", "wwenxt", "nxt", "smackdown",
    "aewdynamite", "aewrampage", "aewcollision", "aewfightforever",
    "oplive", "aewallout", "aewdoubleornothing", "aewfullyloaded",
    "wrestlemania", "royalrumble", "summerslam", "survivorseries",
    "moneyinthebank", "hellinthecell", "eliminationchamber",
    "nightofchampions", "backlash",
    "roman reigns", "cody rhodes", "sami zayn", "kevin owens",
    "seth rollins", "cm punk", "aj styles", "becky lynch",
    "bianca belair", "rhea ripley", "the miz", "undertaker",
    "john cena", "brock lesnar", "goldberg", "rey mysterio",
    "chris jericho", "kenny omega", "moxley", "mjf",
    "jungle boy", "darby allin", "ricky starks", "jade cargill",
    "sasha banks", "bayley", "charlotte flair", "asuka",
    "roman", "usos", "bloodline",
    "wrestling", "kayfabe", "titantron",
    "la knight", "laknight", "andrade",
    "rhea", "sami", "jericho", "bianca", "seth", "brock",
    "vince", "vince mcmahon",
    "punk",
]

SPORTS_NBA = [
    "nba", "nbafinals", "nba finals", "nba draft", "nbadraft",
    "nbaplayoffs", "nba playoffs", "nbaallstar",
    # teams
    "lakers", "lakeshow", "celtics", "warriors", "dubnation",
    "bulls", "heat", "knicks", "nets", "spurs", "bucks",
    "suns", "clippers", "blazers", "nuggets", "jazz", "hawks",
    "pistons", "pacers", "magic", "hornets", "wizards",
    "cavaliers", "cavs", "raptors", "grizzlies", "pelicans",
    "thunder", "timberwolves", "rockets", "mavericks", "mavs",
    "kings", "sixers", "gokingsgo", "lightthebeam", "light the beam",
    "letsgomavs", "mffl", "dubs",
    # players
    "lebron", "bron", "curry", "steph", "durant", "giannis",
    "tatum", "luka", "jokic", "embiid", "kyrie", "klay",
    "draymond", "harden", "westbrook", "kawhi", "booker",
    "brunson", "randle", "maxey", "jrue", "dame", "damian",
    "jaylen brown", "poole", "wiggins", "anthony davis",
    "pat bev", "marcus smart", "ty lue", "dillon brooks",
    "austin reaves", "reaves", "brandon miller", "jeff saturday",
    "paul george", "melo", "carmelo", "ben simmons", "beal",
    "zion", "chris paul", "gobert", "sabonis", "anthony edwards",
    "chet", "derrick white", "jordan poole", "darvin ham",
    "victor wembanyama", "wemby", "scoot henderson",
    "basketball",
    "ja morant", "jamorant", "grayson allen", "graysonallen",
    "scott foster", "scottfoster", "jimmy butler", "jimmybutler",
    "garland", "middleton", "thibs", "memphis grizzlies",
    "chapman", "shai", "donovan mitchell", "drummond",
    "trae young", "76ers", "doc rivers", "demar derozan",
    "aaron gordon", "edey", "reggie jackson",
    "kobe", "kobe bryant",
    "bronny", "bronny james",
    "ayton", "deandre ayton",
    "memphis grizzlies",
    "caruso", "alex caruso",
]

SPORTS_NFL = [
    "nfl", "nfldraft", "nfl draft", "superbowl", "super bowl",
    "nflplayoffs", "nfl playoffs", "nflsundaynight", "mnf",
    # teams
    "cowboys", "eagles", "patriots", "chiefs", "bills", "bengals",
    "rams", "49ers", "niners", "packers", "bears", "lions",
    "vikings", "giants", "falcons", "saints", "buccaneers", "bucs",
    "seahawks", "cardinals", "broncos", "raiders", "chargers",
    "dolphins", "jets", "ravens", "steelers", "browns", "colts",
    "texans", "jaguars", "jags", "titans", "commanders",
    "flyeaglesfly", "hereweго", "hereweго", "herewego",
    "billsmafia", "finsup", "rolltribe",
    # players
    "mahomes", "brady", "rodgers", "lamar", "travis kelce", "kelce",
    "derrick henry", "cooper kupp", "davante adams", "tyreek hill",
    "justin jefferson", "josh allen", "hurts", "burrow", "purdy",
    "stroud", "richardson", "dak", "russ", "wilson", "stafford",
    "goff", "herbert", "carr", "wentz", "justin fields", "fields",
    "caleb williams", "bo nix", "jordan love", "diggs",
    "victory monday", "football",
    "panthers", "gopackgo", "baker", "mac jones", "pats",
    "watson", "deebo", "jimmy g", "jimmyg", "trevor lawrence",
    "bucciovertimechallenge", "vegasborn",
    "philly", "jalen hurts", "jalen",
    "daniel jones", "zeke", "aj brown", "cam newton", "camnewton",
    "flacco", "joe flacco", "kirk cousins", "kirkcousins",
    "pro bowl", "probowl",
    "harbaugh", "go birds", "jameis", "bryce young",
    "matt ryan", "jerry jones", "hookem", "godawgs",
    "buckeyes", "washington commanders", "heisman",
]

SPORTS_MLB = [
    "mlb", "worldseries", "world series", "mlbplayoffs",
    "allstargame", "mlbdraft",
    # teams
    "yankees", "mets", "dodgers", "cubs", "red sox", "astros",
    "braves", "phillies", "padres", "mariners", "orioles",
    "bluejays", "twins", "whitesox", "tigers", "royals",
    "athletics", "guardians", "rays", "nationals", "pirates",
    "reds", "rockies", "diamondbacks", "rangers",
    # players
    "ohtani", "judge", "soto", "acuna", "vlad", "betts", "devers",
    "alonso", "stanton", "kimbrel", "clay holmes", "gerrit cole",
    "nestor", "altuve", "rizzo", "degrom", "mookie", "tatis",
    "machado", "buehler", "kershaw",
    "baseball",
    "hader", "lindor", "rodon", "wheeler", "scherzer",
    "aaron nola", "snell",
]

SPORTS_NHL = [
    "nhl", "stanleycup", "stanley cup", "nhlplayoffs",
    "bruins", "maple leafs", "canadiens", "penguins", "flyers",
    "rangers", "capitals", "lightning", "lightningstrikes",
    "hurricanes", "oilers", "flames", "canucks", "avalanche",
    "goavsgo", "blues", "predators", "wild", "mnwildfirst",
    "stars", "ducks", "sharks", "sabres", "senators", "devils",
    "islanders", "kraken", "golden knights",
    "hockey",
    "mcdavid", "vegas born", "vegasborn",
    "game 7",
]

SPORTS_SOCCER = [
    "premier league", "premierleague", "champions league",
    "championsleague", "laliga", "bundesliga", "seriea",
    "ligue1", "mls", "fifaworldcup", "worldcup", "world cup",
    "euro", "copa america", "usmnt", "uswnt",
    # clubs
    "arsenal", "chelsea", "liverpool", "tottenham", "coys", "coyg",
    "ynwa", "manchester", "barcelona", "barca", "real madrid",
    "psg", "juventus", "bayernmunich", "dortmund", "atletico",
    "inter milan", "acmilan", "mufc", "mcfc", "newcastle",
    "everton", "west ham", "aston villa", "leicester", "fulham",
    "chivas", "americavs", "tigres",
    # players
    "messi", "ronaldo", "haaland", "mbappe", "pulisic",
    "neymar", "lewandowski", "kane", "salah", "maguire",
    "rashford", "de bruyne", "pedri", "vini", "vinicius",
    "benzema", "grealish", "xhaka", "martinelli", "saka",
    "sancho", "casemiro", "lautaro", "sterling", "harry kane",
    "soccer", "futbol",
    "bayern", "foden", "leeds", "dembele", "antony",
    "mctominay", "tuchel",
    "onana", "werner", "lukaku",
    "klopp", "poch", "pochettino", "burnley", "brighton",
    "brentford", "nunez", "httc", "gobolts",
    "argentina", "brazil", "madrid", "spain", "england", "london",
    "ligue 1",
    "xavi", "pogba", "paul pogba",
]

SPORTS_COLLEGE = [
    "ncaa", "marchmadness", "march madness", "finalfour", "final four",
    "cfb", "collegefootball", "college football", "natty",
    "nationalchampionship", "national championship",
    "iowa", "auburn", "bama", "alabama", "ucla", "goblue",
    "michigan", "clemson", "arkansas", "ohio state", "ohiostate",
    "texas", "baylor", "oregon", "rutgers", "mizzou", "louisville",
    "colorado", "purdue", "gonzaga", "kansas", "kentucky", "duke",
    "uconn", "creighton", "houston", "tennessee", "geno", "geno auriemma",
    "rolltide", "roll tide", "warnation",
    "kubball", "iubb",
    "ole miss", "olemiss", "penn state", "pennstate",
    "vandy", "vanderbilt", "nc state", "ncstate",
    "nebraska", "stanford", "game day", "gameday",
    "syracuse", "dabo", "godawgs", "hookem",
    "brian kelly", "briankelly", "ryan day", "ryanday",
    "unlv", "south carolina", "southcarolina",
    "game 6",
    # State names used as college team shorthand
    "arizona", "utah", "florida", "wisconsin", "oklahoma",
    "ohio", "georgia", "saban", "nick saban",
]

SPORTS_WOMENS = [
    "wnba", "nwsl",
    "caitlin clark", "caitlinclark", "angel reese", "angelreese",
    "paige bueckers", "paigebueckers", "aliyah boston",
    "a'ja wilson", "breanna stewart", "sabrina ionescu",
    "megan rapinoe", "rapinoe",
    "uswnt", "usawomen",
    "ncaawomen", "womensbasketball", "womens basketball",
    "womenssoccer", "womens soccer",
    "serena", "serena williams",
    "paige", "paige bueckers",
    "naomi osaka", "osaka",
    "simone biles", "biles",
]

SPORTS_OTHER = [
    # F1
    "formula1", "formula 1", "f1", "grandprix", "grand prix",
    "verstappen", "hamilton", "leclerc", "perez", "checo",
    "norris", "sainz", "alonso", "russell",
    # NASCAR
    "nascar",
    # Golf
    "golf", "pga", "masters", "usopen", "ryder cup",
    "tiger woods", "tigerwoods", "rory mcilroy",
    # Tennis
    "tennis", "wimbledon", "usopen", "rolandgarros",
    "djokovic", "nadal", "federer", "alcaraz",
    # Olympics
    "olympics", "teamusa",
    "rory", "rory mcilroy",
    "lando", "lando norris",
    # Horse racing
    "kentuckyderby", "kentucky derby",
]

COMBAT_SPORTS = [
    "ufc", "mma", "bellator", "boxing", "pbc",
    "mcgregor", "conor mcgregor", "khabib", "jon jones",
    "strickland", "adesanya", "poirier", "gaethje",
    "volkanovski", "holloway", "pimblett", "ngannou",
    "miocic", "cormier", "jake paul", "logan paul",
    "canelo", "fury", "tyson fury", "usyk", "crawford",
    "spence", "haney", "ryan garcia", "devin haney",
    "fight night", "fight week", "ppv",
    "knockout", "submission",
    "dwcs", "izzy", "israel adesanya",
    "floyd", "floyd mayweather",
]

REALITY_TV = [
    # Housewives franchise
    "rhonj", "rhoa", "rhobh", "rhoslc", "rhony", "rhop", "rhoc",
    "rhoatl", "rhod", "rhom", "rhodc", "realhousewives",
    "real housewives",
    # Dating / relationship shows
    "thebachelor", "thebachelorette", "bachelorette", "bachelor",
    "loveisblind", "love is blind", "loveisblindseason",
    "toohottohandle", "too hot to handle",
    "loveisland", "love island",
    "theultimatum", "ultimatum",
    "fboyisland",
    "datingaround",
    "marriedatfirstsight", "mafs",
    "temptationisland",
    "loveafterlockup", "love after lockup",
    "readytolove", "ready to love",
    "singleinferno",
    "areYouTheOne", "ayto",
    "bling empire", "blingempire",
    # Housewife adjacent / Bravo
    "married2med", "married to med",
    "vanderpump", "pumprules", "pump rules",
    "belowdeck", "below deck",
    "southerncharm",
    "summerhouse", "summer house",
    "winterhouse",
    "shahs of sunset",
    "datingnofilter",
    # Competition reality
    "survivorcbs", "survivor",
    "bigbrother", "bb25", "bb26",
    "theamazingrace", "amazing race",
    "themaskedsinger", "masked singer",
    "dragrace", "drag race", "rpdr",
    "americasgottalent", "agt",
    "thevoice", "the voice",
    "americanidol", "american idol",
    "dancingwiththestars", "dwts",
    "projectrunway",
    "masterchef",
    "hell's kitchen", "hellskitchen",
    "topchef",
    "inkmaster",
    # TLC / lifestyle reality
    "90dayfiance", "90day", "90dayfinace",
    "1000lbsisters", "sisterwives", "sister wives",
    "my600lblife",
    "teenmon", "teenmom",
    "welcometoplathville",
    "iamjazz",
    "dancemoms",
    "littlepeople",
    # Celebrity / talk reality
    "thetraitors", "thetraitorsus", "traitors us",
    "thekarDAShians", "kardashians",
    "keeping up",
    "lamh",
    "sistasonbet", "sistas on bet",
    "lhhatl", "love and hip hop",
    "queenscourt",
    "powerbook", "powerghost", "power ghost",
    # Docuseries
    "makingamurderer",
    "thecrownnetflix",
    "inventing anna",
    # Other reality
    "ghostadventures",
    "alaskan bush",
    "90daytheotherway",
    "joseline", "joselines cabaret",
    "loveislandusa", "love island usa",
    "survivor45", "survivor46",
    "raising kanan", "raisingkanan",
]

ENTERTAINMENT = [
    # Prestige / scripted TV
    "yellowstone", "succession", "euphoria",
    "thelastofus", "tlou",
    "themandalorian", "mandalorian",
    "andor", "strangerthings",
    "theboys", "houseofthedragon", "hotd",
    "ringsofpower", "lotr",
    "abbottelementary", "abbott elementary",
    "greys anatomy", "greysanatomy",
    "station19", "chicago fire", "chicagofire",
    "onechicago", "911lone",
    "peakyblinders", "ozark", "bridgerton",
    "squidgame", "squid game",
    "snowfallfx", "thelastdrivein",
    "this is us", "thisis us",
    "allamerican", "all american",
    "power", "powerghost",
    "svengoolie",
    "toonami",
    "criticalrole", "criticalrolespoilers",
    # Movies
    "marvel", "avengers", "blackpanther", "wakanda", "thor",
    "spiderman", "batman", "dcuniverse",
    "avatar", "topgun", "barbie", "oppenheimer", "dune",
    "fastfurious", "jurassicworld",
    "blackadam", "quantumania", "guardians",
    "antman",
    # Music / awards (non-Taylor)
    "grammys", "oscars", "emmys", "goldenglobes", "bafta",
    "vmas", "amas", "billboard", "iheartawards",
    "beyonce", "drake", "badbunny", "bad bunny",
    "sza", "lizzo", "harry styles", "harrystyles",
    "ariana grande", "arianagrande",
    "billie eilish", "billieeilish",
    "doja cat", "dojacat",
    "olivia rodrigo", "oliviarodrigo",
    "the weeknd", "weeknd",
    "kendrick", "kendrick lamar",
    "travis scott", "travisscott",
    "cardi b", "cardib", "cardi",
    "nicki minaj", "nickiminaj", "nicki",
    "post malone", "postmalone",
    "lil baby", "lilbaby",
    "gunna", "young thug", "youngthug",
    "future", "21 savage",
    "morgan wallen", "morganwallen",
    "luke combs", "lukecombs",
    "zach bryan", "zachbryan",
    "rihanna", "ariana",
    "kanye", "ye",
    "carti", "playboi carti",
    "diddy", "sean combs",
    "usher",
    "billie", "megan thee stallion", "megan",
    "durk", "lil durk", "quavo", "boosie", "meek mill", "meek",
    "doja", "halle bailey", "halle",
    "verzuz", "vince staples",
    "911onFOX", "911 on fox", "911onfox",
    "mostRequestedlive",
    "snkrs",
    # Celebrity
    "kim kardashian", "pete davidson",
    "chris rock", "will smith",
    "johnny depp", "amber heard",
    "jada",
    # More musicians / celebs
    "beyonce", "beyonc",
    "zendaya",
    "eminem", "slim shady",
    "disney", "disneyland", "disneyplus",
    "lil wayne", "wayne",
    "snoop", "snoop dogg",
    "jack harlow", "jackharlow",
    "j cole", "jcole",
    "rod wave", "rodwave",
    "alec baldwin", "alecbaldwin",
    "jeezy", "young jeezy",
    "star wars", "starwars",
    "mario", "super mario",
    "gta", "grand theft auto",
    "ticketmaster",
    "spotify",
    "bluesky",
    "metgala", "met gala",
    "friday the 13th",
    "wandavision", "wanda vision",
    "xmen", "x-men", "xmen97",
    "ahsoka",
    "better call saul", "bettercallsaul",
    "cassie",
    "jon stewart", "jonstewarrt",
    "bill maher", "billmaher",
    "don lemon", "donlemon",
    "chuck todd", "chucktodd",
    "tariq",
    # Music
    "lorde",
    "kid rock", "kidrock",
    "coachella",
    "normani",
    "chris brown", "chrisbrown",
    "adele",
    "cudi", "kid cudi",
    "benito", "bad bunny benito",
    "oprah",
    # Movies / TV
    "joker",
    "superman",
    "jack black", "jackblack",
    "loki",
    "she hulk", "shehulk",
    "insecure hbo", "insecurehbo",
]

TAYLOR_SWIFT = [
    "taylor swift", "taylorswift",
    "swifties", "taylornation",
    "tstheerastour", "eras tour", "erastour",
    "theerastour",
    "taylor", "tswift",
    "taylorsversion", "taylors version",
    "midnights", "speaknow", "folklore", "evermore",
    "lover", "reputation", "fearless",
    "1989taylorsversion", "themanmv",
    "travis kelce taylor", "tayvis",
    "swift",
]

FANDOM = [
    # K-pop groups
    "bts", "armybts", "btsfesta",
    "blackpink", "blink",
    "exo", "got7", "twice", "momoland",
    "straykids", "stray kids",
    "enhypen", "nct", "seventeen",
    "txt", "ateez", "itzy", "aespa",
    "newjeans", "lesserafim",
    "monsta x", "monstax",
    # BTS solo members
    "jungkook", "jung kook", "jk",
    "hobi", "jhope", "j-hope",
    "namjoon", "rm",
    "jimin", "park jimin",
    "taehyung", "v bts",
    "jin", "seokjin", "suga", "yoongi", "min yoongi",
    "joon", "namjoon",
    "soobin", "hoseok", "gege",
    "yeonjun",
    "jaehyun", "taeyong",
    # K-pop generic
    "kpop", "k-pop", "idol", "comeback",
    # Anime
    "anime", "manga",
    "naruto", "dragonball", "onepiece",
    "attackontitan", "jujutsukaisen",
    "demonslayer", "kimetsu",
    "bleach", "myheroacademia",
    "hunterxhunter", "blackclover",
    "chainsawman", "chainsaw man",
    "spyxfamily",
    "toonami",
    # Bollywood / international celeb
    "askSRK", "srk", "bollywood",
    "iifa",
    # Gaming fandom / streaming
    "criticalrole",
    "pokimane", "xqc", "asmongold",
    "esports", "lcs", "lck", "worlds",
]

TECH_GAMING = [
    # Gaming hardware
    "xbox", "playstation", "ps5", "ps4", "nintendo", "switch",
    "xboxfreecodefriday", "freecodefridaycontest",
    # Games
    "fortnite", "minecraft", "pokemon", "zelda",
    "callofduty", "warzone", "cod", "halo",
    "eldenring", "elden ring",
    "genshin", "genshinimpact",
    "valorant", "leagueoflegends",
    "roblox", "apex", "apexlegends", "overwatch",
    "diablo", "worldofwarcraft",
    "fifa", "fc24", "nba2k", "madden",
    "cyberpunk", "hogwartslegacy",
    # Crypto / web3
    "bitcoin", "btc", "ethereum", "eth", "crypto",
    "nft", "web3", "defi", "solana",
    "doge", "dogecoin", "shib",
    # AI / tech
    "chatgpt", "gpt", "openai", "gemini",
    "artificialintelligence", "machinelearning", "llm",
    "nvidia", "amd",
    "apple", "iphone", "wwdc",
    "google", "meta", "amazon", "microsoft",
    "gta6", "gta 6",
    "boeing",
]

MANOSPHERE = [
    "andrew tate", "andrewtate",
    "tristan tate", "tristantate",
    "joe rogan", "joerogan", "jre",
    "jordan peterson", "jordanpeterson",
    "fresh and fit", "freshandfit",
    "sneako",
    "valuetainment",
    "tim pool", "timpool",
    "red pill", "redpill",
    "blackpill",
    "mgtow",
    "incel",
    "alpha male", "alphamale",
    "sigma male", "sigmamale",
    "tradwife", "trad wife",
    "hypergamy",
    "manosphere",
    "based and redpilled",
    "rogan",
]

LGBTQ_SOCIAL = [
    # Pride / LGBTQ+
    "pride", "pridemonth", "pride month",
    "lgbtq", "lgbt", "lgbtqia",
    "trans rights", "transrights", "transgender",
    "trans", "nonbinary", "enby",
    "gaymarriage", "gay marriage",
    "queer", "bisexual",
    "transvisibility",
    # Social justice
    "blacklivesmatter", "blm", "black lives matter",
    "sayhername", "sayhisname",
    "metoo", "timesup",
    "stopasianhate",
    "marchforourlives",
    "climatechange", "climate change",
    "greenNewDeal",
    "defundthepolice", "abolishICE",
    # Awareness
    "mentalhealth", "mental health awareness",
    "worldmentalhealthday",
]

RELIGIOUS = [
    "jesus", "jesus christ", "jesuschrist",
    "god", "lord", "holy spirit",
    "christian", "christianity",
    "church", "prayer", "pray",
    "bible", "scripture", "gospel",
    "easter", "christmas", "advent",
    "ramadan", "eid", "eidalfitr", "eidadha",
    "diwali",
    "roshhashanah", "yomkippur", "passover",
    "hanukkah",
    "blessed sunday",
    "jesus christ is alive",
    "good friday",
    "holy week", "holyweek",
]

MUSK_TWITTER = [
    "elon", "elon musk", "elonmusk",
    "twitter", "twitterfiles", "twitter files",
    "twitter blue", "twitterblue",
    "x corp", "xcorp", "xtwitter",
    "musk", "musks",
    "mastodon",
    "twitter migration",
    "leave twitter", "rip twitter",
    "twitterdead", "riptwitter",
    "donald trump twitter",
    "tesla", "spacex",
    "neuralink", "starlink",
    "twitter suspension",
    "verified", "bluecheckmark",
]

HOLIDAYS = [
    # Major holidays
    "christmas", "xmas", "merrychristmas", "merry christmas",
    "halloween", "trickortreat", "trick or treat",
    "thanksgiving", "happythanksgiving",
    "newYear", "newyearseve", "newyearsday",
    "happy new year", "happynewyear",
    "valentinesday", "valentines day",
    "stpatricksday", "st patrick",
    "fourthofjuly", "fourth of july", "july 4th",
    "memorialday", "memorial day",
    "laborday", "labor day",
    "mothersday", "mothers day",
    "fathersday", "fathers day",
    "happyholidays",
    "april fools", "aprilfools",
    "black friday", "blackfriday",
    "santa", "santa claus",
    "st patrick", "stpatrick", "stpatricksday",
    "juneteenth",
    "earth day", "earthday",
    "election day", "electionday",
    "feliz navidad", "feliznavidad", "navidad",
]

TRUE_CRIME = [
    "truecrime", "true crime",
    "dateline", "48 hours", "48hours",
    "investigationdiscovery", "id channel",
    "making a murderer",
    "dahmer", "jeffrey dahmer",
    "gabby petito", "gabbypetito",
    "missing", "cold case",
    "serial killer",
    "murder",
    "criminal minds",
    "crime junkie", "crimejunkie",
    "my favorite murder",
]

POLITICS = [
    # Institutions / process
    "supreme court", "scotus",
    "sotu", "state of the union",
    "senate", "congress", "house of representatives",
    "filibuster", "debt ceiling",
    "jan6", "january 6",
    "impeachment",
    "midterms", "midterm", "primary election",
    "electoral college", "voter", "voting rights",
    "gerrymandering", "ballot",
    # Parties / ideology
    "republican", "democrat", "gop", "dnc", "rnc",
    "maga", "resist",
    "socialism", "communist", "fascist", "fascism",
    "conservative", "liberal", "progressive",
    # Policy issues
    "abortion", "roe v wade", "roevwade", "prolife", "prochoice",
    "gun control", "gun violence", "2ndamendment", "second amendment",
    "obamacare", "aca", "medicare", "medicaid",
    "immigration", "border wall", "daca",
    "student loan", "studentloan",
    "tax cuts", "taxbill",
    # Political figures (not already in news_events)
    "trump", "biden", "harris", "desantis", "newsom",
    "pence", "manchin",
    "pelosi", "mcconnell", "schumer", "aoc",
    "mtg", "marjorie taylor greene",
    "ron paul", "rand paul",
    "bernie", "bernie sanders",
    "buttigieg", "pete buttigieg",
    "morningjoe",
    "liz cheney", "lizcheney",
    "jim jordan", "jimjordan",
    "ted cruz", "tedcruz",
    "jerome powell", "powell",
    "alito", "vivek", "vivek ramaswamy",
    "antifa", "geraldo",
    "tucker", "tucker carlson", "tuckercarlson",
    # Figures
    "fauci", "dr fauci", "anthony fauci",
    "mccarthy", "kevin mccarthy",
    "fetterman", "john fetterman",
    "tim scott", "timscott",
    "nikki haley", "nikkihaley",
    "george santos", "georgesantos",
    "mitch mcconnell", "mitchmcconnell",
    "clarence thomas", "clarencethomas",
    "paul ryan", "paulryan",
    "hillary", "hillary clinton", "hillaryclinton",
    "rudy", "rudy giuliani", "rudygiuliani",
    "jack smith", "jacksmith",
    "bannon", "steve bannon",
    "sinema", "kyrsten sinema",
    "lindsey graham", "lindseygraham",
    "barr", "william barr", "bill barr",
    "barron", "barron trump",
    "ben shapiro", "benshapiro",
    "hannity", "sean hannity",
    "alex jones", "alexjones",
    # Events/topics
    "capitol", "january 6", "jan 6",
    "ar-15", "ar15",
    "george floyd", "georgefloyd",
    "epstein", "jeffrey epstein",
    "secret service", "secretservice",
    "rafah", "rafah crossing",
    "afghanistan",
    "dems", "republicans",
    "florida man",
    "bluesky",
    # More figures
    "kavanaugh", "brett kavanaugh",
    "gaetz", "matt gaetz", "mattgaetz",
    "beto", "beto orourke",
    "white house", "whitehouse",
    "democrats",
    "hunter", "hunter biden",
    "melania", "melania trump",
    "the cdc", "cdc",
    "tulsi", "tulsi gabbard",
    "cuomo", "andrew cuomo",
    "kamala",
    "mar-a-lago", "mar a lago", "maralago",
    "proud boys", "proudboys",
    "sandy hook", "sandyhook",
    "dark brandon", "darkbrandon",
    "candace", "candace owens",
    "susan collins", "susancollins",
    "constitution",
]

NEWS_EVENTS = [
    # Geopolitical
    "ukraine", "russia", "zelensky", "nato", "mariupol",
    "israel", "hamas", "gaza", "palestine", "iran",
    "north korea", "taiwan", "china",
    "putin",
    # Disasters
    "earthquake", "hurricane", "tornado", "flood", "wildfire",
    "tsunami",
    # Breaking
    "rip", "pray for",
    "breaking news", "breakingnews",
    "memorial", "never forget", "neverforget",
    "international womens day", "internationalwomensday",
    "giving tuesday", "givingtuesday",
    "canada", "mexico",
    "uvalde",
    "fema",
    "newsmax",
    "fox news", "foxnews",
]

SOCIAL_FILLER = [
    # Monday
    "mondaymotivation", "monday motivation",
    "mondaythoughts", "mondaymorning",
    "mondayvibes", "mondaymood",
    "mondayblessings",
    "good monday",
    # Tuesday
    "tuesdaythoughts", "tuesdaymotivations", "tuesdayvibe",
    "tuesdayfeeling", "tuesdaymorning",
    "good tuesday",
    # Wednesday
    "wednesdaywisdom", "wednesdaythoughts", "wednesdaymotivation",
    "wednesdayvibe", "wednesdaythought",
    "hump day", "humpday",
    "good wednesday",
    # Thursday
    "thursdaythoughts", "thursdayvibes", "thursdaymorning",
    "thursdaymotivation",
    "good thursday",
    # Friday
    "fridayfeeling", "fridaymotivation", "fridaymorning",
    "fridaykiss", "fridayvibes",
    "finally friday", "happy friday eve", "friday eve",
    "happy friyay", "friyay",
    "good friday",
    # Saturday
    "saturdaymorning", "saturdayvibes", "saturdaythoughts",
    "caturday", "fursuitfriday",
    "good saturday",
    # Sunday
    "sundayvibes", "sundaymorning", "sundaythoughts",
    "sundayfunday", "sunday funday",
    "sundayservice",
    "good sunday",
    # Generic good morning/night
    "good morning", "goodmorning",
    "good night", "goodnight",
    "good afternoon", "goodafternoon",
    "gm", "gn",
    # Motivational fluff
    "motivation", "mondayblessings",
    "blessed", "grateful", "thankful",
    "positivity",
    # Generic social tags
    "tbt", "throwbackthursday",
    "ootd", "selfie",
    "venmome", "venmo me",
    "notifications",
    "taco tuesday", "tacotuesady", "new month",
    "tuesdaymotivaton", "saturdayblessings",
]


# ── main classifier ───────────────────────────────────────────────────────────

# Priority order — first match wins
CATEGORY_ORDER = [
    ("wrestling",       WRESTLING),
    ("combat_sports",   COMBAT_SPORTS),
    ("taylor_swift",    TAYLOR_SWIFT),
    ("manosphere",      MANOSPHERE),
    ("musk_twitter",    MUSK_TWITTER),
    ("lgbtq_social",    LGBTQ_SOCIAL),
    ("religious",       RELIGIOUS),
    ("true_crime",      TRUE_CRIME),
    ("tech_gaming",     TECH_GAMING),
    ("fandom",          FANDOM),
    ("reality_tv",      REALITY_TV),
    ("entertainment",   ENTERTAINMENT),
    ("sports_womens",   SPORTS_WOMENS),
    ("sports_nba",      SPORTS_NBA),
    ("sports_nfl",      SPORTS_NFL),
    ("sports_mlb",      SPORTS_MLB),
    ("sports_nhl",      SPORTS_NHL),
    ("sports_soccer",   SPORTS_SOCCER),
    ("sports_college",  SPORTS_COLLEGE),
    ("sports_other",    SPORTS_OTHER),
    ("holidays",        HOLIDAYS),
    ("politics",        POLITICS),
    ("news_events",     NEWS_EVENTS),
    ("social_filler",   SOCIAL_FILLER),
]

# Merged display groups for charts
DISPLAY_GROUPS = {
    "wrestling":      "Wrestling (WWE/AEW)",
    "combat_sports":  "Combat Sports (UFC/Boxing)",
    "sports_nba":     "Sports — NBA",
    "sports_nfl":     "Sports — NFL",
    "sports_mlb":     "Sports — MLB",
    "sports_nhl":     "Sports — NHL",
    "sports_soccer":  "Sports — Soccer/International",
    "sports_college": "Sports — College",
    "sports_womens":  "Sports — Women's",
    "sports_other":   "Sports — Other (F1/Golf/Tennis)",
    "reality_tv":     "Reality TV",
    "entertainment":  "Entertainment (TV/Film/Music)",
    "taylor_swift":   "Taylor Swift",
    "fandom":         "Fandom (Anime/K-pop)",
    "tech_gaming":    "Tech & Gaming",
    "manosphere":     "Manosphere-adjacent",
    "lgbtq_social":   "LGBTQ+ & Social Justice",
    "religious":      "Religious Content",
    "musk_twitter":   "Musk / Twitter / X",
    "holidays":       "Holidays & Seasonal",
    "true_crime":     "True Crime",
    "politics":       "Politics (Partisan)",
    "news_events":    "News & Events",
    "social_filler":  "Social Filler",
    "other":          "Other / Uncategorised",
}

# Colour palette
CAT_COLORS = {
    "wrestling":      "#d62728",
    "combat_sports":  "#ff7f0e",
    "sports_nba":     "#1f77b4",
    "sports_nfl":     "#4a90d9",
    "sports_mlb":     "#6baed6",
    "sports_nhl":     "#9ecae1",
    "sports_soccer":  "#3182bd",
    "sports_college": "#08519c",
    "sports_womens":  "#74c476",
    "sports_other":   "#c6dbef",
    "reality_tv":     "#9467bd",
    "entertainment":  "#c5b0d5",
    "taylor_swift":   "#e7ba52",
    "fandom":         "#e377c2",
    "tech_gaming":    "#17becf",
    "manosphere":     "#8c564b",
    "lgbtq_social":   "#f781bf",
    "religious":      "#bcbd22",
    "musk_twitter":   "#000000",
    "holidays":       "#98df8a",
    "true_crime":     "#7f7f7f",
    "politics":       "#e41a1c",
    "news_events":    "#525252",
    "social_filler":  "#d9d9d9",
    "other":          "#f0f0f0",
}


def classify_category(topic: str) -> str:
    token_set, _ = _prepare(topic)
    for cat_name, keywords in CATEGORY_ORDER:
        if _hits(keywords, token_set):
            return cat_name
    return "other"
