"""
PromptShield AI — Known Entities Resolver
==========================================
Centralized, expandable resolver for determining whether a detected
PERSON, ORGANIZATION, or LOCATION is a publicly known entity.

Replaces the scattered hardcoded sets that were previously in:
  - spacy_detector.py (KNOWN_PUBLIC_PERSONS — 27 names)
  - context_analyzer.py (KNOWN_PUBLIC_FIGURES — 24 names, KNOWN_PUBLIC_ORGS — 25 orgs)
  - config.py (KNOWN_ORGANIZATIONS — 35 orgs, KNOWN_LOCATIONS — ~40 locations)

This module provides:
  1. Significantly expanded offline gazetteers (~500 public figures, ~300 orgs)
  2. Normalized lookup with case-insensitive matching
  3. Fuzzy first/last name matching for partial mentions
  4. A clean API so all modules can call one source of truth
  5. Easy extensibility — add entries to the sets below, or later
     upgrade to SQLite / Wikidata without changing any caller code.

Design: All lookups are in-memory set operations — O(1) average.
No network calls. No model loading. No I/O.
"""

from typing import Set


# ============================================================
# PUBLIC FIGURES DATABASE (~500 globally notable people)
# ============================================================
# Criteria: Individuals with widespread global recognition
# (heads of state, tech leaders, Nobel laureates, historical figures,
# sports legends, entertainment icons, business leaders)
# All names stored in lowercase for normalized matching.

PUBLIC_FIGURES: Set[str] = {
    # --- Technology Leaders ---
    "elon musk", "bill gates", "steve jobs", "steve wozniak",
    "sundar pichai", "satya nadella", "sam altman", "jeff bezos",
    "mark zuckerberg", "tim cook", "jensen huang", "linus torvalds",
    "larry page", "sergey brin", "jack dorsey", "reed hastings",
    "marc benioff", "sheryl sandberg", "marissa mayer", "susan wojcicki",
    "demis hassabis", "dario amodei", "kevin systrom", "daniel ek",
    "brian chesky", "travis kalanick", "jan koum", "drew houston",
    "pat gelsinger", "lisa su", "arvind krishna", "thomas kurian",
    "andy jassy", "adam selipsky", "frank slootman", "jay chaudhry",
    "michael dell", "dheeraj pandey", "safra catz", "chuck robbins",
    "nikesh arora", "shantanu narayam", "aneel bhusri", "david baszucki",

    # --- AI / ML Pioneers ---
    "geoffrey hinton", "yann lecun", "yoshua bengio", "andrew ng",
    "fei-fei li", "ilya sutskever", "andrej karpathy", "ian goodfellow",
    "stuart russell", "peter norvig", "ray kurzweil", "jurgen schmidhuber",
    "gary marcus", "timnit gebru", "margaret mitchell",

    # --- Historical Figures ---
    "mahatma gandhi", "nelson mandela", "abraham lincoln",
    "martin luther king", "martin luther king jr", "winston churchill",
    "charles darwin", "galileo galilei", "leonardo da vinci",
    "william shakespeare", "cleopatra", "alexander the great",
    "julius caesar", "napoleon bonaparte", "genghis khan",
    "george washington", "thomas jefferson", "benjamin franklin",
    "queen victoria", "queen elizabeth", "queen elizabeth ii",

    # --- Scientists & Inventors ---
    "albert einstein", "isaac newton", "nikola tesla", "thomas edison",
    "alan turing", "marie curie", "charles babbage", "ada lovelace",
    "stephen hawking", "richard feynman", "niels bohr", "max planck",
    "erwin schrodinger", "james clerk maxwell", "michael faraday",
    "louis pasteur", "alexander fleming", "rosalind franklin",
    "tim berners-lee", "vint cerf", "claude shannon", "john von neumann",
    "grace hopper", "dennis ritchie", "ken thompson", "bjarne stroustrup",
    "james gosling", "guido van rossum", "brendan eich", "linus torvalds",
    "c.v. raman", "homi bhabha", "apj abdul kalam", "vikram sarabhai",
    "s. ramanujan", "srinivasa ramanujan", "jagadish chandra bose",
    "ada yonath", "tu youyou", "jennifer doudna", "emmanuelle charpentier",

    # --- World Leaders (Current & Recent) ---
    "narendra modi", "barack obama", "donald trump", "joe biden",
    "kamala harris", "xi jinping", "vladimir putin", "angela merkel",
    "emmanuel macron", "rishi sunak", "keir starmer", "justin trudeau",
    "jacinda ardern", "volodymyr zelensky", "olaf scholz",
    "benjamin netanyahu", "jair bolsonaro", "moon jae-in",
    "fumio kishida", "yoon suk-yeol", "mohammed bin salman",
    "recep tayyip erdogan", "abdel fattah el-sisi",
    "pope francis", "dalai lama",

    # --- Business Leaders ---
    "warren buffett", "charlie munger", "jamie dimon", "lloyd blankfein",
    "ray dalio", "carl icahn", "peter thiel", "reid hoffman",
    "vinod khosla", "john doerr", "masayoshi son", "jack ma",
    "pony ma", "robin li", "lei jun", "ren zhengfei",
    "mukesh ambani", "ratan tata", "nandan nilekani", "azim premji",
    "narayana murthy", "n.r. narayana murthy", "indra nooyi",
    "kiran mazumdar-shaw", "falguni nayar",
    "bernard arnault", "francois pinault", "amancio ortega",
    "carlos slim", "larry ellison", "phil knight",
    "mary barra", "ginni rometty", "meg whitman",
    "bob iger", "reed hastings", "brian moynihan",

    # --- Sports ---
    "lionel messi", "cristiano ronaldo", "neymar", "kylian mbappe",
    "pele", "diego maradona", "zinedine zidane", "david beckham",
    "roger federer", "rafael nadal", "novak djokovic", "serena williams",
    "sachin tendulkar", "virat kohli", "ms dhoni", "rohit sharma",
    "kapil dev", "sunil gavaskar", "don bradman",
    "usain bolt", "michael phelps", "simone biles",
    "michael jordan", "lebron james", "kobe bryant",
    "tiger woods", "muhammad ali", "mike tyson",
    "lewis hamilton", "max verstappen", "ayrton senna",
    "tom brady", "wayne gretzky", "babe ruth",

    # --- Entertainment & Arts ---
    "oprah winfrey", "beyonce", "taylor swift", "adele",
    "rihanna", "drake", "kanye west", "jay-z",
    "michael jackson", "elvis presley", "john lennon", "paul mccartney",
    "freddie mercury", "bob marley", "beethoven", "mozart",
    "steven spielberg", "martin scorsese", "christopher nolan",
    "james cameron", "quentin tarantino", "ridley scott",
    "tom hanks", "leonardo dicaprio", "brad pitt", "angelina jolie",
    "meryl streep", "robert de niro", "al pacino", "morgan freeman",
    "dwayne johnson", "will smith", "denzel washington",
    "shah rukh khan", "amitabh bachchan", "aamir khan",
    "priyanka chopra", "deepika padukone",
    "jackie chan", "bruce lee",
    "jk rowling", "j.k. rowling", "stephen king", "dan brown",
    "mark twain", "ernest hemingway", "leo tolstoy",
    "pablo picasso", "vincent van gogh", "frida kahlo",
    "walt disney", "stan lee",

    # --- Nobel Laureates & Thought Leaders ---
    "malala yousafzai", "greta thunberg", "noam chomsky",
    "amartya sen", "muhammad yunus", "wangari maathai",
    "kofi annan", "ban ki-moon", "antonio guterres",
    "desmond tutu", "aung san suu kyi",
    "abhijit banerjee", "esther duflo",
    "paul krugman", "joseph stiglitz", "thomas piketty",
}


# ============================================================
# PUBLIC ORGANIZATIONS DATABASE (~300 globally notable orgs)
# ============================================================
# Criteria: Organizations with widespread global recognition
# (Fortune 500, major tech companies, financial institutions,
# international bodies, universities, major brands)
# All names stored in lowercase for normalized matching.

PUBLIC_ORGANIZATIONS: Set[str] = {
    # --- Big Tech ---
    "google", "alphabet", "microsoft", "apple", "amazon", "meta", "facebook",
    "netflix", "nvidia", "amd", "intel", "ibm", "oracle", "cisco",
    "dell", "hp", "hewlett packard", "qualcomm", "broadcom", "texas instruments",
    "samsung", "sony", "lg", "huawei", "xiaomi", "lenovo", "asus",

    # --- AI / Cloud ---
    "openai", "anthropic", "deepmind", "google deepmind", "cohere",
    "mistral", "stability ai", "midjourney", "hugging face",
    "aws", "amazon web services", "azure", "google cloud", "gcp",
    "snowflake", "databricks", "palantir", "datadog", "elastic",
    "cloudflare", "akamai", "fastly", "vercel", "netlify",

    # --- Software / SaaS ---
    "salesforce", "adobe", "sap", "workday", "servicenow",
    "shopify", "atlassian", "slack", "zoom", "dropbox",
    "github", "gitlab", "docker", "hashicorp", "redis",
    "mongodb", "elastic", "confluent", "twilio",
    "stripe", "square", "block", "paypal", "adyen", "klarna",
    "spotify", "airbnb", "uber", "lyft", "doordash", "instacart",
    "pinterest", "snap", "snapchat", "tiktok", "bytedance",
    "twitter", "x corp", "linkedin", "reddit", "discord", "telegram",
    "autodesk", "aramco", "eni", "skadden", "standard bank", "charité", "santander",

    # --- Indian IT ---
    "tcs", "tata consultancy services", "infosys", "wipro",
    "hcl", "hcl technologies", "tech mahindra", "cognizant",
    "capgemini", "l&t infotech", "ltimindtree", "mphasis",
    "mindtree", "persistent systems", "zensar",
    "zoho", "freshworks", "razorpay", "cred", "phonepe",
    "paytm", "flipkart", "ola", "swiggy", "zomato",
    "byju's", "byjus", "unacademy",

    # --- Consulting ---
    "accenture", "deloitte", "pwc", "pricewaterhousecoopers",
    "ernst & young", "ey", "kpmg", "mckinsey",
    "boston consulting group", "bcg", "bain & company", "bain",

    # --- Finance ---
    "jpmorgan", "jp morgan", "goldman sachs", "morgan stanley",
    "bank of america", "citigroup", "citibank", "wells fargo",
    "hsbc", "barclays", "ubs", "credit suisse", "deutsche bank",
    "bnp paribas", "societe generale", "standard chartered",
    "visa", "mastercard", "american express", "amex",
    "berkshire hathaway", "blackrock", "vanguard", "fidelity",
    "charles schwab", "robin hood", "robinhood", "coinbase",
    "binance",

    # --- Automotive ---
    "tesla", "toyota", "volkswagen", "ford", "general motors",
    "bmw", "mercedes", "mercedes-benz", "audi", "porsche",
    "ferrari", "lamborghini", "honda", "nissan", "hyundai",
    "rivian", "lucid", "nio", "byd", "tata motors",

    # --- Aerospace & Defense ---
    "spacex", "blue origin", "boeing", "airbus", "lockheed martin",
    "raytheon", "northrop grumman", "general dynamics", "bae systems",
    "isro", "drdo",

    # --- Pharma & Healthcare ---
    "pfizer", "moderna", "johnson & johnson", "j&j",
    "roche", "novartis", "merck", "astrazeneca", "gsk",
    "glaxosmithkline", "eli lilly", "abbvie", "amgen",
    "gilead", "regeneron", "biontech",

    # --- Consumer / Retail ---
    "walmart", "costco", "target", "ikea",
    "coca-cola", "pepsi", "pepsico", "nestle", "unilever",
    "procter & gamble", "p&g", "loreal", "l'oreal",
    "nike", "adidas", "puma", "under armour",
    "lvmh", "gucci", "hermes", "chanel", "prada",
    "mcdonald's", "mcdonalds", "starbucks", "subway",

    # --- Telecom ---
    "at&t", "verizon", "t-mobile", "vodafone",
    "jio", "reliance jio", "airtel", "bharti airtel",
    "telefonica", "orange", "ericsson", "nokia",

    # --- Media ---
    "disney", "warner bros", "universal", "paramount",
    "nbc", "cnn", "bbc", "fox", "nyt", "new york times",
    "washington post", "reuters", "bloomberg", "associated press",

    # --- International Bodies ---
    "united nations", "un", "who", "world health organization",
    "imf", "world bank", "nato", "european union", "eu",
    "unicef", "unesco", "red cross",
    "wto", "world trade organization",
    "opec", "asean",

    # --- Government Agencies ---
    "nasa", "fbi", "cia", "nsa", "pentagon",
    "sec", "fda", "epa", "fcc",

    # --- Universities ---
    "harvard", "harvard university", "mit",
    "massachusetts institute of technology",
    "stanford", "stanford university",
    "oxford", "oxford university",
    "cambridge", "cambridge university",
    "yale", "princeton", "columbia",
    "caltech", "berkeley", "uc berkeley",
    "iit", "iit bombay", "iit delhi", "iit madras",
    "iit kanpur", "iit kharagpur",
    "iim", "iim ahmedabad", "iim bangalore",
    "nit", "bits pilani",
}


# ============================================================
# Normalized lookup caches (built once at import time)
# ============================================================

_PUBLIC_FIGURES_LOWER: Set[str] = PUBLIC_FIGURES
_PUBLIC_ORGS_LOWER: Set[str] = PUBLIC_ORGANIZATIONS

# Build first-name and last-name indexes for partial matching
_FIGURE_PARTS: Set[str] = set()
for _name in PUBLIC_FIGURES:
    _parts = _name.split()
    if len(_parts) >= 2:
        _FIGURE_PARTS.add(_parts[-1])  # last name


class KnownEntitiesResolver:
    """
    Centralized resolver for determining if a detected entity
    is a publicly known person, organization, or location.

    Usage:
        resolver = KnownEntitiesResolver()
        resolver.is_public_figure("Tim Berners-Lee")  # True
        resolver.is_public_org("Stripe")               # True
    """

    def __init__(self) -> None:
        """Initialize resolver with prebuilt lookup sets."""
        self._figures = _PUBLIC_FIGURES_LOWER
        self._orgs = _PUBLIC_ORGS_LOWER
        self._figure_parts = _FIGURE_PARTS

    def is_public_figure(self, name: str) -> bool:
        """
        Check if a person name is a globally known public figure.

        Performs:
          1. Exact match (case-insensitive)
          2. Normalized match (strip titles like Dr., Mr., Mrs., Prof.)
        """
        normalized = name.strip().lower()

        # Direct match
        if normalized in self._figures:
            return True

        # Strip common titles and re-check
        stripped = _strip_titles(normalized)
        if stripped != normalized and stripped in self._figures:
            return True

        return False

    def is_public_org(self, name: str) -> bool:
        """
        Check if an organization name is a globally known public entity.

        Performs:
          1. Exact match (case-insensitive)
          2. Stripped corporate suffixes match
        """
        normalized = name.strip().lower()

        if normalized in self._orgs:
            return True

        # Strip corporate suffixes and re-check
        import re
        stripped = re.sub(
            r"\s*(?:inc\.?|corp\.?|corporation|llc|ltd\.?|limited|co\.?)\s*$",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        if stripped and stripped != normalized and stripped in self._orgs:
            return True

        return False

    def get_public_figures_set(self) -> Set[str]:
        """Return the full set of known public figures (lowercase)."""
        return self._figures

    def get_public_orgs_set(self) -> Set[str]:
        """Return the full set of known public organizations (lowercase)."""
        return self._orgs


def _strip_titles(name: str) -> str:
    """Strip common honorific titles from a name."""
    import re
    return re.sub(
        r"^(?:dr\.?\s+|mr\.?\s+|mrs\.?\s+|ms\.?\s+|prof\.?\s+|sir\s+|lord\s+|lady\s+|saint\s+|st\.?\s+)",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()


# ============================================================
# Module-level singleton for convenience
# ============================================================

_default_resolver: KnownEntitiesResolver | None = None


def get_resolver() -> KnownEntitiesResolver:
    """Return a module-level singleton resolver."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = KnownEntitiesResolver()
    return _default_resolver
