"""Text normalization for business names and addresses.

Country-aware (US / India / France / unknown). Unknown countries fall back to
generic rules, so an unseen country never breaks the pipeline.
Non-Latin tokens are mapped through a dictionary learned from training pairs
(see translit.py), with unidecode as fallback.
"""
import re
from unidecode import unidecode

# ---------------------------------------------------------------- constants
LEGAL = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "pvt", "private", "llp", "plc", "lp", "pllc", "pc", "ms", "the", "public",
    "sarl", "sas", "sasu", "eurl", "sci", "sa", "snc", "selarl", "scp", "gmbh",
}
NAME_STOP = {"and", "of", "et", "de", "du", "des", "la", "le", "les", "d", "l"}
DBA_RE = re.compile(r"\b(?:doing business as|d/b/a|dba|t/a|trading as|a/k/a|aka)\b")
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)((?:\.[a-z]{2,4}){1,2})/?$")
NONLATIN_RE = re.compile(r"[^\x00-ɏ]")
DOTTED_RE = re.compile(r"\b([a-z])\.(?=[a-z]\b)")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
DIGITS = re.compile(r"\d+")
SPLIT_KEEP = re.compile(r"(\s+|[,;()])")

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "new delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "jammu kashmir": "jk",
    "ladakh": "la", "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "lakshadweep": "ld",
}
IN_CODE_ALIAS = {"ts": "tg", "or": "od", "ct": "cg", "ut": "uk", "tl": "tg", "cht": "cg"}

COMMON_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court",
    "cir": "circle", "pl": "place", "hwy": "highway", "pkwy": "parkway", "sq": "square",
    "ter": "terrace", "trl": "trail", "apt": "apartment", "apts": "apartments", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "mt": "mount", "ft": "fort",
    "cres": "crescent", "ctr": "center", "centre": "center", "expy": "expressway", "fwy": "freeway",
    "rte": "route", "rt": "route", "hts": "heights", "jct": "junction", "xing": "crossing",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "opp": "opposite", "nr": "near",
    "marg": "marg", "ngr": "nagar", "clny": "colony", "sec": "sector", "extn": "extension",
    "ext": "extension",
}
FR_ABBR = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
    "blvd": "boulevard", "all": "allee", "imp": "impasse", "che": "chemin", "chem": "chemin",
    "ch": "chemin", "rte": "route", "pl": "place", "sq": "square", "st": "saint", "ste": "sainte",
    "fbg": "faubourg", "crs": "cours", "qu": "quai", "qua": "quai", "prom": "promenade",
    "res": "residence", "lot": "lotissement", "zi": "zone", "za": "zone", "zac": "zone",
    "sen": "sente", "pas": "passage", "ham": "hameau", "vla": "villa", "mte": "montee",
}
ADDR_STOP = {
    "no", "h", "hno", "c", "o", "co", "city", "of", "the", "de", "du", "des", "la", "le", "les", "d",
    "l", "and", "et", "near", "opposite", "null", "none", "na", "nan", "n", "a", "b", "at", "post",
    "po", "dist", "district", "tq", "taluk", "tal", "via", "bis", "ter", "unit", "suite",
    "floor", "apartment", "building", "india", "usa", "us", "france", "county", "cdp", "town",
    "township", "village", "ndeg", "hno",
}
STREET_TYPES = {
    "road", "street", "avenue", "boulevard", "drive", "lane", "court", "circle", "place", "highway",
    "parkway", "square", "terrace", "trail", "way", "rue", "allee", "impasse", "chemin", "route",
    "quai", "cours", "marg", "nagar", "colony", "sector", "north", "south", "east", "west",
    "main", "cross",
}


def _state_regex(states):
    names = sorted(states, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


US_STATE_RE = _state_regex(US_STATES)
IN_STATE_RE = _state_regex(IN_STATES)
US_CODES = set(US_STATES.values())
IN_CODES = set(IN_STATES.values()) | set(IN_CODE_ALIAS)


def country_key(country):
    c = (country or "").strip().lower()
    if c in ("us", "usa", "united states", "united states of america"):
        return "us"
    if c in ("india", "in", "ind"):
        return "in"
    if c in ("france", "fr", "fra"):
        return "fr"
    return c or "unk"


# ---------------------------------------------------------------- helpers
def has_nonlatin(s):
    return bool(NONLATIN_RE.search(s or ""))


def transliterate(s, mapping):
    """Map each non-Latin token through a learned dictionary, else unidecode."""
    if not s or not NONLATIN_RE.search(s):
        return s
    out = []
    for part in SPLIT_KEEP.split(s):
        if part and NONLATIN_RE.search(part):
            key = part.strip().lower()
            out.append(mapping.get(key) if mapping and key in mapping else unidecode(part))
        else:
            out.append(part)
    return "".join(out)


def _basic(s):
    s = unidecode(s).lower()
    s = s.replace("&", " and ").replace("'", "").replace("`", "")
    s = DOTTED_RE.sub(r"\1", s)
    s = DOTTED_RE.sub(r"\1", s)
    return s


# ---------------------------------------------------------------- names
def norm_name(raw, mapping=None):
    raw = raw or ""
    nonlatin = has_nonlatin(raw)
    s = transliterate(raw, mapping)
    s = _basic(s).strip()
    s = re.sub(r"\bm\s*/\s*s\b", " ", s)
    # strip leading/trailing noise punctuation for domain detection
    bare = re.sub(r"^[^a-z0-9]+|[^a-z0-9/]+$", "", s)
    is_domain = 0
    m = DOMAIN_RE.match(bare)
    if m and " " not in bare:
        is_domain = 1
        s = m.group(1).replace("-", "")
    alt = ""
    m = DBA_RE.search(s)
    if m:
        alt = s[:m.start()]
        s = s[m.end():]
        if not NON_ALNUM.sub("", s):  # nothing after the DBA marker
            s, alt = alt, ""
    toks = [t for t in NON_ALNUM.split(s) if t]
    core = [t for t in toks if t not in LEGAL and t not in NAME_STOP]
    legal = sorted({t for t in toks if t in LEGAL and t != "the"})
    if not core:
        core = toks
    alt_toks = [t for t in NON_ALNUM.split(alt) if t and t not in LEGAL and t not in NAME_STOP]
    return {
        "core": " ".join(core),
        "compact": "".join(core),
        "alt": " ".join(alt_toks),
        "legal": " ".join(legal),
        "is_domain": is_domain,
        "nonlatin": int(nonlatin),
    }


# ---------------------------------------------------------------- addresses
def norm_addr(raw, ck, mapping=None):
    raw = raw or ""
    nonlatin = has_nonlatin(raw)
    s = transliterate(raw, mapping)
    s = _basic(s)
    s = re.sub(r"<\s*null\s*>|\bnull\b|\bn/a\b", " ", s)
    s = re.sub(r"\bc\s*/\s*o\b", " ", s)
    s = re.sub(r"\bh\s*\.?\s*no\b", " ", s)
    s = s.replace("n°", " ").replace("no.", " ")
    if ck == "us":
        s = US_STATE_RE.sub(lambda m: " " + US_STATES[m.group(1)] + " ", s)
        abbr, codes, alias = COMMON_ABBR, US_CODES, {}
    elif ck == "in":
        s = IN_STATE_RE.sub(lambda m: " " + IN_STATES[m.group(1)] + " ", s)
        abbr, codes, alias = COMMON_ABBR, IN_CODES, IN_CODE_ALIAS
    elif ck == "fr":
        abbr, codes, alias = FR_ABBR, set(), {}
    else:
        abbr, codes, alias = COMMON_ABBR, set(), {}
    nums = []
    for d in DIGITS.findall(s):
        d = d.lstrip("0") or "0"
        if d not in nums:
            nums.append(d)
    toks = []
    for t in NON_ALNUM.split(s):
        if not t:
            continue
        if t.isdigit():
            toks.append(t.lstrip("0") or "0")
            continue
        # split alnum like "12271d" / "9th" / "sector99" into digits + letters
        if any(ch.isdigit() for ch in t):
            letters = re.sub(r"\d+", " ", t).split()
            toks.extend(abbr.get(x, x) for x in letters if x not in ("st", "nd", "rd", "th"))
            continue
        toks.append(abbr.get(t, t))
    state = ""
    words = []
    for t in toks:
        if t.isdigit():
            continue
        tt = alias.get(t, t)
        if tt in codes and not state:
            state = tt
            continue
        if tt in codes:
            continue
        if t in ADDR_STOP or len(t) < 2:
            continue
        words.append(t)
    norm = " ".join(t for t in toks if t not in ADDR_STOP)
    return {
        "anorm": norm,
        "awords": " ".join(words),
        "nums": " ".join(nums),
        "state": state,
        "aempty": int(not words and not nums),
        "anonlatin": int(nonlatin),
    }


def street_words(awords):
    return [w for w in awords.split() if w not in STREET_TYPES]
