"""
Address Validator API

A lightweight FastAPI service that takes messy/voice-transcribed addresses,
generates phonetic spelling variants, and validates them against geocoding
services. Returns the best match with structured address data.

Usage:
    pip install -r requirements.txt
    python server.py

    POST http://localhost:8100/validate-address
    { "raw_address": "2711 brian hall drive austin tx" }
"""

import asyncio
import os
import re
import time
from itertools import product
from typing import Optional

import httpx
import jellyfish
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="Address Validator", version="0.1.0")

# ── Configurable service URLs ─────────────────────────────────────
# Default to public servers for local dev; override via env vars for self-hosted.
PHOTON_URL = os.environ.get("PHOTON_URL", "https://photon.komoot.io")
OSRM_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit: Nominatim allows 1 req/sec
_last_nominatim_call = 0.0
NOMINATIM_DELAY = 1.1  # seconds between calls


# ── Request / Response Models ──────────────────────────────────────

class AddressRequest(BaseModel):
    raw_address: str
    country: str = "us"
    city: str = ""      # Optional: filter results to this city (e.g., "Austin")
    state: str = ""     # Optional: filter results to this state (e.g., "Texas")


class AddressMatch(BaseModel):
    matched: bool
    confidence: str  # "exact", "high", "medium", "low"
    input_address: str
    matched_query: Optional[str] = None
    formatted_address: Optional[str] = None
    house_number: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    county: Optional[str] = None
    state: Optional[str] = None
    postcode: Optional[str] = None
    country: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    variants_tried: int = 0


# ── Phonetic Variant Generation ────────────────────────────────────

# Common voice-to-text confusions: source sound → possible spellings
VOWEL_SWAPS = {
    "a": ["a", "o", "e"],
    "e": ["e", "a", "i"],
    "i": ["i", "y", "e"],
    "o": ["o", "a", "u"],
    "u": ["u", "o"],
    "y": ["y", "i", "e"],
}

CONSONANT_SWAPS = {
    "b": ["b", "p"],
    "d": ["d", "t"],
    "f": ["f", "ph"],
    "g": ["g", "j"],
    "j": ["j", "g"],
    "k": ["k", "c"],
    "c": ["c", "k"],
    "n": ["n", "m"],
    "m": ["m", "n"],
    "s": ["s", "z", "c"],
    "z": ["z", "s"],
    "t": ["t", "d"],
    "p": ["p", "b"],
    "v": ["v", "b"],
    "w": ["w", "wh"],
}

# ── Local street name dictionary (for fuzzy fallback) ──────────────
# When geocoder variants all fail, fuzzy-match the street words against
# known street names. Loaded from austin_streets.json if present.
_STREET_DICT: list[str] = []
_STREET_DICT_PATH = os.path.join(os.path.dirname(__file__), "austin_streets.json")
if os.path.exists(_STREET_DICT_PATH):
    import json as _json
    with open(_STREET_DICT_PATH) as _f:
        _STREET_DICT = _json.load(_f)


def _fuzzy_street_match(street_words: list[str], street_type: str = "") -> str | None:
    """Find the closest known street name using Jaro-Winkler similarity.

    Tries the raw words, the joined form, and vowel/consonant variants
    against the street dictionary. Returns the best match above threshold.
    """
    if not _STREET_DICT:
        return None

    raw = " ".join(street_words).lower()
    joined = "".join(street_words).lower()

    best_score = 0.0
    best_match = None
    threshold = 0.80  # Jaro-Winkler threshold (0.80 catches "all torf"→"oltorf" at 0.819)

    for known in _STREET_DICT:
        # Extract just the street name (without type like "Drive", "Street")
        known_parts = known.lower().split()
        known_types = {"drive","dr","street","st","road","rd","lane","ln","boulevard",
                       "blvd","avenue","ave","way","court","ct","circle","cir","place",
                       "pl","trail","loop","parkway","expressway","park"}
        known_name_parts = [p for p in known_parts if p not in known_types]
        known_name = " ".join(known_name_parts)
        known_joined = "".join(known_name_parts)

        # Try matching against raw, joined, and individual-word combos
        for candidate in [raw, joined]:
            for target in [known_name, known_joined]:
                if not candidate or not target:
                    continue
                score = jellyfish.jaro_winkler_similarity(candidate, target)
                if score > best_score:
                    best_score = score
                    best_match = known

    if best_score >= threshold and best_match:
        return best_match
    return None


# Common street type abbreviations
STREET_TYPES = {
    "drive": ["dr", "drive", "drv"],
    "dr": ["dr", "drive"],
    "street": ["st", "street", "str"],
    "st": ["st", "street"],
    "avenue": ["ave", "avenue", "av"],
    "ave": ["ave", "avenue"],
    "road": ["rd", "road"],
    "rd": ["rd", "road"],
    "lane": ["ln", "lane"],
    "ln": ["ln", "lane"],
    "boulevard": ["blvd", "boulevard"],
    "blvd": ["blvd", "boulevard"],
    "court": ["ct", "court"],
    "ct": ["ct", "court"],
    "circle": ["cir", "circle"],
    "place": ["pl", "place"],
    "way": ["way", "wy"],
}

# US state abbreviations for normalization
STATE_ABBREVS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
    "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}


def parse_address_parts(raw: str) -> dict:
    """Split a raw address into approximate parts for variant generation."""
    tokens = raw.lower().strip().split()
    parts = {"number": None, "street_words": [], "street_type": None, "city": None, "state": None}

    # Extract house number (first token if numeric)
    idx = 0
    if tokens and re.match(r"^\d+$", tokens[0]):
        parts["number"] = tokens[0]
        idx = 1

    # Extract state (last token if it's a known abbreviation or state name)
    remaining = tokens[idx:]
    if remaining:
        last = remaining[-1]
        if last.upper() in STATE_ABBREVS.values() or last in STATE_ABBREVS:
            parts["state"] = last
            remaining = remaining[:-1]

    # Extract city (token before state, if we found a state)
    # Heuristic: take 1 word for city unless it looks like a street type
    if parts["state"] and remaining:
        candidate = remaining[-1]
        if candidate not in STREET_TYPES:
            parts["city"] = candidate
            remaining = remaining[:-1]

    # Extract street type — scan remaining tokens for known types
    for i, token in enumerate(remaining):
        if token in STREET_TYPES:
            parts["street_type"] = token
            remaining = remaining[:i] + remaining[i + 1:]
            break

    # Everything left is the street name
    if remaining:
        parts["street_words"] = remaining

    return parts


def generate_street_variants(street_words: list[str]) -> list[str]:
    """Generate phonetic variants of street name words.

    Handles the key voice-to-text failure modes:
    - Word boundary errors: "bryonhall" heard as "brian hall"
    - Vowel confusion: "bryon" heard as "brian"
    - Consonant confusion: similar-sounding consonants
    """
    variants = set()
    original = " ".join(street_words)
    variants.add(original)

    # Strategy 1: Join adjacent words (word boundary error)
    # "brian hall" → "brianhall"
    if len(street_words) >= 2:
        for i in range(len(street_words) - 1):
            joined = street_words[:i] + [street_words[i] + street_words[i + 1]] + street_words[i + 2:]
            variants.add(" ".join(joined))

    # Strategy 2: Split single words at different points
    # "brianhall" → "brian hall"
    for word in street_words:
        if len(word) > 4:
            for split_pos in range(2, len(word) - 1):
                left, right = word[:split_pos], word[split_pos:]
                new_words = [w for w in street_words]
                idx = new_words.index(word)
                new_words[idx:idx + 1] = [left, right]
                variants.add(" ".join(new_words))

    # Strategy 3: Vowel substitutions on each word
    # "brian" → "brion", "bryon", "bryen" etc.
    for word in street_words:
        vowel_positions = [(i, c) for i, c in enumerate(word) if c in VOWEL_SWAPS]
        if vowel_positions and len(vowel_positions) <= 3:
            swap_options = [VOWEL_SWAPS[c] for _, c in vowel_positions]
            for combo in product(*swap_options):
                new_word = list(word)
                for (pos, _), replacement in zip(vowel_positions, combo):
                    new_word[pos] = replacement
                new_word = "".join(new_word)
                # Replace this word in the street words
                new_street = [new_word if w == word else w for w in street_words]
                variants.add(" ".join(new_street))
                # Also try joined version
                if len(street_words) >= 2:
                    variants.add("".join(new_street))

    # Strategy 4: Combine join + vowel swaps
    # "brian hall" → "bryonhall", "brionhall", etc.
    if len(street_words) >= 2:
        joined_word = "".join(street_words)
        vowel_positions = [(i, c) for i, c in enumerate(joined_word) if c in VOWEL_SWAPS]
        if vowel_positions and len(vowel_positions) <= 4:
            swap_options = [VOWEL_SWAPS[c] for _, c in vowel_positions]
            for combo in product(*swap_options):
                new_word = list(joined_word)
                for (pos, _), replacement in zip(vowel_positions, combo):
                    new_word[pos] = replacement
                variants.add("".join(new_word))

    # Strategy 5: Consonant substitutions on each word
    # "duball" → "duval", "flugerville" → "pflugerville"
    for word in street_words:
        consonant_positions = [(i, c) for i, c in enumerate(word) if c in CONSONANT_SWAPS]
        # Limit to 2 consonant positions to avoid combinatorial explosion
        if consonant_positions and len(consonant_positions) <= 2:
            swap_options = [CONSONANT_SWAPS[c] for _, c in consonant_positions]
            for combo in product(*swap_options):
                new_word = list(word)
                for (pos, _), replacement in zip(consonant_positions, combo):
                    new_word[pos:pos + 1] = list(replacement)  # handles "ph" multi-char
                new_word = "".join(new_word)
                new_street = [new_word if w == word else w for w in street_words]
                variants.add(" ".join(new_street))

    # Strategy 6: Consonant swaps on joined form
    if len(street_words) >= 2:
        joined_word = "".join(street_words)
        consonant_positions = [(i, c) for i, c in enumerate(joined_word) if c in CONSONANT_SWAPS]
        if consonant_positions and len(consonant_positions) <= 2:
            swap_options = [CONSONANT_SWAPS[c] for _, c in consonant_positions]
            for combo in product(*swap_options):
                new_word = list(joined_word)
                for (pos, _), replacement in zip(consonant_positions, combo):
                    new_word[pos:pos + 1] = list(replacement)
                variants.add("".join(new_word))

    return list(variants)


def build_full_queries(parts: dict, street_variants: list[str]) -> list[str]:
    """Assemble full address queries from parts + street variants.

    Uses only the primary street type (e.g. "drive") to keep the query list small.
    Street type variations are a low-value axis — the street name is where
    voice-to-text errors live.
    """
    queries = []
    stype = parts["street_type"] or ""

    for street in street_variants:
        q_parts = []
        if parts["number"]:
            q_parts.append(parts["number"])
        q_parts.append(street)
        if stype:
            q_parts.append(stype)
        if parts["city"]:
            q_parts.append(parts["city"])
        if parts["state"]:
            q_parts.append(parts["state"])
        queries.append(" ".join(q_parts))

    return queries


def score_variant(variant: str, original_parts: dict) -> float:
    """Score a query for priority. Lower = try first.

    Prioritizes:
    1. Joined words (word-boundary errors are the #1 voice failure)
    2. Fewer character changes from original
    3. Penalizes word splits (less common error)
    """
    original_street = " ".join(original_parts["street_words"])
    # Extract the street portion from the query
    q_lower = variant.lower()
    street_part = q_lower
    if original_parts["number"]:
        street_part = street_part.replace(original_parts["number"], "", 1).strip()
    if original_parts["city"]:
        street_part = street_part.replace(original_parts["city"], "", 1).strip()
    if original_parts["state"]:
        street_part = street_part.replace(original_parts["state"], "", 1).strip()
    if original_parts["street_type"]:
        street_part = street_part.replace(original_parts["street_type"], "", 1).strip()

    score = 0.0

    # Reward: joined words (no spaces in street name) — likely word boundary fix
    has_joined = " " not in street_part
    if has_joined:
        score -= 0.3

    # Reward: similarity to original using Jaro-Winkler
    score += (1.0 - jellyfish.jaro_winkler_similarity(street_part, original_street)) * 0.5

    # Reward: similarity to joined-original (catches "brianhall" vs "bryonhall")
    joined_original = original_street.replace(" ", "")
    score += (1.0 - jellyfish.jaro_winkler_similarity(street_part, joined_original)) * 0.5

    # Penalize: word splits that create very short fragments
    words = street_part.split()
    if any(len(w) < 2 for w in words):
        score += 0.5

    return score


# ── Geocoding ──────────────────────────────────────────────────────

async def geocode_photon(
    query: str, country: str, client: httpx.AsyncClient,
    filter_city: str = "", filter_state: str = "",
) -> Optional[dict]:
    """Call Photon geocoder (Komoot) — no rate limit, fast.
    Returns result in Nominatim-compatible format.
    When filter_city/filter_state are set, requests multiple results
    and picks the first one matching the city/state."""
    # Request more results when filtering by city so we can pick the right one
    limit = 5 if (filter_city or filter_state) else 1
    params = {"q": query, "limit": str(limit)}
    if country == "us":
        params["lang"] = "en"

    try:
        resp = await client.get(
            f"{PHOTON_URL}/api/",
            params=params,
            headers={"User-Agent": "AddressValidator/0.1"},
        )
        data = resp.json()
        features = data.get("features", [])
        if not features:
            return None

        # Score and filter candidates
        filter_city_l = filter_city.lower().strip()
        filter_state_l = filter_state.lower().strip()

        for feat in features:
            props = feat.get("properties", {})
            coords = feat.get("geometry", {}).get("coordinates", [])
            if not coords or len(coords) < 2:
                continue

            # Filter to correct country
            if country and props.get("countrycode", "").lower() != country.lower():
                continue

            # Filter by city if specified
            result_city = (props.get("city", "") or props.get("town", "") or props.get("village", "")).lower()
            if filter_city_l and filter_city_l not in result_city and result_city not in filter_city_l:
                continue

            # Filter by state if specified
            result_state = props.get("state", "").lower()
            if filter_state_l and filter_state_l not in result_state and result_state not in filter_state_l:
                continue

            # Must have a street-level result (not just city)
            # Photon uses "street" for address results and "name" for street-level results
            if filter_city_l and not props.get("street") and not props.get("housenumber") and not props.get("name"):
                continue

            # Convert to Nominatim-like format
            return {
                "lat": str(coords[1]),
                "lon": str(coords[0]),
                "display_name": ", ".join(filter(None, [
                    props.get("housenumber", ""),
                    props.get("street", "") or props.get("name", ""),
                    props.get("city", "") or props.get("town", "") or props.get("village", ""),
                    props.get("state", ""),
                    props.get("postcode", ""),
                    props.get("country", ""),
                ])),
            "address": {
                "house_number": props.get("housenumber"),
                "road": props.get("street") or props.get("name"),
                "city": props.get("city") or props.get("town") or props.get("village"),
                "county": props.get("county"),
                "state": props.get("state"),
                "postcode": props.get("postcode"),
                "country": props.get("country"),
            },
        }
    except Exception:
        return None


async def geocode_nominatim(query: str, country: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Call Nominatim geocoder with rate limiting. Used as fallback for verification."""
    global _last_nominatim_call

    elapsed = time.time() - _last_nominatim_call
    if elapsed < NOMINATIM_DELAY:
        await asyncio.sleep(NOMINATIM_DELAY - elapsed)

    _last_nominatim_call = time.time()

    params = {
        "q": query,
        "format": "json",
        "limit": "1",
        "countrycodes": country,
        "addressdetails": "1",
    }

    resp = await client.get(
        "https://nominatim.openstreetmap.org/search",
        params=params,
        headers={"User-Agent": "AddressValidator/0.1"},
    )
    results = resp.json()
    if results:
        return results[0]
    return None


# ── Main Endpoint ──────────────────────────────────────────────────

def _spoken_number_to_digits(text: str) -> str:
    """Convert spoken numbers to digits in address text.

    Handles patterns like:
      "fifteen oh three" → "1503"
      "twenty nine oh one" → "2901"
      "forty four seventy seven" → "4477"
      "eleven hundred" → "1100"
      "twelve twenty one" → "1221"
      "eighty eight sixty eight" → "8868"
      "one hundred" → "100"
      "three hundred" → "300"
      "five twelve" → "512"

    Only converts the leading number portion — leaves the rest of the address intact.
    """
    ONES = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19,
    }
    TENS = {
        "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }

    tokens = text.lower().split()
    if not tokens:
        return text

    # Try to consume number tokens from the front
    i = 0
    number_parts = []

    def _parse_small_number(idx):
        """Parse a 1-2 digit number starting at idx. Returns (value, tokens_consumed)."""
        if idx >= len(tokens):
            return None, 0
        t = tokens[idx]
        # "oh" = 0 (as in "oh one" = 01)
        if t == "oh" and idx + 1 < len(tokens) and tokens[idx + 1] in ONES:
            return ONES[tokens[idx + 1]], 2
        if t in ONES:
            return ONES[t], 1
        if t in TENS:
            # "twenty" alone or "twenty one"
            if idx + 1 < len(tokens) and tokens[idx + 1] in ONES and ONES[tokens[idx + 1]] < 10:
                return TENS[t] + ONES[tokens[idx + 1]], 2
            return TENS[t], 1
        return None, 0

    def _try_parse_number(idx):
        """Try to parse a full house number starting at idx.
        Returns (digit_string, tokens_consumed) or (None, 0)."""
        if idx >= len(tokens):
            return None, 0

        total_consumed = 0
        parts = []

        # Pattern 1: "N hundred" (e.g., "eleven hundred" = 1100, "three hundred" = 300)
        val, consumed = _parse_small_number(idx)
        if val is not None and consumed > 0:
            next_idx = idx + consumed
            if next_idx < len(tokens) and tokens[next_idx] == "hundred":
                hundred_val = val * 100
                next_idx += 1
                total_consumed = next_idx - idx
                # Check for trailing small number: "eleven hundred fifteen" = 1115
                trail_val, trail_consumed = _parse_small_number(next_idx)
                if trail_val is not None and trail_val < 100:
                    return str(hundred_val + trail_val), total_consumed + trail_consumed
                return str(hundred_val), total_consumed

            # Pattern 2: "N thousand N" (e.g., "eleven thousand four hundred" = 11400)
            if next_idx < len(tokens) and tokens[next_idx] == "thousand":
                thousand_val = val * 1000
                next_idx += 1
                total_consumed = next_idx - idx
                trail_val, trail_consumed = _parse_small_number(next_idx)
                if trail_val is not None:
                    # Could be "eleven thousand four hundred"
                    trail_next = next_idx + trail_consumed
                    if trail_next < len(tokens) and tokens[trail_next] == "hundred":
                        return str(thousand_val + trail_val * 100), total_consumed + trail_consumed + 1
                    return str(thousand_val + trail_val), total_consumed + trail_consumed
                return str(thousand_val), total_consumed

            # Pattern 3: Two-part number (e.g., "forty four seventy seven" = 4477, "twelve twenty one" = 1221)
            second_val, second_consumed = _parse_small_number(next_idx)
            if second_val is not None and second_consumed > 0:
                # "twelve twenty one" → 12 * 100 + 21 = 1221
                # "forty four seventy seven" → 44 * 100 + 77 = 4477
                # "five twelve" → 5 * 100 + 12 = 512
                combined = val * 100 + second_val
                if combined > 99:  # Only if it makes a plausible house number
                    return str(combined), consumed + second_consumed

            # Pattern 4: "N oh N" (e.g., "twenty nine oh one" = 2901)
            if next_idx < len(tokens) and tokens[next_idx] == "oh":
                oh_next = next_idx + 1
                if oh_next < len(tokens) and tokens[oh_next] in ONES:
                    return str(val * 100 + ONES[tokens[oh_next]]), consumed + 2

            # Single number at the start — only valid if > 0
            if val > 0:
                return str(val), consumed

        return None, 0

    digit_str, consumed = _try_parse_number(0)
    if digit_str and consumed > 0:
        remaining = " ".join(tokens[consumed:])
        return f"{digit_str} {remaining}".strip()

    return text


def _ordinals_to_numbers(text: str) -> str:
    """Convert spoken ordinals to numeric ordinals in address text.

    "west sixth street" → "west 6th street"
    "east forty second street" → "east 42nd street"
    "south first street" → "south 1st street"
    """
    ORDINALS = {
        "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th",
        "fifth": "5th", "sixth": "6th", "seventh": "7th", "eighth": "8th",
        "ninth": "9th", "tenth": "10th", "eleventh": "11th", "twelfth": "12th",
        "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th",
        "sixteenth": "16th", "seventeenth": "17th", "eighteenth": "18th",
        "nineteenth": "19th", "twentieth": "20th",
        "twenty first": "21st", "twenty second": "22nd", "twenty third": "23rd",
        "twenty fourth": "24th", "twenty fifth": "25th", "twenty sixth": "26th",
        "twenty seventh": "27th", "twenty eighth": "28th", "twenty ninth": "29th",
        "thirtieth": "30th",
        "thirty first": "31st", "thirty second": "32nd", "thirty third": "33rd",
        "thirty fourth": "34th", "thirty fifth": "35th", "thirty sixth": "36th",
        "thirty seventh": "37th", "thirty eighth": "38th", "thirty ninth": "39th",
        "fortieth": "40th",
        "forty first": "41st", "forty second": "42nd", "forty third": "43rd",
        "forty fourth": "44th", "forty fifth": "45th", "forty sixth": "46th",
        "forty seventh": "47th", "forty eighth": "48th", "forty ninth": "49th",
        "fiftieth": "50th",
        "fifty first": "51st", "fifty second": "52nd", "fifty third": "53rd",
        "fifty fourth": "54th", "fifty fifth": "55th",
    }
    result = text.lower()
    # Try two-word ordinals first (longer match wins)
    for spoken, numeric in sorted(ORDINALS.items(), key=lambda x: -len(x[0])):
        if spoken in result:
            result = result.replace(spoken, numeric, 1)
            break  # Only convert one ordinal per address
    return result


@app.post("/validate-address", response_model=AddressMatch)
async def validate_address(req: AddressRequest):
    raw = req.raw_address.strip()

    # Pre-process: convert spoken numbers and ordinals to digits
    raw = _spoken_number_to_digits(raw)
    raw = _ordinals_to_numbers(raw)

    filter_city = req.city
    filter_state = req.state

    # Auto-detect city/state from the raw address if not explicitly provided
    if not filter_city or not filter_state:
        parts = parse_address_parts(raw)
        if not filter_city and parts.get("city"):
            filter_city = parts["city"]
        if not filter_state and parts.get("state"):
            filter_state = parts["state"]

    # Step 0: Dictionary pre-correction — if the street name fuzzy-matches
    # a known street, rewrite the query BEFORE geocoding. This prevents the
    # geocoder from confidently returning the wrong street.
    parts_pre = parse_address_parts(raw)
    dict_corrected_query = None
    if parts_pre.get("street_words"):
        dict_match = _fuzzy_street_match(parts_pre["street_words"], parts_pre.get("street_type", ""))
        if dict_match:
            # Only use dictionary if it actually corrects something.
            # Compare the raw street words to dictionary name words.
            # Skip ONLY if the words themselves are identical (same count, same spelling).
            raw_words_lower = [w.lower() for w in parts_pre["street_words"]]
            generic = {"street","st","road","rd","drive","dr","lane","ln","boulevard",
                       "blvd","avenue","ave","way","court","ct","loop","parkway","expressway","trail","park"}
            dict_name_words = [w.lower() for w in dict_match.split() if w.lower() not in generic]
            if raw_words_lower != dict_name_words:  # Different words or word count = needs correction
                number = parts_pre.get("number", "")
                city_part = parts_pre.get("city") or filter_city or ""
                state_part = parts_pre.get("state") or filter_state or ""
                dict_corrected_query = f"{number} {dict_match} {city_part} {state_part}".strip()

    # Step 1: Try the dictionary-corrected query first (if available), then raw
    async with httpx.AsyncClient(timeout=10) as client:
        if dict_corrected_query:
            result = await geocode_photon(dict_corrected_query, req.country, client,
                                          filter_city=filter_city, filter_state=filter_state)
            if result:
                return build_response(raw, dict_corrected_query, result, "high", 1)

        result = await geocode_photon(raw, req.country, client,
                                      filter_city=filter_city, filter_state=filter_state)
        if result:
            return build_response(raw, raw, result, "exact", 2)

        result = await geocode_nominatim(raw, req.country, client)
        if result:
            return build_response(raw, raw, result, "exact", 3)

        # Step 2: Parse and generate variants
        parts = parse_address_parts(raw)

        if not parts["street_words"]:
            return AddressMatch(
                matched=False,
                confidence="low",
                input_address=raw,
                variants_tried=2,
            )

        street_variants = generate_street_variants(parts["street_words"])
        queries = build_full_queries(parts, street_variants)

        # Sort by priority — joined+vowel-swapped variants first
        queries.sort(key=lambda q: score_variant(q, parts))

        # Remove the original (already tried) and deduplicate
        seen = {raw.lower()}
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        # Step 3: Try variants using Photon (no rate limit — batch concurrently)
        batch_size = 10
        max_tries = min(len(unique_queries), 50)
        tried = 2

        for batch_start in range(0, max_tries, batch_size):
            batch = unique_queries[batch_start:batch_start + batch_size]
            tasks = [geocode_photon(q, req.country, client,
                                    filter_city=filter_city, filter_state=filter_state)
                     for q in batch]
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                tried += 1
                if result:
                    idx = batch_start + i
                    confidence = "high" if idx < 10 else "medium"
                    return build_response(raw, batch[i], result, confidence, tried)

        # Step 4: Nominatim fallback for top variants (different data coverage)
        top_variants = unique_queries[:3]
        for q in top_variants:
            tried += 1
            result = await geocode_nominatim(q, req.country, client)
            if result:
                # Apply city/state filter manually for Nominatim results
                addr = result.get("address", {})
                res_city = (addr.get("city", "") or addr.get("town", "") or addr.get("village", "")).lower()
                res_state = addr.get("state", "").lower()
                if filter_city and filter_city.lower() not in res_city and res_city not in filter_city.lower():
                    continue
                if filter_state and filter_state.lower() not in res_state and res_state not in filter_state.lower():
                    continue
                # Must have street-level result
                if filter_city and not addr.get("road") and not addr.get("house_number"):
                    continue
                return build_response(raw, q, result, "medium", tried)

        # Step 5: Local street dictionary fuzzy fallback
        # When all geocoder queries fail, fuzzy-match against known street names
        if parts.get("street_words"):
            dict_match = _fuzzy_street_match(parts["street_words"], parts.get("street_type", ""))
            if dict_match:
                # Rebuild the query with the dictionary street name
                number = parts.get("number", "")
                city = parts.get("city") or filter_city or ""
                state = parts.get("state") or filter_state or ""
                dict_query = f"{number} {dict_match} {city} {state}".strip()
                tried += 1

                result = await geocode_photon(dict_query, req.country, client,
                                              filter_city=filter_city, filter_state=filter_state)
                if result:
                    return build_response(raw, dict_query, result, "medium", tried)

                # Try Nominatim too
                tried += 1
                result = await geocode_nominatim(dict_query, req.country, client)
                if result:
                    return build_response(raw, dict_query, result, "medium", tried)

        return AddressMatch(
            matched=False,
            confidence="low",
            input_address=raw,
            variants_tried=tried,
        )


def build_response(
    raw: str, matched_query: str, geo: dict, confidence: str, variants_tried: int
) -> AddressMatch:
    addr = geo.get("address", {})
    return AddressMatch(
        matched=True,
        confidence=confidence,
        input_address=raw,
        matched_query=matched_query,
        formatted_address=geo.get("display_name"),
        house_number=addr.get("house_number"),
        street=addr.get("road"),
        city=addr.get("city") or addr.get("town") or addr.get("village"),
        county=addr.get("county"),
        state=addr.get("state"),
        postcode=addr.get("postcode"),
        country=addr.get("country"),
        lat=float(geo["lat"]),
        lng=float(geo["lon"]),
        variants_tried=variants_tried,
    )


# ── Route / Distance Models ────────────────────────────────────────

class RoutePoint(BaseModel):
    """A point can be an address string OR explicit lat/lng coordinates."""
    address: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None


class RouteRequest(BaseModel):
    origin: RoutePoint
    destination: RoutePoint
    waypoints: list[RoutePoint] = []  # optional intermediate stops


class RouteLeg(BaseModel):
    from_address: Optional[str] = None
    from_lat: float
    from_lng: float
    to_address: Optional[str] = None
    to_lat: float
    to_lng: float
    distance_miles: float
    distance_meters: float
    duration_minutes: float
    duration_seconds: float


class RouteResponse(BaseModel):
    success: bool
    total_distance_miles: float = 0.0
    total_distance_meters: float = 0.0
    total_duration_minutes: float = 0.0
    total_duration_seconds: float = 0.0
    legs: list[RouteLeg] = []
    geometry: Optional[dict] = None  # GeoJSON LineString for map rendering
    error: Optional[str] = None


# ── Route Helpers ──────────────────────────────────────────────────

async def resolve_point(point: RoutePoint, country: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Resolve a RoutePoint to lat/lng. If coordinates given, use them.
    If address given, geocode it (with fuzzy matching)."""
    if point.lat is not None and point.lng is not None:
        # Reverse geocode to get address for display
        address = None
        try:
            result = await geocode_photon(
                f"{point.lat}, {point.lng}", country, client
            )
            if result:
                address = result.get("display_name")
        except Exception:
            pass
        return {"lat": point.lat, "lng": point.lng, "address": address or f"{point.lat}, {point.lng}"}

    if point.address:
        # Use the same fuzzy validation logic
        req = AddressRequest(raw_address=point.address, country=country)
        match = await validate_address(req)
        if match.matched:
            return {"lat": match.lat, "lng": match.lng, "address": match.formatted_address}

    return None


async def fetch_osrm_route(coordinates: list[list[float]], client: httpx.AsyncClient) -> Optional[dict]:
    """Call OSRM for driving route between ordered coordinates.
    coordinates: [[lng, lat], [lng, lat], ...]
    """
    coord_str = ";".join(f"{c[0]},{c[1]}" for c in coordinates)
    url = f"{OSRM_URL}/route/v1/driving/{coord_str}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
        "annotations": "duration,distance",
    }

    resp = await client.get(url, params=params)
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    return data["routes"][0]


# ── Route Endpoint ─────────────────────────────────────────────────

@app.post("/route", response_model=RouteResponse)
async def calculate_route(req: RouteRequest):
    """Calculate driving route, distance, and time between points.

    Accepts addresses (with fuzzy matching) or lat/lng coordinates.
    Returns leg-by-leg distances/times and full route geometry for map rendering.

    Use cases:
    - Store/HQ to customer address
    - Driver's current location to next delivery
    - Multi-stop route planning
    """
    async with httpx.AsyncClient(timeout=15) as client:
        # Resolve all points to coordinates
        all_points = [req.origin] + req.waypoints + [req.destination]
        resolved = []

        for i, point in enumerate(all_points):
            result = await resolve_point(point, "us", client)
            if not result:
                label = "origin" if i == 0 else ("destination" if i == len(all_points) - 1 else f"waypoint {i}")
                addr = point.address or f"{point.lat}, {point.lng}"
                return RouteResponse(
                    success=False,
                    error=f"Could not resolve {label}: {addr}",
                )
            resolved.append(result)

        # Build coordinate list for OSRM
        coordinates = [[r["lng"], r["lat"]] for r in resolved]

        # Fetch route from OSRM
        route = await fetch_osrm_route(coordinates, client)
        if not route:
            return RouteResponse(
                success=False,
                error="OSRM routing failed — points may be unreachable by road",
            )

        # Build leg-by-leg breakdown
        route_legs = route.get("legs", [])
        legs = []
        for i, leg in enumerate(route_legs):
            legs.append(RouteLeg(
                from_address=resolved[i]["address"],
                from_lat=resolved[i]["lat"],
                from_lng=resolved[i]["lng"],
                to_address=resolved[i + 1]["address"],
                to_lat=resolved[i + 1]["lat"],
                to_lng=resolved[i + 1]["lng"],
                distance_miles=round(leg["distance"] / 1609.344, 1),
                distance_meters=round(leg["distance"], 0),
                duration_minutes=round(leg["duration"] / 60, 1),
                duration_seconds=round(leg["duration"], 0),
            ))

        total_dist = route["distance"]
        total_time = route["duration"]

        return RouteResponse(
            success=True,
            total_distance_miles=round(total_dist / 1609.344, 1),
            total_distance_meters=round(total_dist, 0),
            total_duration_minutes=round(total_time / 60, 1),
            total_duration_seconds=round(total_time, 0),
            legs=legs,
            geometry=route.get("geometry"),
        )


# ── Health Check ───────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
