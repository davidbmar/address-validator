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

async def geocode_photon(query: str, country: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Call Photon geocoder (Komoot) — no rate limit, fast.
    Returns result in Nominatim-compatible format."""
    params = {"q": query, "limit": "1"}
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

        feat = features[0]
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords or len(coords) < 2:
            return None

        # Filter to correct country
        if country and props.get("countrycode", "").lower() != country.lower():
            return None

        # Convert to Nominatim-like format
        return {
            "lat": str(coords[1]),
            "lon": str(coords[0]),
            "display_name": ", ".join(filter(None, [
                props.get("housenumber", ""),
                props.get("street", ""),
                props.get("city", "") or props.get("town", "") or props.get("village", ""),
                props.get("state", ""),
                props.get("postcode", ""),
                props.get("country", ""),
            ])),
            "address": {
                "house_number": props.get("housenumber"),
                "road": props.get("street"),
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

@app.post("/validate-address", response_model=AddressMatch)
async def validate_address(req: AddressRequest):
    raw = req.raw_address.strip()

    # Step 1: Try the exact input with both geocoders
    async with httpx.AsyncClient(timeout=10) as client:
        result = await geocode_photon(raw, req.country, client)
        if result:
            return build_response(raw, raw, result, "exact", 1)

        result = await geocode_nominatim(raw, req.country, client)
        if result:
            return build_response(raw, raw, result, "exact", 2)

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
            tasks = [geocode_photon(q, req.country, client) for q in batch]
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                tried += 1
                if result:
                    idx = batch_start + i
                    confidence = "high" if idx < 10 else "medium"
                    return build_response(raw, batch[i], result, confidence, tried)

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
