# Address Validator API

A lightweight FastAPI service that validates messy or voice-transcribed addresses against real geocoding data. Handles common speech-to-text errors like word boundary mistakes and vowel confusion.

## The Problem

Voice-to-text often garbles addresses:

| Voice heard | Actual address |
|-------------|---------------|
| "2711 brian hall drive austin tx" | 2711 Bryonhall Drive, Austin, TX 78745 |

Standard geocoders fail on these because they do exact matching. This service generates phonetic variants and tries them until it finds a real address.

## How It Works

1. **Parse** the raw address into parts (number, street, type, city, state)
2. **Try exact match** against Photon (Komoot) geocoder
3. If no match, **generate ~40 phonetic variants** — join/split words, swap vowels (i↔y, a↔o, etc.)
4. **Fire variants concurrently** at Photon (no rate limit)
5. **Return** the first hit with structured address data + coordinates

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Server runs on `http://localhost:8100`.

## API

### `POST /validate-address`

```bash
curl -X POST http://localhost:8100/validate-address \
  -H "Content-Type: application/json" \
  -d '{"raw_address": "2711 brian hall drive austin tx"}'
```

Response:

```json
{
  "matched": true,
  "confidence": "high",
  "input_address": "2711 brian hall drive austin tx",
  "matched_query": "2711 brianhall drive austin tx",
  "formatted_address": "2711, Bryonhall Drive, Austin, Texas, 78745, United States",
  "house_number": "2711",
  "street": "Bryonhall Drive",
  "city": "Austin",
  "state": "Texas",
  "postcode": "78745",
  "country": "United States",
  "lat": 30.2054615,
  "lng": -97.8178795,
  "variants_tried": 3
}
```

### Fields

| Field | Description |
|-------|-------------|
| `matched` | Whether a valid address was found |
| `confidence` | `"exact"`, `"high"`, `"medium"`, or `"low"` |
| `matched_query` | The variant that matched (shows what transformation worked) |
| `formatted_address` | Full standardized address string |
| `lat` / `lng` | Coordinates |
| `variants_tried` | Number of geocoding attempts made |

### `GET /health`

Returns `{"status": "ok"}`.

## Dependencies

- **FastAPI** + **Uvicorn** — HTTP server
- **httpx** — async HTTP client for geocoder calls
- **jellyfish** — phonetic algorithms (Jaro-Winkler similarity for variant scoring)

## External Services (free, no API keys)

- **[Photon](https://photon.komoot.io/)** (Komoot) — primary geocoder, no rate limit, fuzzy matching
- **[Nominatim](https://nominatim.openstreetmap.org/)** (OpenStreetMap) — fallback geocoder, 1 req/sec rate limit

## Limitations

- US addresses only (configurable via `country` parameter)
- No live traffic data — coordinates only
- Depends on OpenStreetMap data completeness
- Very creative misspellings (3+ errors) may not resolve
