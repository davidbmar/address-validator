# Address Validator & Route Calculator API

A lightweight FastAPI service that validates messy or voice-transcribed addresses and calculates driving routes, distances, and travel times using OpenStreetMap data. No API keys required.

## The Problem

Voice-to-text often garbles addresses:

| Voice heard | Actual address |
|-------------|---------------|
| "2711 brian hall drive austin tx" | 2711 Bryonhall Drive, Austin, TX 78745 |

Standard geocoders fail on these because they do exact matching. This service generates phonetic variants and tries them until it finds a real address. Once validated, you can calculate drive times and distances to any address — from a store, HQ, or a driver's current location.

## How It Works

### Address Validation
1. **Parse** the raw address into parts (number, street, type, city, state)
2. **Try exact match** against Photon (Komoot) geocoder
3. If no match, **generate ~40 phonetic variants** — join/split words, swap vowels (i↔y, a↔o, etc.)
4. **Fire variants concurrently** at Photon (no rate limit)
5. **Return** the first hit with structured address data + coordinates

### Route Calculation
1. **Resolve** origin and destination — accepts addresses (with fuzzy matching) or lat/lng coordinates
2. **Route** via OSRM (Open Source Routing Machine) — follows real roads, respects one-way streets, turn restrictions
3. **Return** leg-by-leg distance/time breakdown + full route geometry for map rendering

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Server runs on `http://localhost:8100`.

## API

### `POST /validate-address`

Validate and resolve a raw address string.

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

| Field | Description |
|-------|-------------|
| `matched` | Whether a valid address was found |
| `confidence` | `"exact"`, `"high"`, `"medium"`, or `"low"` |
| `matched_query` | The variant that matched (shows what transformation worked) |
| `formatted_address` | Full standardized address string |
| `lat` / `lng` | Coordinates |
| `variants_tried` | Number of geocoding attempts made |

---

### `POST /route`

Calculate driving distance and time between two or more points. Accepts addresses (with the same fuzzy matching as `/validate-address`) or raw lat/lng coordinates. Mix and match freely.

#### Example 1: Store to customer address

```bash
curl -X POST http://localhost:8100/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": { "address": "1234 Commerce Dr, Austin, TX" },
    "destination": { "address": "2711 bryonhall drive austin tx" }
  }'
```

#### Example 2: Driver's GPS location to next delivery

```bash
curl -X POST http://localhost:8100/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": { "lat": 30.2672, "lng": -97.7431 },
    "destination": { "address": "2711 bryonhall drive austin tx" }
  }'
```

#### Example 3: Multi-stop route (HQ → job site A → job site B → return)

```bash
curl -X POST http://localhost:8100/route \
  -H "Content-Type: application/json" \
  -d '{
    "origin": { "address": "1234 Commerce Dr, Austin, TX" },
    "waypoints": [
      { "address": "2711 bryonhall drive austin tx" },
      { "address": "500 E Riverside Dr, Austin, TX" }
    ],
    "destination": { "address": "1234 Commerce Dr, Austin, TX" }
  }'
```

#### Response

```json
{
  "success": true,
  "total_distance_miles": 12.3,
  "total_distance_meters": 19795.0,
  "total_duration_minutes": 18.5,
  "total_duration_seconds": 1110.0,
  "legs": [
    {
      "from_address": "1234, Commerce Drive, Austin, Texas, 78745, United States",
      "from_lat": 30.2345,
      "from_lng": -97.7654,
      "to_address": "2711, Bryonhall Drive, Austin, Texas, 78745, United States",
      "to_lat": 30.2054,
      "to_lng": -97.8178,
      "distance_miles": 12.3,
      "distance_meters": 19795.0,
      "duration_minutes": 18.5,
      "duration_seconds": 1110.0
    }
  ],
  "geometry": {
    "type": "LineString",
    "coordinates": [[...]]
  }
}
```

| Field | Description |
|-------|-------------|
| `total_distance_miles` | Total driving distance across all legs |
| `total_duration_minutes` | Total estimated drive time (based on speed limits, not live traffic) |
| `legs` | Per-leg breakdown with addresses, coordinates, distance, and time |
| `geometry` | GeoJSON LineString of the full route — render on a map with MapLibre/Leaflet/Mapbox |

#### Input options for origin/destination/waypoints

| Format | Example | Use case |
|--------|---------|----------|
| Address string | `{"address": "123 Main St, Austin TX"}` | Customer addresses, store locations |
| Lat/lng coordinates | `{"lat": 30.267, "lng": -97.743}` | Driver GPS position, known coordinates |
| Messy/voice address | `{"address": "brian hall drive austin"}` | Voice-transcribed input (auto-corrected) |

---

### `GET /health`

Returns `{"status": "ok"}`.

## Typical Workflow

For a plumbing/service dispatch AI:

```
1. Customer calls, gives address via phone
   → Voice-to-text: "2711 brian hall drive austin texas"

2. AI calls POST /validate-address
   → Gets back: "2711 Bryonhall Drive, Austin, TX 78745" ✓

3. AI calls POST /route with HQ as origin
   → Gets back: 12.3 miles, 18 minutes

4. Or, AI calls POST /route with driver's current GPS
   → Gets back: 4.1 miles, 8 minutes (driver is already nearby)

5. AI confirms with customer: "We can have someone there in about 20 minutes"
```

## Dependencies

- **FastAPI** + **Uvicorn** — HTTP server
- **httpx** — async HTTP client for geocoder/router calls
- **jellyfish** — phonetic algorithms (Jaro-Winkler similarity for variant scoring)

## External Services (free, no API keys)

| Service | Used for | Rate limit |
|---------|----------|------------|
| [Photon](https://photon.komoot.io/) (Komoot) | Primary geocoder, fuzzy matching | None |
| [Nominatim](https://nominatim.openstreetmap.org/) (OpenStreetMap) | Fallback geocoder | 1 req/sec |
| [OSRM](https://router.project-osrm.org/) (Project OSRM) | Driving routes, distances, times | Public demo server |

## Limitations

- **US addresses only** (configurable via `country` parameter)
- **No live traffic** — drive times based on speed limits and road type, not current congestion. For traffic-aware routing, swap in Google Directions API or Mapbox
- **OSRM demo server** — for production use, consider [self-hosting OSRM](https://github.com/Project-OSRM/osrm-backend) or using a paid routing provider
- **OpenStreetMap coverage** — depends on community mapping completeness
- **Very creative misspellings** (3+ simultaneous errors) may not resolve
