#!/usr/bin/env bash
#
# Download and preprocess US road data for OSRM.
# Run this ONCE before docker-compose up.
#
# This takes 10-20 minutes and ~5GB disk for the full US extract.
# For faster setup, swap the URL for a single state (see comments below).
#
set -euo pipefail

# Use podman if docker isn't available (e.g. macOS with Podman Desktop)
if command -v docker &>/dev/null; then
    CONTAINER_CMD="docker"
elif command -v podman &>/dev/null; then
    CONTAINER_CMD="podman"
else
    echo "Error: neither docker nor podman found in PATH" >&2
    exit 1
fi

DATA_DIR="./data/osrm"
mkdir -p "$DATA_DIR"

REGION="${1:-austin}"  # pass region name as argument, defaults to austin
PBF_FILE="$DATA_DIR/${REGION}.osm.pbf"

# ── Download road data ──────────────────────────────────────────────
# City-level extracts from BBBike (~50-100MB, fast to process):
#   ./setup-osrm.sh austin       (default)
#   ./setup-osrm.sh SanAntonio
#   ./setup-osrm.sh Houston
#   See full list: https://download.bbbike.org/osm/bbbike/
#
# State-level extracts from Geofabrik (~500MB+):
#   ./setup-osrm.sh texas
#
# Full US from Geofabrik (~8GB):
#   ./setup-osrm.sh us

case "$REGION" in
    us)
        URL="https://download.geofabrik.de/north-america/us-latest.osm.pbf"
        PBF_FILE="$DATA_DIR/us-latest.osm.pbf"
        ;;
    texas|california|florida|new-york|arizona|colorado|georgia|illinois|ohio|pennsylvania|virginia|washington)
        URL="https://download.geofabrik.de/north-america/us/${REGION}-latest.osm.pbf"
        PBF_FILE="$DATA_DIR/${REGION}-latest.osm.pbf"
        ;;
    *)
        # Assume BBBike city extract (capitalized name)
        CITY="$(echo "${REGION}" | awk '{print toupper(substr($0,1,1)) substr($0,2)}')"
        URL="https://download.bbbike.org/osm/bbbike/${CITY}/${CITY}.osm.pbf"
        PBF_FILE="$DATA_DIR/${REGION}.osm.pbf"
        ;;
esac

PBF_NAME="$(basename "$PBF_FILE")"
OSRM_NAME="${PBF_NAME%.osm.pbf}.osrm"

if [ ! -f "$PBF_FILE" ]; then
    echo "Downloading ${REGION} road data..."
    curl -L -o "$PBF_FILE" "$URL"
else
    echo "PBF file already exists, skipping download."
fi

# ── Preprocess for OSRM (MLD algorithm) ────────────────────────────
echo "Running osrm-extract (this takes a while)..."
$CONTAINER_CMD run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-extract -p /opt/car.lua /data/${PBF_NAME}

echo "Running osrm-partition..."
$CONTAINER_CMD run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-partition /data/${OSRM_NAME}

echo "Running osrm-customize..."
$CONTAINER_CMD run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-customize /data/${OSRM_NAME}

echo ""
echo "Done! Now run: docker-compose up"
echo "Note: update docker-compose.yml osrm command to use /data/${OSRM_NAME}"
