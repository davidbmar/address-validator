#!/usr/bin/env bash
#
# Download and preprocess US road data for OSRM.
# Run this ONCE before docker-compose up.
#
# This takes 10-20 minutes and ~5GB disk for the full US extract.
# For faster setup, swap the URL for a single state (see comments below).
#
set -euo pipefail

DATA_DIR="./data/osrm"
mkdir -p "$DATA_DIR"

PBF_FILE="$DATA_DIR/us-latest.osm.pbf"

# ── Download US road data from Geofabrik ────────────────────────────
# Full US (~8GB download, ~5GB processed):
URL="https://download.geofabrik.de/north-america/us-latest.osm.pbf"

# For testing, use a single state instead (much faster):
# URL="https://download.geofabrik.de/north-america/us/texas-latest.osm.pbf"
# PBF_FILE="$DATA_DIR/texas-latest.osm.pbf"  # also update docker-compose command

if [ ! -f "$PBF_FILE" ]; then
    echo "Downloading US road data (~8GB)..."
    curl -L -o "$PBF_FILE" "$URL"
else
    echo "PBF file already exists, skipping download."
fi

# ── Preprocess for OSRM (MLD algorithm) ────────────────────────────
echo "Running osrm-extract (this takes a while)..."
docker run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-extract -p /opt/car.lua /data/us-latest.osm.pbf

echo "Running osrm-partition..."
docker run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-partition /data/us-latest.osrm

echo "Running osrm-customize..."
docker run --rm -v "$(pwd)/data/osrm:/data" osrm/osrm-backend:latest \
    osrm-customize /data/us-latest.osrm

echo ""
echo "Done! Now run: docker-compose up"
