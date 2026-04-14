#!/usr/bin/env bash
# Download multiple Brawl Stars gameplay clips from YouTube search queries.
# Targets: clean gameplay, esports spectator views, no-commentary recordings.
#
# Usage: ./download-batch.sh [max-per-query]
# Each query downloads up to max-per-query clips (default: 3)

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"
CLIPS_DIR="$PROJECT_ROOT/capture/clips"
VENV="$PROJECT_ROOT/.venv/bin"
MAX="${1:-3}"

ARCHIVE="$PROJECT_ROOT/capture/download_history.txt"

mkdir -p "$CLIPS_DIR"

# Curated search queries for clean gameplay footage
QUERIES=(
    "brawl stars gameplay no commentary 2026"
    "brawl stars ranked match no commentary"
    "brawl stars pro gameplay raw 2025"
    "brawl stars esports spectator view 2026"
    "brawl stars championship finals replay 2026"
    "brawl stars gem grab gameplay 2026"
    "brawl stars showdown gameplay no facecam"
    "brawl stars brawl ball pro gameplay"
)

echo "Downloading up to $MAX clips per query, ${#QUERIES[@]} queries total."
echo "Output: $CLIPS_DIR"
echo ""

for i in "${!QUERIES[@]}"; do
    QUERY="${QUERIES[$i]}"
    PREFIX="batch_$(printf '%02d' $i)"
    echo "=== [$((i+1))/${#QUERIES[@]}] Query: $QUERY ==="

    "$VENV/yt-dlp" \
        -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best" \
        --merge-output-format mp4 \
        -o "$CLIPS_DIR/${PREFIX}_%(title).50s.%(ext)s" \
        --no-playlist \
        --download-sections "*0:00-3:00" \
        --max-downloads "$MAX" \
        --download-archive "$ARCHIVE" \
        "ytsearch${MAX}:${QUERY}" \
        2>&1 || echo "  (some downloads may have failed, continuing...)"

    echo ""
done

echo "=== Done ==="
ls -lh "$CLIPS_DIR"/batch_*.mp4 2>/dev/null | wc -l
echo " clips downloaded to $CLIPS_DIR"
