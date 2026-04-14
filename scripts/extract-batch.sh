#!/usr/bin/env bash
# Extract frames for every downloaded clip that does not already have frames.
#
# Usage:
#   ./scripts/extract-batch.sh [fps]
#
# Example:
#   ./scripts/extract-batch.sh 2

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"
CLIPS_DIR="$PROJECT_ROOT/capture/clips"
FRAMES_DIR="$PROJECT_ROOT/capture/frames"
FPS="${1:-2}"

mkdir -p "$FRAMES_DIR"

TOTAL=0
EXTRACTED=0
SKIPPED=0

for video in "$CLIPS_DIR"/*.mp4; do
    if [[ ! -e "$video" ]]; then
        echo "No clips found in $CLIPS_DIR"
        exit 0
    fi

    TOTAL=$((TOTAL + 1))
    basename="$(basename "${video%.*}")"
    outdir="$FRAMES_DIR/$basename"

    if [[ -d "$outdir" ]] && find "$outdir" -maxdepth 1 -type f -name '*.jpg' | grep -q . 2>/dev/null; then
        echo "Skipping existing frames: $basename"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    bash "$PROJECT_ROOT/scripts/extract-frames.sh" "$video" "$FPS" "$outdir"
    EXTRACTED=$((EXTRACTED + 1))
done

echo ""
echo "Batch extraction complete."
echo "Total clips: $TOTAL"
echo "Extracted: $EXTRACTED"
echo "Skipped: $SKIPPED"
