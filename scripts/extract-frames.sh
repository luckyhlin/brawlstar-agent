#!/usr/bin/env bash
# Extract frames from a video clip
# Usage: ./extract-frames.sh <video-file> [fps] [output-dir]
# Example: ./extract-frames.sh capture/clips/match01.mp4 2
#   → extracts 2 frames per second into capture/frames/match01/

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"

VIDEO="${1:?Usage: $0 <video-file> [fps] [output-dir]}"
FPS="${2:-2}"
BASENAME="$(basename "${VIDEO%.*}")"
OUTDIR="${3:-$PROJECT_ROOT/capture/frames/$BASENAME}"

mkdir -p "$OUTDIR"

echo "Extracting frames from: $VIDEO"
echo "FPS: $FPS"
echo "Output: $OUTDIR"

ffmpeg -i "$VIDEO" \
    -vf "fps=$FPS" \
    -q:v 2 \
    "$OUTDIR/frame_%06d.jpg" \
    -hide_banner -loglevel warning

COUNT=$(ls "$OUTDIR"/*.jpg 2>/dev/null | wc -l)
echo "Extracted $COUNT frames to $OUTDIR"
