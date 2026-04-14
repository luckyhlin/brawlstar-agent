#!/usr/bin/env bash
# Extract a time segment from a video, then optionally extract frames
# Usage: ./extract-segment.sh <video> <start> <duration> [output-name]
# Example: ./extract-segment.sh capture/clips/match.mp4 01:30 00:30 teamfight_01
#   → cuts 30 seconds starting at 1:30 into capture/clips/teamfight_01.mp4

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"

VIDEO="${1:?Usage: $0 <video> <start> <duration> [output-name]}"
START="${2:?Provide start time (e.g., 01:30)}"
DURATION="${3:?Provide duration (e.g., 00:30)}"
NAME="${4:-segment_$(date +%s)}"

OUTFILE="$PROJECT_ROOT/capture/clips/${NAME}.mp4"

echo "Cutting segment: start=$START duration=$DURATION"
ffmpeg -ss "$START" -i "$VIDEO" -t "$DURATION" -c copy "$OUTFILE" \
    -hide_banner -loglevel warning

echo "Saved to: $OUTFILE"
ls -lh "$OUTFILE"
