#!/usr/bin/env bash
# Download Brawl Stars gameplay from YouTube
# Usage: ./download-gameplay.sh <youtube-url> [output-name]
# Example: ./download-gameplay.sh "https://www.youtube.com/watch?v=XXXXX" ranked_match_01

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"
CLIPS_DIR="$PROJECT_ROOT/capture/clips"
VENV="$PROJECT_ROOT/.venv/bin"

URL="${1:?Usage: $0 <youtube-url> [output-name]}"
NAME="${2:-$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$CLIPS_DIR"

echo "Downloading: $URL"
echo "Output name: $NAME"

ARCHIVE="$PROJECT_ROOT/capture/download_history.txt"

"$VENV/yt-dlp" \
    -f "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best" \
    --merge-output-format mp4 \
    -o "$CLIPS_DIR/${NAME}.%(ext)s" \
    --no-playlist \
    --download-archive "$ARCHIVE" \
    "$URL"

echo ""
echo "Downloaded to: $CLIPS_DIR/${NAME}.mp4"
ls -lh "$CLIPS_DIR/${NAME}".*
