#!/usr/bin/env bash
# Prepare headless review assets for every extracted clip.
#
# Usage:
#   ./scripts/prepare-review-batch.sh [sample]

set -euo pipefail

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"
FRAMES_DIR="$PROJECT_ROOT/capture/frames"
SAMPLE="${1:-20}"

TOTAL=0

for dir in "$FRAMES_DIR"/*; do
    if [[ ! -d "$dir" ]]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))
    PYTHONPATH="$PROJECT_ROOT/src" "$PROJECT_ROOT/.venv/bin/python" \
        "$PROJECT_ROOT/scripts/prepare-review.py" \
        "$dir" \
        --sample "$SAMPLE"
done

echo ""
echo "Prepared review assets for $TOTAL frame directories."
