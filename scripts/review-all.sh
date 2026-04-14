#!/usr/bin/env bash
# Open review.html files one at a time, wait for you to save, then auto-move the result.
#
# Workflow per clip:
#   1. Opens review.html in browser
#   2. You review/correct labels, click Save (downloads review_manifest.json)
#   3. Press Enter in this terminal when done
#   4. Script moves ~/Downloads/review_manifest.json back to the clip folder
#   5. Opens the next clip
#
# Usage: bash scripts/review-all.sh
#        bash scripts/review-all.sh --skip-done   # skip clips already labeled

set -euo pipefail

FRAMES_ROOT="/media/lin/disk2/brawlstar-agent/capture/frames"
DOWNLOADS="$HOME/Downloads"
SKIP_DONE=false

[[ "${1:-}" == "--skip-done" ]] && SKIP_DONE=true

# Build array properly handling spaces in paths
DIRS=()
while IFS= read -r -d '' htmlfile; do
    DIRS+=("$(dirname "$htmlfile")")
done < <(find "$FRAMES_ROOT" -name "review.html" -print0 | sort -z)

TOTAL=${#DIRS[@]}
echo "Found $TOTAL clips with review.html"
echo ""

for i in "${!DIRS[@]}"; do
    DIR="${DIRS[$i]}"
    CLIP_NAME="$(basename "$DIR")"
    MANIFEST="$DIR/review_manifest.json"

    if $SKIP_DONE && [ -f "$MANIFEST" ]; then
        UNKNOWNS=$(grep -c '"unknown"' "$MANIFEST" 2>/dev/null || echo 0)
        if [ "$UNKNOWNS" -eq 0 ]; then
            echo "[$((i+1))/$TOTAL] SKIP (already labeled): $CLIP_NAME"
            continue
        fi
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$((i+1))/$TOTAL] $CLIP_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    HTML_PATH="$DIR/review.html"
    echo "  Opening: $HTML_PATH"
    xdg-open "$HTML_PATH" 2>/dev/null &

    echo ""
    echo "  Review in browser, click Save when done."
    echo "  Then press ENTER here to move the file and continue."
    echo "  Type 's' + ENTER to skip this clip."
    echo "  Type 'q' + ENTER to quit."
    echo ""
    read -r RESPONSE

    case "$RESPONSE" in
        q|Q) echo "Quitting."; exit 0 ;;
        s|S) echo "Skipped."; continue ;;
    esac

    # Move the downloaded manifest back
    DL_FILE="$DOWNLOADS/review_manifest.json"
    if [ -f "$DL_FILE" ]; then
        mv "$DL_FILE" "$MANIFEST"
        echo "  ✓ Saved → $MANIFEST"

        GAMEPLAY=$(grep -c '"gameplay"' "$MANIFEST" 2>/dev/null || echo 0)
        TOTAL_F=$(grep -c '"frame_' "$MANIFEST" 2>/dev/null || echo 0)
        echo "  Gameplay: $GAMEPLAY/$TOTAL_F frames"
    else
        echo "  No download found at $DL_FILE — manifest not updated."
        echo "  (Check your browser downloads folder)"
    fi
    echo ""
done

echo ""
echo "=== All clips reviewed ==="
