#!/usr/bin/env bash
# Install a split APK bundle (XAPK) via ADB
# Usage: ./sideload-xapk.sh <path-to-xapk-file>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

XAPK_FILE="${1:?Usage: $0 <path-to-xapk-file>}"
WORK_DIR="/media/lin/disk2/brawlstar-agent/emulator/apks/extracted"

if [[ ! -f "$XAPK_FILE" ]]; then
    echo "ERROR: File not found: $XAPK_FILE"
    exit 1
fi

echo "=== Sideloading XAPK: $XAPK_FILE ==="

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "[1/4] Extracting XAPK (it's a zip)..."
unzip -o "$XAPK_FILE" -d "$WORK_DIR"

echo "[2/4] Finding APK files..."
APK_FILES=()
for f in "$WORK_DIR"/*.apk; do
    if [[ -f "$f" ]]; then
        APK_FILES+=("$f")
        echo "  Found: $(basename "$f") ($(du -h "$f" | cut -f1))"
    fi
done

if [[ ${#APK_FILES[@]} -eq 0 ]]; then
    echo "ERROR: No .apk files found in XAPK. Listing contents:"
    ls -la "$WORK_DIR"
    exit 1
fi

echo "[3/4] Verifying ADB connection..."
adb devices | grep -q "device$" || { echo "ERROR: No ADB device connected"; exit 1; }

echo "[4/4] Installing ${#APK_FILES[@]} APK(s) via adb install-multiple..."
adb install-multiple "${APK_FILES[@]}"

echo ""
echo "=== Installation complete ==="
echo "Verify with: adb shell pm list packages | grep brawl"
