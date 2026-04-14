#!/usr/bin/env bash
# Start the Brawl Stars research emulator
# Usage: ./emu-start.sh [--headless]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

AVD_NAME="brawlstars-research"
EXTRA_ARGS=""

if [[ "$1" == "--headless" ]]; then
    EXTRA_ARGS="-no-window"
    echo "Starting emulator in headless mode..."
else
    echo "Starting emulator with display..."
fi

emulator -avd "$AVD_NAME" \
    -gpu host \
    -memory 4096 \
    -partition-size 16384 \
    -no-snapshot-load \
    $EXTRA_ARGS \
    "$@" &

EMU_PID=$!
echo "Emulator PID: $EMU_PID"
echo "Waiting for device to boot..."

adb wait-for-device
adb shell 'while [[ -z $(getprop sys.boot_completed) ]]; do sleep 1; done'

echo "Emulator booted successfully."
