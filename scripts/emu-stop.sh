#!/usr/bin/env bash
# Stop all running emulator instances

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "Stopping emulator..."
adb emu kill 2>/dev/null || echo "No emulator running or adb not connected."
echo "Done."
