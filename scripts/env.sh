#!/usr/bin/env bash
# Source this file to set up the Android SDK environment:
#   source /media/lin/disk2/brawlstar-agent/scripts/env.sh

PROJECT_ROOT="/media/lin/disk2/brawlstar-agent"

export JAVA_HOME="$PROJECT_ROOT/emulator/jdk-17.0.18+8"
export ANDROID_HOME="$PROJECT_ROOT/emulator/android-sdk"
export ANDROID_AVD_HOME="$PROJECT_ROOT/emulator/avd"
export ANDROID_USER_HOME="$PROJECT_ROOT/emulator/android-user"
export ANDROID_SDK_ROOT="$ANDROID_HOME"

export PATH="$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"

echo "Android SDK environment loaded."
echo "  JAVA_HOME=$JAVA_HOME"
echo "  ANDROID_HOME=$ANDROID_HOME"
echo "  ANDROID_AVD_HOME=$ANDROID_AVD_HOME"
echo "  adb: $(which adb 2>/dev/null || echo 'not on PATH')"
echo "  emulator: $(which emulator 2>/dev/null || echo 'not on PATH')"
