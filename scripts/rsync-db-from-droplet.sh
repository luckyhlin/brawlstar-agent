#!/usr/bin/env bash
# Safely copy the droplet's live SQLite DB to local using sqlite3 ".backup".
#
# Why we can't just rsync the .db file directly: WAL-mode SQLite writes recent
# transactions to brawlstars.db-wal first and only periodically checkpoints
# them back into the main .db file. If rsync runs while the droplet's crawler
# is committing, the main .db file ends up referencing pages that exist only
# in the WAL — opening it standalone reports `database disk image is malformed`.
#
# `.backup` is the online-safe alternative: SQLite's page-locked snapshot API
# produces a single self-contained file even while writers are active. We don't
# need to stop the timers.
#
# Usage:
#   bash scripts/rsync-db-from-droplet.sh                       # default: brawl → data/brawlstars.db
#   bash scripts/rsync-db-from-droplet.sh brawl data/foo.db     # override host + path
#   bash scripts/rsync-db-from-droplet.sh --keep-remote         # don't delete remote snapshot

set -euo pipefail

REMOTE_HOST="${1:-brawl}"
LOCAL_PATH="${2:-data/brawlstars.db}"
KEEP_REMOTE=0
for arg in "$@"; do
    [ "$arg" = "--keep-remote" ] && KEEP_REMOTE=1
done

REMOTE_DB="\$HOME/brawlstar-agent/data/brawlstars.db"
SNAPSHOT="/tmp/brawl-snapshot-$(date +%Y%m%d-%H%M%S).db"

say() { printf "[rsync-db] %s\n" "$*"; }

# Take a hot backup on the droplet. `ssh host 'cmd'` runs in a non-interactive
# shell so the fish-auto-exec in ~/.bashrc cannot fire (it requires a TTY).
# `.backup` is online-safe — no need to pause timers.
say "Hot-backup on $REMOTE_HOST → $SNAPSHOT"
ssh "$REMOTE_HOST" "sqlite3 $REMOTE_DB '.backup $SNAPSHOT'"

REMOTE_SIZE=$(ssh "$REMOTE_HOST" "stat -c %s $SNAPSHOT")
say "  Snapshot size: $(numfmt --to=iec --suffix=B $REMOTE_SIZE)"

# Make sure the local directory exists
LOCAL_DIR=$(dirname "$LOCAL_PATH")
mkdir -p "$LOCAL_DIR"

# Pull. We use --inplace so a partial transfer can resume cleanly next time.
say "Pulling to $LOCAL_PATH..."
rsync -avz --progress --inplace "${REMOTE_HOST}:${SNAPSHOT}" "$LOCAL_PATH"

# Verify integrity
say "Verifying integrity..."
RESULT=$(sqlite3 "$LOCAL_PATH" "PRAGMA integrity_check;")
if [ "$RESULT" != "ok" ]; then
    say "FAIL: integrity check returned: $RESULT"
    exit 1
fi
say "  ✓ integrity_check: ok"

# Quick row counts
ROWS=$(sqlite3 "$LOCAL_PATH" "SELECT COUNT(*) FROM battles;")
say "  battles: $ROWS"

# Cleanup remote snapshot
if (( ! KEEP_REMOTE )); then
    say "Cleaning up remote snapshot..."
    ssh "$REMOTE_HOST" "rm -f $SNAPSHOT"
fi

say "Done."
