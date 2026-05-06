#!/usr/bin/env bash
# Copy the droplet's SQLite DB to local. Two modes:
#
#   --backup-mode (default): online-safe via `sqlite3 .backup`. Works while
#                  writers are active. Needs ~DB-size free disk on droplet for
#                  the temporary snapshot file. Best when timers are running.
#
#   --direct (alternative): direct rsync of the live .db file, NO intermediate
#                  snapshot. Requires that all writers are stopped AND the WAL
#                  has been checkpointed (`PRAGMA wal_checkpoint(TRUNCATE)`)
#                  so the main file holds the full database. Use this when the
#                  droplet is disk-constrained (no room for a snapshot file).
#
# Why this matters: WAL-mode SQLite writes recent transactions to
# brawlstars.db-wal first and only periodically checkpoints them back into the
# main .db file. A naive rsync of just the main file while a writer is active
# can end up referencing pages that exist only in the WAL — opening it
# standalone reports `database disk image is malformed`.
#
# Usage:
#   bash scripts/rsync-db-from-droplet.sh                       # backup mode
#   bash scripts/rsync-db-from-droplet.sh --direct              # direct mode (writers must be off)
#   bash scripts/rsync-db-from-droplet.sh brawl data/foo.db     # override host + path
#   bash scripts/rsync-db-from-droplet.sh --keep-remote         # backup mode, don't delete remote snapshot

set -euo pipefail

MODE="backup"
KEEP_REMOTE=0
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --direct) MODE="direct" ;;
        --backup-mode) MODE="backup" ;;
        --keep-remote) KEEP_REMOTE=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

REMOTE_HOST="${POSITIONAL[0]:-brawl}"
LOCAL_PATH="${POSITIONAL[1]:-data/brawlstars.db}"
REMOTE_DB="\$HOME/brawlstar-agent/data/brawlstars.db"

say() { printf "[rsync-db] %s\n" "$*"; }

LOCAL_DIR=$(dirname "$LOCAL_PATH")
mkdir -p "$LOCAL_DIR"

if [ "$MODE" = "direct" ]; then
    # ----- direct mode -----
    say "DIRECT mode (no .backup): rsyncing live .db file"
    say "  ⚠ Assumes timers are stopped + WAL is checkpointed"
    say "  Verify on droplet first:"
    say "    sudo systemctl is-active brawl-collect.service brawl-collect-pinned.service brawl-analytics.service"
    say "    (all should be 'inactive' or 'failed', not 'active')"
    say "    sqlite3 \$HOME/brawlstar-agent/data/brawlstars.db 'PRAGMA wal_checkpoint(TRUNCATE);'"
    say
    say "Pulling $REMOTE_HOST:$REMOTE_DB → $LOCAL_PATH"
    rsync -avz --progress --inplace "${REMOTE_HOST}:${REMOTE_DB#\\}" "$LOCAL_PATH"
else
    # ----- backup mode (default) -----
    SNAPSHOT="/tmp/brawl-snapshot-$(date +%Y%m%d-%H%M%S).db"
    say "BACKUP mode: hot-snapshot on $REMOTE_HOST → $SNAPSHOT"
    say "  (online-safe; no need to pause timers — but uses ~DB-size scratch space)"
    ssh "$REMOTE_HOST" "sqlite3 $REMOTE_DB '.backup $SNAPSHOT'"

    REMOTE_SIZE=$(ssh "$REMOTE_HOST" "stat -c %s $SNAPSHOT")
    say "  Snapshot size: $(numfmt --to=iec --suffix=B $REMOTE_SIZE)"

    say "Pulling to $LOCAL_PATH..."
    rsync -avz --progress --inplace "${REMOTE_HOST}:${SNAPSHOT}" "$LOCAL_PATH"

    if (( ! KEEP_REMOTE )); then
        say "Cleaning up remote snapshot..."
        ssh "$REMOTE_HOST" "rm -f $SNAPSHOT"
    fi
fi

# Verify integrity in both modes
say "Verifying integrity..."
RESULT=$(sqlite3 "$LOCAL_PATH" "PRAGMA integrity_check;")
if [ "$RESULT" != "ok" ]; then
    say "FAIL: integrity check returned: $RESULT"
    exit 1
fi
say "  ✓ integrity_check: ok"

ROWS=$(sqlite3 "$LOCAL_PATH" "SELECT COUNT(*) FROM battles;")
say "  battles: $ROWS"
say "Done."
