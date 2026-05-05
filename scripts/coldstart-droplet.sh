#!/usr/bin/env bash
# Cold-start the droplet DB:
#   purge pre-CLEAN_CUTOFF battles → reset fetch state → kick off aggressive re-crawl.
#
# Two-phase invocation. Phase 1 takes a backup and prints the rsync command;
# Phase 2 does the destructive ops AFTER you've verified the backup on local.
#
# Usage (run as `lin` on the droplet):
#   bash scripts/coldstart-droplet.sh                      # PHASE 1: backup + rsync hint
#   bash scripts/coldstart-droplet.sh --really-purge       # PHASE 2: stop, purge, crawl
#
# Tunables (env vars):
#   COLDSTART_RPS=3                          API requests per second (default: 3)
#   COLDSTART_LIMIT=200000                   max players per crawl run (default: 200000)
#   COLDSTART_OLDER_THAN=24                  don't refetch a player within N hours
#                                            (default 24, so each player gets 1 fresh fetch)
#   COLDSTART_CUTOFF=2026-05-03T01:00:00Z    purge battles strictly before this time
#
# After the crawl finishes (you'll see the script return; check `pgrep collect-battles`):
#   sudo systemctl start brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer

set -euo pipefail

# -------- config --------
REPO="${REPO:-$HOME/brawlstar-agent}"
RPS="${COLDSTART_RPS:-3}"
LIMIT="${COLDSTART_LIMIT:-200000}"
OLDER_THAN="${COLDSTART_OLDER_THAN:-24}"
CUTOFF="${COLDSTART_CUTOFF:-2026-05-03T01:00:00Z}"

DB="$REPO/data/brawlstars.db"
BACKUP_DIR="$REPO/data/backups"
LOG_DIR="$REPO/logs"

DO_PURGE=0
for arg in "$@"; do
    case "$arg" in
        --really-purge) DO_PURGE=1 ;;
        --help|-h) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \{0,1\}//' | head -30; exit 0 ;;
    esac
done

# -------- helpers --------
say() { printf "[coldstart] %s\n" "$*"; }
fail() { say "ERROR: $*"; exit 1; }

# -------- pre-flight --------
[ -f "$DB" ] || fail "$DB not found. Are you on the droplet?"
mkdir -p "$BACKUP_DIR" "$LOG_DIR"

if (( ! DO_PURGE )); then
    # ===== PHASE 1: backup only =====
    TS=$(date +%Y%m%d-%H%M%S)
    BACKUP="$BACKUP_DIR/brawlstars-pre-coldstart-$TS.db"
    say "Phase 1/2: hot backup of $DB"
    sqlite3 "$DB" ".backup '$BACKUP'"
    sz=$(du -h "$BACKUP" | cut -f1)
    rows=$(sqlite3 "$BACKUP" "SELECT COUNT(*) FROM battles;")
    say "  ✓ backup OK: $BACKUP ($sz, $rows battles)"
    say
    say "NEXT — from your LOCAL laptop, in another terminal:"
    say "  rsync -avz --progress brawl:$BACKUP /media/lin/disk2/brawlstar-agent/data/backups/"
    say
    say "Verify the local copy has the same row count, THEN run on droplet:"
    say "  bash scripts/coldstart-droplet.sh --really-purge"
    exit 0
fi

# ===== PHASE 2: destructive ops + crawl =====

# Sanity: a recent backup must exist
LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/brawlstars-pre-coldstart-*.db 2>/dev/null | head -1 || true)
[ -n "$LATEST_BACKUP" ] || fail "no backup in $BACKUP_DIR. Run phase 1 first."
BACKUP_AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_BACKUP") ))
say "Latest backup: $LATEST_BACKUP (age: $((BACKUP_AGE/60)) min)"
if (( BACKUP_AGE > 7200 )); then
    say "WARN: backup older than 2 hours. Consider re-running phase 1 to refresh it."
    say "      Continuing anyway in 5s; Ctrl+C to abort..."
    sleep 5
fi

# Stop all timers (idempotent)
say "Stopping cron timers..."
sudo systemctl stop brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer 2>/dev/null || true

# Wait for any in-flight oneshot service to drain
say "Waiting for any in-flight services to finish..."
for svc in brawl-collect.service brawl-collect-pinned.service brawl-analytics.service; do
    while sudo systemctl is-active --quiet "$svc"; do
        say "  $svc still active; sleeping 30s"
        sleep 30
    done
    say "  $svc idle"
done

# Purge pre-cutoff battles + reset fetch state + VACUUM
say "Purging battles with battle_time_iso < '$CUTOFF'..."
sqlite3 "$DB" <<EOF
.headers on
SELECT 'before  battles=' || COUNT(*) FROM battles;
SELECT 'before  battle_players=' || COUNT(*) FROM battle_players;
SELECT 'before  players=' || COUNT(*) FROM players;

BEGIN IMMEDIATE;
DELETE FROM battle_players
 WHERE battle_id IN (SELECT battle_id FROM battles WHERE battle_time_iso < '$CUTOFF');
DELETE FROM battles
 WHERE battle_time_iso < '$CUTOFF';
UPDATE players SET last_battlelog_at = NULL;
COMMIT;

VACUUM;

SELECT 'after   battles=' || COUNT(*) FROM battles;
SELECT 'after   battle_players=' || COUNT(*) FROM battle_players;
SELECT 'after   players=' || COUNT(*) FROM players;
EOF
say "  ✓ purge + reset + VACUUM complete"

# Quick file-size readout (VACUUM should have shrunk it)
size_after=$(du -h "$DB" | cut -f1)
say "  DB size after VACUUM: $size_after"

# Launch aggressive crawl in background
TS=$(date +%Y%m%d-%H%M%S)
LOG="$LOG_DIR/coldstart-crawl-$TS.log"
say "Launching aggressive crawl: rps=$RPS  limit=$LIMIT  older-than=${OLDER_THAN}h"
say "  Log: $LOG"

nohup bash -c "
cd '$REPO' && \
PYTHONPATH=src \
BRAWL_API_KEY_VAR=BRAWL_STAR_API_DO \
UV_CACHE_DIR=\$HOME/.cache/uv \
\$HOME/.local/bin/uv run python scripts/collect-battles.py \
    --collect-only \
    --battlelog-limit '$LIMIT' \
    --older-than '$OLDER_THAN' \
    --rps '$RPS'
" > "$LOG" 2>&1 &
PID=$!
sleep 2

if ! kill -0 "$PID" 2>/dev/null; then
    say "ERROR: crawl process died immediately. Tail of log:"
    tail -40 "$LOG" || true
    exit 1
fi

# Save PID for later monitoring / kill
echo "$PID" > "$REPO/data/coldstart-crawl.pid"
say "  ✓ crawl launched: PID=$PID  (saved to data/coldstart-crawl.pid)"

cat <<MSG

────────────────────────────────────────────────────────────
COLD-START IN PROGRESS

Monitor:
  tail -f $LOG
  watch -n 60 'sqlite3 $DB "SELECT COUNT(*), MIN(battle_time_iso), MAX(battle_time_iso) FROM battles;"'

Stop early (graceful):
  kill \$(cat $REPO/data/coldstart-crawl.pid)

Hard kill if needed:
  pkill -f collect-battles.py

When the crawl finishes (or you stop it), RESTART the cron timers:
  sudo systemctl start brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer

If anything went wrong, restore from backup:
  cp $LATEST_BACKUP $DB
────────────────────────────────────────────────────────────
MSG
