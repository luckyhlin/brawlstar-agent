#!/usr/bin/env bash
# Push the local SQLite DB to the droplet. Companion to rsync-db-from-droplet.sh.
#
# Designed for a low-RAM, disk-constrained droplet on a flaky network path:
#
# - --inplace via --append-verify: the droplet only needs ~DB-size free disk
#   (no room for a separate temp file) AND we can resume a partial upload from
#   wherever the last attempt died. --append-verify MD5-checks the existing
#   prefix on both ends before appending, so a corrupt prefix gets caught
#   rather than silently extended.
# - --bwlimit: caps the transfer rate. Sustained large-file uploads at line
#   rate get killed by NAT/ISP/DO ingress shaping mid-stream, not by SSH idle
#   timeouts. A bandwidth cap of 5-10 MB/s reliably keeps the connection alive
#   through to completion.
# - SSH keepalives: belt-and-suspenders for the idle-disconnect case.
# - Retry loop: each retry resumes via --append-verify, so even on a flaky
#   network the file ultimately arrives. Attempt count + cumulative time
#   logged so you can see whether the network is working with you or against.
#
# Caller MUST stop the droplet's timers before running; they hold writers on
# the DB and the script will refuse to overwrite a live database. Restart
# timers manually after the script verifies the new DB.
#
# Usage:
#   bash scripts/rsync-db-to-droplet.sh                      # defaults: 10 MB/s, host=brawl, src=data/brawlstars.db
#   bash scripts/rsync-db-to-droplet.sh --bwlimit 5          # gentler 5 MB/s cap
#   bash scripts/rsync-db-to-droplet.sh --bwlimit 0          # unlimited (ill-advised on flaky paths)
#   bash scripts/rsync-db-to-droplet.sh --no-verify          # skip post-transfer integrity_check
#   bash scripts/rsync-db-to-droplet.sh --host other         # different SSH host alias
#   bash scripts/rsync-db-to-droplet.sh --src data/foo.db    # different local path

set -euo pipefail

REMOTE_HOST="brawl"
LOCAL_PATH="data/brawlstars.db"
# Relative path: resolved against the SSH user's $HOME by both the remote
# shell (`ssh "$host" "stat $REMOTE_PATH ..."`) and the remote rsync
# (`rsync ... "$host:$REMOTE_PATH"`).
#
# Do NOT use a literal "$HOME" / '$HOME' here. rsync >= 3.2.4 enables
# --secluded-args by default, which sends path arguments straight to the
# remote rsync WITHOUT shell expansion, so "$HOME" stays literal and the
# path becomes "/home/<user>/$HOME/brawlstar-agent/...".
REMOTE_PATH='brawlstar-agent/data/brawlstars.db'
BWLIMIT="10M"
DO_VERIFY=1
RETRY_BACKOFF=15
MAX_ATTEMPTS=20

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        REMOTE_HOST="$2"; shift 2 ;;
        --src)         LOCAL_PATH="$2"; shift 2 ;;
        --remote-path) REMOTE_PATH="$2"; shift 2 ;;
        --bwlimit)     BWLIMIT="${2}M"; [[ "$2" == "0" ]] && BWLIMIT=""; shift 2 ;;
        --no-verify)   DO_VERIFY=0; shift ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; /^set -/d'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

say() { printf "[rsync-to-droplet] %s\n" "$*"; }

# ----- Pre-flight -----
[[ -f "$LOCAL_PATH" ]] || { say "FAIL: local DB not found at $LOCAL_PATH"; exit 1; }
LOCAL_SIZE=$(stat -c %s "$LOCAL_PATH")
say "Local source: $LOCAL_PATH ($(numfmt --to=iec --suffix=B "$LOCAL_SIZE"))"

say "Probing droplet ($REMOTE_HOST)..."
ssh "$REMOTE_HOST" "test -d \$(dirname $REMOTE_PATH) && df -B1 \$HOME | tail -1" \
    > /tmp/droplet-df.txt 2>&1 || {
    say "FAIL: cannot reach $REMOTE_HOST or remote dir missing"
    cat /tmp/droplet-df.txt; exit 1
}
REMOTE_FREE=$(awk '{print $4}' /tmp/droplet-df.txt)
REMOTE_FREE_HR=$(numfmt --to=iec --suffix=B "$REMOTE_FREE")
say "Droplet free space: $REMOTE_FREE_HR"

# Existing partial / current DB on droplet?
REMOTE_EXISTING=$(ssh "$REMOTE_HOST" "stat -c %s $REMOTE_PATH 2>/dev/null || echo 0")
if (( REMOTE_EXISTING > 0 )); then
    REMOTE_EXISTING_HR=$(numfmt --to=iec --suffix=B "$REMOTE_EXISTING")
    if (( REMOTE_EXISTING == LOCAL_SIZE )); then
        say "  Remote already same size ($REMOTE_EXISTING_HR); rsync will quick-verify and exit"
    elif (( REMOTE_EXISTING < LOCAL_SIZE )); then
        say "  Remote is partial: $REMOTE_EXISTING_HR / $(numfmt --to=iec --suffix=B "$LOCAL_SIZE") — will resume via --append-verify"
    else
        say "FAIL: remote file ($REMOTE_EXISTING_HR) is LARGER than local ($(numfmt --to=iec --suffix=B "$LOCAL_SIZE")). Refusing to truncate."
        say "  To inspect, ssh $REMOTE_HOST and run:"
        say "    ls -lh $REMOTE_PATH"
        exit 1
    fi
fi

# Refuse if WAL/SHM exist remotely → writers are or were active without checkpoint
WAL_EXISTS=$(ssh "$REMOTE_HOST" "ls $REMOTE_PATH-wal $REMOTE_PATH-shm 2>/dev/null | wc -l")
if (( WAL_EXISTS > 0 )); then
    say "FAIL: $REMOTE_HOST has $REMOTE_PATH-wal / -shm files."
    say "  Stop timers + rm those files, then re-run this script."
    say "  On the droplet (ssh $REMOTE_HOST), paste:"
    say "    sudo systemctl stop brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer"
    say "    rm -f ${REMOTE_PATH}-wal ${REMOTE_PATH}-shm"
    exit 1
fi

# Make sure droplet has enough space if remote is empty (peak = LOCAL_SIZE - REMOTE_EXISTING)
NEEDED=$(( LOCAL_SIZE - REMOTE_EXISTING ))
if (( REMOTE_FREE < NEEDED + 100*1024*1024 )); then  # 100MB headroom
    say "FAIL: droplet has $REMOTE_FREE_HR free but transfer needs ~$(numfmt --to=iec --suffix=B "$NEEDED") + headroom"
    exit 1
fi

# ----- Transfer with retry-on-failure -----
RSYNC_OPTS=(-avh --append-verify --progress --inplace)
[[ -n "$BWLIMIT" ]] && RSYNC_OPTS+=(--bwlimit="$BWLIMIT")
RSYNC_OPTS+=(-e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes")

START=$(date +%s)
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    say "Attempt $attempt/$MAX_ATTEMPTS — rsync ${BWLIMIT:+(bwlimit=$BWLIMIT)} ..."
    if rsync "${RSYNC_OPTS[@]}" "$LOCAL_PATH" "$REMOTE_HOST:$REMOTE_PATH"; then
        ELAPSED=$(( $(date +%s) - START ))
        say "  ✓ Transfer complete in ${ELAPSED}s after $attempt attempt(s)"
        break
    fi
    if (( attempt == MAX_ATTEMPTS )); then
        say "FAIL: rsync failed $MAX_ATTEMPTS times; giving up"
        exit 1
    fi
    say "  rsync exited non-zero; retrying in ${RETRY_BACKOFF}s..."
    sleep "$RETRY_BACKOFF"
done

# ----- Verify -----
if (( DO_VERIFY )); then
    say "Verifying integrity on droplet..."
    REMOTE_INTEGRITY=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'PRAGMA integrity_check;'")
    if [[ "$REMOTE_INTEGRITY" != "ok" ]]; then
        say "FAIL: remote integrity check returned: $REMOTE_INTEGRITY"
        exit 1
    fi
    say "  ✓ integrity_check: ok"

    REMOTE_BATTLES=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'SELECT COUNT(*) FROM battles;'")
    REMOTE_PLAYERS=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'SELECT COUNT(*) FROM battle_players;'")
    REMOTE_RANGE=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH \"SELECT MIN(battle_time_iso) || ' .. ' || MAX(battle_time_iso) FROM battles;\"")
    say "  battles=$REMOTE_BATTLES  battle_players=$REMOTE_PLAYERS"
    say "  time range: $REMOTE_RANGE"
fi

say "Done."
say "Reminder: timers are still stopped on the droplet."
say "  To restart, ssh $REMOTE_HOST and paste:"
say "    sudo systemctl start brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer"
