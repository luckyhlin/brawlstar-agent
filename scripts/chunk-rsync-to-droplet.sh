#!/usr/bin/env bash
# Robust DB push for unstable SSH paths.
#
# Splits the local DB into N-byte chunks, transfers each chunk in its own
# short SSH session (~50 s at 10 MB/s for 500 MB chunks), verifies via
# SHA-256, then appends-and-deletes on the droplet. Survives an SSH session
# that dies every minute because each chunk-attempt is ~50 s of work and
# retries are independent — losing one chunk costs at most one chunk's time,
# not the entire transfer.
#
# Trade-offs vs scripts/rsync-db-to-droplet.sh:
# - More overhead: each chunk = one full SSH connection + stat + append.
# - By default STARTS FRESH (existing remote partial discarded). Pass
#   --continue-remote to instead truncate the remote partial to a chunk
#   boundary, SHA-verify the prefix matches local, and skip already-transferred
#   chunks. Useful after a previous --append-verify rsync left a partial on
#   the droplet that you don't want to re-transfer.
# - Returns: 100% reliable progression even on a path that drops sustained
#   transfers in <60 s. The chunk-and-cat strategy is byte-exact.
#
# Disk requirements:
# - Local: ~CHUNK_SIZE temp space per chunk (in /tmp by default; override via
#   --chunk-dir-local). For default 500M chunks, peak local usage is ~CHUNK_SIZE.
#   Note: chunks are split + sha256'd up-front, so peak is actually ~DB_SIZE.
# - Droplet: peak = DB_SIZE (target) + CHUNK_SIZE (one chunk in flight). For a
#   15 GB DB + 500 MB chunk, peak is ~15.5 GB. Caller is responsible for
#   ensuring this fits.
#
# Usage:
#   bash scripts/chunk-rsync-to-droplet.sh                       # defaults (fresh transfer)
#   bash scripts/chunk-rsync-to-droplet.sh --chunk-size 250M     # smaller chunks (~25s each at 10MB/s)
#   bash scripts/chunk-rsync-to-droplet.sh --bwlimit 5           # gentler 5 MB/s
#   bash scripts/chunk-rsync-to-droplet.sh --continue-remote     # resume from existing partial on droplet
#   bash scripts/chunk-rsync-to-droplet.sh --keep-chunks         # keep local chunk dir for debugging
#   bash scripts/chunk-rsync-to-droplet.sh --resume              # skip local split if chunk dir exists

set -euo pipefail

REMOTE_HOST="brawl"
LOCAL_PATH="data/brawlstars.db"
REMOTE_PATH_REL="brawlstar-agent/data/brawlstars.db"  # relative to remote $HOME
CHUNK_SIZE=500M
BWLIMIT=10M
CHUNK_DIR_LOCAL="/media/lin/disk2/brawlstar-agent/tmp_db_chunks"
CHUNK_DIR_REMOTE_REL="db_chunks"  # relative to remote $HOME
PER_CHUNK_RETRIES=20
PER_CHUNK_BACKOFF=10
DO_VERIFY=1
KEEP_CHUNKS=0
RESUME_SPLIT=0
CONTINUE_REMOTE=0

REMOTE_PATH_OVERRIDE=""
CHUNK_DIR_REMOTE_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)             REMOTE_HOST="$2"; shift 2 ;;
        --src)              LOCAL_PATH="$2"; shift 2 ;;
        --remote-path)      REMOTE_PATH_OVERRIDE="$2"; shift 2 ;;
        --chunk-size)       CHUNK_SIZE="$2"; shift 2 ;;
        --bwlimit)          BWLIMIT="${2}M"; [[ "$2" == "0" ]] && BWLIMIT=""; shift 2 ;;
        --chunk-dir-local)  CHUNK_DIR_LOCAL="$2"; shift 2 ;;
        --chunk-dir-remote) CHUNK_DIR_REMOTE_OVERRIDE="$2"; shift 2 ;;
        --no-verify)        DO_VERIFY=0; shift ;;
        --keep-chunks)      KEEP_CHUNKS=1; shift ;;
        --resume)           RESUME_SPLIT=1; shift ;;
        --continue-remote)  CONTINUE_REMOTE=1; shift ;;
        --retries)          PER_CHUNK_RETRIES="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -/p' "$0" | sed 's/^# \{0,1\}//; /^set -/d'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

say() { printf "[chunk-rsync] %s\n" "$*"; }

# Retry helper for transient SSH issues at connection setup time
# (kex resets, MaxStartups races, etc). Args: <description> -- <command...>
ssh_retry() {
    local desc="$1"; shift
    [[ "$1" == "--" ]] && shift
    local out
    for attempt in 1 2 3 4 5 6 7 8; do
        if out=$("$@" 2>&1); then
            printf '%s' "$out"
            return 0
        fi
        if [[ "$attempt" -lt 8 ]]; then
            say "  $desc: attempt $attempt failed (${out##*$'\n'} ...) — retrying in 10s"
            sleep 10
        fi
    done
    say "FAIL: $desc — 8 attempts exhausted"
    printf '%s\n' "$out" >&2
    return 1
}

# ----- Pre-flight -----
[[ -f "$LOCAL_PATH" ]] || { say "FAIL: $LOCAL_PATH missing"; exit 1; }
LOCAL_SIZE=$(stat -c %s "$LOCAL_PATH")
say "Local source: $LOCAL_PATH ($(numfmt --to=iec --suffix=B "$LOCAL_SIZE"))"

# Resolve absolute remote paths once (rsync doesn't shell-expand $HOME or ~ reliably).
REMOTE_HOME=$(ssh_retry "probe remote \$HOME" -- ssh -o ConnectTimeout=10 "$REMOTE_HOST" 'printf %s "$HOME"') \
    || exit 1
[[ -n "$REMOTE_HOME" ]] || { say "FAIL: empty REMOTE_HOME from $REMOTE_HOST"; exit 1; }
if [[ -n "$REMOTE_PATH_OVERRIDE" ]]; then
    REMOTE_PATH="$REMOTE_PATH_OVERRIDE"
else
    REMOTE_PATH="$REMOTE_HOME/$REMOTE_PATH_REL"
fi
if [[ -n "$CHUNK_DIR_REMOTE_OVERRIDE" ]]; then
    CHUNK_DIR_REMOTE="$CHUNK_DIR_REMOTE_OVERRIDE"
else
    CHUNK_DIR_REMOTE="$REMOTE_HOME/$CHUNK_DIR_REMOTE_REL"
fi
say "Remote: $REMOTE_HOST  target=$REMOTE_PATH  staging=$CHUNK_DIR_REMOTE"

# Refuse if WAL/SHM exist remotely → writers are or were active without checkpoint
WAL_EXISTS=$(ssh "$REMOTE_HOST" "ls $REMOTE_PATH-wal $REMOTE_PATH-shm 2>/dev/null | wc -l")
if (( WAL_EXISTS > 0 )); then
    say "FAIL: $REMOTE_HOST has $REMOTE_PATH-wal / -shm files. Stop timers + rm them first."
    exit 1
fi

# ----- Stage 1: Split locally -----
if (( RESUME_SPLIT )) && [[ -d "$CHUNK_DIR_LOCAL" ]] && ls "$CHUNK_DIR_LOCAL"/c_* >/dev/null 2>&1; then
    say "Resume: reusing chunks in $CHUNK_DIR_LOCAL (use without --resume to re-split)"
else
    say "Splitting $LOCAL_PATH into ${CHUNK_SIZE} chunks at $CHUNK_DIR_LOCAL ..."
    rm -rf "$CHUNK_DIR_LOCAL"
    mkdir -p "$CHUNK_DIR_LOCAL"
    split -b "$CHUNK_SIZE" -d --suffix-length=4 "$LOCAL_PATH" "$CHUNK_DIR_LOCAL/c_"
    (cd "$CHUNK_DIR_LOCAL" && sha256sum c_* > chunks.sha256)
fi

CHUNKS=( "$CHUNK_DIR_LOCAL"/c_* )
N_CHUNKS=${#CHUNKS[@]}
# awk's default print uses %g for large integers (>2^31 or so) → scientific notation
# breaks downstream arithmetic. Use printf "%d" to force integer output.
TOTAL_CHUNK_BYTES=$(stat -c %s "$CHUNK_DIR_LOCAL"/c_* | awk '{s+=$1} END{printf "%d\n", s}')
say "Created $N_CHUNKS chunks totaling $(numfmt --to=iec --suffix=B "$TOTAL_CHUNK_BYTES")"
if (( TOTAL_CHUNK_BYTES != LOCAL_SIZE )); then
    say "FAIL: chunk total ($TOTAL_CHUNK_BYTES) != source size ($LOCAL_SIZE). Aborting."
    exit 1
fi

# ----- Stage 2: Set up destination on droplet -----
SKIP_CHUNKS=0
if (( CONTINUE_REMOTE )); then
    REMOTE_SIZE=$(ssh "$REMOTE_HOST" "stat -c %s $REMOTE_PATH 2>/dev/null || echo 0")
    if (( REMOTE_SIZE == 0 )); then
        say "--continue-remote: remote DB missing/empty; doing fresh transfer"
    elif (( REMOTE_SIZE >= LOCAL_SIZE )); then
        say "--continue-remote: remote ($(numfmt --to=iec --suffix=B "$REMOTE_SIZE")) >= local ($(numfmt --to=iec --suffix=B "$LOCAL_SIZE"))"
        say "  Refusing to truncate. Inspect: ssh $REMOTE_HOST 'ls -lh $REMOTE_PATH'"
        exit 1
    else
        # Compute chunk-boundary alignment. CHUNK_SIZE is e.g. "250M" or "500M".
        CHUNK_SIZE_BYTES=$(numfmt --from=iec "$CHUNK_SIZE")
        ALIGN=$(( REMOTE_SIZE / CHUNK_SIZE_BYTES * CHUNK_SIZE_BYTES ))
        SKIP_CHUNKS=$(( ALIGN / CHUNK_SIZE_BYTES ))

        say "--continue-remote: remote partial = $(numfmt --to=iec --suffix=B "$REMOTE_SIZE")"
        say "  Aligning to chunk boundary: $(numfmt --to=iec --suffix=B "$ALIGN") = $SKIP_CHUNKS × $CHUNK_SIZE chunks"
        say "  (Discarding $(numfmt --to=iec --suffix=B "$(( REMOTE_SIZE - ALIGN ))") of unaligned tail)"

        if (( REMOTE_SIZE > ALIGN )); then
            ssh "$REMOTE_HOST" "truncate -s $ALIGN $REMOTE_PATH"
        fi

        if (( ALIGN > 0 )); then
            say "  Verifying SHA-256 of truncated prefix on droplet (expect ~$(( ALIGN / (200*1024*1024) ))s remote read)..."
            REMOTE_PREFIX_HASH=$(ssh "$REMOTE_HOST" "sha256sum $REMOTE_PATH | awk '{print \$1}'")
            LOCAL_PREFIX_HASH=$(head -c "$ALIGN" "$LOCAL_PATH" | sha256sum | awk '{print $1}')
            if [[ "$REMOTE_PREFIX_HASH" != "$LOCAL_PREFIX_HASH" ]]; then
                say "FAIL: remote prefix doesn't match local — partial is corrupt or stale."
                say "  Local  ($ALIGN bytes): $LOCAL_PREFIX_HASH"
                say "  Remote ($ALIGN bytes): $REMOTE_PREFIX_HASH"
                say "  Re-run without --continue-remote to start fresh."
                exit 1
            fi
            say "  ✓ remote prefix matches local; resuming from chunk c_$(printf '%04d' $SKIP_CHUNKS)"
        fi
    fi
fi

if (( ! CONTINUE_REMOTE )) || (( SKIP_CHUNKS == 0 && REMOTE_SIZE == 0 )); then
    say "Resetting droplet destination (fresh transfer)"
    ssh "$REMOTE_HOST" "
        rm -f $REMOTE_PATH $REMOTE_PATH-wal $REMOTE_PATH-shm
        mkdir -p $CHUNK_DIR_REMOTE
        rm -f $CHUNK_DIR_REMOTE/c_*
        : > $REMOTE_PATH
    "
else
    say "Preserving existing remote DB; cleaning chunk staging dir only"
    ssh "$REMOTE_HOST" "mkdir -p $CHUNK_DIR_REMOTE; rm -f $CHUNK_DIR_REMOTE/c_*"
fi

# Send the manifest so we can re-verify if needed
rsync -a "$CHUNK_DIR_LOCAL/chunks.sha256" "$REMOTE_HOST:$CHUNK_DIR_REMOTE/chunks.sha256"

# ----- Stage 3: For each chunk: rsync → sha verify → cat to target → rm -----
START=$(date +%s)
TOTAL_RETRIES=0
chunk_index=0
for chunk_path in "${CHUNKS[@]}"; do
    chunk_index=$((chunk_index + 1))
    chunk_name=$(basename "$chunk_path")
    chunk_size=$(stat -c %s "$chunk_path")
    expected_hash=$(awk -v n="$chunk_name" '$2==n {print $1}' "$CHUNK_DIR_LOCAL/chunks.sha256")
    progress_pct=$(( chunk_index * 100 / N_CHUNKS ))

    # Skip chunks already covered by the (verified) remote prefix
    if (( chunk_index <= SKIP_CHUNKS )); then
        say "[$progress_pct%] $chunk_name SKIPPED (already on droplet via --continue-remote)"
        continue
    fi

    say "[$progress_pct%] $chunk_name ($(numfmt --to=iec --suffix=B "$chunk_size"))"

    # Per-chunk retry loop
    for attempt in $(seq 1 "$PER_CHUNK_RETRIES"); do
        if rsync -a --bwlimit="$BWLIMIT" \
            -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10 -o TCPKeepAlive=yes' \
            "$chunk_path" "$REMOTE_HOST:$CHUNK_DIR_REMOTE/$chunk_name" 2>/dev/null; then
            break
        fi
        if (( attempt == PER_CHUNK_RETRIES )); then
            say "FAIL: chunk $chunk_name after $PER_CHUNK_RETRIES retries"
            exit 1
        fi
        TOTAL_RETRIES=$((TOTAL_RETRIES + 1))
        sleep "$PER_CHUNK_BACKOFF"
    done

    # Verify hash on droplet
    actual_hash=$(ssh "$REMOTE_HOST" "sha256sum $CHUNK_DIR_REMOTE/$chunk_name | awk '{print \$1}'")
    if [[ "$actual_hash" != "$expected_hash" ]]; then
        say "FAIL: chunk $chunk_name SHA mismatch on droplet"
        say "  expected: $expected_hash"
        say "  actual:   $actual_hash"
        exit 1
    fi

    # Append to target + delete chunk
    ssh "$REMOTE_HOST" "cat $CHUNK_DIR_REMOTE/$chunk_name >> $REMOTE_PATH && rm $CHUNK_DIR_REMOTE/$chunk_name"
done
ELAPSED=$(( $(date +%s) - START ))
say "All $N_CHUNKS chunks transferred in ${ELAPSED}s ($TOTAL_RETRIES retries total)"

# ----- Stage 4: Verify destination -----
REMOTE_SIZE=$(ssh "$REMOTE_HOST" "stat -c %s $REMOTE_PATH")
if (( REMOTE_SIZE != LOCAL_SIZE )); then
    say "FAIL: remote size $REMOTE_SIZE != local size $LOCAL_SIZE"
    exit 1
fi
say "Remote size: $(numfmt --to=iec --suffix=B "$REMOTE_SIZE") (matches local)"

if (( DO_VERIFY )); then
    say "Verifying remote SHA-256 (full file)..."
    LOCAL_FULL_HASH=$(sha256sum "$LOCAL_PATH" | awk '{print $1}')
    REMOTE_FULL_HASH=$(ssh "$REMOTE_HOST" "sha256sum $REMOTE_PATH | awk '{print \$1}'")
    if [[ "$LOCAL_FULL_HASH" != "$REMOTE_FULL_HASH" ]]; then
        say "FAIL: full-file SHA mismatch"
        say "  local:  $LOCAL_FULL_HASH"
        say "  remote: $REMOTE_FULL_HASH"
        exit 1
    fi
    say "  ✓ SHA-256 matches"

    say "Running PRAGMA integrity_check on droplet..."
    REMOTE_INTEGRITY=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'PRAGMA integrity_check;'")
    [[ "$REMOTE_INTEGRITY" != "ok" ]] && { say "FAIL: $REMOTE_INTEGRITY"; exit 1; }
    say "  ✓ integrity_check: ok"

    REMOTE_BATTLES=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'SELECT COUNT(*) FROM battles;'")
    REMOTE_PLAYERS=$(ssh "$REMOTE_HOST" "sqlite3 $REMOTE_PATH 'SELECT COUNT(*) FROM battle_players;'")
    say "  battles=$REMOTE_BATTLES  battle_players=$REMOTE_PLAYERS"
fi

# ----- Stage 5: Cleanup -----
ssh "$REMOTE_HOST" "rmdir $CHUNK_DIR_REMOTE 2>/dev/null || rm -rf $CHUNK_DIR_REMOTE"
if (( ! KEEP_CHUNKS )); then
    rm -rf "$CHUNK_DIR_LOCAL"
fi
say "Done. Reminder: timers are still stopped on the droplet."
say "  ssh $REMOTE_HOST 'sudo systemctl start brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer'"
