#!/bin/bash
# Full, consistent backup: Qdrant snapshot + PostgreSQL dump + uploads/
# originals + a checksum manifest, built in a temp directory and only
# published under its final dated name once every step (and the checksum
# self-check) has actually succeeded — a script that died partway through
# used to still leave a same-named, half-written directory sitting in
# backups/ indistinguishable from a good one.
#
# Consistency across the three systems: an EXCLUSIVE flock on .backup.lock
# (same file lock.py's SHARED flock uses — see lock.py, api/main.py's
# require_not_backing_up(), and the maintenance scripts) is held from
# before the Qdrant snapshot starts until right after the local backup is
# published. A plain file-existence check (an earlier version of this
# script) is a gate, not mutual exclusion: it's checked once at the start
# of a request, so an upload already past that check keeps mutating Qdrant/
# Postgres/disk for the rest of its own duration, unseen by a backup that
# only looked for the file's presence. flock actually waits for every
# in-flight SHARED holder to finish before this EXCLUSIVE acquire succeeds,
# and blocks any new SHARED holder from starting once acquired — and, being
# a kernel-held lock tied to an open file descriptor rather than a flag
# file a trap has to remember to delete, it can never be left stuck forever
# by a `kill -9` or power loss either: the kernel drops it the instant this
# process's file descriptors close, for any reason.
#
# Использование: ./backup_qdrant.sh
# Cron (ежедневно в 3:00): 0 3 * * * cd /path/to/rag-knowledge-system && bash backup_qdrant.sh
# The lock wait (below) can still time out and exit non-zero if something
# legitimately holds the shared lock the whole window — point cron's
# MAILTO, or whatever wraps this invocation, at that exit code; a silently
# skipped backup is worse than a noisy alert about one.
#
# Optional off-host copy: set BACKUP_REMOTE_DEST (e.g. an rsync target
# like user@host:/path/ or a mounted second disk's path) in .env — a local
# disk failure, accidental `rm -rf`, or ransomware takes out backups/
# right along with the live data otherwise, since both sit on the same
# disk. The exclusive lock is released BEFORE this step (see below), so a
# slow or unreachable remote doesn't hold live uploads/deletes hostage for
# its duration.

set -Eeuo pipefail

# Resolved from this script's own location, not the caller's CWD — must
# land on the exact same absolute path lock.py computes from ITS own
# location (both files live at the repo root), or the API and this script
# would each be locking a different file without either of them noticing.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

# `source .env` in the crontab entry (as previously documented) only sets
# shell variables in cron's invoking shell — a plain `KEY=value` line with
# no `export` is never inherited by `bash backup_qdrant.sh`'s own process,
# so QDRANT_API_KEY (and everything else) would silently come through
# empty regardless of what the crontab does. Loading .env here directly,
# with `set -a` to auto-export every assignment, makes this script correct
# no matter how it's invoked.
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_API_KEY="${QDRANT_API_KEY:-}"
COLLECTION="${QDRANT_COLLECTION:-knowledge_base}"
POSTGRES_URL="${POSTGRES_URL:-postgresql://raguser:ragpass@localhost:5432/ragdb}"
UPLOAD_DIR="uploads"
BACKUP_ROOT="backups"
KEEP_DAYS=7
LOCK_FILE=".backup.lock"
LOCK_WAIT_SECONDS="${BACKUP_LOCK_WAIT_SECONDS:-600}"

# ── Exclusive lock: fd 200 stays open only for the span that actually
# needs exclusivity (snapshot -> dump -> tar -> checksum -> publish);
# released explicitly right after, well before the optional off-host copy.
# Bounded wait, not -n/non-blocking: a single long-running upload (a big
# OCR job, say) starting right as cron fires would otherwise make this
# give up and skip the whole day's backup outright instead of just
# starting a few seconds late once that one operation finishes. A cron
# monitor should still alert on a non-zero exit — this can still time out
# and fail if something holds the shared lock for longer than
# LOCK_WAIT_SECONDS (default 10 minutes; override via BACKUP_LOCK_WAIT_SECONDS).
exec 200>"$LOCK_FILE"
if ! flock -w "$LOCK_WAIT_SECONDS" -x 200; then
    echo "ERROR: could not acquire the exclusive lock on $LOCK_FILE within ${LOCK_WAIT_SECONDS}s — a mutation held it too long, or another backup is already running." >&2
    exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
TMP_DEST="$BACKUP_ROOT/.tmp_${TS}"
FINAL_DEST="$BACKUP_ROOT/$TS"
rm -rf "$TMP_DEST"   # leftover staging dir from a prior run that crashed before ever reaching the trap below
mkdir -p "$TMP_DEST"

AUTH_HEADER=()
if [ -n "$QDRANT_API_KEY" ]; then
    AUTH_HEADER=(-H "api-key: ${QDRANT_API_KEY}")
fi

# SNAPSHOT_NAME is set once creation succeeds (below); the trap only tries
# to delete it server-side if it's non-empty. If curl's own 3 retries all
# fail to download it (network blip, Qdrant restart mid-copy, whatever),
# `set -e` kills the script before the script's own explicit DELETE step
# ever runs — without this, that snapshot sits in Qdrant's storage volume
# forever, since local retention (below) only ever looks at backups/, not
# at what's still inside Qdrant itself. Cleared back to "" right after the
# explicit DELETE succeeds, so the trap doesn't try to delete it twice on
# the normal path (harmless either way — deleting an already-gone snapshot
# just 404s — but there's no reason to make that the expected case).
SNAPSHOT_NAME=""
_cleanup() {
    rm -rf "$TMP_DEST"
    if [ -n "$SNAPSHOT_NAME" ]; then
        echo "Cleaning up orphaned server-side snapshot: $SNAPSHOT_NAME" >&2
        curl -sS --retry 3 "${AUTH_HEADER[@]}" -X DELETE \
            "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}" >/dev/null 2>&1 || true
    fi
}
trap _cleanup EXIT

echo "=== [$TS] Backing up (staging in $TMP_DEST) ==="

# ── 1. Qdrant snapshot ───────────────────────────────────────────────────
echo "--- Qdrant: creating snapshot for '$COLLECTION' ---"
RESPONSE=$(curl -sS --fail-with-body --retry 3 -X POST "${AUTH_HEADER[@]}" \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots")
SNAPSHOT_NAME=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['name'])" 2>/dev/null)

if [ -z "$SNAPSHOT_NAME" ]; then
    echo "ERROR: Failed to create Qdrant snapshot. Response: $RESPONSE" >&2
    exit 1
fi

echo "Downloading snapshot: $SNAPSHOT_NAME"
curl -sS --fail-with-body --retry 3 "${AUTH_HEADER[@]}" -o "$TMP_DEST/qdrant_${SNAPSHOT_NAME}" \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}"
echo "Saved: qdrant_${SNAPSHOT_NAME} ($(du -sh "$TMP_DEST/qdrant_${SNAPSHOT_NAME}" | cut -f1))"

# Every cron run otherwise leaves its server-side snapshot behind inside
# Qdrant's own storage volume forever — local retention (below) only ever
# looks at backups/, never at what's still sitting inside Qdrant itself.
echo "Deleting server-side snapshot (already downloaded locally)..."
curl -sS --fail-with-body --retry 3 "${AUTH_HEADER[@]}" -X DELETE \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots/${SNAPSHOT_NAME}" >/dev/null
echo "Server-side snapshot deleted."
SNAPSHOT_NAME=""  # done — the EXIT trap's cleanup is now a no-op for this run

# ── 2. PostgreSQL dump ───────────────────────────────────────────────────
# pipefail (set above) is what makes this actually safe — without it, a
# failing pg_dump piped into gzip still exits 0 (gzip's own status),
# `set -e` never fires, and the script would checksum + publish a gzip of
# a truncated/empty dump as if it were a real backup.
echo "--- PostgreSQL: dumping ---"
pg_dump "$POSTGRES_URL" | gzip > "$TMP_DEST/postgres.sql.gz"
echo "Saved: postgres.sql.gz ($(du -sh "$TMP_DEST/postgres.sql.gz" | cut -f1))"

# ── 3. Original uploaded files ───────────────────────────────────────────
echo "--- uploads/: archiving originals ---"
if [ -d "$UPLOAD_DIR" ] && [ "$(ls -A "$UPLOAD_DIR" 2>/dev/null)" ]; then
    tar -czf "$TMP_DEST/uploads.tar.gz" -C "$(dirname "$UPLOAD_DIR")" "$(basename "$UPLOAD_DIR")"
    echo "Saved: uploads.tar.gz ($(du -sh "$TMP_DEST/uploads.tar.gz" | cut -f1))"
else
    echo "uploads/ is empty or missing — skipping."
fi

# ── 4. Checksums ─────────────────────────────────────────────────────────
echo "--- Computing + verifying checksums ---"
(cd "$TMP_DEST" && sha256sum -- * > checksums.sha256)
(cd "$TMP_DEST" && sha256sum -c checksums.sha256 >/dev/null)
echo "Checksums OK"

# ── 5. Publish: rename only now that everything above succeeded ────────
rm -rf "$FINAL_DEST"
mv "$TMP_DEST" "$FINAL_DEST"
echo "Published: $FINAL_DEST"

# ── 6. Release the exclusive lock NOW — the consistent, atomic local
# snapshot already exists on disk; nothing past this point needs Qdrant/
# Postgres/uploads/ to stay frozen, so there's no reason to keep blocking
# live mutations for however long an off-host copy or retention sweep
# takes. ─────────────────────────────────────────────────────────────────
flock -u 200
exec 200>&-
echo "Exclusive lock released — uploads/deletes can proceed again."

# ── 7. Optional off-host copy ────────────────────────────────────────────
if [ -n "${BACKUP_REMOTE_DEST:-}" ]; then
    echo "--- Copying to off-host destination: $BACKUP_REMOTE_DEST ---"
    rsync -a "$FINAL_DEST" "$BACKUP_REMOTE_DEST"
    echo "Copied."
else
    echo "BACKUP_REMOTE_DEST not set — backup stays local only (same disk as the data it protects)."
fi

# ── 8. Retention ─────────────────────────────────────────────────────────
find "$BACKUP_ROOT" -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' -mtime +$KEEP_DAYS -exec rm -rf {} \;
echo "Cleaned up local backup runs older than $KEEP_DAYS days."
echo "=== Done: $FINAL_DEST ==="
