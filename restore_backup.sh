#!/bin/bash
# Restores a single backup_qdrant.sh backup (Qdrant snapshot + Postgres
# dump + uploads/ originals) into an EXPLICIT target — every target flag is
# required, none defaults to this repo's own .env, because a restore
# silently applied to the wrong place overwrites live data with a past
# state. See verify_restore.sh for the automated drill that runs this
# against a disposable scratch environment.
#
# Usage:
#   ./restore_backup.sh \
#       --backup-dir backups/20260718_030000 \
#       --postgres-url postgresql://raguser:ragpass@localhost:15432/ragdb \
#       --qdrant-url http://localhost:16333 \
#       --uploads-dir /path/to/scratch/uploads \
#       --i-understand-this-overwrites-target
#
#   --backup-dir latest   resolves to the most recent backups/YYYYMMDD_HHMMSS/
#
# Refuses to run at all without --i-understand-this-overwrites-target (this
# DROPs and replaces whatever Postgres schema/Qdrant collection/uploads dir
# already exists at the target), and separately refuses to run against
# targets that match THIS repo's own .env (i.e. what looks like production)
# unless --force-prod is also given.
#
# Exit codes: 0 = restored and every count matches manifest.json;
# 1 = refused to start, or a restore step itself failed;
# 2 = every restore step completed, but at least one post-restore count
# doesn't match what manifest.json recorded at backup time.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# What "production" means for the guard below — same resolution
# backup_qdrant.sh itself uses.
PROD_POSTGRES_URL="${POSTGRES_URL:-postgresql://raguser:ragpass@localhost:5432/ragdb}"
PROD_QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
PROD_UPLOAD_DIR="$SCRIPT_DIR/uploads"

BACKUP_DIR=""
TARGET_POSTGRES_URL=""
TARGET_QDRANT_URL=""
TARGET_QDRANT_COLLECTION=""
TARGET_QDRANT_API_KEY="${QDRANT_API_KEY:-}"
TARGET_UPLOADS_DIR=""
FORCE_PROD=0
CONSENT=0

usage() {
    cat <<'EOF'
Usage: ./restore_backup.sh --backup-dir DIR --postgres-url URL --qdrant-url URL
           --uploads-dir DIR --i-understand-this-overwrites-target
           [--qdrant-collection NAME] [--qdrant-api-key KEY] [--force-prod]
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --backup-dir) BACKUP_DIR="$2"; shift 2 ;;
        --postgres-url) TARGET_POSTGRES_URL="$2"; shift 2 ;;
        --qdrant-url) TARGET_QDRANT_URL="$2"; shift 2 ;;
        --qdrant-collection) TARGET_QDRANT_COLLECTION="$2"; shift 2 ;;
        --qdrant-api-key) TARGET_QDRANT_API_KEY="$2"; shift 2 ;;
        --uploads-dir) TARGET_UPLOADS_DIR="$2"; shift 2 ;;
        --force-prod) FORCE_PROD=1; shift ;;
        --i-understand-this-overwrites-target) CONSENT=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [ "$BACKUP_DIR" = "latest" ]; then
    BACKUP_DIR=$(find backups -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' | sort | tail -1)
fi

[ -n "$BACKUP_DIR" ] || { echo "ERROR: --backup-dir is required" >&2; usage; exit 1; }
[ -n "$TARGET_POSTGRES_URL" ] || { echo "ERROR: --postgres-url is required (no default — deliberate)" >&2; exit 1; }
[ -n "$TARGET_QDRANT_URL" ] || { echo "ERROR: --qdrant-url is required" >&2; exit 1; }
[ -n "$TARGET_UPLOADS_DIR" ] || { echo "ERROR: --uploads-dir is required" >&2; exit 1; }
if [ "$CONSENT" != "1" ]; then
    echo "ERROR: refusing to run without --i-understand-this-overwrites-target" >&2
    echo "This DROPs and replaces whatever is currently at the target Postgres schema, Qdrant collection, and uploads dir." >&2
    exit 1
fi

[ -d "$BACKUP_DIR" ] || { echo "ERROR: backup dir not found: $BACKUP_DIR" >&2; exit 1; }
[ -f "$BACKUP_DIR/checksums.sha256" ] || { echo "ERROR: $BACKUP_DIR has no checksums.sha256 — refusing to trust an unmanifested backup" >&2; exit 1; }
[ -f "$BACKUP_DIR/manifest.json" ] || { echo "ERROR: $BACKUP_DIR has no manifest.json (backup predates this check, or is otherwise incomplete) — refusing" >&2; exit 1; }

# Resolve target uploads dir to an absolute path up front — both for the
# prod-guard comparison below and so every later step doesn't have to
# re-derive it relative to whatever CWD it happens to run from.
case "$TARGET_UPLOADS_DIR" in
    /*) TARGET_UPLOADS_ABS="$TARGET_UPLOADS_DIR" ;;
    *) TARGET_UPLOADS_ABS="$(pwd)/$TARGET_UPLOADS_DIR" ;;
esac

if [ -z "$TARGET_QDRANT_COLLECTION" ]; then
    TARGET_QDRANT_COLLECTION=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['qdrant']['collection'])")
fi

# ── Prod guard ────────────────────────────────────────────────────────────
# A restore target that happens to match this repo's own .env is almost
# certainly a mistake (a copy-pasted command, a missing flag override in a
# wrapper script) rather than an intentional "restore over production" —
# the one case where that IS intentional (real disaster recovery) is rare
# enough to afford one extra explicit flag.
LOOKS_LIKE_PROD=0
[ "$TARGET_POSTGRES_URL" = "$PROD_POSTGRES_URL" ] && LOOKS_LIKE_PROD=1
[ "$TARGET_QDRANT_URL" = "$PROD_QDRANT_URL" ] && LOOKS_LIKE_PROD=1
[ "$TARGET_UPLOADS_ABS" = "$PROD_UPLOAD_DIR" ] && LOOKS_LIKE_PROD=1

if [ "$LOOKS_LIKE_PROD" = "1" ] && [ "$FORCE_PROD" != "1" ]; then
    echo "ERROR: target matches this repo's own .env (production) endpoints:" >&2
    echo "  postgres-url: $TARGET_POSTGRES_URL" >&2
    echo "  qdrant-url:   $TARGET_QDRANT_URL" >&2
    echo "  uploads-dir:  $TARGET_UPLOADS_ABS" >&2
    echo "Refusing to overwrite what looks like production. Pass --force-prod if this is really a disaster-recovery restore." >&2
    exit 1
fi

echo "=== Restoring $BACKUP_DIR ==="
echo "  target postgres:  $TARGET_POSTGRES_URL"
echo "  target qdrant:    $TARGET_QDRANT_URL (collection: $TARGET_QDRANT_COLLECTION)"
echo "  target uploads:   $TARGET_UPLOADS_ABS"

# ── 1. Checksums — before anything destructive happens ──────────────────
echo "--- Verifying checksums ---"
(cd "$BACKUP_DIR" && sha256sum -c checksums.sha256)
echo "Checksums OK"

MANIFEST_DOCUMENTS=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['postgres']['documents'])")
MANIFEST_FILE_HASHES=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['postgres']['file_hashes'])")
MANIFEST_FOLDERS=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['postgres']['folders'])")
MANIFEST_DOCUMENT_PAGES=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['postgres']['document_pages'])")
MANIFEST_POINT_COUNT=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['qdrant']['point_count'])")
MANIFEST_UPLOAD_COUNT=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['uploads']['file_count'])")

# ── 2. PostgreSQL ─────────────────────────────────────────────────────────
echo "--- PostgreSQL: restoring into target ---"
psql "$TARGET_POSTGRES_URL" -v ON_ERROR_STOP=1 -q -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
gunzip -c "$BACKUP_DIR/postgres.sql.gz" | psql "$TARGET_POSTGRES_URL" -v ON_ERROR_STOP=1 -q
echo "PostgreSQL restored."

read -r RESTORED_DOCUMENTS RESTORED_FILE_HASHES RESTORED_FOLDERS RESTORED_DOCUMENT_PAGES < <(
    psql "$TARGET_POSTGRES_URL" -tAc \
        "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM file_hashes), (SELECT count(*) FROM folders), (SELECT count(*) FROM document_pages);" \
    | tr '|' ' '
)

# ── 3. Qdrant ─────────────────────────────────────────────────────────────
echo "--- Qdrant: restoring snapshot into target ---"
SNAPSHOT_FILE=$(python3 -c "import json; print(json.load(open('$BACKUP_DIR/manifest.json'))['qdrant']['snapshot_file'])")
SNAPSHOT_PATH="$BACKUP_DIR/$SNAPSHOT_FILE"
[ -f "$SNAPSHOT_PATH" ] || { echo "ERROR: snapshot file referenced by manifest.json not found: $SNAPSHOT_PATH" >&2; exit 1; }

TARGET_AUTH_HEADER=()
if [ -n "$TARGET_QDRANT_API_KEY" ]; then
    TARGET_AUTH_HEADER=(-H "api-key: ${TARGET_QDRANT_API_KEY}")
fi

# Drop any pre-existing collection at the target first so recovery never
# merges with leftovers from an earlier restore run at the same target —
# a 404 here is the expected, harmless first-run case.
curl -sS --retry 3 "${TARGET_AUTH_HEADER[@]}" -X DELETE \
    "${TARGET_QDRANT_URL}/collections/${TARGET_QDRANT_COLLECTION}" >/dev/null 2>&1 || true

curl -sS --fail-with-body --retry 3 "${TARGET_AUTH_HEADER[@]}" \
    -X POST "${TARGET_QDRANT_URL}/collections/${TARGET_QDRANT_COLLECTION}/snapshots/upload?wait=true&priority=snapshot" \
    -F "snapshot=@${SNAPSHOT_PATH}" >/dev/null
echo "Qdrant collection restored from snapshot."

RESTORED_POINT_COUNT=$(curl -sS --fail-with-body --retry 3 "${TARGET_AUTH_HEADER[@]}" \
    "${TARGET_QDRANT_URL}/collections/${TARGET_QDRANT_COLLECTION}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])")

# ── 4. uploads/ ───────────────────────────────────────────────────────────
echo "--- uploads/: restoring originals ---"
mkdir -p "$TARGET_UPLOADS_ABS"
find "$TARGET_UPLOADS_ABS" -mindepth 1 -delete
UPLOADS_TAR="$BACKUP_DIR/uploads.tar.gz"
if [ -f "$UPLOADS_TAR" ]; then
    # --strip-components=1 drops the archive's own top-level "uploads/" dir
    # name, so this works regardless of what the target dir is itself named.
    tar -xzf "$UPLOADS_TAR" --strip-components=1 -C "$TARGET_UPLOADS_ABS"
else
    echo "No uploads.tar.gz in this backup (uploads/ was empty at backup time) — target left empty."
fi
RESTORED_UPLOAD_COUNT=$(find "$TARGET_UPLOADS_ABS" -type f | wc -l)

# ── 5. Compare against manifest.json ─────────────────────────────────────
MISMATCH=0
check() {  # check <label> <expected> <actual>
    if [ "$2" != "$3" ]; then
        echo "MISMATCH: $1 expected=$2 actual=$3" >&2
        MISMATCH=1
    else
        echo "OK: $1 = $3"
    fi
}

echo "--- Comparing restored counts against manifest.json ---"
check "postgres.documents"      "$MANIFEST_DOCUMENTS"      "$RESTORED_DOCUMENTS"
check "postgres.file_hashes"    "$MANIFEST_FILE_HASHES"    "$RESTORED_FILE_HASHES"
check "postgres.folders"        "$MANIFEST_FOLDERS"        "$RESTORED_FOLDERS"
check "postgres.document_pages" "$MANIFEST_DOCUMENT_PAGES" "$RESTORED_DOCUMENT_PAGES"
check "qdrant.point_count"      "$MANIFEST_POINT_COUNT"    "$RESTORED_POINT_COUNT"
check "uploads.file_count"      "$MANIFEST_UPLOAD_COUNT"   "$RESTORED_UPLOAD_COUNT"

echo "=== Restore summary: $BACKUP_DIR -> target ==="
echo "  Postgres: documents=$RESTORED_DOCUMENTS file_hashes=$RESTORED_FILE_HASHES folders=$RESTORED_FOLDERS document_pages=$RESTORED_DOCUMENT_PAGES"
echo "  Qdrant:   collection=$TARGET_QDRANT_COLLECTION points=$RESTORED_POINT_COUNT"
echo "  Uploads:  files=$RESTORED_UPLOAD_COUNT"

if [ "$MISMATCH" = "1" ]; then
    echo "=== Restore completed WITH count mismatches — see above ===" >&2
    exit 2
fi
echo "=== Restore verified: every count matches manifest.json ==="
