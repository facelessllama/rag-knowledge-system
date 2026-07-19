#!/bin/bash
# Restore drill: spins up an ISOLATED, disposable Postgres + Qdrant
# (docker/docker-compose.restore-test.yml — different container names,
# ports, and volumes than the real stack, so nothing here can ever touch
# production even if this script has a bug), restores the most recent
# backup into them via restore_backup.sh, runs semantic verification
# (scripts/verify_restore_semantics.py) against the restored data compared
# to manifest.json (never against live production, which has moved on
# since the backup was taken), tears the scratch environment down, and
# records the outcome as a durable artifact — not just pass/fail, but
# enough to debug or audit a run without re-running it:
#   backups/.last_restore_check.{json,log} — always the MOST RECENT run
#     (overwritten every time) — fast to check, but not evidence on its
#     own: a lost/corrupted backup host takes this with it.
#   backups/restore_checks/<TS>.{json,log} — one dated, retained copy per
#     run (kept RESTORE_CHECK_KEEP_DAYS days, default 30 — longer than
#     backups/ itself, since these are tiny compared to a full backup and
#     the drill HISTORY matters more than any single backup's retention
#     window). Set RESTORE_CHECK_REMOTE_DEST (rsync target) to also ship
#     each dated copy off-host — the same failure mode that destroys your
#     backups can otherwise destroy the record that restores were ever
#     verified.
# The JSON captures: timestamp, backup_dir, result, git commit (+ dirty
# flag) of the scripts that ran, restore/verify durations, exit codes, and
# the full semantic-check detail. The paired .log is the whole run's
# combined stdout/stderr, secrets redacted (POSTGRES_PASSWORD,
# QDRANT_API_KEY, and any embedded URL credentials).
#
# Usage:
#   ./verify_restore.sh                          # most recent backup
#   ./verify_restore.sh --backup-dir backups/20260718_030000
#
# Run this manually and confirm it passes before wiring it into cron.
# Recommended schedule (see README.md): backup daily, this drill weekly,
# check_restore_freshness.sh daily, a manual full disaster-recovery
# walkthrough quarterly. Note: running this manually proves the mechanism
# works NOW — it does not by itself satisfy "restore is regularly
# verified" until it's actually on a schedule, with an owner and a
# wired-up alert channel.
#
# Optional: set RESTORE_CHECK_ALERT_CMD to the path of a single executable
# (a script — NOT a shell command line) to run when the drill fails.
# Invoked directly as `"$RESTORE_CHECK_ALERT_CMD" drill_failed`, never via
# eval or sh -c, with the run's JSON status piped to its stdin — nothing
# derived from logs/environment ever gets re-interpreted by a shell. If
# you need several steps (curl + jq, say), put them inside that script;
# don't build a shell one-liner in the env var.

set -Eeuo pipefail
umask 077   # restores the same sensitive documents/dump backup_qdrant.sh protects
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

COMPOSE_FILE="docker/docker-compose.restore-test.yml"
PROJECT="rag_restore_test"
STATUS_FILE="backups/.last_restore_check.json"
LOG_FILE="backups/.last_restore_check.log"
HISTORY_DIR="backups/restore_checks"
HISTORY_KEEP_DAYS="${RESTORE_CHECK_KEEP_DAYS:-30}"
mkdir -p "$HISTORY_DIR"

# Computed once, up front, so both the JSON (written inside _drill, before
# the log is redacted) and the log (written after, once the pipe below
# drains) file under the same run identifier.
RUN_TS="$(date -u +%Y%m%d_%H%M%S)"
RUN_TS_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HISTORY_JSON="$HISTORY_DIR/${RUN_TS}.json"
HISTORY_LOG="$HISTORY_DIR/${RUN_TS}.log"

# scripts/verify_restore_semantics.py needs psycopg2 + qdrant-client, which
# live in this project's venv, not whatever bare `python3` resolves to.
if [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

BACKUP_DIR_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --backup-dir) BACKUP_DIR_ARG="$2"; shift 2 ;;
        -h|--help) echo "Usage: ./verify_restore.sh [--backup-dir backups/YYYYMMDD_HHMMSS]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# The whole drill runs inside this function so its entire stdout/stderr can
# be captured for the sanitized log below — `trap ... EXIT` here fires on
# THIS function's subshell exit (the first stage of a pipe always runs in
# a subshell in bash), which is what makes teardown fire even if `set -e`
# kills us partway through, not just on a clean `return`.
_drill() {
    local backup_dir="$1"

    if [ -z "$backup_dir" ]; then
        backup_dir=$(find backups -maxdepth 1 -mindepth 1 -type d -name '[0-9]*' | sort | tail -1)
    fi
    [ -n "$backup_dir" ] && [ -d "$backup_dir" ] || { echo "ERROR: no backup found under backups/ (run backup_qdrant.sh first)" >&2; return 1; }
    echo "=== Restore drill: $backup_dir ==="

    local git_commit git_dirty
    git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then git_dirty=true; else git_dirty=false; fi
    echo "repo commit: $git_commit (dirty=$git_dirty)"

    # Not `local` — _teardown (below) references these too, and a trap
    # handler firing on this function's `return` can run after `local`
    # variables from this call frame have already gone out of scope.
    restore_target_pg="postgresql://${POSTGRES_USER:-raguser}:${POSTGRES_PASSWORD:-ragpass}@localhost:15432/${POSTGRES_DB:-ragdb}"
    restore_target_qdrant="http://localhost:16333"
    restore_target_uploads="$SCRIPT_DIR/.restore_test_uploads"
    report_file="$(mktemp)"

    _teardown() {
        echo "--- Tearing down scratch environment ---"
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT" down -v >/dev/null 2>&1 || true
        rm -rf "$restore_target_uploads" "$report_file"
    }
    trap _teardown EXIT

    echo "--- Starting scratch Postgres + Qdrant (project: $PROJECT) ---"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT" down -v >/dev/null 2>&1 || true
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT" up -d

    echo "--- Waiting for scratch Postgres ---"
    local pg_ready=0
    for _ in $(seq 1 60); do
        if pg_isready -h localhost -p 15432 >/dev/null 2>&1; then pg_ready=1; break; fi
        sleep 2
    done
    [ "$pg_ready" = "1" ] || { echo "ERROR: scratch Postgres never became ready" >&2; return 1; }

    echo "--- Waiting for scratch Qdrant ---"
    local qdrant_ready=0
    for _ in $(seq 1 60); do
        if curl -sS -o /dev/null "http://localhost:16333/collections"; then qdrant_ready=1; break; fi
        sleep 2
    done
    [ "$qdrant_ready" = "1" ] || { echo "ERROR: scratch Qdrant never became ready" >&2; return 1; }

    local collection_name
    collection_name=$(python3 -c "import json; print(json.load(open('$backup_dir/manifest.json'))['qdrant']['collection'])")

    echo "--- Restoring into scratch targets ---"
    local restore_start restore_end restore_exit=0
    restore_start=$(date +%s)
    ./restore_backup.sh \
        --backup-dir "$backup_dir" \
        --postgres-url "$restore_target_pg" \
        --qdrant-url "$restore_target_qdrant" \
        --uploads-dir "$restore_target_uploads" \
        --i-understand-this-overwrites-target || restore_exit=$?
    restore_end=$(date +%s)

    echo "--- Running semantic verification ---"
    local verify_start verify_end verify_exit=0
    verify_start=$(date +%s)
    "$PYTHON_BIN" scripts/verify_restore_semantics.py \
        --manifest "$backup_dir/manifest.json" \
        --postgres-url "$restore_target_pg" \
        --qdrant-url "$restore_target_qdrant" \
        --qdrant-collection "$collection_name" \
        --qdrant-api-key "${QDRANT_API_KEY:-}" \
        --uploads-dir "$restore_target_uploads" \
        --sample-size 20 \
        > "$report_file" || verify_exit=$?
    verify_end=$(date +%s)
    cat "$report_file"

    local result="pass"
    [ "$restore_exit" = "0" ] && [ "$verify_exit" = "0" ] || result="fail"

    python3 -c "
import json
try:
    report = json.load(open('$report_file'))
except Exception as e:
    report = {'error': f'could not parse verify_restore_semantics.py output: {e}'}
status = {
    'timestamp': '$RUN_TS_ISO',
    'backup_dir': '$backup_dir',
    'result': '$result',
    'git_commit': '$git_commit',
    'git_dirty': ${git_dirty^},
    'restore_exit_code': $restore_exit,
    'verify_exit_code': $verify_exit,
    'restore_duration_seconds': $((restore_end - restore_start)),
    'verify_duration_seconds': $((verify_end - verify_start)),
    'log_file': '$LOG_FILE',
    'history_json': '$HISTORY_JSON',
    'history_log': '$HISTORY_LOG',
    'details': report,
}
with open('$STATUS_FILE', 'w') as f:
    json.dump(status, f, indent=2)
    f.write('\n')
"
    cp "$STATUS_FILE" "$HISTORY_JSON"
    echo "=== Restore drill result: $result (recorded in $STATUS_FILE and $HISTORY_JSON) ==="
    [ "$result" = "pass" ]
}

RAW_LOG="$(mktemp)"
_drill "$BACKUP_DIR_ARG" 2>&1 | tee "$RAW_LOG"
DRILL_EXIT=${PIPESTATUS[0]}

# Redact before this log becomes a saved artifact — .env secrets and any
# embedded URL credentials (restore_backup.sh prints its target URLs) —
# written to both the "last" pointer and this run's dated history copy.
python3 - "$RAW_LOG" "$LOG_FILE" "$HISTORY_LOG" <<'PYEOF'
import os
import re
import sys

raw_path, out_path, history_path = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(raw_path).read()
for secret in (os.environ.get("POSTGRES_PASSWORD", ""), os.environ.get("QDRANT_API_KEY", "")):
    if secret:
        text = text.replace(secret, "***REDACTED***")
text = re.sub(r'(://[^:@/\s]+):[^@/\s]+@', r'\1:***REDACTED***@', text)
for path in (out_path, history_path):
    with open(path, "w") as f:
        f.write(text)
PYEOF
rm -f "$RAW_LOG"

# Retention for the dated history — independent of, and normally longer
# than, backups/ itself (see header comment).
find "$HISTORY_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.log' \) -mtime "+$HISTORY_KEEP_DAYS" -delete

# Off-host copy of THIS run's dated artifacts — not the "last" pointer
# files, which get overwritten next run; the dated copies are the ones
# worth shipping somewhere that a dead backup host can't take with it.
if [ -n "${RESTORE_CHECK_REMOTE_DEST:-}" ]; then
    rsync -a "$HISTORY_JSON" "$HISTORY_LOG" "${RESTORE_CHECK_REMOTE_DEST}" \
        || echo "WARNING: off-host copy of restore-check artifacts to $RESTORE_CHECK_REMOTE_DEST failed" >&2
fi

if [ "$DRILL_EXIT" != "0" ] && [ -n "${RESTORE_CHECK_ALERT_CMD:-}" ]; then
    # Direct invocation — never eval/sh -c. $RESTORE_CHECK_ALERT_CMD must be
    # a single executable path; the JSON goes over stdin, not interpolated
    # into any command string.
    "$RESTORE_CHECK_ALERT_CMD" "drill_failed" < "$STATUS_FILE" \
        || echo "WARNING: RESTORE_CHECK_ALERT_CMD failed" >&2
fi

exit "$DRILL_EXIT"
