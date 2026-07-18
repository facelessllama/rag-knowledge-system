#!/bin/bash
# Freshness monitor for the restore drill. Run this on its OWN schedule
# (daily — see README.md), independent of verify_restore.sh's weekly
# cadence: if verify_restore.sh's own cron entry silently stops firing
# (misconfigured, host issue, cron daemon down), verify_restore.sh never
# gets a chance to alert on its own failure — this script is the outside
# check that catches THAT failure mode.
#
# Looks at the timestamp of the most recent SUCCESSFUL (result=="pass")
# run recorded in backups/restore_checks/*.json — deliberately not just
# the mtime of backups/.last_restore_check.json and not "any run": a
# fresh FAILED run would otherwise look "fresh" by mtime alone and mask a
# restore that hasn't actually verified successfully in weeks.
#
# Usage: ./check_restore_freshness.sh [max_days]
#   max_days defaults to RESTORE_FRESHNESS_MAX_DAYS or 9 — sized for a
#   weekly drill (7-day cadence + ~2 days slack before paging).
#
# Exit 0 = a successful drill within max_days. Exit 1 = stale (or none
# ever recorded). Optional: set RESTORE_CHECK_ALERT_CMD to the path of a
# single executable (a script, not a shell command line) — invoked
# directly, never via eval/sh -c, as `"$RESTORE_CHECK_ALERT_CMD" stale`
# with a small JSON payload on its stdin. Reused from verify_restore.sh
# since both represent the same "restore verification isn't healthy"
# condition to whoever is paged.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MAX_DAYS="${1:-${RESTORE_FRESHNESS_MAX_DAYS:-9}}"
HISTORY_DIR="backups/restore_checks"

LAST_PASS_ISO=$(python3 -c "
import glob
import json

best = None
for path in glob.glob('$HISTORY_DIR/*.json'):
    try:
        data = json.load(open(path))
    except Exception:
        continue
    if data.get('result') == 'pass':
        ts = data.get('timestamp')
        if ts and (best is None or ts > best):
            best = ts
print(best or '')
")

if [ -z "$LAST_PASS_ISO" ]; then
    echo "STALE: no successful restore drill found in $HISTORY_DIR" >&2
    DAYS_SINCE=""
    STALE=1
else
    DAYS_SINCE=$(python3 -c "
from datetime import datetime, timezone
last = datetime.strptime('$LAST_PASS_ISO', '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
now = datetime.now(timezone.utc)
print((now - last).days)
")
    if [ "$DAYS_SINCE" -gt "$MAX_DAYS" ]; then
        echo "STALE: last successful restore drill was $DAYS_SINCE day(s) ago ($LAST_PASS_ISO) — exceeds ${MAX_DAYS}-day threshold" >&2
        STALE=1
    else
        echo "OK: last successful restore drill was $DAYS_SINCE day(s) ago ($LAST_PASS_ISO)"
        STALE=0
    fi
fi

if [ "$STALE" = "1" ] && [ -n "${RESTORE_CHECK_ALERT_CMD:-}" ]; then
    printf '{"reason": "stale", "last_successful": %s, "days_since": %s, "max_days": %s}\n' \
        "$([ -n "$LAST_PASS_ISO" ] && echo "\"$LAST_PASS_ISO\"" || echo null)" \
        "${DAYS_SINCE:-null}" \
        "$MAX_DAYS" \
    | "$RESTORE_CHECK_ALERT_CMD" "stale" \
    || echo "WARNING: RESTORE_CHECK_ALERT_CMD failed" >&2
fi

[ "$STALE" = "0" ]
