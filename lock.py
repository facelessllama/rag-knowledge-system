"""
Reader/writer lock coordinating live mutations (API uploads/deletes/folder
changes, and maintenance scripts like backfill_document_pages.py and
migrate_doc_ids_to_uuid.py) against backup_qdrant.sh's need for a single
consistent instant across Qdrant + Postgres + uploads/. A file-existence
check alone (an earlier approach) is a gate, not mutual exclusion — it's
checked once at the start of a request, so an upload that passed the check
a moment before backup created its lock file keeps mutating Qdrant/
Postgres/disk for the rest of its own duration, unseen by backup. flock is
an actual kernel-held lock: any number of SHARED holders (mutating
operations) can run at once, but they all block a concurrent EXCLUSIVE
holder (backup) until every one of them releases — and a new mutation
started after backup already holds the exclusive lock fails immediately
rather than being let through. Held for a whole request/script's duration,
not just checked once at the start.

Also incidentally fixes the stale-lock problem a plain flag file has: the
kernel releases a process's flock automatically on exit for ANY reason —
normal return, uncaught exception, SIGKILL, even a host crash — so there's
no "forgot to clean up in a trap that never got a chance to run" failure
mode to begin with.

LOCK_FILE_PATH is resolved from this file's own location, not the caller's
CWD — api/main.py, the scripts, and backup_qdrant.sh's bash-side flock
otherwise risk locking three different files if any of them is launched
from an unexpected working directory (systemd, Docker, a cron entry with
its own `cd`, ...).

Every maintenance script that mutates Qdrant/Postgres/uploads/ (backfills,
migrations, schema changes — anything backup_qdrant.sh's snapshot could
otherwise catch mid-write) should call run_locked() around its actual work,
not hand-roll its own try/except BlockingIOError: a repeated-by-hand
pattern is a pattern someone eventually forgets to copy into the next new
script.
"""
import fcntl
import os
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE_PATH = Path(__file__).resolve().parent / ".backup.lock"


@contextmanager
def shared_lock():
    """Hold for the duration of any operation that mutates Qdrant/Postgres/
    uploads/ (API request or maintenance script). Non-blocking: raises
    BlockingIOError immediately if backup_qdrant.sh currently holds the
    exclusive lock, rather than queuing — callers decide what that means
    (api/main.py turns it into an HTTP 503; run_locked() below turns it
    into a logged error + non-zero exit for scripts)."""
    LOCK_FILE_PATH.touch(exist_ok=True)
    fd = os.open(str(LOCK_FILE_PATH), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def run_locked(work, logger):
    """Standard entry point for a maintenance script's main(): runs work()
    (a zero-arg callable — typically a lambda wrapping the script's real
    logic function and its parsed args) inside shared_lock(), and turns a
    failed acquire into one consistent log message + `SystemExit(1)` instead
    of every script reimplementing its own try/except BlockingIOError.
    Usage:
        def main():
            args = argparse.ArgumentParser()...parse_args()
            run_locked(lambda: _run(args), logger)

    Acquisition and execution are deliberately separate try blocks — a
    `try: with shared_lock(): work() except BlockingIOError` here would
    also catch a BlockingIOError work() itself happens to raise for an
    unrelated reason (its own non-blocking I/O, say) and misreport it as
    "backup in progress" instead of letting it surface as whatever error
    it actually is. Same fix as api/main.py's require_not_backing_up()."""
    cm = shared_lock()
    try:
        cm.__enter__()
    except BlockingIOError:
        logger.error("backup_qdrant.sh currently holds the exclusive lock — refusing to run "
                     "while a backup is in progress. Try again once it finishes.")
        raise SystemExit(1)
    try:
        return work()
    finally:
        cm.__exit__(None, None, None)
