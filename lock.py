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
import asyncio
import fcntl
import logging
import math
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2

_logger = logging.getLogger(__name__)

LOCK_FILE_PATH = Path(__file__).resolve().parent / ".backup.lock"

# Arbitrary fixed key identifying this specific advisory lock's purpose.
# pg_advisory_lock's key space is a single flat namespace per Postgres
# database — shared with anything else that ever takes an advisory lock
# against the same database — so this is deliberately a large, unlikely-to-
# collide constant rather than a small int like 1.
SINGLE_INSTANCE_LOCK_KEY = 727_942_016_618_820_711

# Holds the dedicated connection acquire_single_instance_guard() opens, kept
# alive here (not just a local variable in that function) so nothing garbage
# collects it — see the function's docstring for why the connection itself
# IS the lock.
_single_instance_conn = None


def acquire_single_instance_guard(postgres_url: str):
    """Acquires — and deliberately never releases — a Postgres session-level
    advisory lock (pg_try_advisory_lock) proving this process is the only one
    holding api/main.py's documents_registry/file_hashes/folders_registry.
    Those are plain in-memory dicts loaded once from Postgres at startup
    (init_db()) and mutated only in this process from then on; a second
    process (`uvicorn --workers 2`, a second replica, ...) would load its own
    separate copy that never sees the first process's writes, so
    list/page/pdf/delete would silently return stale data or 404 depending on
    which process a given request happens to land on, with no self-healing
    short of restarting every process.

    An earlier version of this guard used a local flock (see shared_lock()
    below) on a file next to this one. That only proves single-process-per-
    filesystem: two containers with separate root filesystems, or two
    separate hosts, each get their own independent inode to lock and would
    each happily start up believing they're the only instance. Postgres has
    no such seam — every instance of this app already MUST point at the same
    Postgres database (it's where documents_registry itself is loaded from),
    so an advisory lock's session table, which lives inside that one
    database, is contended for identically regardless of which host,
    container, or filesystem a given instance happens to run on.

    Caveat this guard does NOT cover: a session-level advisory lock is only
    as session-scoped as the connection actually is. If POSTGRES_URL is ever
    pointed at a connection pooler running in transaction-pooling mode (e.g.
    PgBouncer in its default `transaction` pool_mode — session mode is fine),
    the pooler can hand this dedicated connection's underlying server
    connection to a completely different client between transactions,
    silently detaching the lock from the process that thinks it's still
    holding it. Nothing in this function can detect that from the client
    side; keep POSTGRES_URL pointed directly at Postgres (or a
    session-pooling-mode pooler) for this guard to mean what it says.

    Deliberately a plain function, not a `with`-scoped contextmanager: this
    lock (session-level pg_try_advisory_lock, not the transaction-level
    variant) is released when its session — this dedicated connection —
    closes. Closing the connection right after acquiring it would
    immediately release the lock, so it's kept open (via the module-level
    _single_instance_conn reference above) for this process's entire
    lifetime instead — there is no explicit unlock to get wrong across
    restarts under the one failure mode where "session ends" and "this
    process is done" are the same event: normal shutdown, an uncaught
    exception, SIGKILL, a host crash. Call release_single_instance_guard()
    below to end it deliberately (clean shutdown, or a later startup step
    failing) without waiting for the process itself to exit.

    Session end and process end are NOT the same event under every failure
    mode, though, and that gap is exactly the one this function cannot close
    by itself: a network partition between this process and Postgres, or
    Postgres itself restarting, ends the session — Postgres releases the
    advisory lock on its side — while this process keeps running, still
    holding a `conn` object that LOOKS fine locally (psycopg2 doesn't
    proactively notice a half-dead TCP connection) and still serving
    requests against registries it can no longer prove it's the sole owner
    of. A second process can then acquire the now-free lock and start
    serving too, right up until this one's next query fails — the exact
    split-brain this whole module exists to prevent, just relocated from
    "at startup" to "sometime later, silently." watch_single_instance_guard()
    below exists specifically to close this gap by continuously proving the
    session is still alive rather than assuming it after the fact; run it
    for as long as this process claims to hold the lock.

    Raises RuntimeError if another process already holds it (pg_try_
    advisory_lock returns false rather than blocking — same fail-fast
    contract flock's LOCK_NB gave). Raises whatever psycopg2 raises if
    postgres_url itself can't be connected to at all — a distinct failure
    from "another instance is running" that callers should let surface
    as-is rather than mask. Either way, the connection this call opened for
    itself is always closed before it raises — nothing is left dangling for
    garbage collection to eventually get around to, which matters here
    because an un-collected connection would itself hold a Postgres session
    open indefinitely."""
    global _single_instance_conn
    conn = psycopg2.connect(postgres_url)
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (SINGLE_INSTANCE_LOCK_KEY,))
            acquired = cur.fetchone()[0]
        if not acquired:
            raise RuntimeError(
                f"another instance already holds the single-instance Postgres advisory "
                f"lock (key={SINGLE_INSTANCE_LOCK_KEY})"
            )
    except BaseException:
        conn.close()
        raise
    _single_instance_conn = conn


def release_single_instance_guard():
    """Explicitly ends the session acquire_single_instance_guard() opened,
    releasing the advisory lock immediately rather than waiting for this
    process to exit and its socket to be torn down. Call this from a clean
    shutdown handler, or from any startup step that fails AFTER the guard
    already succeeded — the process is going to exit either way in that
    case, but an explicit release here means the NEXT instance doesn't have
    to wait out however long that exit takes.

    Safe to call even if the guard was never successfully acquired (no-op) —
    callers don't need their own None-check."""
    global _single_instance_conn
    conn = _single_instance_conn
    _single_instance_conn = None
    if conn is not None:
        conn.close()


async def check_single_instance_guard(timeout_seconds: float = 5.0):
    """Proves the dedicated advisory-lock session is still usable.

    The libpq call runs off the event-loop thread because psycopg2 is
    synchronous.  ``wait_for`` is load-bearing here: a clean server-side
    disconnect fails the query immediately, but a network blackhole can
    otherwise leave libpq blocked for minutes.  Callers must treat either a
    query failure or this deadline expiring as loss of the guard.

    Timing out does not stop the worker thread executing the libpq call, but
    that is intentional for the watchdog caller: it immediately terminates
    the whole process with os._exit(), which also terminates that thread.
    During startup a failed check is fatal too, so the process never becomes
    ready on a session it can no longer prove is alive.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("single-instance guard ping timeout must be greater than zero")

    conn = _single_instance_conn
    if conn is None or conn.closed:
        raise RuntimeError("_single_instance_conn is not open")

    def _ping():
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

    await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout_seconds)


async def watch_single_instance_guard(
    interval_seconds: float = 5.0,
    timeout_seconds: float = 5.0,
    on_failure=None,
):
    """Runs forever (until cancelled) periodically proving
    _single_instance_conn's session — and therefore the advisory lock riding
    on it — is still alive, by round-tripping a trivial query. See the
    "Session end and process end are NOT the same event" paragraph in
    acquire_single_instance_guard()'s docstring above for exactly which
    failure this closes: without it, a network partition or a Postgres
    restart silently frees the lock on Postgres's side while this process
    keeps running and serving requests as if it still held it.

    On ANY failure of that query (connection closed, network timeout, query
    error, _single_instance_conn having gone missing entirely), this is
    fatal by design, in this order:
      1. on_failure() is called synchronously, if given — e.g. flipping a
         readiness flag an in-process /health endpoint reads, so a
         concurrent request has the best available chance of seeing
         "unhealthy" before anything else happens.
      2. os._exit(1) — NOT sys.exit()/raise: those unwind through Python
         (running `finally` blocks, and anything upstream is free to catch a
         plain exception and keep going), which is precisely the "keep
         serving on unsafe state" outcome this function exists to prevent.
         os._exit() is immediate and unconditional — the one thing this
         function must never do on failure is return control to any caller
         that might decide to keep the server up.
    No retry-then-recover: this guard's entire job is telling the truth
    about whether this process can currently prove it's the sole lock
    holder, and "we can't tell right now" is not provably different from
    "we don't hold it" — treating a transient blip as fine on the hope it
    self-resolves is exactly how split-brain reappears here. A restart
    triggered by a transient blip is cheap; staying up on a lock this
    process can no longer verify is the actual bug.

    This narrows, but does not close to zero, the window where two
    processes could both believe they hold the lock — there is inherently
    some delay between Postgres releasing it and this watchdog's next tick
    noticing. Closing that window fully needs either a fencing token
    checked on every request (each acquire gets a monotonic generation
    number; every read/write of documents_registry/etc. validates it's
    still current) or moving those registries off process-local memory
    entirely and onto Postgres as the live source of truth — either is a
    real architecture change, not a tuning knob on this function."""
    if not math.isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("single-instance watchdog interval must be greater than zero")

    while True:
        try:
            # Check first, then sleep: the task proves the session as soon as
            # it starts instead of adding a full interval before its first
            # observation.  api/main.py also performs an awaited check after
            # heavy startup initialization and before declaring readiness.
            await check_single_instance_guard(timeout_seconds)
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            _logger.critical(
                "single-instance guard watchdog lost its Postgres session (%s) — "
                "exiting immediately: this process can no longer prove it is the "
                "only instance holding the in-memory document registries.", e,
            )
            try:
                if on_failure is not None:
                    on_failure()
            finally:
                # Even a broken readiness callback (including a
                # BaseException) must never suppress the fail-stop exit.
                os._exit(1)
        await asyncio.sleep(interval_seconds)


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
