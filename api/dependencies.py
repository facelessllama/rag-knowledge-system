"""
Shared FastAPI dependencies with no dependency on api/main.py's state —
split out purely so api/documents.py and api/upload.py (the two modules
that mutate Qdrant/Postgres/uploads/ and therefore need to be excluded
while a backup holds the exclusive lock) share one implementation instead
of two independently-maintained copies that could silently drift apart.
"""
from fastapi import HTTPException

from lock import shared_lock


async def require_not_backing_up():
    """FastAPI dependency — holds lock.py's SHARED flock for the entire
    request, not just a check at the start of it. A plain existence-check
    of a flag file (an earlier approach) is a gate, not mutual exclusion:
    checked once, an upload that passed a moment before backup created its
    lock file kept mutating Qdrant/Postgres/disk for the rest of its own
    duration, invisible to backup_qdrant.sh's EXCLUSIVE flock hold. Holding
    a real kernel lock for the whole request means backup's exclusive
    acquire (a bounded `flock -w` wait — see backup_qdrant.sh) genuinely
    waits for every in-flight mutation to finish first instead of just
    giving up, and any NEW mutation that starts after backup already holds
    the exclusive lock fails fast with 503 instead of being let through.

    BlockingIOError is caught ONLY around acquiring the lock (the
    __enter__ call, before yield) — not wrapped around `yield` itself. A
    `with shared_lock(): yield` here would also catch a BlockingIOError
    the endpoint's own body happened to raise for an unrelated reason (a
    non-blocking read on some other fd, say) and misreport it as "backup
    in progress" instead of letting it surface as whatever error it
    actually is.

    Deliberately NOT imported from api.main (which has its own history
    with this exact function) — this module has zero dependency on
    api.main's state, so api/documents.py and api/upload.py both import it
    directly rather than reaching into api.main lazily like everything
    else they need from there."""
    cm = shared_lock()
    try:
        cm.__enter__()
    except BlockingIOError:
        raise HTTPException(503, "Backup in progress — try again in a few minutes")
    try:
        yield
    finally:
        cm.__exit__(None, None, None)
