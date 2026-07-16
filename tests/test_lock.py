"""
Tests for lock.py's shared_lock() — the reader/writer lock coordinating
live API mutations (uploads, deletes, folder changes) and maintenance
scripts against backup_qdrant.sh's exclusive hold. Uses the real flock(2)
syscall (no fake/mock — this is exactly the kind of thing that's only
meaningful tested against the actual kernel primitive), but ALWAYS against
an isolated tmp_path file, never the real repo-root .backup.lock.

That last point matters beyond tidiness: if these tests ever ran while a
real backup_qdrant.sh held its exclusive flock on the production lock
file, and a fixture here `unlink()`'d that file by its real path, the
backup's already-open file descriptor would keep the lock associated with
the now-deleted inode — while shared_lock()'s next `LOCK_FILE_PATH.touch()`
would silently create a brand-new file (new inode) at that same path.
Every subsequent shared acquire (a live upload/delete) would then lock the
NEW inode, completely unrelated to the backup's lock on the orphaned OLD
one — new uploads would run right through backup's exclusive hold as if
it didn't exist. The one thing a test suite for this lock must never do is
weaken the very guarantee it's testing.
"""
import fcntl
import logging
import os

import pytest

import lock as lock_module
from lock import shared_lock, run_locked


@pytest.fixture(autouse=True)
def _isolated_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(lock_module, "LOCK_FILE_PATH", tmp_path / ".backup.lock")


def _lock_path():
    return lock_module.LOCK_FILE_PATH


def test_shared_lock_can_be_acquired_and_released():
    with shared_lock():
        pass  # no exception raised = acquired and released cleanly


def test_multiple_shared_locks_can_coexist():
    # SHARED locks don't exclude each other — this is what lets many
    # concurrent uploads/deletes run at once, only backup's EXCLUSIVE hold
    # is mutually exclusive against them.
    with shared_lock():
        with shared_lock():
            pass


def test_shared_lock_fails_fast_when_exclusive_lock_held():
    """Simulates backup_qdrant.sh's `flock -x` hold — a concurrent shared
    acquire attempt (an incoming upload/delete request) must fail
    immediately (non-blocking), not hang the request."""
    path = _lock_path()
    path.touch(exist_ok=True)
    fd = os.open(str(path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            with shared_lock():
                pass
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_shared_lock_released_on_exception():
    """A shared lock held during a request that then raises must still be
    released — otherwise one failed upload would permanently wedge out
    every future backup, since the exclusive acquire would never succeed."""
    with pytest.raises(ValueError):
        with shared_lock():
            raise ValueError("simulated request failure")

    # If the shared lock had leaked, this exclusive, non-blocking acquire
    # attempt would raise BlockingIOError instead of succeeding.
    fd = os.open(str(_lock_path()), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ── run_locked() ─────────────────────────────────────────────────────────────

def test_run_locked_exits_when_exclusive_lock_held():
    """Simulates backup_qdrant.sh holding the exclusive flock — a
    maintenance script calling run_locked() must refuse to start its work
    at all, exiting non-zero rather than running unlocked."""
    path = _lock_path()
    path.touch(exist_ok=True)
    fd = os.open(str(path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        called = []
        with pytest.raises(SystemExit) as exc_info:
            run_locked(lambda: called.append(True), logging.getLogger("test"))
        assert exc_info.value.code == 1
        assert called == []  # work() must never run if the lock couldn't be acquired
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_run_locked_does_not_mask_a_blocking_io_error_from_work():
    """The exact regression this guards: work() raising its own
    BlockingIOError (for a completely unrelated reason — some non-blocking
    I/O the script's own logic does) must propagate as-is, not get
    reinterpreted as "backup in progress" and turned into a SystemExit."""
    def _work():
        raise BlockingIOError("unrelated non-blocking I/O failure inside the script itself")

    with pytest.raises(BlockingIOError, match="unrelated non-blocking I/O failure"):
        run_locked(_work, logging.getLogger("test"))

    # And the lock must still have been released despite work() raising —
    # otherwise this one buggy script would permanently wedge out backup.
    fd = os.open(str(_lock_path()), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_run_locked_returns_works_return_value():
    result = run_locked(lambda: 42, logging.getLogger("test"))
    assert result == 42
