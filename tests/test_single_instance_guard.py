"""
Subprocess-level tests for lock.py's acquire_single_instance_guard() — the
Postgres advisory-lock guard that stops a second process (`uvicorn --workers
N>1`, a second replica, a crash-loop respawn of a rejected worker) from ever
running with its own independent copy of api/main.py's documents_registry/
file_hashes/folders_registry.

Unlike shared_lock() (tested in test_lock.py against real file descriptors
within THAT one process), what actually needs proving here is cross-PROCESS
exclusion against the real Postgres session primitive — two connections
opened from within the same Python process wouldn't exercise the thing that
is actually load-bearing: that separate OS processes, potentially on
separate hosts or containers, genuinely contend for the same lock because it
lives in Postgres rather than on either process's local filesystem.

Requires a live, reachable Postgres (POSTGRES_URL env var, same default as
api/main.py) — skips the whole module otherwise, same spirit as eval/'s
live-service tests (see tests/test_document_page_endpoint.py's docstring).

Each test uses its own randomly chosen advisory lock key (never the real
app's SINGLE_INSTANCE_LOCK_KEY) so this suite can never contend with, or be
disrupted by, a real API instance that happens to be running against the
same Postgres database during a dev/test session.
"""
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import psycopg2
import pytest

import lock as lock_module

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")


def _postgres_reachable():
    try:
        conn = psycopg2.connect(POSTGRES_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_reachable(), reason=f"Postgres not reachable at {POSTGRES_URL}")


def _open_connection_count():
    """Live server-side connection count for the current Postgres user —
    used to prove a failed acquire attempt doesn't leak a connection."""
    conn = psycopg2.connect(POSTGRES_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM pg_stat_activity WHERE usename = current_user")
            return cur.fetchone()[0]
    finally:
        conn.close()


# Runs as a genuinely separate OS process: imports lock.py fresh, points it
# at a test-only key (never the real app's), acquires the guard, announces
# success on stdout, then blocks on stdin until the parent test kills it or
# closes the pipe — simulating a live worker holding the lock for its whole
# lifetime, same as api/main.py's startup() does.
_HOLDER_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {repo_root!r})
    import lock
    lock.SINGLE_INSTANCE_LOCK_KEY = int(sys.argv[1])
    lock.acquire_single_instance_guard(sys.argv[2])
    print("ACQUIRED", flush=True)
    sys.stdin.readline()
""")


@pytest.fixture(autouse=True)
def _cleanup_single_instance_conn():
    """acquire_single_instance_guard() deliberately never closes the
    connection it succeeds with (see its docstring) — fine in production
    (one process, one guard, whole process lifetime), but left unchecked
    across a test session this would accumulate one leaked live Postgres
    connection per test that successfully acquires. Close it after every
    test regardless of outcome."""
    yield
    if lock_module._single_instance_conn is not None:
        lock_module._single_instance_conn.close()
        lock_module._single_instance_conn = None


@pytest.fixture
def lock_key():
    # Random per test — see module docstring for why this must never be the
    # real app's SINGLE_INSTANCE_LOCK_KEY.
    return random.randint(10 ** 9, 2 ** 62)


@pytest.fixture
def holder_process(tmp_path, lock_key):
    """Spawns the real, separate holder process described above and waits
    for its own confirmation that it acquired the lock before handing
    control to the test."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    script = tmp_path / "holder.py"
    script.write_text(_HOLDER_SCRIPT.format(repo_root=repo_root))
    proc = subprocess.Popen(
        [sys.executable, str(script), str(lock_key), POSTGRES_URL],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    if line.strip() != "ACQUIRED":
        proc.kill()
        proc.wait(timeout=5)
        pytest.fail(f"holder process failed to acquire the lock: stderr={proc.stderr.read()}")
    yield proc
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=5)


def test_second_process_is_refused_while_holder_is_alive(holder_process, lock_key, monkeypatch):
    monkeypatch.setattr(lock_module, "SINGLE_INSTANCE_LOCK_KEY", lock_key)
    with pytest.raises(RuntimeError, match="another instance already holds"):
        lock_module.acquire_single_instance_guard(POSTGRES_URL)


def test_failed_acquire_does_not_leak_a_connection(holder_process, lock_key, monkeypatch):
    monkeypatch.setattr(lock_module, "SINGLE_INSTANCE_LOCK_KEY", lock_key)
    before = _open_connection_count()
    with pytest.raises(RuntimeError):
        lock_module.acquire_single_instance_guard(POSTGRES_URL)
    assert _open_connection_count() == before


def test_lock_is_released_when_holder_process_dies(holder_process, lock_key, monkeypatch):
    monkeypatch.setattr(lock_module, "SINGLE_INSTANCE_LOCK_KEY", lock_key)
    holder_process.kill()
    holder_process.wait(timeout=5)
    # No explicit unlock ever happens (see acquire_single_instance_guard()'s
    # docstring) — the whole point being tested here is that Postgres itself
    # releases a session-level advisory lock automatically when its session
    # ends, for ANY reason, including a hard kill with no chance to clean up.
    lock_module.acquire_single_instance_guard(POSTGRES_URL)  # must NOT raise


def test_lock_can_be_reacquired_after_a_clean_release(holder_process, lock_key, monkeypatch):
    monkeypatch.setattr(lock_module, "SINGLE_INSTANCE_LOCK_KEY", lock_key)
    holder_process.stdin.write("\n")  # unblocks the holder's stdin.readline(), letting it exit normally
    holder_process.stdin.close()
    holder_process.wait(timeout=5)
    lock_module.acquire_single_instance_guard(POSTGRES_URL)  # must NOT raise
