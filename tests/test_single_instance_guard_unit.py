"""
Fake-connection unit tests for lock.py's acquire_single_instance_guard(),
release_single_instance_guard(), and watch_single_instance_guard() — no live
Postgres required (unlike test_single_instance_guard.py's real-subprocess
tests), since what's under test here is this module's own cleanup/control
flow, not the actual cross-process guarantee.

Covers:
  - acquire_single_instance_guard() closes its connection on ANY failure
    between opening it and successfully handing it to the module-level
    owner (P2: previously cursor()/execute()/fetchone() raising left the
    connection open, relying on GC).
  - release_single_instance_guard() closes and resets cleanly, and is a
    no-op when nothing is held.
  - watch_single_instance_guard() marks unhealthy and hard-exits on a lost
    connection (P1's watchdog), and does neither while the connection is
    healthy.
"""
import asyncio

import pytest

import lock as lock_module


class _FakeCursor:
    def __init__(self, raise_on_execute=None, fetch_value=None):
        self._raise = raise_on_execute
        self._fetch_value = fetch_value

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def execute(self, *args, **kwargs):
        if self._raise is not None:
            raise self._raise

    def fetchone(self):
        return (self._fetch_value,)


class _FakeConn:
    def __init__(self, cursor, closed=False):
        self._cursor = cursor
        self.closed = closed
        self.close_calls = 0
        self.autocommit = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.close_calls += 1
        self.closed = True


# ── acquire_single_instance_guard(): P2 cleanup-on-failure ──────────────────

def test_acquire_closes_connection_when_query_raises(monkeypatch):
    boom = RuntimeError("network blip mid-query")
    fake_conn = _FakeConn(_FakeCursor(raise_on_execute=boom))
    monkeypatch.setattr(lock_module.psycopg2, "connect", lambda url: fake_conn)
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    with pytest.raises(RuntimeError, match="network blip"):
        lock_module.acquire_single_instance_guard("postgresql://fake")

    assert fake_conn.close_calls == 1
    assert lock_module._single_instance_conn is None


def test_acquire_closes_connection_when_lock_already_held(monkeypatch):
    """The pre-existing 'not acquired' path — was already closing the
    connection before P2, this just pins that behavior alongside the new
    exception-cleanup path above."""
    fake_conn = _FakeConn(_FakeCursor(fetch_value=False))  # pg_try_advisory_lock returned false
    monkeypatch.setattr(lock_module.psycopg2, "connect", lambda url: fake_conn)
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    with pytest.raises(RuntimeError, match="another instance already holds"):
        lock_module.acquire_single_instance_guard("postgresql://fake")

    assert fake_conn.close_calls == 1
    assert lock_module._single_instance_conn is None


def test_acquire_succeeds_and_keeps_connection_open(monkeypatch):
    fake_conn = _FakeConn(_FakeCursor(fetch_value=True))  # pg_try_advisory_lock returned true
    monkeypatch.setattr(lock_module.psycopg2, "connect", lambda url: fake_conn)
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    lock_module.acquire_single_instance_guard("postgresql://fake")

    assert fake_conn.close_calls == 0
    assert lock_module._single_instance_conn is fake_conn


# ── release_single_instance_guard() ──────────────────────────────────────────

def test_release_closes_and_resets(monkeypatch):
    fake_conn = _FakeConn(_FakeCursor())
    monkeypatch.setattr(lock_module, "_single_instance_conn", fake_conn)

    lock_module.release_single_instance_guard()

    assert fake_conn.close_calls == 1
    assert lock_module._single_instance_conn is None


def test_release_is_noop_when_nothing_held(monkeypatch):
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    lock_module.release_single_instance_guard()  # must not raise

    assert lock_module._single_instance_conn is None


# ── watch_single_instance_guard(): P1 watchdog ───────────────────────────────

class _FakeExit(Exception):
    """Stands in for os._exit(): real os._exit() never returns and can't be
    observed by pytest (it would kill the test process), so it's monkeypatched
    to raise this instead, letting the test assert it *would* have fired."""


async def test_watchdog_marks_unhealthy_and_hard_exits_on_lost_connection(monkeypatch):
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))
    boom = RuntimeError("connection lost")
    monkeypatch.setattr(lock_module, "_single_instance_conn", _FakeConn(_FakeCursor(raise_on_execute=boom)))
    calls = []

    with pytest.raises(_FakeExit):
        await lock_module.watch_single_instance_guard(interval_seconds=0.01, on_failure=lambda: calls.append(True))

    assert calls == [True]


async def test_watchdog_treats_missing_connection_as_failure(monkeypatch):
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)
    calls = []

    with pytest.raises(_FakeExit):
        await lock_module.watch_single_instance_guard(interval_seconds=0.01, on_failure=lambda: calls.append(True))

    assert calls == [True]


async def test_watchdog_checks_immediately_before_first_interval(monkeypatch):
    """A dead session at task creation must not get an extra full polling
    interval before detection."""
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    with pytest.raises(_FakeExit):
        await asyncio.wait_for(
            lock_module.watch_single_instance_guard(interval_seconds=3600),
            timeout=0.1,
        )


async def test_watchdog_hard_exits_when_ping_exceeds_deadline(monkeypatch):
    """A blackholed libpq call must have a bounded detection time; the
    polling interval alone does not bound a query already in progress."""
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))
    monkeypatch.setattr(lock_module, "_single_instance_conn", _FakeConn(_FakeCursor(fetch_value=1)))

    async def _never_returns(_func):
        await asyncio.Event().wait()

    monkeypatch.setattr(lock_module.asyncio, "to_thread", _never_returns)

    with pytest.raises(_FakeExit):
        await lock_module.watch_single_instance_guard(
            interval_seconds=3600,
            timeout_seconds=0.01,
        )


async def test_watchdog_survives_on_failure_callback_itself_raising(monkeypatch):
    """on_failure() raising must not stop os._exit() from still being
    called — the exit is the part that must never be skipped."""
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))
    monkeypatch.setattr(lock_module, "_single_instance_conn", None)

    def _broken_callback():
        raise KeyboardInterrupt("readiness flag plumbing itself broke")

    with pytest.raises(_FakeExit):
        await lock_module.watch_single_instance_guard(interval_seconds=0.01, on_failure=_broken_callback)


async def test_watchdog_does_not_exit_while_connection_is_healthy(monkeypatch):
    exit_calls = []
    monkeypatch.setattr(lock_module.os, "_exit", lambda code: exit_calls.append(code))
    monkeypatch.setattr(lock_module, "_single_instance_conn", _FakeConn(_FakeCursor(fetch_value=1)))

    task = asyncio.create_task(
        lock_module.watch_single_instance_guard(interval_seconds=0.01, on_failure=lambda: None)
    )
    await asyncio.sleep(0.05)  # several ticks
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert exit_calls == []
