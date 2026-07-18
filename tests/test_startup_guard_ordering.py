"""
Tests that api/main.py's startup() acquires lock.py's single-instance guard
BEFORE constructing anything expensive (embedder/reranker load real models
onto the GPU), that a guard failure surfaces as a RuntimeError rather than
being swallowed, and that the watchdog + explicit release are correctly
wired into startup()'s later-failure path and shutdown(). Everything
lock-related is mocked here — real guard/watchdog behavior against a live
Postgres is covered by test_single_instance_guard.py and
test_single_instance_guard_unit.py; this file is only about how api/main.py
sequences and reacts to them, so it needs no live Postgres/Qdrant/GPU.
"""
import asyncio
from types import SimpleNamespace

import pytest

import api.main as m


class _FakeExit(BaseException):
    pass


async def test_startup_reraises_guard_failure(monkeypatch):
    def _raise(_url):
        raise RuntimeError("another instance already holds the single-instance Postgres advisory lock")
    monkeypatch.setattr(m, "acquire_single_instance_guard", _raise)

    with pytest.raises(RuntimeError, match="already holds"):
        await m.startup()


async def test_startup_does_not_construct_heavy_globals_when_guard_fails(monkeypatch):
    """The actual point of moving the guard first: a rejected worker must
    never pay embedder/reranker's GPU load cost. Proven here by asserting
    those globals are still unset after a failed startup() — if the guard
    ran after construction (the ordering bug this guards against), they'd be
    non-None despite the guard having failed."""
    monkeypatch.setattr(m, "acquire_single_instance_guard", lambda _url: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(m, "embedder", None)
    monkeypatch.setattr(m, "chunker", None)
    monkeypatch.setattr(m, "parser", None)

    with pytest.raises(RuntimeError):
        await m.startup()

    assert m.embedder is None
    assert m.chunker is None
    assert m.parser is None


async def _fake_watchdog_blocks_until_cancelled(interval_seconds, on_failure):
    """Stands in for lock.watch_single_instance_guard() in tests below — the
    real one needs a live _single_instance_conn and loops on a real timer;
    this just occupies the task slot the same way, until startup()/shutdown()
    cancels it, same as production does to the real one at process exit."""
    await asyncio.Event().wait()


def _stub_successful_heavy_init(monkeypatch):
    """Makes startup's post-guard initialization fast and offline."""
    plain = lambda *args, **kwargs: SimpleNamespace()
    embedder = SimpleNamespace(get_vector_size=lambda: 1024)
    vector_store = SimpleNamespace(create_collection=lambda **kwargs: None)

    monkeypatch.setattr(m, "PDFParser", plain)
    monkeypatch.setattr(m, "TxtParser", plain)
    monkeypatch.setattr(m, "SmartChunker", plain)
    monkeypatch.setattr(m, "EmbeddingService", lambda *args, **kwargs: embedder)
    monkeypatch.setattr(m, "VectorStore", lambda *args, **kwargs: vector_store)
    monkeypatch.setattr(m, "HybridRetriever", plain)
    monkeypatch.setattr(m, "CrossEncoderReranker", plain)
    monkeypatch.setattr(m, "PromptBuilder", plain)
    monkeypatch.setattr(m, "LLMGenerator", plain)
    monkeypatch.setattr(m, "QueryExpander", plain)
    monkeypatch.setattr(m, "Langfuse", plain)
    monkeypatch.setattr(m, "init_db", lambda: None)


async def _drain_cancelled_task(task):
    if task is None:
        return
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_startup_rechecks_guard_before_starting_watchdog(monkeypatch):
    order = []
    monkeypatch.setattr(m, "acquire_single_instance_guard", lambda url: order.append("acquire"))
    monkeypatch.setattr(m, "release_single_instance_guard", lambda: None)
    monkeypatch.setattr(m, "_single_instance_watchdog_task", None)
    _stub_successful_heavy_init(monkeypatch)

    async def _check(timeout_seconds):
        order.append("check")

    monkeypatch.setattr(m, "check_single_instance_guard", _check)

    # A plain (non-async) function standing in for watch_single_instance_
    # guard(): startup() calls it synchronously to build the coroutine object
    # it hands to asyncio.create_task(), so recording the call here happens
    # regardless of whether the event loop ever actually steps that
    # coroutine's body — which, for the failure path below, it won't (the
    # rest of startup() is synchronous, so a synchronous failure cancels the
    # task before the loop gets a chance to run even its first line).
    def _fake_watchdog(interval_seconds, timeout_seconds, on_failure):
        order.append("watchdog")
        return _fake_watchdog_blocks_until_cancelled(interval_seconds, on_failure)

    monkeypatch.setattr(m, "watch_single_instance_guard", _fake_watchdog)
    await m.startup()

    assert order == ["acquire", "check", "watchdog"]
    assert m._single_instance_healthy is True

    m._single_instance_watchdog_task.cancel()
    await _drain_cancelled_task(m._single_instance_watchdog_task)


async def test_startup_refuses_readiness_if_guard_died_during_heavy_init(monkeypatch):
    monkeypatch.setattr(m, "acquire_single_instance_guard", lambda url: None)
    monkeypatch.setattr(m, "_single_instance_watchdog_task", None)
    _stub_successful_heavy_init(monkeypatch)
    released = []
    watchdog_calls = []
    monkeypatch.setattr(m, "release_single_instance_guard", lambda: released.append(True))

    async def _dead_guard(timeout_seconds):
        raise RuntimeError("guard session died during model loading")

    monkeypatch.setattr(m, "check_single_instance_guard", _dead_guard)
    monkeypatch.setattr(m, "watch_single_instance_guard", lambda **kwargs: watchdog_calls.append(True))

    with pytest.raises(RuntimeError, match="died during model loading"):
        await m.startup()

    assert released == [True]
    assert watchdog_calls == []
    assert m._single_instance_healthy is False


async def test_startup_ping_timeout_hard_exits_instead_of_closing_busy_connection(monkeypatch):
    monkeypatch.setattr(m, "acquire_single_instance_guard", lambda url: None)
    monkeypatch.setattr(m, "_single_instance_watchdog_task", None)
    _stub_successful_heavy_init(monkeypatch)
    released = []
    monkeypatch.setattr(m, "release_single_instance_guard", lambda: released.append(True))

    async def _timed_out_guard(timeout_seconds):
        raise TimeoutError("simulated blackholed libpq query")

    monkeypatch.setattr(m, "check_single_instance_guard", _timed_out_guard)
    monkeypatch.setattr(m.os, "_exit", lambda code: (_ for _ in ()).throw(_FakeExit(code)))

    with pytest.raises(_FakeExit):
        await m.startup()

    assert released == [True]  # reached only because os._exit is faked
    assert m._single_instance_healthy is False


async def test_startup_releases_guard_and_cancels_watchdog_when_heavy_init_fails(monkeypatch):
    monkeypatch.setattr(m, "acquire_single_instance_guard", lambda url: None)
    monkeypatch.setattr(m, "watch_single_instance_guard", _fake_watchdog_blocks_until_cancelled)
    monkeypatch.setattr(m, "PDFParser", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    released = []
    monkeypatch.setattr(m, "release_single_instance_guard", lambda: released.append(True))

    with pytest.raises(RuntimeError, match="boom"):
        await m.startup()

    assert released == [True]

    assert m._single_instance_watchdog_task is None


async def test_shutdown_releases_guard_and_cancels_watchdog(monkeypatch):
    order = []
    started = asyncio.Event()

    async def _reconciliation():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            order.append("reconciliation_stopped")

    monkeypatch.setattr(m, "release_single_instance_guard", lambda: order.append("guard_released"))
    task = asyncio.create_task(_fake_watchdog_blocks_until_cancelled(5.0, lambda: None))
    reconciliation_task = asyncio.create_task(_reconciliation())
    await started.wait()
    monkeypatch.setattr(m, "_single_instance_watchdog_task", task)
    monkeypatch.setattr(m, "_startup_reconciliation_task", reconciliation_task)

    await m.shutdown()

    assert order == ["reconciliation_stopped", "guard_released"]
    await _drain_cancelled_task(task)
    assert task.cancelled()
    assert reconciliation_task.cancelled()
