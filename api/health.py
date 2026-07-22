"""
Health/readiness and /models status endpoints. Split out of api/main.py
purely to shrink that file — see the refactor plan.

Every reference to api/main.py's shared state (Postgres/Qdrant config, the
single-instance-guard flag, the generator/embedder instances,
CLOUD_GENERATOR_ENABLED/LANGFUSE_ENABLED) goes through a LAZY
`import api.main as m` done INSIDE each function, at the point of use —
never a top-level `from api.main import X`. Two independent reasons:
  1. `api.main` imports THIS module to wire up its routers, so a top-level
     `import api.main` here would be circular.
  2. Tests do `monkeypatch.setattr(api.main, "X", fake)` against the
     `api.main` module object itself and `startup()` reassigns several of
     these via `global` — a binding captured once at import time (this
     module's or anywhere else's) would freeze at whatever value existed
     then and silently stop seeing either kind of update.

Two routers, not one: `/models` is protected (requires the API key) while
`/health`, `/health/live`, `/health/ready` are deliberately public — a load
balancer/orchestrator has no API key to send. Mounting both under one
router, one way or the other, would silently change one of their auth
requirements.
"""
import asyncio

import psycopg2
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

import logging

logger = logging.getLogger(__name__)

public_router = APIRouter()
protected_router = APIRouter()


async def _check_postgres(timeout: float = 3.0) -> bool:
    # connect_timeout only bounds the TCP connect phase; it does nothing
    # once a connection is established, so a hung/overloaded server could
    # still leave SELECT 1 blocking indefinitely. `options=-c
    # statement_timeout=...` sets a server-side bound on the query itself,
    # so the DB (not just our client) enforces it. asyncio.wait_for around
    # asyncio.to_thread below cannot force-stop the underlying OS thread —
    # it only stops the *caller* from waiting — so these two real,
    # server/libpq-enforced timeouts are what actually end _ping(); the
    # wait_for is just a backstop with headroom over them, not the primary
    # bound.
    import api.main as m

    postgres_url = m.POSTGRES_URL

    def _ping():
        conn = psycopg2.connect(
            postgres_url,
            connect_timeout=int(timeout),
            options=f"-c statement_timeout={int(timeout * 1000)}",
        )
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        finally:
            conn.close()
    try:
        await asyncio.wait_for(asyncio.to_thread(_ping), timeout=timeout * 2 + 1)
        return True
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")
        return False


async def _check_qdrant(timeout: float | None = None) -> dict | None:
    # get_collection_info() catches its own exceptions and returns
    # {"collection": ..., "error": str(e)} instead of raising — so a failure
    # here is only visible by checking for the "error" key, not via except.
    # vector_store's own client was built with timeout=QDRANT_REQUEST_TIMEOUT_SECONDS
    # (api/main.py's VectorStore(...) call) — that's the real, request-level
    # bound on the blocking HTTP call inside get_collection_info(). The
    # wait_for here can't stop that call once it's running in its thread, so
    # it's given headroom over the client's own timeout rather than racing it.
    import api.main as m

    if timeout is None:
        timeout = m.QDRANT_REQUEST_TIMEOUT_SECONDS
    try:
        info = await asyncio.wait_for(asyncio.to_thread(m.vector_store.get_collection_info), timeout=timeout + 2)
        return None if "error" in info else info
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return None


async def _check_ollama(timeout: float = 3.0) -> bool:
    import httpx

    import api.main as m

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{m.generator.ollama_url}/api/tags")
            return r.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


async def _readiness() -> JSONResponse:
    import api.main as m

    # See _single_instance_healthy's declaration in api/main.py, above
    # startup(): flipped by the single-instance watchdog the instant it
    # loses the Postgres advisory lock's session, immediately before the
    # process exits. Checked first and unconditionally — none of the checks
    # below mean anything if this process can no longer prove it's the only
    # one holding documents_registry/file_hashes/folders_registry. This one
    # failure mode stays a plain HTTPException (not the flat JSONResponse
    # body below) — it's "this process isn't even a valid replica", a
    # different shape of failure than "a dependency is down", so
    # {"detail": "..."} is fine here.
    if not m._single_instance_healthy:
        raise HTTPException(503, "single-instance guard lost its Postgres session — this process is exiting")

    qdrant_info, postgres_ok, ollama_ok = await asyncio.gather(
        _check_qdrant(), _check_postgres(), _check_ollama()
    )
    qdrant_ok = qdrant_info is not None
    all_ok = qdrant_ok and postgres_ok and ollama_ok
    body = {
        "status": "healthy" if all_ok else "degraded",
        "components": {
            "qdrant": "ok" if qdrant_ok else "error",
            "postgres": "ok" if postgres_ok else "error",
            "ollama": "ok" if ollama_ok else "error",
            "embedding_model": m.embedder.model_name,
            "llm_model": m.generator.model,
            "langfuse": "ok" if m.LANGFUSE_ENABLED else "disabled",
            "cloud_generator": "available" if (m.CLOUD_GENERATOR_ENABLED and m.generator.deepseek is not None)
                               else ("enabled_no_key" if m.CLOUD_GENERATOR_ENABLED else "disabled"),
        },
        "vector_store": qdrant_info or {}
    }
    # Returned as a plain JSONResponse, not raised as an HTTPException(detail=body)
    # — FastAPI wraps HTTPException.detail in {"detail": ...}, which would nest
    # this body one level deeper than what it actually documents/looks like.
    return JSONResponse(status_code=200 if all_ok else 503, content=body)


@public_router.get("/health/live")
async def health_live():
    """Liveness: is this process itself still able to serve traffic at all —
    no calls to Postgres/Qdrant/Ollama. A load balancer/orchestrator should
    restart the process on failure here; a downstream dependency being down
    is NOT a liveness failure (see /health/ready for that)."""
    import api.main as m

    if not m._single_instance_healthy:
        raise HTTPException(503, "single-instance guard lost its Postgres session — this process is exiting")
    return {"status": "alive"}


@public_router.get("/health/ready")
async def health_ready():
    """Readiness: process alive AND Postgres/Qdrant/Ollama actually reachable.
    Returns HTTP 503 (not 200 with a 'degraded' field buried in the body) the
    moment any of them fails, so orchestrators/load balancers can act on the
    status code alone."""
    return await _readiness()


@public_router.get("/health")
async def health_check():
    """Back-compat alias for /health/ready — same real checks, same 503 on
    failure. Kept because README/start_rag.sh/frontend/scripts already poll
    this path; new integrations should prefer /health/live or /health/ready
    directly."""
    return await _readiness()


@protected_router.get("/models")
async def list_models():
    """List available local models from Ollama, plus whether the cloud
    (DeepSeek) generator is selectable at all right now — the frontend
    uses `cloud` to decide whether to show the provider toggle. `cloud.
    available` requires BOTH ENABLE_CLOUD_GENERATOR=true and a configured
    DEEPSEEK_API_KEY (see GeneratorRouter) — it does not mean any request
    is currently using it, only that provider="deepseek" would be
    accepted."""
    import httpx
    import os

    import api.main as m

    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11435")
    cloud_info = {
        "enabled_by_admin": m.CLOUD_GENERATOR_ENABLED,
        "available": m.CLOUD_GENERATOR_ENABLED and m.generator.deepseek is not None,
        "model": m.generator.deepseek.model if m.generator.deepseek is not None else None,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_url}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = []
                for ollama_model in data.get("models", []):
                    name = ollama_model.get("name", "")
                    size_bytes = ollama_model.get("size", 0)
                    size_gb = round(size_bytes / 1e9, 1) if size_bytes else 0
                    models.append({
                        "name": name,
                        "size_gb": size_gb,
                        "active": name == m.generator.model,
                    })
                return {"models": models, "current": m.generator.model, "cloud": cloud_info}
    except Exception as e:
        logger.warning(f"Ollama models fetch failed: {e}")
    return {"models": [{"name": m.generator.model, "size_gb": 0, "active": True}],
            "current": m.generator.model, "cloud": cloud_info}
