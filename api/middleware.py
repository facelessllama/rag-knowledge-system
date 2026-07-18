"""
ASGI middleware for api/main.py. Split out purely to shrink that file — no
dependency on anything else in it.
"""
import json

from fastapi import HTTPException


class MaxBodySizeMiddleware:
    """Raw ASGI middleware (not Starlette's BaseHTTPMiddleware, which
    doesn't give clean control over a streaming request body) — rejects a
    request whose body exceeds its route's limit, checked two ways: an
    honest Content-Length header is rejected before a single byte is read
    off the wire; failing that (header absent, chunked transfer encoding,
    or a client that simply lies about it), actual bytes are counted as
    they arrive from the ASGI server and the request is aborted the
    moment the running total crosses the limit.

    This exists because MAX_UPLOAD_MB enforced inside
    _stream_upload_to_disk() (api/main.py) runs too late to be the only
    protection: FastAPI's `UploadFile = File(...)` dependency is resolved
    BEFORE the endpoint body ever runs, which means Starlette's own
    multipart parser has already read the ENTIRE request body — spooling
    it to a temp file once past its internal in-memory threshold — just
    to construct that UploadFile. A huge body already occupies this
    process's own temp storage by the time _stream_upload_to_disk() gets
    its first chunk. This middleware is the earliest point in the process
    a request can be rejected, before routing or form-parsing touch it —
    the equivalent of a reverse-proxy `client_max_body_size`, run here
    because this deployment doesn't put one in front of uvicorn (see
    README's Reliability section if one is added later: it should set
    its own limit too, as defense in depth, not instead of this).

    Route-aware because a legitimate single upload and a legitimate
    20-file batch differ by more than an order of magnitude; anything not
    explicitly listed gets `default_limit`, which should stay small — no
    other route in this app has a reason to receive a large body.

    Raises plain `HTTPException`, not a custom exception type: FastAPI's
    own request-body-parsing wrapper (fastapi/routing.py's
    `request_body_to_args` caller) has `except HTTPException: raise` /
    `except Exception: raise HTTPException(400, "There was an error
    parsing the body")` around exactly the code path that calls into
    Starlette's multipart parser, which is what ends up invoking
    `receive()` (below) — a custom exception type gets silently flattened
    into that generic 400 by the second clause; only HTTPException itself
    is passed through by the first, giving a real, correctly-coded 413."""

    def __init__(self, app, limits: dict, default_limit: int):
        self.app = app
        self.limits = limits
        self.default_limit = default_limit

    def _limit_for(self, scope) -> int:
        return self.limits.get(scope.get("path"), self.default_limit)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit_for(scope)
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = None
            if declared is not None and declared > limit:
                # Rejected before self.app is ever invoked — there is no
                # FastAPI exception-handling machinery out here to catch a
                # raise, so the response is sent directly, ASGI-style.
                await self._send_413(send, limit)
                return

        total = 0

        async def guarded_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    # Raised from inside a `receive()` call made by
                    # Starlette's own multipart parser, deep inside
                    # self.app — propagates up through FastAPI's request-
                    # body handling (see docstring above for why this must
                    # be HTTPException specifically) to become a real 413.
                    raise HTTPException(413, f"Request body exceeds the {limit // (1024 * 1024)}MB limit for this endpoint")
            return message

        await self.app(scope, guarded_receive, send)

    @staticmethod
    async def _send_413(send, limit_bytes: int):
        body = json.dumps({"detail": f"Request body exceeds the {limit_bytes // (1024 * 1024)}MB limit for this endpoint"}).encode()
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        })
        await send({"type": "http.response.body", "body": body})
