"""
Shared thread pool for GPU-bound calls (embedding, reranking).

Bounded to 1 worker so concurrent requests don't fight over the same GPU,
while still freeing the asyncio event loop to serve other work (Qdrant/
Postgres I/O, other users' requests up to this step) instead of blocking
it outright, which is what happens when these calls run inline in an
`async def` route handler.
"""
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-worker")


async def run_on_gpu(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        func = functools.partial(func, **kwargs)
    return await loop.run_in_executor(GPU_EXECUTOR, func, *args)
