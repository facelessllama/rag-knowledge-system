"""
/query and /query/stream endpoints. Split out of api/main.py purely to
shrink that file — see the refactor plan.

Every reference to api/main.py's shared state (`_query_semaphore`,
`RELEVANCE_THRESHOLD`, `LANGFUSE_ENABLED`/`langfuse`, `documents_registry`)
goes through a LAZY `import api.main as m` done INSIDE each function, at the
point of use — never a top-level `from api.main import X`. Same two reasons
as api/documents.py and api/health.py: `api.main` imports this module to
wire up its router (a top-level `import api.main` here would be circular),
and tests do `monkeypatch.setattr(api.main, "X", fake)` against the
`api.main` module object itself — a binding captured once at import time
elsewhere would freeze at whatever value existed then.

ONE deliberate exception, spelled out in the refactor plan: `get_query_
expander`/`get_retriever`/`get_reranker`/`get_prompt_builder`/`get_generator`
ARE imported directly at this module's top level, because they're used as
`Depends(...)` default argument values below — evaluated once at route-
decoration time, which is also import time, so a lazy in-function import
can't help here. This is safe specifically because tests key
`app.dependency_overrides` by the FUNCTION OBJECT itself (`m.get_generator`),
not by patching the name — so this module must import the exact same
function objects `api.main` holds, not redefine equivalent-looking local
ones (which would have a different identity and silently stop being
override-able). `api/main.py` only imports this module AFTER these five
getters are defined, specifically so this top-level import can resolve.
"""
import json
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.documents import _active_chunks, _resolve_scope_document_ids, _validate_query_document_scope
from api.main import get_generator, get_prompt_builder, get_query_expander, get_reranker, get_retriever
from api.schemas import QueryRequest, QueryResponse
from rag.executors import run_on_gpu
from rag.generator import PartialStreamError, ProviderNotAvailable, is_refusal
from rag.retriever import promote_document_opening_chunks, promote_identity_matches, promote_missing_compare_documents

logger = logging.getLogger(__name__)

router = APIRouter()


def _augment_compare_queries(expanded_queries: list[str], document_ids: list[str] | None) -> list[str]:
    # query_expander.expand() decomposes within an 80-token LLM budget — for
    # a "compare A, B, C" question that spells out every filename plus
    # per-aspect instructions, it commonly runs out of budget after the
    # first document's name (observed: all sub-queries referenced only the
    # first of two compared documents). doc_filter in hybrid_search only
    # restricts which documents are *eligible*, it doesn't pull one in on
    # its own — a document never named in any query text simply never
    # enters the candidate pool, so retrieve_expanded's "at least 1 chunk
    # per document" guarantee has nothing to promote for it. One
    # deterministic query per named document closes that gap regardless of
    # what the free-form expander produced.
    if not document_ids or len(document_ids) < 2:
        return expanded_queries
    import api.main as m

    for doc_id in document_ids:
        filename = m.documents_registry.get(doc_id, {}).get("filename")
        if filename:
            expanded_queries.append(f"main subject, parties, and conclusions of {filename}")
    return expanded_queries


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    query_expander=Depends(get_query_expander),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    prompt_builder=Depends(get_prompt_builder),
    generator=Depends(get_generator),
):
    import api.main as m

    # Blank/whitespace-only question is rejected by QueryRequest's own
    # validator (api/schemas.py) before this body ever runs — see
    # _question_not_blank.
    #
    # Conflict (422) before existence (404), deliberately: a document_id
    # that disagrees with document_ids is a malformed request regardless of
    # whether that document_id happens to exist — it doesn't need a
    # documents_registry lookup to be wrong. Checking existence first would
    # make {"document_id": "missing", "document_ids": ["d1"]} 404 instead of
    # the 422 the conflict itself warrants, purely because "missing" also
    # doesn't exist — the wrong reason, and not reproducible in a case where
    # the conflicting document_id DOES exist. See test_conflicting_
    # document_id_and_document_ids_with_nonexistent_singular_id_returns_422.
    scope_document_ids = _resolve_scope_document_ids(request)
    _validate_query_document_scope(request)

    if m._query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")

    async with m._query_semaphore:
        return await _do_query(request, scope_document_ids, query_expander, retriever, reranker, prompt_builder, generator)


async def _do_query(request: QueryRequest, scope_document_ids, query_expander, retriever, reranker, prompt_builder, generator):
    import api.main as m

    start_time = time.time()

    trace = None
    if m.LANGFUSE_ENABLED:
        try:
            trace = m.langfuse.trace(name="rag_query", input=request.question, tags=["query"])
        except Exception as e:
            logger.warning(f"Langfuse trace failed: {e}")

    # chat_history is otherwise only used by prompt_builder.build() below,
    # for the final ANSWER generation — retrieval/reranking never saw it at
    # all until contextualize(), so a follow-up like "What was the court's
    # final decision?" used to search/rerank on that literal text alone,
    # with nothing pointing retrieval at which case "the court" even means.
    # search_query is what actually goes to expand()/retrieve_expanded()/
    # rerank() below; request.question (the user's literal wording) still
    # goes to prompt_builder and the trace, unchanged.
    chat_history_dicts = [t.model_dump() for t in request.chat_history] if request.chat_history else []
    search_query = await query_expander.contextualize(request.question, chat_history_dicts)

    t0 = time.time()
    expanded_queries = await query_expander.expand(search_query)
    expanded_queries = _augment_compare_queries(expanded_queries, scope_document_ids)
    chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=scope_document_ids)
    chunks = _active_chunks(chunks)
    retrieval_ms = int((time.time() - t0) * 1000)

    if trace:
        try:
            trace.span(name="retrieval", input=request.question,
                      output={"chunks_found": len(chunks)}, metadata={"duration_ms": retrieval_ms})
        except Exception as e:
            logger.warning(f"Langfuse span failed: {e}")

    if not chunks:
        return QueryResponse(answer="No relevant information found in the knowledge base.",
                           sources=[], model=generator.model, tokens_used=0)

    top_chunks = await run_on_gpu(reranker.rerank, search_query, chunks, top_k=request.top_k)
    top_chunks = promote_identity_matches(chunks, top_chunks, m.RELEVANCE_THRESHOLD)
    top_chunks = promote_document_opening_chunks(chunks, top_chunks)
    top_chunks = promote_missing_compare_documents(chunks, top_chunks, scope_document_ids)

    best_score = max((c.get("rerank_score", 0) for c in top_chunks), default=0)
    # RELEVANCE_THRESHOLD exists to catch "this isn't in the knowledge base at
    # all" for open-ended questions — moot when a scope was explicitly given
    # (document_ids, or document_id normalized into it — see
    # _resolve_scope_document_ids), since that scope already came from the
    # user explicitly picking real documents out of the library, not from
    # the cross-encoder's opinion
    # of them. Observed directly: a legitimate 3-document compare (all three
    # in scope, all three retrieved — 60 candidates) refused outright because
    # the generic "for each document provide: 1)... 2)... 3)..." compare
    # template scores low against some documents' content even when that
    # content is exactly what was asked to compare. Gating a guaranteed-
    # in-scope answer on a threshold tuned for natural single-topic questions
    # (see eval/golden_dataset.json calibration) rejects real compares for no
    # benefit — there's no "not in the KB" case left to catch here.
    if best_score < m.RELEVANCE_THRESHOLD and not scope_document_ids:
        logger.info(f"Best rerank score {best_score:.3f} below threshold {m.RELEVANCE_THRESHOLD} — not answering")
        return QueryResponse(answer="I couldn't find relevant information in the knowledge base to answer this question.",
                           sources=[], model=generator.model, tokens_used=0,
                           debug={"best_rerank_score": round(best_score, 4), "threshold": m.RELEVANCE_THRESHOLD,
                                  "chunks_retrieved": len(chunks), "chunks_after_rerank": len(top_chunks),
                                  "search_query": search_query if search_query != request.question else None})

    messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                   chat_history=chat_history_dicts)

    t1 = time.time()
    try:
        result = await generator.generate_with_refusal_retry(
            messages, model=request.model or None, provider=request.provider
        )
    except ProviderNotAvailable as e:
        # A clear 4xx, not a silent fall-back to local — see GeneratorRouter's
        # own docstring on why a request that explicitly asked for the cloud
        # provider must never be answered by a different one without saying so.
        raise HTTPException(400, str(e))
    generation_ms = int((time.time() - t1) * 1000)

    # Provider-usage log line — deliberately separate from Langfuse tracing
    # (which already captures full prompt/answer content when enabled) and
    # deliberately NEVER includes question/document/answer text, per the
    # product requirement that cloud-mode usage be auditable without any
    # document content appearing in ordinary logs.
    logger.info(
        f"generator_call provider={result['provider']} model={result['model']} "
        f"latency_ms={generation_ms} tokens={result['total_tokens']} "
        f"result={'refused' if is_refusal(result['answer']) else 'answered'}"
    )

    if trace:
        try:
            trace.generation(name="llm_generation", model=result["model"],
                           input=messages, output=result["answer"],
                           usage={"input": result.get("prompt_tokens", 0),
                                  "output": result.get("completion_tokens", 0),
                                  "total": result["total_tokens"]},
                           metadata={"duration_ms": generation_ms, "provider": result["provider"]})
        except Exception as e:
            logger.warning(f"Langfuse generation failed: {e}")

    seen_docs = {}
    for c in top_chunks:
        doc_id = c.get("document_id")
        score = c.get("rerank_score", c.get("score", 0))
        if doc_id not in seen_docs or score > seen_docs[doc_id]["relevance_score"]:
            raw = c["text"].strip().replace("\n", " ")
            excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
            seen_docs[doc_id] = {
                "page": c.get("page_num"),
                "document": doc_id,
                "excerpt": excerpt,
                "chunk_text": raw,
                "relevance_score": round(score, 3),
                "char_start": c.get("char_start"),
                "char_end": c.get("char_end"),
            }
    sources = sorted(seen_docs.values(), key=lambda x: x["relevance_score"], reverse=True)

    total_ms = int((time.time() - start_time) * 1000)

    if trace:
        try:
            trace.update(output=result["answer"],
                        metadata={"total_ms": total_ms, "retrieval_ms": retrieval_ms,
                                  "generation_ms": generation_ms, "tokens_used": result["total_tokens"],
                                  "sources_count": len(sources)})
            m.langfuse.flush()
        except Exception as e:
            logger.warning(f"Langfuse update failed: {e}")

    return QueryResponse(answer=result["answer"], sources=sources,
                        model=result["model"], provider=result["provider"], tokens_used=result["total_tokens"],
                        debug={
                            "search_query": search_query if search_query != request.question else None,
                            "expanded_queries": expanded_queries,
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms,
                            "total_ms": total_ms,
                            "chunks_retrieved": len(chunks),
                            "chunks_after_rerank": len(top_chunks),
                            "top_chunks": [
                                {
                                    "chunk_id": c.get("chunk_id", ""),
                                    "document_id": c.get("document_id", ""),
                                    "page_num": c.get("page_num", 0),
                                    "score": round(c.get("rerank_score", c.get("score", 0)), 4),
                                    "source": c.get("source", ""),
                                    "text_preview": c.get("text", "")[:100],
                                    # char_start/char_end (already on every
                                    # chunk dict — see vector_db/qdrant_
                                    # client.py's _to_result_dicts) let a
                                    # caller check whether the SPECIFIC span
                                    # a fact lives on reached the model, not
                                    # just whether some chunk from the right
                                    # page did — see eval/mixed_corpus/
                                    # build_golden_dataset.py's evidence_
                                    # char_start/char_end, computed in the
                                    # same normalize_whitespace() coordinate
                                    # space so the two can be intersected
                                    # directly without re-fetching chunk text.
                                    "char_start": c.get("char_start"),
                                    "char_end": c.get("char_end"),
                                }
                                for c in top_chunks
                            ]
                        })


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    query_expander=Depends(get_query_expander),
    retriever=Depends(get_retriever),
    reranker=Depends(get_reranker),
    prompt_builder=Depends(get_prompt_builder),
    generator=Depends(get_generator),
):
    import api.main as m

    # Blank/whitespace-only question is rejected by QueryRequest's own
    # validator (api/schemas.py) before this body ever runs — see
    # _question_not_blank.
    # query_expander/retriever/reranker/prompt_builder/generator above are
    # captured by event_stream()'s closure below — no need to thread them
    # through explicitly the way _do_query() needs them passed in, since
    # event_stream is defined inside this same function.
    #
    # Conflict (422) before existence (404) — see query_knowledge_base's
    # identical comment above for why.
    scope_document_ids = _resolve_scope_document_ids(request)
    _validate_query_document_scope(request)

    if m._query_semaphore.locked():
        raise HTTPException(429, "Too many concurrent requests, please try again shortly")
    await m._query_semaphore.acquire()

    async def event_stream():
        trace = None
        try:
            start_time = time.time()

            if m.LANGFUSE_ENABLED:
                try:
                    trace = m.langfuse.trace(name="rag_stream", input=request.question, tags=["stream"])
                except Exception:
                    pass

            # See _do_query()'s identical comment: chat_history was otherwise
            # only used for the final answer generation below — retrieval/
            # reranking never saw it, so a context-dependent follow-up
            # ("What was the court's final decision?") searched/reranked on
            # that literal text alone, with nothing pointing at which case
            # "the court" refers to.
            chat_history_dicts = [t.model_dump() for t in request.chat_history] if request.chat_history else []
            search_query = await query_expander.contextualize(request.question, chat_history_dicts)

            t0 = time.time()
            expanded_queries = await query_expander.expand(search_query)
            expanded_queries = _augment_compare_queries(expanded_queries, scope_document_ids)
            expansion_ms = int((time.time() - t0) * 1000)

            t1 = time.time()
            chunks = await retriever.retrieve_expanded(expanded_queries, top_k=max(20, request.top_k * 5), folder=request.folder or None, document_ids=scope_document_ids)
            chunks = _active_chunks(chunks)
            retrieval_ms = int((time.time() - t1) * 1000)

            retrieval_scores = [c.get("score", 0) for c in chunks] if chunks else []
            retrieval_best = max(retrieval_scores) if retrieval_scores else 0
            score_meta = {
                "best": round(retrieval_best, 3),
                "avg": round(sum(retrieval_scores) / len(retrieval_scores), 3) if retrieval_scores else 0,
                "chunks_found": len(chunks),
                "queries_expanded": len(expanded_queries),
            }

            if trace:
                try:
                    trace.span(name="query_expansion", input=request.question,
                               output={"queries": expanded_queries},
                               metadata={"duration_ms": expansion_ms, "search_query": search_query})
                    trace.span(name="retrieval", input=expanded_queries,
                               output=score_meta,
                               metadata={"duration_ms": retrieval_ms})
                except Exception:
                    pass

            if not chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant information found in the knowledge base.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            t2 = time.time()
            top_chunks = await run_on_gpu(reranker.rerank, search_query, chunks, top_k=request.top_k)
            top_chunks = promote_identity_matches(chunks, top_chunks, m.RELEVANCE_THRESHOLD)
            top_chunks = promote_document_opening_chunks(chunks, top_chunks)
            top_chunks = promote_missing_compare_documents(chunks, top_chunks, scope_document_ids)
            rerank_ms = int((time.time() - t2) * 1000)
            reranker_type = type(reranker).__name__

            rerank_scores = [c.get("rerank_score", c.get("score", 0)) for c in top_chunks]
            best_score = max(rerank_scores) if rerank_scores else 0
            score_meta = {
                "best": round(best_score, 3),
                "avg": round(sum(rerank_scores) / len(rerank_scores), 3) if rerank_scores else 0,
                "chunks_found": len(chunks),
                "queries_expanded": len(expanded_queries),
            }

            if best_score < m.RELEVANCE_THRESHOLD and not scope_document_ids:
                logger.info(f"Best rerank score {best_score:.3f} below threshold {m.RELEVANCE_THRESHOLD} — not answering")
                msg = "I couldn't find relevant information in the knowledge base to answer this question."
                yield f"data: {json.dumps({'type': 'token', 'content': msg})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'debug': {**score_meta, 'threshold': m.RELEVANCE_THRESHOLD, 'search_query': search_query if search_query != request.question else None}})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            messages = prompt_builder.build(query=request.question, chunks=top_chunks,
                                            chat_history=chat_history_dicts)

            t2 = time.time()
            answer_tokens = []
            async for token in generator.generate_stream_with_refusal_retry(
                messages, model=request.model or None, provider=request.provider
            ):
                answer_tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            generation_ms = int((time.time() - t2) * 1000)

            # Both providers can stream now (GeneratorRouter.
            # generate_stream_with_refusal_retry dispatches to whichever
            # backend _resolve() picked — a request asking for a disabled/
            # misconfigured cloud provider would already have raised
            # ProviderNotAvailable above, so getting here means
            # request.provider (or "local" if unset) is exactly what ran.
            # model_for() exists because a token stream has no return value
            # to carry the resolved model name back the way
            # generate_with_refusal_retry()'s result dict does.
            #
            # request.model is ONLY meaningful for the local provider — the
            # router itself drops it for any other backend before ever
            # generating (rag/generator.py's GeneratorRouter, "cloud
            # backends must never take a client-supplied model string"), so
            # trusting it here for a non-local resolved_provider would make
            # observability (this log line, the Langfuse generation below,
            # and the debug payload's 'model') report whatever Ollama model
            # name the client happened to send alongside provider="deepseek"
            # — wrong, even though the actual generation already correctly
            # used DeepSeek's own configured model throughout.
            resolved_provider = request.provider or "local"
            resolved_model = (
                generator.model_for(resolved_provider)
                if resolved_provider == "deepseek"
                else request.model or generator.model_for("local")
            )
            logger.info(
                f"generator_call provider={resolved_provider} model={resolved_model} "
                f"latency_ms={generation_ms} "
                f"result={'refused' if is_refusal(''.join(answer_tokens)) else 'answered'}"
            )

            if trace:
                try:
                    trace.generation(name="llm_stream", model=resolved_model,
                                     input=messages, output="".join(answer_tokens),
                                     metadata={"duration_ms": generation_ms, "provider": resolved_provider})
                    trace.update(metadata={
                        "total_ms": int((time.time() - start_time) * 1000),
                        "expansion_ms": expansion_ms,
                        "retrieval_ms": retrieval_ms,
                        "rerank_ms": rerank_ms,
                        "generation_ms": generation_ms,
                        "reranker": reranker_type,
                        **score_meta,
                    })
                    m.langfuse.flush()
                except Exception:
                    pass

            # Best chunk per document from all retrieved chunks (not just top_chunks),
            # so compare queries always show sources for every document found.
            # Best chunk per document from top_chunks (reranker already decided relevance)
            seen_docs = {}
            for c in top_chunks:
                doc_id = c.get("document_id")
                score = c.get("rerank_score", c.get("score", 0))
                if doc_id not in seen_docs or score > seen_docs[doc_id]["relevance_score"]:
                    raw = c["text"].strip().replace("\n", " ")
                    excerpt = raw[:150].rsplit(" ", 1)[0] + "…" if len(raw) > 150 else raw
                    seen_docs[doc_id] = {
                        "page": c.get("page_num"),
                        "document": doc_id,
                        "excerpt": excerpt,
                        "chunk_text": raw,
                        "relevance_score": round(score, 3),
                        "char_start": c.get("char_start"),
                        "char_end": c.get("char_end"),
                    }
            sources = sorted(seen_docs.values(), key=lambda x: x["relevance_score"], reverse=True)

            total_ms = int((time.time() - start_time) * 1000)
            debug_payload = {
                'search_query': search_query if search_query != request.question else None,
                'expanded_queries': expanded_queries,
                'total_ms': total_ms,
                'expansion_ms': expansion_ms,
                'retrieval_ms': retrieval_ms,
                'rerank_ms': rerank_ms,
                'generation_ms': generation_ms,
                'chunks_retrieved': len(chunks),
                'chunks_after_rerank': len(top_chunks),
                'best_score': float(score_meta['best']),
                'avg_score': float(score_meta['avg']),
                'reranker': reranker_type,
                'model': resolved_model,
                'provider': resolved_provider,
                'top_chunks': [
                    {
                        'chunk_id': str(c.get('chunk_id', '')),
                        'document_id': str(c.get('document_id', '')),
                        'page_num': c.get('page_num'),
                        'score': float(c.get('rerank_score', c.get('score', 0))),
                        'source': str(c.get('source', '')),
                        'text_preview': str(c.get('text', ''))[:100],
                        'char_start': c.get('char_start'),
                        'char_end': c.get('char_end'),
                    }
                    for c in top_chunks
                ],
            }
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'debug': debug_payload})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except PartialStreamError as e:
            logger.error(str(e))
            if trace:
                try: trace.update(metadata={"error": "partial_stream", "chunks_sent": e.chunks_yielded})
                except Exception: pass
            yield f"data: {json.dumps({'type': 'error', 'error_type': 'partial_stream', 'message': 'The response was cut off midway. Please try again.', 'partial': True, 'chunks_sent': e.chunks_yielded})}\n\n"
        except ProviderNotAvailable as e:
            # Same discipline as the non-streaming endpoint: a clear error
            # event, never a silent switch to a different provider — see
            # GeneratorRouter's own docstring. Both providers can stream
            # now, so this only fires for the same reasons the non-
            # streaming endpoint's 400 does: cloud disabled by the
            # administrator, no DEEPSEEK_API_KEY configured, or an unknown
            # provider name — never "provider=deepseek on /query/stream"
            # by itself anymore.
            logger.warning(f"Stream provider request rejected: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error_type': 'provider_not_available', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            if trace:
                try: trace.update(metadata={"error": str(e)})
                except Exception: pass
            yield f"data: {json.dumps({'type': 'error', 'message': 'Could not get a response from the model. Please try again.'})}\n\n"
        finally:
            m._query_semaphore.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
