"""
Pydantic request/response models shared across api/main.py's query endpoints.
Split out of main.py purely to shrink that file — these have no dependency
on anything else in it and no behavior of their own beyond validation.
"""
from typing import Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    # Hard cap regardless of caller — retrieve_expanded() sizes its candidate
    # pool off this (max(20, top_k*5)), so an unbounded top_k (e.g. a
    # frontend "compare all documents in a 150-doc folder" bug computing
    # top_k = doc_count*4) turns into a multi-thousand-candidate rerank pass.
    # MAX_CONTEXT_CHARS (prompt_builder.py) already truncates the prompt to
    # ~3000 tokens regardless, so nothing above this cap could even reach
    # the LLM — it would just be wasted retrieval/rerank work.
    # Deliberately not Optional[int] — Optional[int] = Field(..., ge=1, le=20)
    # only enforces ge/le when a value is actually given; an explicit
    # "top_k": null in the request body still passes validation as None,
    # and request.top_k * 5 downstream then throws a bare TypeError -> 500.
    # A plain int with a default rejects null outright (422) instead.
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[str] = None
    # Restricts retrieval to exactly these documents — used by the frontend's
    # "compare N specific documents" flow. Previously that flow only ever
    # *hinted* at scope via filenames spelled out in the question text plus
    # a folder filter, which left hybrid search (and the reranker) free to
    # pull in other documents from the same folder — see
    # rag/retriever.py::retrieve_expanded's document_ids param.
    document_ids: Optional[list[str]] = None
    chat_history: Optional[list[ChatTurn]] = []
    model: Optional[str] = None
    rerank: Optional[bool] = True
    folder: Optional[str] = None
    channel: Optional[str] = None   # "telegram" | None (web)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    tokens_used: int
    debug: Optional[dict] = None
