"""
Pydantic request/response models shared across api/main.py's query endpoints.
Split out of main.py purely to shrink that file — these have no dependency
on anything else in it and no behavior of their own beyond validation.
"""
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Bounds exist to close two separate problems, not just "reject huge input":
# 1. rag/prompt_builder.py's MAX_CONTEXT_CHARS/MAX_HISTORY_CHARS budgets are
#    enforced by silently truncating — a single chat_history turn bigger than
#    the whole budget survives _trim_history() untouched (it only drops
#    whole turns, never shortens one), so it alone can crowd out the actual
#    retrieved context. Capping turn/question length here means that
#    truncation logic is never the only thing standing between a client and
#    a degenerate prompt.
# 2. Every accepted field ends up either inside the LLM prompt (question,
#    chat_history) or used as a lookup key (document_id/document_ids) — an
#    unbounded list of either is free amplification (n turns * m chars each
#    all headed into one prompt; n document IDs all headed into a Qdrant
#    filter).
MAX_QUESTION_CHARS = 2000
MAX_CHAT_HISTORY_TURNS = 20
MAX_TURN_CONTENT_CHARS = 4000
MAX_DOCUMENT_IDS = 50
MAX_DOCUMENT_ID_CHARS = 200


class ChatTurn(BaseModel):
    # Was `role: str` — accepted literally any string, including "system".
    # chat_history turns are passed straight through to the LLM's messages
    # list (api/main.py -> rag/prompt_builder.py's `messages.append(turn)`)
    # with no re-labeling, so a client-supplied {"role": "system", "content":
    # "..."} was indistinguishable from this server's own system prompt by
    # the time it reached the model. Restricting to the two roles a chat
    # turn can legitimately be closes that off at the validation layer
    # (422) instead of relying on the LLM to notice.
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_TURN_CONTENT_CHARS)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
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
    document_id: Optional[str] = Field(default=None, max_length=MAX_DOCUMENT_ID_CHARS)
    # Restricts retrieval to exactly these documents — used by the frontend's
    # "compare N specific documents" flow. Previously that flow only ever
    # *hinted* at scope via filenames spelled out in the question text plus
    # a folder filter, which left hybrid search (and the reranker) free to
    # pull in other documents from the same folder — see
    # rag/retriever.py::retrieve_expanded's document_ids param.
    document_ids: Optional[list[Annotated[str, Field(max_length=MAX_DOCUMENT_ID_CHARS)]]] = Field(
        default=None, max_length=MAX_DOCUMENT_IDS
    )
    chat_history: Optional[list[ChatTurn]] = Field(default=[], max_length=MAX_CHAT_HISTORY_TURNS)
    model: Optional[str] = None
    rerank: Optional[bool] = True
    folder: Optional[str] = None
    # Which generator backend answers this specific request — "local"
    # (default, Qwen via Ollama, nothing leaves this server) or "deepseek"
    # (opt-in cloud mode, requires ENABLE_CLOUD_GENERATOR=true on the
    # server AND this field set explicitly — see rag/generator.py's
    # GeneratorRouter, the single place that enforces both gates). A
    # Literal, not a bare str, so an unknown provider name is a 422 at
    # validation time, not a confusing runtime error deep in the router.
    provider: Optional[Literal["local", "deepseek"]] = None

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, v: str) -> str:
        # min_length=1 only rejects an empty string outright — " " or
        # "\n\t" still satisfy it and used to reach api/main.py's own
        # `if not request.question.strip(): raise HTTPException(400, ...)`
        # — a second check, at the endpoint, for what's really the same
        # validation failure this model should reject directly (422).
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    model: str
    # "local" or "deepseek" — which generator actually answered, so a caller
    # never has to infer this from the model name string. Defaults to
    # "local" for the two early-return paths (no chunks / below relevance
    # threshold) that never invoke a generator at all — see GeneratorRouter.
    provider: str = "local"
    tokens_used: int
    debug: Optional[dict] = None
