"""
Sparse (BM25-style) vector encoding for Qdrant hybrid search.

Term frequency only — Qdrant applies IDF weighting server-side at query time
via the collection's sparse vector Modifier.IDF, using corpus-wide stats it
maintains incrementally. No client-side global index or rebuild needed.
"""
import hashlib
import re
from collections import Counter

from qdrant_client.models import SparseVector

_CYRILLIC_RE = re.compile(r"[а-яё]")

# Words that carry no search signal
STOP_WORDS_EN = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "as", "if", "not", "no", "nor",
    "so", "yet", "both", "either", "each", "any", "all", "some",
}

STOP_WORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под",
    "будет", "ж", "тогда", "кто", "этот", "того", "потому", "этого",
    "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "были", "куда", "зачем", "всех", "никогда",
    "можно", "при", "наконец", "два", "об", "другой", "хоть", "после",
    "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "него", "какая", "много", "разве", "три", "эту", "моя", "впрочем",
    "хорошо", "свою", "этой", "перед", "иногда", "лучше", "чуть",
    "том", "нельзя", "такой", "им", "более", "всегда", "конечно",
    "всю", "между",
}

STOP_WORDS = STOP_WORDS_EN | STOP_WORDS_RU

# Feature-hashing bucket count for sparse vector indices. Large enough that
# collisions are rare for a corpus vocabulary of a few hundred thousand terms.
_HASH_BUCKETS = 2 ** 31 - 1


def _stem(token: str) -> str:
    for suffix in ("ment", "tion", "ing", "ness", "ies", "ied", "ed", "er", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words. Stems English tokens only."""
    lowered = text.lower()
    tokens = re.sub(r"[^a-zа-яё0-9\s]", " ", lowered).split()
    result = []
    for t in tokens:
        if len(t) <= 1 or t in STOP_WORDS:
            continue
        # Stem only ASCII (English) tokens; Russian morphology is too complex for a suffix stripper
        result.append(t if _CYRILLIC_RE.search(t) else _stem(t))
    return result


def _stable_token_index(token: str) -> int:
    # hashlib, not Python's builtin hash() — builtin hash() is randomized per
    # process via PYTHONHASHSEED and would silently desync the sparse index
    # across restarts/replicas.
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _HASH_BUCKETS


def build_sparse_vector(text: str) -> SparseVector:
    """Raw term-frequency sparse vector. IDF weighting applied by Qdrant at query time."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVector(indices=[], values=[])
    counts = Counter(_stable_token_index(t) for t in tokens)
    return SparseVector(indices=list(counts.keys()), values=[float(v) for v in counts.values()])
