"""In-memory BM25 Search over Chunk text."""
from rank_bm25 import BM25Okapi

from rag.lexical import tokenize_normalized


class Bm25Index:
    """BM25 index built from the loaded chunk list (no on-disk FTS artifact)."""

    def __init__(self, chunks, tokenize=None):
        self.chunks = chunks
        self.tokenize = tokenize or tokenize_normalized
        self._corpus = [self.tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus) if chunks else None

    def search(self, query: str, k: int):
        if not self.chunks or self._bm25 is None or k <= 0:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        query_terms = set(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            # Tiny corpora can yield negative BM25 for true matches; require term overlap
            # so unrelated zero-score rows never enter RRF as fake ranks.
            if not query_terms.intersection(self._corpus[i]):
                continue
            hits.append(self.chunks[i])
            if len(hits) >= k:
                break
        return hits
