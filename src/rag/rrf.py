"""Reciprocal Rank Fusion for Hybrid Search."""


def chunk_identity(chunk: dict) -> tuple:
    """Stable Chunk Identity across search lanes."""
    return (chunk["source"], chunk["chunk_id"])


def rrf_fuse(ranked_lists, rrf_k: int = 60, top_k: int = 5):
    """
    Merge ranked Chunk lists with RRF.

    Score is sum of 1/(rrf_k + rank) per lane; ranks are 1-based best-first.
    Dedupes by Chunk Identity (source, chunk_id). Single-lane hits remain eligible.
    """
    scores = {}
    by_id = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            cid = chunk_identity(chunk)
            by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

    ordered = sorted(scores.keys(), key=lambda cid: (-scores[cid], cid))
    return [by_id[cid] for cid in ordered[:top_k]]
