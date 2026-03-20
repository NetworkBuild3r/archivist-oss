"""Pure retrieval filters — shared by rlm_retriever and tests."""


def apply_retrieval_threshold(results: list[dict], threshold: float) -> list[dict]:
    """Filter out results with vector score below threshold."""
    return [r for r in results if r["score"] >= threshold]
