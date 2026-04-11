"""Contextual Chunk Augmentation — enrich chunks with metadata before embedding.

Inspired by Anthropic's "Contextual Retrieval" paper: prepending document
context to chunks before embedding reduces retrieval failures by up to 49%.

The augmented text is used for embedding only; the original text is kept
in the payload for display.  This is a pure function with no external
dependencies (no LLM calls for the basic variant).
"""

from pre_extractor import pre_extract


def augment_chunk(
    text: str,
    agent_id: str = "",
    file_path: str = "",
    date: str = "",
    topic: str = "",
    thought_type: str = "",
) -> str:
    """Prepend a contextual header to chunk text for embedding.

    The header includes structured metadata (agent, file, date, topic)
    and key phrases extracted deterministically from the text.
    The original text follows after a separator.
    """
    parts: list[str] = []

    meta_fields: list[str] = []
    if agent_id:
        meta_fields.append(f"Agent: {agent_id}")
    if file_path:
        meta_fields.append(f"File: {file_path}")
    if date:
        meta_fields.append(f"Date: {date}")
    if topic:
        meta_fields.append(f"Topic: {topic}")
    if thought_type and thought_type != "general":
        meta_fields.append(f"Type: {thought_type}")

    if meta_fields:
        parts.append("[" + " | ".join(meta_fields) + "]")

    hints = pre_extract(text)
    entities = hints.get("entities", [])
    if entities:
        names = [e["name"] for e in entities[:8]]
        parts.append("Key entities: " + ", ".join(names))

    dates = hints.get("dates", [])
    if dates:
        parts.append("Dates: " + ", ".join(dates[:5]))

    if parts:
        return "\n".join(parts) + "\n---\n" + text
    return text


def strip_augmentation_header(text: str) -> str:
    """Extract the raw content from augmented text by removing the metadata header.

    The header is separated from content by a ``---\\n`` delimiter.
    If no delimiter is found, the text is returned unchanged.
    """
    sep = "\n---\n"
    idx = text.find(sep)
    if idx == -1:
        return text
    return text[idx + len(sep):]
