def split_text(text, max_words=500):
    """
    Split long research text into manageable word-based chunks.

    Args:
        text: Input research text.
        max_words: Maximum approximate number of words per chunk.

    Returns:
        List of text chunks.
    """

    words = text.split()

    chunks = []

    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start:start + max_words])

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def summarize_text(text, summarizer):
    """
    Generate a summary for biomedical research text.

    Long documents are summarized chunk-by-chunk and the
    intermediate summaries are combined into a final summary.

    Args:
        text: Research text.
        summarizer: Loaded Transformer summarization pipeline.

    Returns:
        Generated summary string.
    """

    text = text.strip()

    if not text:
        return ""

    # Very short text does not need summarization.
    if len(text.split()) < 80:
        return text

    chunks = split_text(text)

    chunk_summaries = []

    for chunk in chunks:

        result = summarizer(
            chunk,
            max_length=120,
            min_length=30,
            do_sample=False
        )

        chunk_summary = result[0]["summary_text"].strip()

        if chunk_summary:
            chunk_summaries.append(chunk_summary)

    if not chunk_summaries:
        return ""

    # If there is only one chunk, return its summary directly.
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Combine intermediate summaries.
    combined_summary = " ".join(chunk_summaries)

    # If the combined summary is still long, summarize it again.
    if len(combined_summary.split()) > 180:

        final_result = summarizer(
            combined_summary,
            max_length=180,
            min_length=60,
            do_sample=False
        )

        return final_result[0]["summary_text"].strip()

    return combined_summary
