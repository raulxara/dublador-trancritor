import re


def synthesis_chunks(text, limit=120):
    """Bound synthesis length without dropping words or relying on punctuation."""
    words = re.findall(r"\S+", text)
    chunk = ""
    for word in words:
        if chunk and len(chunk) + len(word) + 1 > limit:
            yield chunk
            chunk = ""
        # A single long token still needs a bound for the model tokenizer.
        while len(word) > limit:
            if chunk:
                yield chunk
                chunk = ""
            yield word[:limit]
            word = word[limit:]
        chunk = (chunk + " " + word).strip()
    if chunk:
        yield chunk
