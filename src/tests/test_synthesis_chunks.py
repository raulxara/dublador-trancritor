from app.engines.text_chunks import synthesis_chunks


def test_long_unpunctuated_speech_preserves_words_with_bounded_phrases():
    text = "Esta é uma fala longa sem pausas " * 50
    chunks = list(synthesis_chunks(text))
    assert len(chunks) > 1
    assert all(len(part) <= 120 for part in chunks)
    assert " ".join(chunks).split() == text.split()


def test_empty_and_long_token():
    assert list(synthesis_chunks("  ")) == []
    chunks = list(synthesis_chunks("a" * 301))
    assert all(len(part) <= 120 for part in chunks)
    assert "".join(chunks) == "a" * 301
