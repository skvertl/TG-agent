import pytest
from adapters.telegram.splitter import split_message

class TestSplitMessage:
    def test_empty_string(self):
        assert split_message("") == []

    def test_short_text_no_split(self):
        text = "Hello world! This is a short message."
        chunks = split_message(text, max_chunk_size=100)
        assert chunks == [text]

    def test_exact_limit(self):
        text = "a" * 100
        chunks = split_message(text, max_chunk_size=100)
        assert chunks == [text]

    def test_split_on_paragraphs(self):
        p1 = "First paragraph." * 5  # ~80 chars
        p2 = "Second paragraph." * 5 # ~85 chars
        text = f"{p1}\n\n{p2}"
        chunks = split_message(text, max_chunk_size=100)
        assert len(chunks) == 2
        assert chunks[0] == p1
        assert chunks[1] == p2

    def test_split_on_newlines(self):
        line1 = "Line number one is relatively long" # 34 chars
        line2 = "Line number two is also long text" # 33 chars
        text = f"{line1}\n{line2}"
        chunks = split_message(text, max_chunk_size=40)
        assert len(chunks) == 2
        assert chunks[0] == line1
        assert chunks[1] == line2

    def test_split_on_space(self):
        words = "word1 word2 word3 word4 word5 word6"
        chunks = split_message(words, max_chunk_size=15)
        for chunk in chunks:
            assert len(chunk) <= 15
        assert " ".join(chunks) == words

    def test_hard_split_without_spaces(self):
        long_token = "abcdefghijklmnopqrstuvwxyz"
        chunks = split_message(long_token, max_chunk_size=10)
        assert chunks == ["abcdefghij", "klmnopqrst", "uvwxyz"]

    def test_preserves_markdown_code_blocks(self):
        code_body = "x = 1\n" * 20 # 120 chars
        text = f"```python\n{code_body}```"
        chunks = split_message(text, max_chunk_size=60)
        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= 60
            if i == 0:
                assert chunk.startswith("```python\n")
                assert chunk.endswith("```")
            elif i == len(chunks) - 1:
                assert chunk.startswith("```python\n")
                assert chunk.endswith("```")
            else:
                assert chunk.startswith("```python\n")
                assert chunk.endswith("```")
