"""
RLM Utilities
=============
Helper functions for file loading and text splitting.
"""

import os
import tiktoken


def load_file(path: str) -> str:
    """
    Read and return the full text content of a file.

    Args:
        path: Absolute or relative path to a text file.

    Returns:
        The file contents as a single string.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text(text: str, chunk_size: int = 4000, model: str = "cl100k_base") -> list[str]:
    """
    Split text into chunks of approximately `chunk_size` tokens.

    Uses tiktoken to count tokens so chunks align with LLM context limits.

    Args:
        text:       The input text to split.
        chunk_size: Maximum number of tokens per chunk.
        model:      Tiktoken encoding name (default: cl100k_base).

    Returns:
        A list of text chunks, each ≤ chunk_size tokens.
    """
    enc = tiktoken.get_encoding(model)
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i : i + chunk_size]
        chunks.append(enc.decode(chunk_tokens))

    return chunks
