from __future__ import annotations

from codecs import getincrementaldecoder
from collections.abc import AsyncIterator
from pathlib import Path

import aiofiles

VARIANT_SOURCES = {
    0: "server_log",
    1: "json_events",
    2: "csv_metrics",
}


async def read_chunks(path: str | Path, chunk_size: int = 8192) -> AsyncIterator[bytes]:
    """Read file content asynchronously in fixed-size byte chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    async with aiofiles.open(path, mode="rb") as file:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def parse_lines(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Reconstruct full text lines from a byte-chunk async iterator."""
    decoder = getincrementaldecoder("utf-8")()
    buffer = ""

    async for chunk in chunks:
        buffer += decoder.decode(chunk)
        while True:
            newline_index = buffer.find("\n")
            if newline_index < 0:
                break
            line = buffer[:newline_index]
            if line.endswith("\r"):
                line = line[:-1]
            yield line
            buffer = buffer[newline_index + 1 :]

    tail = buffer + decoder.decode(b"", final=True)
    if tail:
        if tail.endswith("\r"):
            tail = tail[:-1]
        yield tail


async def filter_lines(lines: AsyncIterator[str], pattern: str) -> AsyncIterator[str]:
    """Yield only lines that contain the provided pattern."""
    async for line in lines:
        if pattern in line:
            yield line


async def batch(items: AsyncIterator[str], size: int) -> AsyncIterator[list[str]]:
    """Group async stream items into batches of at most size elements."""
    if size <= 0:
        raise ValueError("size must be > 0")

    items_batch: list[str] = []
    async for item in items:
        items_batch.append(item)
        if len(items_batch) >= size:
            yield items_batch
            items_batch = []

    if items_batch:
        yield items_batch
