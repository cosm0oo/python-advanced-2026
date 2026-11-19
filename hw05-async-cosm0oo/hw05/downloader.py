from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

import aiofiles
import aiohttp
from tqdm.asyncio import tqdm_asyncio


def _filename_from_url(url: str) -> str:
    name = Path(urlsplit(url).path).name
    if name:
        return name
    digest = hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"download_{digest}.bin"


async def _download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    destination: Path,
) -> tuple[str, bool]:
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status >= 400:
                    return url, False
                payload = await response.read()
        except (aiohttp.ClientError, TimeoutError, OSError):
            return url, False

        file_path = destination / _filename_from_url(url)
        async with aiofiles.open(file_path, mode="wb") as file:
            await file.write(payload)
        return url, True


async def download_all(
    urls: list[str],
    dest_dir: str | Path,
    max_concurrent: int = 5,
) -> dict[str, bool]:
    """Download URLs concurrently with a semaphore and return per-URL status."""
    if max_concurrent <= 0:
        raise ValueError("max_concurrent must be > 0")

    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not urls:
        return {}

    timeout = aiohttp.ClientTimeout(total=10.0)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            _download_one(
                session=session,
                semaphore=semaphore,
                url=url,
                destination=destination,
            )
            for url in urls
        ]
        results = await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="download")

    return {url: is_success for url, is_success in results}
