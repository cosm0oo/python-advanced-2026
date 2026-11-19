from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp

API_BASE = "https://jsonplaceholder.typicode.com"
DEFAULT_OUTPUT_DIR = "./output"


async def _fetch_json(url: str, params: dict[str, Any] | None = None) -> Any:
    timeout = aiohttp.ClientTimeout(total=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()


async def fetch_user_data(user_id: int) -> dict[str, Any]:
    """Fetch one user profile from the external API."""
    # requests.get(...) блокирует event loop на время сети.
    # aiohttp не блокирует и отдаёт управление другим корутинам.
    payload = await _fetch_json(f"{API_BASE}/users/{user_id}")
    if not isinstance(payload, dict):
        raise ValueError("Expected user payload to be a JSON object")
    return payload


async def fetch_user_posts(user_id: int) -> list[dict[str, Any]]:
    """Fetch all posts for a specific user."""
    # Синхронный HTTP-вызов блокирует event loop.
    # Используем aiohttp и await для неблокирующего I/O.
    payload = await _fetch_json(f"{API_BASE}/posts", params={"userId": user_id})
    if not isinstance(payload, list):
        raise ValueError("Expected posts payload to be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


async def save_to_file(
    data: dict[str, Any] | list[dict[str, Any]] | list[Any], filepath: str
) -> None:
    """Save JSON-serializable data to a UTF-8 file asynchronously."""
    # open(...)/json.dump(...) в async-функции блокирует loop.
    # aiofiles выполняет файловый I/O без блокировки event loop.
    serialized = json.dumps(data, indent=2, ensure_ascii=False)
    async with aiofiles.open(filepath, mode="w", encoding="utf-8") as file:
        await file.write(serialized)


async def load_from_file(filepath: str) -> dict[str, Any] | list[Any]:
    """Load JSON data from a UTF-8 file asynchronously."""
    # Синхронное чтение файла блокирует event loop.
    # async with aiofiles.open(...) позволяет выполнять другие корутины.
    async with aiofiles.open(filepath, encoding="utf-8") as file:
        content = await file.read()
    parsed = json.loads(content)
    if not isinstance(parsed, (dict, list)):
        raise ValueError("Unexpected JSON root type")
    return parsed


def _compute_statistics_sync(users: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    for value in range(1_000_000):
        total += value * value

    names = sorted(str(user.get("name", "")) for user in users)
    companies = {
        str(user.get("company", {}).get("name", "N/A"))
        for user in users
        if isinstance(user.get("company", {}), dict)
    }

    return {
        "user_count": len(users),
        "checksum": total,
        "names": names,
        "unique_companies": sorted(companies),
    }


async def compute_statistics(users: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregated statistics for a list of users."""
    # CPU-bound цикл в корутине блокирует event loop.
    # Переносим вычисление в отдельный поток через asyncio.to_thread.
    return await asyncio.to_thread(_compute_statistics_sync, users)


async def enrich_user(user: dict[str, Any]) -> dict[str, Any]:
    """Enrich a user profile with posts and post count."""
    user_id_raw = user.get("id")
    if not isinstance(user_id_raw, int):
        raise ValueError("User object must contain integer 'id'")

    posts = await fetch_user_posts(user_id_raw)
    user["posts"] = posts
    user["post_count"] = len(posts)
    return user


async def wait_for_service(url: str, timeout: float = 2.0) -> bool:
    """Wait for external service availability signal."""
    # time.sleep(...) полностью блокирует event loop.
    # await asyncio.sleep(...) приостанавливает только текущую корутину.
    await asyncio.sleep(timeout)
    return bool(url)


async def log_message(message: str, logfile: str = "service.log") -> None:
    """Append a timestamped message to the service log asynchronously."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    async with aiofiles.open(logfile, mode="a", encoding="utf-8") as file:
        await file.write(f"[{timestamp}] {message}\n")


async def process_users(
    user_ids: list[int], output_dir: str = DEFAULT_OUTPUT_DIR
) -> list[dict[str, Any]]:
    """Run full pipeline: fetch, enrich, persist, and summarize users."""
    await asyncio.to_thread(Path(output_dir).mkdir, parents=True, exist_ok=True)

    await log_message("Pipeline started")
    await wait_for_service(API_BASE)

    users = await asyncio.gather(*(fetch_user_data(user_id) for user_id in user_ids))
    await log_message(f"Fetched {len(users)} users")

    enriched = await asyncio.gather(*(enrich_user(user) for user in users))

    for user in enriched:
        filepath = Path(output_dir) / f"user_{user['id']}.json"
        await save_to_file(user, str(filepath))

    stats = await compute_statistics(enriched)
    await save_to_file(stats, str(Path(output_dir) / "stats.json"))

    await log_message("Pipeline finished")
    return enriched


async def main() -> None:
    """Run the processing pipeline for a small default user set."""
    user_ids = [1, 2, 3, 4, 5]
    users = await process_users(user_ids)
    print(f"Processed {len(users)} users")
    for user in users:
        print(f"  - {user['name']}: {user['post_count']} posts")


if __name__ == "__main__":
    asyncio.run(main())
