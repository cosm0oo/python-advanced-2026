"""Task 4.2: sequential and parallel Monte Carlo integration for pi."""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter


def _count_inside_unit_quarter_circle(task: tuple[int, int]) -> int:
    n_samples, seed = task
    rng = random.Random(seed)
    next_random = rng.random
    inside = 0
    for _ in range(n_samples):
        x = next_random()
        y = next_random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def _split_samples(n_samples: int, n_parts: int) -> list[int]:
    base, remainder = divmod(n_samples, n_parts)
    chunks = [base + (1 if idx < remainder else 0) for idx in range(n_parts)]
    return [chunk for chunk in chunks if chunk > 0]


def monte_carlo_pi(n_samples: int) -> float:
    """Estimate pi using sequential Monte Carlo integration."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    inside = _count_inside_unit_quarter_circle((n_samples, 1_000_003))
    return 4.0 * inside / n_samples


def monte_carlo_pi_parallel(n_samples: int, n_workers: int | None = None) -> float:
    """Estimate pi using parallel Monte Carlo integration with ``ProcessPoolExecutor``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")

    workers = n_workers if n_workers is not None else min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, n_samples))

    chunks = _split_samples(n_samples, workers)
    tasks = [(chunk, 1_000_003 + idx * 97_531) for idx, chunk in enumerate(chunks)]
    mp_context = None
    if "fork" in mp.get_all_start_methods():
        mp_context = mp.get_context("fork")

    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=mp_context) as executor:
        inside_total = sum(executor.map(_count_inside_unit_quarter_circle, tasks))

    return 4.0 * inside_total / n_samples


def compare_performance(
    samples_list: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Compare sequential and parallel Monte Carlo implementations."""
    values = samples_list or [10**5, 10**6, 10**7]
    comparison: dict[int, dict[str, float]] = {}

    for n_samples in values:
        start = perf_counter()
        sequential_pi = monte_carlo_pi(n_samples)
        sequential_time = perf_counter() - start

        start = perf_counter()
        parallel_pi = monte_carlo_pi_parallel(n_samples)
        parallel_time = perf_counter() - start

        speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
        comparison[n_samples] = {
            "sequential_time": float(sequential_time),
            "parallel_time": float(parallel_time),
            "sequential_pi": float(sequential_pi),
            "parallel_pi": float(parallel_pi),
            "speedup": float(speedup),
        }

    return comparison
