"""Task 4.1: compare parallelism approaches for fib(35)."""

from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
from time import perf_counter
from typing import Any

InterpreterPoolExecutor: type[Any] | None = getattr(
    concurrent.futures,
    "InterpreterPoolExecutor",
    None,
)


def _call_with_args(payload: tuple[Callable[..., Any], tuple[Any, ...]]) -> Any:
    fn, args = payload
    return fn(*args)


def _build_payload(
    fn: Callable[..., Any],
    args_list: list[tuple[Any, ...]],
) -> list[tuple[Callable[..., Any], tuple[Any, ...]]]:
    return [(fn, tuple(args)) for args in args_list]


def _workers_for_tasks(task_count: int) -> int:
    return max(1, min(task_count, os.cpu_count() or 1))


def fib(n: int) -> int:
    """Compute Fibonacci number recursively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def run_threading(fn: Callable[..., Any], args_list: list[tuple[Any, ...]]) -> list[Any]:
    """Compute ``fn(*args)`` for each ``args`` using ``threading.Thread``."""
    results: list[Any] = [None] * len(args_list)
    threads: list[Thread] = []

    # Для CPU-bound задач потоки в CPython не дают реального ускорения из-за GIL:
    # в один момент времени Python-байткод исполняет только один поток.
    # Начиная с Python 3.12 у каждого субинтерпретатора свой GIL, а в 3.14
    # это удобно использовать через InterpreterPoolExecutor для параллельного
    # выполнения Python-байткода на нескольких ядрах.
    def worker(index: int, args: tuple[Any, ...]) -> None:
        results[index] = fn(*args)

    for index, args in enumerate(args_list):
        thread = Thread(target=worker, args=(index, tuple(args)))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results


def run_multiprocessing_pool(fn: Callable[..., Any], args_list: list[tuple[Any, ...]]) -> list[Any]:
    """Compute ``fn(*args)`` for each ``args`` using ``multiprocessing.Pool``."""
    if not args_list:
        return []
    payload = _build_payload(fn, args_list)
    with mp.Pool(processes=_workers_for_tasks(len(payload))) as pool:
        return list(pool.map(_call_with_args, payload))


def run_executor(fn: Callable[..., Any], args_list: list[tuple[Any, ...]]) -> list[Any]:
    """Compute ``fn(*args)`` for each ``args`` using ``ProcessPoolExecutor``."""
    if not args_list:
        return []
    payload = _build_payload(fn, args_list)
    with ProcessPoolExecutor(max_workers=_workers_for_tasks(len(payload))) as executor:
        return list(executor.map(_call_with_args, payload))


def run_interpreters(fn: Callable[..., Any], args_list: list[tuple[Any, ...]]) -> list[Any]:
    """Compute ``fn(*args)`` for each ``args`` using subinterpreters."""
    if not args_list:
        return []
    if InterpreterPoolExecutor is None:
        return run_executor(fn, args_list)
    payload = _build_payload(fn, args_list)
    with InterpreterPoolExecutor(max_workers=_workers_for_tasks(len(payload))) as executor:
        return list(executor.map(_call_with_args, payload))


def compare_all(n: int = 35, repeat: int = 8) -> dict[str, dict[str, list[int] | float]]:
    """Run all approaches and return timing comparison data."""
    if repeat < 1:
        raise ValueError("repeat must be >= 1")

    args_list: list[tuple[int, ...]] = [(n,)] * repeat
    comparison: dict[str, dict[str, list[int] | float]] = {}

    start = perf_counter()
    sequential_results = [fib(n) for _ in range(repeat)]
    sequential_time = perf_counter() - start
    comparison["sequential"] = {
        "results": sequential_results,
        "time": float(sequential_time),
        "speedup": 1.0,
    }

    runners: list[tuple[str, Callable[[Callable[..., int], list[tuple[int, ...]]], list[int]]]] = [
        ("threading", run_threading),
        ("multiprocessing", run_multiprocessing_pool),
        ("executor", run_executor),
        ("interpreters", run_interpreters),
    ]

    for method_name, runner in runners:
        start = perf_counter()
        method_results = runner(fib, args_list)
        elapsed = perf_counter() - start
        speedup = sequential_time / elapsed if elapsed > 0 else float("inf")
        comparison[method_name] = {
            "results": method_results,
            "time": float(elapsed),
            "speedup": float(speedup),
        }

    return comparison
