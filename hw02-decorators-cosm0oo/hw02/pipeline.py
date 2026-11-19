from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")
T = TypeVar("T")

MISSING = object()


@overload
def pipe(__f1: Callable[P, A], /) -> Callable[P, A]: ...


@overload
def pipe(__f1: Callable[P, A], __f2: Callable[[A], B], /) -> Callable[P, B]: ...


@overload
def pipe(
    __f1: Callable[P, A],
    __f2: Callable[[A], B],
    __f3: Callable[[B], C],
    /,
) -> Callable[P, C]: ...


@overload
def pipe(
    __f1: Callable[P, A],
    __f2: Callable[[A], B],
    __f3: Callable[[B], C],
    __f4: Callable[[C], D],
    /,
) -> Callable[P, D]: ...


def pipe(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """Apply functions left to right."""
    if not fns:

        def identity(value: T) -> T:
            return value

        return identity

    first, *rest = fns

    def run(*args: Any, **kwargs: Any) -> Any:
        result = first(*args, **kwargs)
        for fn in rest:
            result = fn(result)
        return result

    return run


@overload
def compose(__f1: Callable[P, A], /) -> Callable[P, A]: ...


@overload
def compose(__f1: Callable[[A], B], __f2: Callable[P, A], /) -> Callable[P, B]: ...


@overload
def compose(
    __f1: Callable[[B], C],
    __f2: Callable[[A], B],
    __f3: Callable[P, A],
    /,
) -> Callable[P, C]: ...


@overload
def compose(
    __f1: Callable[[C], D],
    __f2: Callable[[B], C],
    __f3: Callable[[A], B],
    __f4: Callable[P, A],
    /,
) -> Callable[P, D]: ...


def compose(*fns: Callable[..., Any]) -> Callable[..., Any]:
    """Apply functions right to left."""
    return pipe(*reversed(fns))


def _read_value(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, MISSING)
    return getattr(item, key, MISSING)


def filter_by(**kwargs: Any) -> Callable[[list[Any]], list[Any]]:
    """Return a closure filtering dicts/objects by provided fields."""

    def run(items: list[Any]) -> list[Any]:
        def matches(item: Any) -> bool:
            for key, expected in kwargs.items():
                if _read_value(item, key) != expected:
                    return False
            return True

        return [item for item in items if matches(item)]

    return run


def sort_by(key: str, *, reverse: bool = False) -> Callable[[list[Any]], list[Any]]:
    """Return a closure sorting dicts/objects by a field."""

    def extract(item: Any) -> Any:
        value = _read_value(item, key)
        if value is MISSING:
            if isinstance(item, Mapping):
                raise KeyError(key)
            raise AttributeError(key)
        return value

    def run(items: list[Any]) -> list[Any]:
        return sorted(items, key=extract, reverse=reverse)

    return run


def take(n: int) -> Callable[[list[Any]], list[Any]]:
    """Return a closure taking first n elements."""
    limit = max(0, n)

    def run(items: list[Any]) -> list[Any]:
        return items[:limit]

    return run
