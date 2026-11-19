from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Hashable
from functools import wraps
from types import NoneType, UnionType
from typing import Any, TypeVar, Union, cast, get_args, get_origin, get_type_hints

F = TypeVar("F", bound=Callable[..., Any])
MISSING = object()


def _format_expected_type(expected: Any) -> str:
    origin = get_origin(expected)
    if expected is Any:
        return "Any"
    if expected in {None, NoneType}:
        return "None"
    if origin in {UnionType, Union}:
        return " | ".join(_format_expected_type(item) for item in get_args(expected))
    if hasattr(expected, "__name__"):
        return cast(str, expected.__name__)
    text = str(expected)
    return text.replace("typing.", "")


def _is_instance_of_annotation(value: Any, annotation: Any) -> bool:
    if annotation is Any:
        return True
    if annotation in {None, NoneType}:
        return value is None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in {UnionType, Union}:
        return any(_is_instance_of_annotation(value, option) for option in args)

    if origin is list:
        if not isinstance(value, list):
            return False
        if not args:
            return True
        item_type = args[0]
        return all(_is_instance_of_annotation(item, item_type) for item in value)

    if origin is dict:
        if not isinstance(value, dict):
            return False
        if len(args) != 2:
            return True
        key_type, value_type = args
        return all(
            _is_instance_of_annotation(item_key, key_type)
            and _is_instance_of_annotation(item_value, value_type)
            for item_key, item_value in value.items()
        )

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if not args:
            return True
        if len(args) == 2 and args[1] is Ellipsis:
            return all(_is_instance_of_annotation(item, args[0]) for item in value)
        if len(value) != len(args):
            return False
        pairs = zip(value, args)
        return all(_is_instance_of_annotation(item, item_type) for item, item_type in pairs)

    if origin is set:
        if not isinstance(value, set):
            return False
        if not args:
            return True
        item_type = args[0]
        return all(_is_instance_of_annotation(item, item_type) for item in value)

    if origin is frozenset:
        if not isinstance(value, frozenset):
            return False
        if not args:
            return True
        item_type = args[0]
        return all(_is_instance_of_annotation(item, item_type) for item in value)

    if origin is Callable:
        return callable(value)

    if origin is not None:
        try:
            return isinstance(value, origin)
        except TypeError:
            return True

    try:
        return isinstance(value, annotation)
    except TypeError:
        return True


def validate_types(func: F) -> F:
    """Validate function arguments and return value against annotations in runtime."""
    signature = inspect.signature(func)
    annotations = get_type_hints(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        for name, value in bound.arguments.items():
            expected = annotations.get(name, MISSING)
            if expected is MISSING:
                continue
            if not _is_instance_of_annotation(value, expected):
                expected_name = _format_expected_type(expected)
                actual_name = type(value).__name__
                raise TypeError(f"argument '{name}' expected {expected_name}, got {actual_name}")

        result = func(*args, **kwargs)
        expected_return = annotations.get("return", MISSING)
        if expected_return is not MISSING and not _is_instance_of_annotation(
            result, expected_return
        ):
            expected_name = _format_expected_type(expected_return)
            actual_name = type(result).__name__
            raise TypeError(f"return value expected {expected_name}, got {actual_name}")
        return result

    return cast(F, wrapper)


def curry(func: F) -> F:
    """Return a curried version of a function."""
    signature = inspect.signature(func)
    required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is inspect._empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    def can_invoke(*args: Any, **kwargs: Any) -> bool:
        bound = signature.bind_partial(*args, **kwargs)
        return all(name in bound.arguments for name in required)

    @wraps(func)
    def curried(*args: Any, **kwargs: Any) -> Any:
        signature.bind_partial(*args, **kwargs)
        if can_invoke(*args, **kwargs):
            return func(*args, **kwargs)

        def collector(*next_args: Any, **next_kwargs: Any) -> Any:
            merged_kwargs = {**kwargs, **next_kwargs}
            return curried(*args, *next_args, **merged_kwargs)

        return collector

    return cast(F, curried)


def _to_hashable(value: Any) -> Hashable:
    if isinstance(value, Hashable):
        return value
    if isinstance(value, list):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, dict):
        converted = ((_to_hashable(key), _to_hashable(item)) for key, item in value.items())
        return tuple(sorted(converted, key=repr))
    if isinstance(value, set):
        return tuple(sorted((_to_hashable(item) for item in value), key=repr))
    if isinstance(value, tuple):
        return tuple(_to_hashable(item) for item in value)
    return repr(value)


def _make_cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    hashable_args = tuple(_to_hashable(item) for item in args)
    hashable_kwargs = tuple(sorted((key, _to_hashable(value)) for key, value in kwargs.items()))
    return hashable_args, hashable_kwargs


def memoize(*, ttl: float | None = None) -> Callable[[F], F]:
    """Cache function results with optional TTL in seconds."""

    def decorator(func: F) -> F:
        cache: dict[Hashable, tuple[Any, float | None]] = {}

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = _make_cache_key(args, kwargs)
            now = time.monotonic()
            if key in cache:
                value, expires_at = cache[key]
                if expires_at is None or now <= expires_at:
                    return value
            result = func(*args, **kwargs)
            expires_at = None if ttl is None else time.monotonic() + ttl
            cache[key] = (result, expires_at)
            return result

        return cast(F, wrapper)

    return decorator


def retry(
    *,
    max_retries: int = 3,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    backoff: float = 1.0,
) -> Callable[[F], F]:
    """Stub decorator kept for import compatibility."""
    _ = max_retries, exceptions, backoff

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def deprecated(*, message: str = "", removal_version: str | None = None) -> Callable[[F], F]:
    """Stub decorator kept for import compatibility."""
    _ = message, removal_version

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def trace(*, logger_name: str = __name__) -> Callable[[F], F]:
    """Stub decorator kept for import compatibility."""
    _ = logger_name

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def throttle(*, rate: float) -> Callable[[F], F]:
    """Stub decorator kept for import compatibility."""
    _ = rate

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
