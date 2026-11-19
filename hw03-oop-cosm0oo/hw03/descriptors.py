from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any
from weakref import WeakKeyDictionary

_MISSING = object()
Observer = Callable[[str, Any, Any], None]


class Validated:
    """Descriptor with type and range validation on assignment."""

    def __init__(
        self,
        expected_type: type[Any],
        min_value: Any = None,
        max_value: Any = None,
    ) -> None:
        """Store expected type and optional bounds."""
        self.expected_type = expected_type
        self.min_value = min_value
        self.max_value = max_value
        self.public_name = ""
        self.storage_name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Return stored value or descriptor itself for class access."""
        if obj is None:
            return self
        if self.storage_name not in obj.__dict__:
            raise AttributeError(f"{self.public_name} is not set")
        return obj.__dict__[self.storage_name]

    def __set__(self, obj: Any, value: Any) -> None:
        """Validate and store assigned value."""
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.public_name} must be {self.expected_type.__name__}")
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.public_name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.public_name} must be <= {self.max_value}")
        obj.__dict__[self.storage_name] = value


class Logged:
    """Descriptor that logs every read and write using logging."""

    def __init__(self, default: Any = _MISSING) -> None:
        """Store optional default value for missing attribute."""
        self.default = default
        self.public_name = ""
        self.storage_name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Return value and emit debug log for read operation."""
        if obj is None:
            return self
        if self.storage_name in obj.__dict__:
            value = obj.__dict__[self.storage_name]
            logging.debug("Read %s=%r", self.public_name, value)
            return value
        if self.default is not _MISSING:
            logging.debug("Read %s=%r", self.public_name, self.default)
            return self.default
        raise AttributeError(f"{self.public_name} is not set")

    def __set__(self, obj: Any, value: Any) -> None:
        """Store value and emit debug log for write operation."""
        obj.__dict__[self.storage_name] = value
        logging.debug("Set %s=%r", self.public_name, value)


class Cached:
    """Descriptor with lazy computation and per-instance cache."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        """Store factory used for lazy value initialization."""
        self.factory = factory
        self.public_name = ""
        self.storage_name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Compute value once per instance and return cached result."""
        if obj is None:
            return self
        if self.storage_name not in obj.__dict__:
            obj.__dict__[self.storage_name] = self.factory()
        return obj.__dict__[self.storage_name]

    def __set__(self, obj: Any, value: Any) -> None:
        """Override cached value for this instance."""
        obj.__dict__[self.storage_name] = value


class Typed:
    """Descriptor with strict type check on assignment."""

    def __init__(self, expected_type: type[Any]) -> None:
        """Store expected type for assignments."""
        self.expected_type = expected_type
        self.public_name = ""
        self.storage_name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Return stored value or descriptor itself for class access."""
        if obj is None:
            return self
        if self.storage_name not in obj.__dict__:
            raise AttributeError(f"{self.public_name} is not set")
        return obj.__dict__[self.storage_name]

    def __set__(self, obj: Any, value: Any) -> None:
        """Validate type and store value."""
        if not isinstance(value, self.expected_type):
            raise TypeError(f"{self.public_name} must be {self.expected_type.__name__}")
        obj.__dict__[self.storage_name] = value


class ReadOnly:
    """Descriptor that allows assignment once per instance."""

    def __init__(self, default: Any = _MISSING) -> None:
        """Store optional default value and internal storage names."""
        self.default = default
        self.public_name = ""
        self.storage_name = ""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Return stored value, default value, or descriptor itself."""
        if obj is None:
            return self
        if self.storage_name in obj.__dict__:
            return obj.__dict__[self.storage_name]
        if self.default is not _MISSING:
            return self.default
        raise AttributeError(f"{self.public_name} is not set")

    def __set__(self, obj: Any, value: Any) -> None:
        """Allow writing value only once per instance."""
        if self.storage_name in obj.__dict__:
            raise AttributeError(f"{self.public_name} is read-only")
        obj.__dict__[self.storage_name] = value


class Observable:
    """Descriptor that notifies registered callbacks on value changes."""

    def __init__(self, default: Any = _MISSING) -> None:
        """Initialize descriptor with optional default and observer storage."""
        self.default = default
        self.public_name = ""
        self.storage_name = ""
        self._observers: WeakKeyDictionary[Any, list[Observer]] = WeakKeyDictionary()

    def __set_name__(self, owner: type[Any], name: str) -> None:
        """Bind descriptor to attribute name in owner class."""
        self.public_name = name
        self.storage_name = f"_{name}"

    def __get__(self, obj: Any, objtype: type[Any] | None = None) -> Any:
        """Return stored value, default value, or descriptor itself."""
        if obj is None:
            return self
        if self.storage_name in obj.__dict__:
            return obj.__dict__[self.storage_name]
        if self.default is not _MISSING:
            return self.default
        raise AttributeError(f"{self.public_name} is not set")

    def __set__(self, obj: Any, value: Any) -> None:
        """Store value and notify all observers for this instance."""
        old_value = obj.__dict__.get(self.storage_name, self.default)
        obj.__dict__[self.storage_name] = value
        for callback in self._observers.get(obj, []):
            callback(self.public_name, old_value, value)

    def add_observer(self, obj: Any, callback: Observer) -> None:
        """Register callback for changes of this descriptor on an instance."""
        if obj not in self._observers:
            self._observers[obj] = []
        self._observers[obj].append(callback)
