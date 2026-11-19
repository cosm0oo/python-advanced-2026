"""Container implementations for homework variants."""

from collections.abc import Hashable, Iterable, Iterator, Mapping
from typing import TypeVar, cast

KT = TypeVar("KT", bound=Hashable)
VT = TypeVar("VT")


class FrozenDict(Mapping[KT, VT]):
    """Immutable mapping with deterministic value-based hashing."""

    __slots__ = ("_data", "_hash")

    def __init__(
        self,
        source: Mapping[KT, VT] | Iterable[tuple[KT, VT]] | None = None,
        **kwargs: VT,
    ) -> None:
        self._data: dict[KT, VT] = dict(source or ())
        if kwargs:
            self._data.update(cast(dict[KT, VT], kwargs))
        self._hash: int | None = None

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[KT]:
        return iter(self._data)

    def __reversed__(self) -> Iterator[KT]:
        return reversed(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __getitem__(self, key: KT) -> VT:
        return self._data[key]

    def __setitem__(self, key: KT, value: VT) -> None:
        raise TypeError(f"{self.__class__.__name__} is immutable")

    def __delitem__(self, key: KT) -> None:
        raise TypeError(f"{self.__class__.__name__} is immutable")

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(frozenset(self._data.items()))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    def __str__(self) -> str:
        return str(self._data)
