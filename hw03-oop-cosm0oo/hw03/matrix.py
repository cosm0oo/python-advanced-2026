from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from types import NotImplementedType, TracebackType
from typing import Self

Number = int | float


class _MatrixFileContext(AbstractContextManager["Matrix"]):
    """Context manager that loads matrix data from a text file."""

    def __init__(self, matrix_cls: type[Matrix], path: str) -> None:
        """Initialize context manager with matrix class and path."""
        self._matrix_cls = matrix_cls
        self._path = path

    def __enter__(self) -> Matrix:
        """Read the file and create a matrix instance."""
        rows: list[list[Number]] = []
        with open(self._path, encoding="utf-8") as file:
            for raw_line in file:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                rows.append([Matrix.parse_number(token) for token in stripped.split()])
        return self._matrix_cls(rows)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish context manager protocol."""
        return None


class Matrix:
    """Matrix with arithmetic operators, hashing and formatting."""

    def __init__(self, data: Iterable[Iterable[Number]]) -> None:
        """Build a rectangular numeric matrix from nested iterables."""
        rows = [tuple(row) for row in data]
        if not rows:
            raise ValueError("matrix must contain at least one row")
        col_count = len(rows[0])
        if col_count == 0:
            raise ValueError("matrix rows must contain at least one element")
        if any(len(row) != col_count for row in rows):
            raise ValueError("matrix must be rectangular")
        for row in rows:
            for value in row:
                if not isinstance(value, int | float):
                    raise TypeError("matrix elements must be int or float")
        self._data: tuple[tuple[Number, ...], ...] = tuple(rows)

    @property
    def rows(self) -> int:
        """Return number of rows."""
        return len(self._data)

    @property
    def cols(self) -> int:
        """Return number of columns."""
        return len(self._data[0])

    def __add__(self, other: object) -> Self | NotImplementedType:
        """Return element-wise sum of two matrices with equal shape."""
        if not isinstance(other, Matrix):
            return NotImplemented
        self._assert_same_shape(other)
        return type(self)(
            [
                [left + right for left, right in zip(left_row, right_row, strict=True)]
                for left_row, right_row in zip(self._data, other._data, strict=True)
            ]
        )

    def __sub__(self, other: object) -> Self | NotImplementedType:
        """Return element-wise difference of two matrices with equal shape."""
        if not isinstance(other, Matrix):
            return NotImplemented
        self._assert_same_shape(other)
        return type(self)(
            [
                [left - right for left, right in zip(left_row, right_row, strict=True)]
                for left_row, right_row in zip(self._data, other._data, strict=True)
            ]
        )

    def __mul__(self, scalar: object) -> Self | NotImplementedType:
        """Return matrix scaled by a numeric factor."""
        if not isinstance(scalar, int | float):
            return NotImplemented
        return type(self)([[value * scalar for value in row] for row in self._data])

    def __rmul__(self, scalar: object) -> Self | NotImplementedType:
        """Return matrix scaled by a numeric factor from the left side."""
        return self.__mul__(scalar)

    def __matmul__(self, other: object) -> Self | NotImplementedType:
        """Return result of matrix multiplication."""
        if not isinstance(other, Matrix):
            return NotImplemented
        if self.cols != other.rows:
            raise ValueError("incompatible matrix shapes for matmul")
        right_columns = list(zip(*other._data, strict=True))
        return type(self)(
            [
                [
                    sum(left * right for left, right in zip(row, column, strict=True))
                    for column in right_columns
                ]
                for row in self._data
            ]
        )

    def __eq__(self, other: object) -> bool:
        """Return True when both matrices have identical type and data."""
        if not isinstance(other, Matrix):
            return False
        if type(self) is not type(other):
            return False
        return self._data == other._data

    def __hash__(self) -> int:
        """Return hash based on immutable matrix data."""
        return hash(self._data)

    def __repr__(self) -> str:
        """Return Python expression style representation."""
        return f"{type(self).__name__}({[list(row) for row in self._data]!r})"

    def __str__(self) -> str:
        """Return readable aligned table representation."""
        return self._format_with_spec("")

    def __format__(self, format_spec: str) -> str:
        """Return formatted string with element formatting spec."""
        return self._format_with_spec(format_spec)

    @classmethod
    def from_file(cls, path: str) -> AbstractContextManager[Matrix]:
        """Return a context manager that reads a matrix from file on enter."""
        return _MatrixFileContext(cls, path)

    @staticmethod
    def parse_number(token: str) -> Number:
        """Parse integer token first, then fallback to float."""
        try:
            return int(token)
        except ValueError:
            return float(token)

    def _assert_same_shape(self, other: Matrix) -> None:
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("matrices must have the same shape")

    def _format_with_spec(self, format_spec: str) -> str:
        formatter = (
            (lambda value: format(value, format_spec))
            if format_spec
            else (lambda value: str(value))
        )
        rendered_rows = [[formatter(value) for value in row] for row in self._data]
        col_widths = [
            max(len(rendered_rows[row][col]) for row in range(self.rows))
            for col in range(self.cols)
        ]
        return "\n".join(
            " ".join(text.rjust(col_widths[col]) for col, text in enumerate(row_texts))
            for row_texts in rendered_rows
        )
