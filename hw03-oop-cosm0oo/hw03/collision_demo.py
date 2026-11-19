from __future__ import annotations

from typing import TypedDict

from hw03.matrix import Matrix


class _CollisionMatrix(Matrix):
    """Matrix subclass with forced hash collision for demo purposes."""

    def __hash__(self) -> int:
        """Return fixed hash value to force collision."""
        return 1


class SetDictDemo(TypedDict):
    """Return type for set/dict demonstration."""

    matrices_in_set: set[Matrix]
    matrix_dict: dict[Matrix, str]


def demonstrate_set_dict() -> SetDictDemo:
    """Demonstrate Matrix usage in set and dict."""
    first = Matrix([[1, 2], [3, 4]])
    equal_to_first = Matrix([[1, 2], [3, 4]])
    second = Matrix([[5, 6], [7, 8]])

    matrices_in_set = {first, equal_to_first, second}
    matrix_dict = {first: "first", second: "second"}

    return {
        "matrices_in_set": matrices_in_set,
        "matrix_dict": matrix_dict,
    }


def demonstrate_collision() -> tuple[Matrix, Matrix]:
    """Create two different matrices with equal hashes."""
    m1 = _CollisionMatrix([[1, 0], [0, 1]])
    m2 = _CollisionMatrix([[2, 0], [0, 2]])
    return m1, m2


if __name__ == "__main__":
    result = demonstrate_set_dict()
    print("Matrices in set:", result["matrices_in_set"])
    print("Matrix dict:", result["matrix_dict"])

    m1, m2 = demonstrate_collision()
    print("m1 =", repr(m1))
    print("m2 =", repr(m2))
    print("m1 == m2:", m1 == m2)
    print("hash(m1) == hash(m2):", hash(m1) == hash(m2))
    print("Both in set:", {m1, m2})
