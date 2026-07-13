"""Small deterministic reproduction assertions."""

from iter8ml.domain.hashing import digest


def assert_same_digest(left: object, right: object) -> str:
    left_digest = digest(left)
    right_digest = digest(right)
    if left_digest != right_digest:
        raise AssertionError(f"values differ: {left_digest} != {right_digest}")
    return left_digest
