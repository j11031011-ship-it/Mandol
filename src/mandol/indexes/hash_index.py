"""Equality index for metadata fields (``eq`` / ``in`` / ``ne`` / ``nin``).

The structure is a value-to-UIDs hash map mirroring ``==`` semantics exactly:
buckets are keyed by the payload value itself, so Python's own equality rules
(``1 == True``, ``None`` for missing fields) carry over without translation.
Units are tracked in a full UID set as well, which turns the negative
operators into complements over the indexed population.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from ..constraints.candidate_set import CandidateSet
from .base_index import BaseIndex

_OPERATORS = frozenset({"eq", "in", "ne", "nin"})


def _is_hashable(value: Any) -> bool:
    """Return whether ``value`` can key a dict bucket."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


class HashIndex(BaseIndex):
    """Value-bucket index serving equality and set-membership operators.

    Units whose field value is unhashable cannot be bucketed, so the index
    reports itself unservable and the resolver falls back to the scan — that is
    the only way to keep ``eq`` results identical for such payloads.
    """

    def __init__(self, field: str):
        """Initialize the index.

        Args:
            field: Payload field this index serves.
        """
        super().__init__(field)
        self._buckets: Dict[Any, Set[str]] = {}
        self._uid_values: Dict[str, Any] = {}
        self._all_uids: Set[str] = set()
        self._unhashable_uids: Set[str] = set()

    @property
    def uid_count(self) -> int:
        """Number of UIDs currently indexed."""
        return len(self._all_uids)

    def add(self, uid: str, value: Any) -> None:
        """Index ``value`` for ``uid``, replacing any previous entry."""
        uid = str(uid)
        if uid in self._all_uids:
            self.remove(uid)
        self._all_uids.add(uid)
        self._uid_values[uid] = value
        if _is_hashable(value):
            self._buckets.setdefault(value, set()).add(uid)
        else:
            self._unhashable_uids.add(uid)

    def remove(self, uid: str) -> None:
        """Drop ``uid`` from the index if present."""
        uid = str(uid)
        value = self._uid_values.pop(uid, None)
        if uid not in self._all_uids:
            return
        self._all_uids.discard(uid)
        self._unhashable_uids.discard(uid)
        if _is_hashable(value):
            bucket = self._buckets.get(value)
            if bucket is not None:
                bucket.discard(uid)
                if not bucket:
                    del self._buckets[value]

    def can_serve(self, conditions: Dict[str, Any]) -> bool:
        """Serve non-empty conditions whose operator and operands are hashable."""
        if not conditions or self._unhashable_uids:
            return False
        for operator, expected in conditions.items():
            if operator not in _OPERATORS:
                return False
            if operator in ("in", "nin"):
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    return False
                if any(not _is_hashable(item) for item in expected):
                    return False
            elif not _is_hashable(expected):
                return False
        return True

    def lookup(self, conditions: Dict[str, Any]) -> CandidateSet:
        """Evaluate the conditions, mirroring ``filter_memory_units`` semantics."""
        matched: Optional[Set[str]] = None
        for operator, expected in conditions.items():
            part = self._match_set(operator, expected)
            matched = part if matched is None else matched & part
        return CandidateSet.from_iterable(
            matched or (), source=f"index:{self.field}"
        )

    def estimate_selectivity(self, conditions: Dict[str, Any]) -> Optional[float]:
        """Return the match fraction over indexed units."""
        if not self._all_uids:
            return None
        return len(self.lookup(conditions)) / len(self._all_uids)

    def _match_set(self, operator: str, expected: Any) -> Set[str]:
        """Return the UIDs satisfying one operator condition."""
        if operator == "eq":
            return set(self._buckets.get(expected, ()))
        if operator == "in":
            return {uid for item in expected for uid in self._buckets.get(item, ())}
        if operator == "ne":
            return self._all_uids - set(self._buckets.get(expected, ()))
        if operator == "nin":
            excluded = {uid for item in expected for uid in self._buckets.get(item, ())}
            return self._all_uids - excluded
        raise ValueError(
            f"HashIndex cannot evaluate operator {operator!r} on field {self.field!r}."
        )