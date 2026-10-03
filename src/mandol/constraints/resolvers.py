"""Constraint resolvers: leaf predicates evaluated into candidate sets.

P0 ships the linear-scan resolver family. Every resolver answers the same
question — "which UIDs satisfy this leaf constraint?" — so the planner can
combine them with set algebra without knowing how each predicate is evaluated.
The metadata index (P1) and the graph BFS resolver (P2) will plug into the same
``BaseConstraintResolver.resolve`` contract, replacing the scans without
touching the planner or the retrieval backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from .candidate_set import CandidateSet
from .query_constraints import TimeRange

if TYPE_CHECKING:
    from ..core.memory_unit import MemoryUnit
    from ..core.semantic_map import SemanticMap

_ORDERING_OPERATORS = ("gt", "gte", "lt", "lte")


def _resolve_field_value(unit: "MemoryUnit", field: str) -> Any:
    """Resolve a field with the same lookup order as ``filter_memory_units``."""
    value = getattr(unit, field, None)
    if value is None:
        value = unit.raw_data.get(field)
    return value


def _compare_ordering(value: Any, operator: str, expected: Any) -> bool:
    """Compare ordering operators, treating missing values as non-matches.

    ``SemanticMap.filter_memory_units`` raises ``TypeError`` when a unit lacks
    the filtered field (``None > expected``); resolvers must not abort a whole
    scan on one payload, so an absent value simply fails the predicate.
    """
    if value is None:
        return False
    try:
        if operator == "gt":
            return value > expected
        if operator == "gte":
            return value >= expected
        if operator == "lt":
            return value < expected
        return value <= expected
    except TypeError:
        return False


def _unit_matches(unit: "MemoryUnit", filter_condition: Dict[str, Dict[str, Any]]) -> bool:
    """Evaluate a field condition using ``filter_memory_units`` semantics."""
    for field, conditions in filter_condition.items():
        value = _resolve_field_value(unit, field)
        for operator, expected in conditions.items():
            if operator == "eq" and value != expected:
                return False
            if operator == "ne" and value == expected:
                return False
            if operator == "in" and value not in expected:
                return False
            if operator == "nin" and value in expected:
                return False
            if operator in _ORDERING_OPERATORS and not _compare_ordering(
                value, operator, expected
            ):
                return False
            if operator == "contain" and expected not in str(value):
                return False
            if operator == "not_contain" and expected in str(value):
                return False
    return True


class BaseConstraintResolver(ABC):
    """Resolve one leaf constraint into a ``CandidateSet``.

    Subclasses receive the retrieval source and their leaf predicate at
    construction time and stay stateless afterwards; the planner builds one
    resolver per declared leaf constraint and intersects the results.
    """

    def __init__(self, semantic_map: "SemanticMap"):
        """Bind the resolver to the map whose payloads back the predicate.

        Args:
            semantic_map: Map owning the L1 unit population scanned by P0
                resolvers and, from P1 on, the metadata index.
        """
        self.semantic_map = semantic_map

    @abstractmethod
    def resolve(self) -> CandidateSet:
        """Return the UID candidate set satisfying this leaf constraint."""

    def _l1_units(self) -> List["MemoryUnit"]:
        """Return L1-resident payloads.

        Candidate resolution currently scans the same population as
        ``filter_memory_units`` (L1 payloads only); paging cold records in is
        deliberately left to the index milestone instead of materializing the
        whole corpus per query.
        """
        return list(self.semantic_map.memory_units.values())

    def _total_units(self) -> int:
        """Return the L1 unit count used for selectivity estimation."""
        return len(self.semantic_map.memory_units)


class MetadataConstraintResolver(BaseConstraintResolver):
    """Linear-scan resolver for metadata field conditions (P0)."""

    source = "metadata_filters"

    def __init__(
        self,
        semantic_map: "SemanticMap",
        filters: Dict[str, Dict[str, Any]],
    ):
        """Initialize the resolver.

        Args:
            semantic_map: Map providing the unit population.
            filters: Field conditions using the ``filter_memory_units``
                operator vocabulary.
        """
        super().__init__(semantic_map)
        self.filters = filters

    def resolve(self) -> CandidateSet:
        """Scan L1 payloads and return the UIDs matching every field condition."""
        total = self._total_units()
        uids: Set[str] = {
            unit.uid
            for unit in self._l1_units()
            if _unit_matches(unit, self.filters)
        }
        return CandidateSet.from_iterable(uids, source=self.source, total=total)


class TimeRangeConstraintResolver(MetadataConstraintResolver):
    """Linear-scan resolver for inclusive time windows (P0).

    The range is expressed as ``gte`` / ``lte`` conditions on the range field
    and evaluated with the metadata scan. P1 replaces the scan with the sorted
    epoch index while keeping this class name and contract.
    """

    source = "time_range"

    def __init__(self, semantic_map: "SemanticMap", time_range: TimeRange):
        """Initialize the resolver.

        Args:
            semantic_map: Map providing the unit population.
            time_range: Inclusive window applied to ``time_range.field``.
        """
        conditions: Dict[str, Any] = {}
        if time_range.start is not None:
            conditions["gte"] = time_range.start
        if time_range.end is not None:
            conditions["lte"] = time_range.end
        super().__init__(semantic_map, {time_range.field: conditions})


class SpaceConstraintResolver(BaseConstraintResolver):
    """Adapter exposing MemorySpace membership as a candidate set.

    The resolver delegates to the map's space-filter resolution — the same
    membership logic the dense/BM25/SPLADE backends use for ``space_names`` — so
    the space predicate stays isomorphic to the other constraints without
    materializing payloads.
    """

    source = "space_names"

    def __init__(self, semantic_map: "SemanticMap", space_names: List[str]):
        """Initialize the resolver.

        Args:
            semantic_map: Map owning the MemorySpace hierarchy.
            space_names: Space names with or without the ``ms:`` prefix.
        """
        super().__init__(semantic_map)
        self.space_names = [str(name) for name in space_names]

    def resolve(self) -> CandidateSet:
        """Return the UIDs belonging to any requested MemorySpace."""
        space_uids: Optional[Set[str]] = self.semantic_map._get_candidate_uids_set(
            self.space_names, None
        )
        return CandidateSet.from_iterable(
            space_uids or (),
            source=self.source,
            total=self._total_units(),
        )