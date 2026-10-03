"""Declarative query constraint value objects for constrained retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TimeRange:
    """Inclusive time window over a memory-unit payload field.

    The field defaults to ``timestamp`` because benchmark payloads store the
    observation time in ``raw_data["timestamp"]`` (for example LoCoMo episodic
    units). When the field is registered as a sorted index, bounds and values
    are normalized to epoch floats by ``TimestampNormalizer`` and the window is
    answered with bisect; otherwise the raw values are compared with the same
    operator semantics as ``SemanticMap.filter_memory_units``.

    Attributes:
        start: Inclusive lower bound; ``None`` means unbounded below.
        end: Inclusive upper bound; ``None`` means unbounded above.
        field: Payload field holding the timestamp value.
    """

    start: Optional[Any] = None
    end: Optional[Any] = None
    field: str = "timestamp"


@dataclass(frozen=True)
class RelationConstraint:
    """Graph-neighborhood constraint over explicit relationships.

    Declared with ``QueryConstraints`` and resolved by the depth-limited BFS
    resolver over ``SemanticGraph.rx_graph`` (``RelationConstraintResolver``).
    The default values describe the assignment's "neighbors of an entity" case:
    one hop from the seed units in both edge directions.

    Attributes:
        seed_uids: UIDs whose graph neighborhood defines the constraint.
        max_depth: BFS depth; ``1`` means direct neighbors only.
        relation_types: Optional edge ``type`` filter.
        direction: ``"successors"``, ``"predecessors"``, or ``"both"``.
    """

    seed_uids: List[str] = field(default_factory=list)
    max_depth: int = 1
    relation_types: Optional[List[str]] = None
    direction: str = "both"


@dataclass(frozen=True)
class QueryConstraints:
    """Declarative constraint bundle for constraint-aware hybrid retrieval.

    Every declared leaf constraint is resolved into a candidate UID set and the
    sets are intersected (AND semantics) before retrieval results are filtered.

    Attributes:
        metadata_filters: Field conditions expressed with the same operator
            vocabulary as ``SemanticMap.filter_memory_units``
            (``eq / ne / in / nin / gt / gte / lt / lte / contain /
            not_contain``).
        time_range: Optional inclusive time window.
        relation: Optional graph-neighborhood constraint (P2).
        space_names: Optional MemorySpace names; membership is resolved to a
            candidate set like any other predicate.
    """

    metadata_filters: Optional[Dict[str, Dict[str, Any]]] = None
    time_range: Optional[TimeRange] = None
    relation: Optional[RelationConstraint] = None
    space_names: Optional[List[str]] = None

    def is_empty(self) -> bool:
        """Return True when no constraint is declared."""
        return not (
            self.metadata_filters
            or self.time_range
            or self.relation
            or self.space_names
        )