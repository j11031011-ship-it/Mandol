"""Constraint-aware hybrid retrieval primitives.

The package adds the "intersection + predicate push-down" capability that the
existing fusion pipeline lacks (it only combines parallel recalls with score
fusion):

- ``QueryConstraints`` declares metadata, time, relation, and space predicates
  as value objects instead of untyped dictionaries;
- ``BaseConstraintResolver`` subclasses evaluate one leaf predicate each into a
  ``CandidateSet``;
- ``CandidateSet`` provides the UID set algebra that decouples predicates from
  retrieval;
- ``ConstraintAwarePlanner`` coordinates resolution and retrieval and is reached
  through ``SemanticGraph.search_constrained``; selective candidate sets are
  pushed down as the ``candidate_uids`` pre-filter and broader ones are applied
  to fused results as a post-filter.

All of it composes with the existing backends through the ``candidate_uids``
parameter of ``BaseRetriever.search``; no retrieval backend is modified. The
incrementally maintained index-backed fast paths live in ``mandol.indexes``.
"""

from .candidate_set import CandidateSet
from .planner import ConstraintAwarePlanner, ConstraintExecutionPlan
from .query_constraints import QueryConstraints, RelationConstraint, TimeRange
from .resolvers import (
    BaseConstraintResolver,
    MetadataConstraintResolver,
    RelationConstraintResolver,
    SpaceConstraintResolver,
    TimeRangeConstraintResolver,
)

__all__ = [
    "CandidateSet",
    "ConstraintAwarePlanner",
    "ConstraintExecutionPlan",
    "QueryConstraints",
    "RelationConstraint",
    "TimeRange",
    "BaseConstraintResolver",
    "MetadataConstraintResolver",
    "RelationConstraintResolver",
    "SpaceConstraintResolver",
    "TimeRangeConstraintResolver",
]