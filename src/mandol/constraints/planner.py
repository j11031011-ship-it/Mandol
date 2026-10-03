"""Constraint-aware planner for constrained hybrid retrieval.

The planner is the coordination layer requested by the assignment: it resolves
every declared constraint into a ``CandidateSet``, intersects the sets (AND
semantics), and drives ``MultiRetriever.smart_search`` with the resulting
execution plan. Selectivity decides how the candidate set reaches retrieval:
sets covering less than the configured fraction of the corpus are pushed down
as the ``candidate_uids`` pre-filter, while broad sets are applied to fused
results afterwards (post-filter). The scheduling milestone (P3) builds on the
same explicit ``mode`` field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..utils.logging_config import create_module_logger
from .candidate_set import CandidateSet
from .query_constraints import QueryConstraints
from .resolvers import (
    BaseConstraintResolver,
    MetadataConstraintResolver,
    RelationConstraintResolver,
    SpaceConstraintResolver,
    TimeRangeConstraintResolver,
)

if TYPE_CHECKING:
    from ..core.memory_unit import MemoryUnit
    from ..core.semantic_graph import SemanticGraph

logger = create_module_logger("constraint_planner")

DEFAULT_PREFILTER_SELECTIVITY_THRESHOLD = 0.2


def _expand_top_k(top_k: int) -> int:
    """Return the retrieval pool size used before post-filtering.

    Mirrors the expansion already applied by ``_execute_base_retrieval``
    (``max(top_k * 3, 50)``) so post-filtering still has enough fused results to
    fill ``top_k`` after unsatisfying candidates are dropped.
    """
    return max(top_k * 3, 50)


@dataclass(frozen=True)
class ConstraintExecutionPlan:
    """Execution plan assembled by ``ConstraintAwarePlanner.plan``.

    Attributes:
        candidate_set: Intersected candidate UIDs, or ``None`` when the query
            declares no constraints and therefore needs no filtering.
        mode: ``"pre_filter"`` when the candidate set is pushed down as the
            ``candidate_uids`` retrieval filter, ``"post_filter"`` when it is
            applied to fused results, or ``"none"`` for unconstrained queries.
        retrieval_top_k: Pool size requested from ``smart_search``.
    """

    candidate_set: Optional[CandidateSet]
    mode: str
    retrieval_top_k: int


class ConstraintAwarePlanner:
    """Resolve constraints and execute constrained retrieval.

    The planner owns the coordination responsibilities and nothing else: it
    never encodes, ranks, or fuses results. Semantic similarity remains a pure
    ranking signal applied by the existing retrieval pipeline, while constraints
    only ever restrict which UIDs are allowed through.
    """

    def __init__(
        self,
        semantic_graph: "SemanticGraph",
        prefilter_selectivity_threshold: float = DEFAULT_PREFILTER_SELECTIVITY_THRESHOLD,
    ):
        """Initialize the planner.

        Args:
            semantic_graph: Graph layer providing the SemanticMap used for
                constraint resolution and the MultiRetriever used for ranking.
            prefilter_selectivity_threshold: Candidate fraction below which the
                plan pushes ``candidate_uids`` down into retrieval instead of
                post-filtering fused results.
        """
        self.semantic_graph = semantic_graph
        self.prefilter_selectivity_threshold = float(prefilter_selectivity_threshold)

    def _build_resolvers(
        self, constraints: QueryConstraints
    ) -> List[BaseConstraintResolver]:
        """Build one resolver per declared leaf constraint."""
        semantic_map = self.semantic_graph.semantic_map
        resolvers: List[BaseConstraintResolver] = []
        if constraints.metadata_filters:
            resolvers.append(
                MetadataConstraintResolver(semantic_map, constraints.metadata_filters)
            )
        if constraints.time_range is not None:
            resolvers.append(
                TimeRangeConstraintResolver(semantic_map, constraints.time_range)
            )
        if constraints.relation is not None:
            resolvers.append(
                RelationConstraintResolver(semantic_map, constraints.relation)
            )
        if constraints.space_names:
            resolvers.append(
                SpaceConstraintResolver(semantic_map, constraints.space_names)
            )
        return resolvers

    def resolve_candidates(self, constraints: QueryConstraints) -> Optional[CandidateSet]:
        """Resolve every leaf constraint and intersect the results.

        Returns:
            The intersected candidate set, or ``None`` when the query declares
            no constraints.
        """
        resolvers = self._build_resolvers(constraints)
        if not resolvers:
            return None

        combined: Optional[CandidateSet] = None
        for resolver in resolvers:
            resolved = resolver.resolve()
            logger.debug(
                "Constraint %s resolved %d candidates.", resolver.source, len(resolved)
            )
            combined = resolved if combined is None else combined.intersect(resolved)
        return combined

    def plan(self, constraints: QueryConstraints, top_k: int) -> ConstraintExecutionPlan:
        """Assemble the execution plan for one constrained query.

        Selectivity chooses the execution mode: candidate sets covering less
        than ``prefilter_selectivity_threshold`` of the corpus are pushed down
        as a retrieval filter, while broad sets (and sets of unknown size) keep
        the post-filter path — push-down only pays off when it removes enough
        candidates to shrink the retrieval work.
        """
        candidate_set = self.resolve_candidates(constraints)
        if candidate_set is None:
            return ConstraintExecutionPlan(
                candidate_set=None, mode="none", retrieval_top_k=top_k
            )
        selectivity = candidate_set.selectivity
        mode = (
            "pre_filter"
            if selectivity is not None
            and selectivity < self.prefilter_selectivity_threshold
            else "post_filter"
        )
        return ConstraintExecutionPlan(
            candidate_set=candidate_set,
            mode=mode,
            retrieval_top_k=_expand_top_k(top_k),
        )

    def execute_query(
        self,
        query_text: str,
        constraints: QueryConstraints,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple["MemoryUnit", float]]:
        """Run constrained retrieval end to end.

        Args:
            query_text: Query text passed to ``smart_search``.
            constraints: Declarative constraint bundle (AND semantics).
            top_k: Final number of results.
            **kwargs: Additional ``MultiRetriever.smart_search`` options
                (``methods``, ``fusion_method``, ``rerank_method``, ...). A
                caller-provided ``candidate_uids`` is intersected with the
                resolved constraint candidates before any push-down.

        Returns:
            Fused results satisfying every declared constraint, ordered by the
            existing ranking pipeline and truncated to ``top_k``.
        """
        if kwargs.pop("return_detailed", False):
            raise ValueError(
                "search_constrained does not support return_detailed=True; use "
                "MultiRetriever.smart_search for detailed execution diagnostics."
            )

        plan = self.plan(constraints, top_k)
        if plan.candidate_set is not None and len(plan.candidate_set) == 0:
            logger.info(
                "Constraint resolution produced an empty candidate set; "
                "skipping retrieval and returning no results."
            )
            return []

        caller_candidates = kwargs.pop("candidate_uids", None)
        search_kwargs: Dict[str, Any] = dict(kwargs)
        effective_set = plan.candidate_set
        if caller_candidates is not None:
            if plan.candidate_set is None:
                search_kwargs["candidate_uids"] = caller_candidates
            else:
                effective_set = plan.candidate_set.intersect(
                    CandidateSet.from_iterable(
                        caller_candidates, source="caller_candidates"
                    )
                )
                if len(effective_set) == 0:
                    logger.info(
                        "Caller candidate filter and constraints share no UIDs; "
                        "skipping retrieval and returning no results."
                    )
                    return []
                search_kwargs["candidate_uids"] = list(effective_set)
        elif plan.mode == "pre_filter":
            search_kwargs["candidate_uids"] = list(effective_set)

        multi_retriever = self.semantic_graph.get_multi_retriever()
        results = multi_retriever.smart_search(
            query_text, top_k=plan.retrieval_top_k, **search_kwargs
        )

        if plan.candidate_set is None:
            return results[:top_k]

        # Final verification: backends and graph expansion may return units the
        # push-down could not exclude, so constraints are always re-checked.
        filtered = [
            (unit, score) for unit, score in results if unit.uid in effective_set
        ]
        logger.debug(
            "Constrained retrieval kept %d of %d fused results (mode=%s).",
            len(filtered),
            len(results),
            plan.mode,
        )
        return filtered[:top_k]