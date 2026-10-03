"""Offline tests for constraint-aware hybrid retrieval (P0).

The suite follows the offline paradigm of ``test_rocksdb_tiered_cache.py``:
``global_model_manager`` is patched with deterministic dummies and every test
runs without network access or real model downloads.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandol.constraints import (
    CandidateSet,
    ConstraintAwarePlanner,
    MetadataConstraintResolver,
    QueryConstraints,
    RelationConstraint,
    SpaceConstraintResolver,
    TimeRange,
    TimeRangeConstraintResolver,
)
from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.utils.model_manager import global_model_manager

# uid, text, timestamp, speaker, space
DEFAULT_UNITS = [
    ("u1", "morning coffee with alice", "2024-01-05T08:00:00", "speaker_a", "s-daily"),
    ("u2", "project deadline discussion", "2024-02-10T14:30:00", "speaker_b", "s-work"),
    ("u3", "coffee beans shopping trip", "2024-03-20T19:00:00", "speaker_a", "s-daily"),
    ("u4", "weekend hiking adventure", "2024-04-02T07:15:00", "speaker_c", "s-travel"),
    ("u5", "coffee machine repair notes", "2024-05-11T21:45:00", "speaker_b", "s-work"),
    ("u6", "evening reading list update", "2024-06-18T22:30:00", "speaker_a", "s-daily"),
]


class _DummyEmbeddingModel:
    """Deterministic 2-dim encoder so cosine ranking varies with the text."""

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = sum(ord(ch) for ch in str(text))
        return np.array([1.0 + seed % 5, 1.0 + (seed // 5) % 7], dtype=np.float32)


class _DummySpladeModel:
    def encode_query(self, text):
        del text
        return {7: 1.0}


class _StubMultiRetriever:
    """Records smart_search calls and replays a fixed result list."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def smart_search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return list(self.results)


@pytest.fixture
def graph_factory(monkeypatch):
    original_loader = global_model_manager.get_or_load_model

    def load_model(**kwargs):
        if kwargs.get("model_type") == "text_embedding":
            return _DummyEmbeddingModel()
        return original_loader(**kwargs)

    monkeypatch.setattr(global_model_manager, "get_or_load_model", load_model)
    monkeypatch.setattr(
        global_model_manager,
        "get_splade_model",
        lambda *args, **kwargs: _DummySpladeModel(),
    )
    monkeypatch.setattr(
        SemanticMap,
        "_incremental_aux_retriever_add",
        lambda self, units: None,
    )

    def create() -> SemanticGraph:
        semantic_map = SemanticMap(
            embedding_model_name="test/dummy",
            embedding_dim=2,
            use_flash_attention=False,
        )
        return SemanticGraph(semantic_map)

    return create


def _build_graph(graph_factory, units) -> SemanticGraph:
    graph = graph_factory()
    for uid, text, timestamp, speaker, space in units:
        raw_data = {"text_content": text, "speaker": speaker}
        if timestamp is not None:
            raw_data["timestamp"] = timestamp
        graph.semantic_map.add_unit(
            MemoryUnit(uid, raw_data),
            explicit_content_for_embedding=text,
            space_names=[space],
            generate_sparse_embedding=False,
        )
    return graph


def test_candidate_set_algebra() -> None:
    first = CandidateSet.from_iterable({"u1", "u2", "u3"}, source="meta", total=10)
    second = CandidateSet.from_iterable({"u2", "u3", "u4"}, source="time", total=10)

    intersection = first.intersect(second)
    assert intersection.uids == {"u2", "u3"}
    assert intersection.source == "meta & time"
    assert intersection.selectivity == pytest.approx(0.2)

    union = first.union(second)
    assert union.uids == {"u1", "u2", "u3", "u4"}
    assert union.selectivity == pytest.approx(0.4)

    assert first.difference(second).uids == {"u1"}
    assert len(first) == 3
    assert "u1" in first
    assert set(first) == {"u1", "u2", "u3"}
    assert CandidateSet.from_iterable([]).selectivity is None


def test_metadata_resolver_matches_filter_memory_units_semantics(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map

    contains = {"speaker": {"in": ["speaker_a", "speaker_b"]}, "text_content": {"contain": "coffee"}}
    contains_uids = {
        unit.uid for unit in semantic_map.filter_memory_units(filter_condition=contains)
    }
    resolved_contains = MetadataConstraintResolver(semantic_map, contains).resolve()
    assert resolved_contains.uids == contains_uids == {"u1", "u3", "u5"}
    assert resolved_contains.total == len(DEFAULT_UNITS)

    excludes = {"speaker": {"in": ["speaker_a", "speaker_b"]}, "text_content": {"not_contain": "coffee"}}
    excludes_uids = {
        unit.uid for unit in semantic_map.filter_memory_units(filter_condition=excludes)
    }
    resolved_excludes = MetadataConstraintResolver(semantic_map, excludes).resolve()
    assert resolved_excludes.uids == excludes_uids == {"u2", "u6"}


def test_time_range_resolver_filters_raw_timestamps(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map

    after = TimeRangeConstraintResolver(
        semantic_map, TimeRange(start="2024-03-01T00:00:00")
    ).resolve()
    assert after.uids == {"u3", "u4", "u5", "u6"}

    window = TimeRangeConstraintResolver(
        semantic_map,
        TimeRange(start="2024-02-01T00:00:00", end="2024-04-30T23:59:59"),
    ).resolve()
    assert window.uids == {"u2", "u3", "u4"}


def test_time_range_resolver_skips_units_without_timestamp(graph_factory) -> None:
    units = [
        ("n1", "has a timestamp", "2024-03-01T10:00:00", "speaker_a", "s-daily"),
        ("n2", "no timestamp at all", None, "speaker_a", "s-daily"),
    ]
    graph = _build_graph(graph_factory, units)

    resolved = TimeRangeConstraintResolver(
        graph.semantic_map, TimeRange(start="2024-01-01T00:00:00")
    ).resolve()
    assert resolved.uids == {"n1"}


def test_space_resolver_returns_space_membership(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)

    resolved = SpaceConstraintResolver(
        graph.semantic_map, ["s-daily", "ms:s-work"]
    ).resolve()
    assert resolved.uids == {"u1", "u2", "u3", "u5", "u6"}


def test_resolvers_on_empty_map_return_empty_sets(graph_factory) -> None:
    semantic_map = graph_factory().semantic_map

    assert (
        MetadataConstraintResolver(semantic_map, {"speaker": {"eq": "speaker_a"}})
        .resolve()
        .uids
        == frozenset()
    )
    assert (
        TimeRangeConstraintResolver(semantic_map, TimeRange(start="2024-01-01"))
        .resolve()
        .uids
        == frozenset()
    )
    assert (
        SpaceConstraintResolver(semantic_map, ["s-daily"]).resolve().uids
        == frozenset()
    )


def test_plan_without_constraints_keeps_plain_retrieval(graph_factory) -> None:
    planner = ConstraintAwarePlanner(graph_factory())

    plan = planner.plan(QueryConstraints(), top_k=3)
    assert plan.mode == "none"
    assert plan.candidate_set is None
    assert plan.retrieval_top_k == 3


def test_plan_post_filter_expands_pool_and_intersects_leaves(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)

    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_a"}},
        time_range=TimeRange(start="2024-01-01T00:00:00"),
        space_names=["s-daily"],
    )
    plan = planner.plan(constraints, top_k=2)

    assert plan.mode == "post_filter"
    assert plan.retrieval_top_k == 50
    assert plan.candidate_set.uids == {"u1", "u3", "u6"}
    assert plan.candidate_set.selectivity == pytest.approx(0.5)


def test_relation_constraint_is_deferred_to_p2(graph_factory) -> None:
    planner = ConstraintAwarePlanner(graph_factory())
    constraints = QueryConstraints(relation=RelationConstraint(seed_uids=["u1"]))

    with pytest.raises(NotImplementedError):
        planner.resolve_candidates(constraints)


def test_execute_query_post_filters_without_candidate_pushdown(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [
            (semantic_map.get_unit("u1"), 0.9),
            (semantic_map.get_unit("u6"), 0.8),
            (semantic_map.get_unit("u3"), 0.7),
            (semantic_map.get_unit("u4"), 0.6),
        ]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(time_range=TimeRange(start="2024-03-01T00:00:00"))
    results = planner.execute_query("coffee", constraints, top_k=2)

    # u1 falls outside the time window and is dropped; ranking order is kept.
    assert [unit.uid for unit, _ in results] == ["u6", "u3"]
    _, call_kwargs = stub.calls[0]
    assert call_kwargs["top_k"] == 50
    assert "candidate_uids" not in call_kwargs


def test_execute_query_skips_retrieval_when_candidates_are_empty(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    stub = _StubMultiRetriever([])
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_z"}})

    assert planner.execute_query("coffee", constraints, top_k=3) == []
    assert stub.calls == []


def test_search_constrained_u1_equivalence_with_linear_baseline(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(time_range=TimeRange(start="2024-02-01T00:00:00"))
    top_k = 3

    constrained = graph.search_constrained(
        "coffee",
        constraints,
        top_k=top_k,
        return_score=True,
        methods=["cosine_similarity"],
    )

    # Baseline: full retrieval, then the existing linear filter.
    retriever = graph.get_multi_retriever()
    baseline_pool = retriever.smart_search(
        "coffee", top_k=50, methods=["cosine_similarity"]
    )
    allowed = {
        unit.uid
        for unit in graph.semantic_map.filter_memory_units(
            filter_condition={"timestamp": {"gte": "2024-02-01T00:00:00"}}
        )
    }
    baseline = [(unit, score) for unit, score in baseline_pool if unit.uid in allowed][:top_k]

    assert constrained
    assert [unit.uid for unit, _ in constrained] == [unit.uid for unit, _ in baseline]


def test_search_constrained_without_constraints_matches_plain_smart_search(
    graph_factory,
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    retriever = graph.get_multi_retriever()

    plain = retriever.smart_search("coffee", top_k=3, methods=["cosine_similarity"])
    constrained = graph.search_constrained(
        "coffee", top_k=3, return_score=True, methods=["cosine_similarity"]
    )

    assert [unit.uid for unit, _ in constrained] == [unit.uid for unit, _ in plain]


def test_search_constrained_satisfies_all_constraints_end_to_end(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"in": ["speaker_a", "speaker_b"]}},
        time_range=TimeRange(end="2024-04-30T23:59:59"),
        space_names=["s-daily", "s-work"],
    )

    results = graph.search_constrained(
        "coffee", constraints, top_k=5, methods=["cosine_similarity"]
    )

    assert {unit.uid for unit in results} == {"u1", "u2", "u3"}
    for unit in results:
        assert isinstance(unit, MemoryUnit)
        assert unit.raw_data["speaker"] in {"speaker_a", "speaker_b"}
        assert unit.raw_data["timestamp"] <= "2024-04-30T23:59:59"
        assert unit.uid in {"u1", "u2", "u3", "u5", "u6"}


def test_search_constrained_returns_empty_for_unsatisfiable_constraints(
    graph_factory,
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_z"}})

    assert graph.search_constrained("coffee", constraints) == []