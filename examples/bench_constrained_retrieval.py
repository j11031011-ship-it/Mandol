"""Offline micro-benchmark for the constrained-retrieval stack (P3).

Measures the layers that P1/P2/P2+ replaced, on a synthetic corpus, without any
network access or real model download (a deterministic dummy encoder stands in
for the embedding model). The point is to quantify the *mechanisms*:

* metadata equality / time-window resolution: index lookup vs linear scan;
* relation one-hop resolution: adjacency BFS vs a full edge scan;
* relation-type edge lookup: inverted index vs a full edge scan;
* execution-mode accounting: for a range of selectivities, which mode the
  planner picks, the pool it requests, and how many units the backend has to
  process compared with an unconstrained scan;
* post-filter pool sizing: when the selectivity-aware expansion grows the pool
  past the baseline ``max(top_k * 3, 50)`` expansion;
* fallback policy: what a degenerate intersection yields under
  ``RETURN_EMPTY`` (default) and under ``DROP_CONSTRAINT``.

Run from the repository root:

    .venv/Scripts/python examples/bench_constrained_retrieval.py
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Tuple

import numpy as np

from mandol.constraints import (
    DROP_CONSTRAINT,
    FallbackPolicy,
    MetadataConstraintResolver,
    QueryConstraints,
    RelationConstraint,
    TimeRange,
)
from mandol.constraints import planner as planner_module
from mandol.constraints.planner import ConstraintAwarePlanner
from mandol.constraints.resolvers import RelationConstraintResolver
from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.utils.logging_config import set_log_level
from mandol.utils.model_manager import global_model_manager

UNITS = 2000
SPEAKERS = 40
SPACES = 10
EDGES = 2000
RELATION_TYPES = ("FOLLOWED_BY", "MENTION_OF", "NEXT", "SHARED_TOPIC", "LOCATED_IN")
REPEATS = 5
TOP_K = 10
START = datetime(2024, 1, 1)


class _DummyEmbeddingModel:
    """Deterministic 8-dim encoder; no download, no network."""

    DIM = 8

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    def _vector(self, text: str) -> np.ndarray:
        rng = random.Random(str(text))
        return np.array([rng.random() for _ in range(self.DIM)], dtype=np.float32)


class _DummySpladeModel:
    def encode_query(self, text):
        del text
        return {7: 1.0}


def _build_corpus() -> SemanticGraph:
    """Create an in-memory graph with units, filterable fields, and edges."""
    global_model_manager.get_or_load_model = lambda **kwargs: _DummyEmbeddingModel()
    global_model_manager.get_splade_model = lambda *args, **kwargs: _DummySpladeModel()
    SemanticMap._incremental_aux_retriever_add = lambda self, units: None

    graph = SemanticGraph(
        SemanticMap(
            embedding_model_name="test/dummy",
            embedding_dim=_DummyEmbeddingModel.DIM,
            use_flash_attention=False,
        )
    )

    rng = random.Random(20240301)
    for index in range(UNITS):
        uid = f"u{index}"
        timestamp = START + timedelta(minutes=rng.randrange(366 * 24 * 60))
        payload = {
            "text_content": f"memory {index} about topic {rng.randrange(200)}",
            "speaker": f"speaker_{rng.randrange(SPEAKERS)}",
            "timestamp": timestamp.isoformat(),
        }
        graph.semantic_map.add_unit(
            MemoryUnit(uid, payload),
            explicit_content_for_embedding=payload["text_content"],
            space_names=[f"space_{rng.randrange(SPACES)}"],
            generate_sparse_embedding=False,
        )

    # Registering after the fact rebuilds the indexes over the whole population.
    graph.semantic_map.register_filterable_fields(
        {"speaker": "hash", "timestamp": "sorted"}
    )

    for _ in range(EDGES):
        source = f"u{rng.randrange(UNITS)}"
        target = f"u{rng.randrange(UNITS)}"
        if source != target:
            graph.add_relationship(source, target, rng.choice(RELATION_TYPES))
    return graph


def _time(func: Callable[[], object]) -> float:
    """Return the best-of-REPEATS wall time of ``func`` in milliseconds."""
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


def _format_row(label: str, baseline_ms: float, optimized_ms: float) -> str:
    speedup = baseline_ms / optimized_ms if optimized_ms else float("inf")
    return f"  {label:<44} {baseline_ms:>9.3f} {optimized_ms:>11.3f} {speedup:>8.1f}x"


def _header(title: str) -> None:
    print(f"\n{title}")
    print(f"  {'case':<44} {'baseline':>9} {'optimized':>11} {'speedup':>9}")
    print(f"  {'':-<44} {'ms':>9} {'ms':>11}")


def bench_metadata(graph: SemanticGraph) -> None:
    semantic_map = graph.semantic_map
    conditions = {"speaker": {"eq": "speaker_7"}}
    resolver = MetadataConstraintResolver(semantic_map, conditions)

    with_index = _time(resolver.resolve)
    index = semantic_map._metadata_index
    semantic_map._metadata_index = None  # forces the P0 linear-scan path
    try:
        with_scan = _time(resolver.resolve)
    finally:
        semantic_map._metadata_index = index

    _header("A. metadata equality (1/40 of units)")
    print(_format_row("index hash lookup vs linear scan", with_scan, with_index))
    print(f"  -> {len(resolver.resolve())} candidates out of {UNITS} units")


def bench_time_range(graph: SemanticGraph) -> None:
    semantic_map = graph.semantic_map
    window = TimeRange(start=START.isoformat(), end=(START + timedelta(days=30)).isoformat())
    resolver = MetadataConstraintResolver(
        semantic_map, {window.field: {"gte": window.start, "lte": window.end}}
    )

    with_index = _time(resolver.resolve)
    index = semantic_map._metadata_index
    semantic_map._metadata_index = None
    try:
        with_scan = _time(resolver.resolve)
    finally:
        semantic_map._metadata_index = index

    _header("B. timestamp range (30 of 366 days)")
    print(_format_row("sorted bisect vs linear scan", with_scan, with_index))
    print(f"  -> {len(resolver.resolve())} candidates out of {UNITS} units")


def bench_relation(graph: SemanticGraph) -> None:
    semantic_map = graph.semantic_map
    seed = "u17"
    resolver = RelationConstraintResolver(
        semantic_map, RelationConstraint(seed_uids=[seed])
    )

    def naive_scan() -> None:
        """Baseline: enumerate every edge and keep those touching the seed."""
        index_to_uid = graph._index_to_uid
        seed_index = graph._uid_to_index[seed]
        neighbours = set()
        for edge_idx in graph.rx_graph.edge_indices():
            source, target = graph.rx_graph.get_edge_endpoints_by_index(edge_idx)
            if source == seed_index:
                neighbours.add(index_to_uid[target])
            elif target == seed_index:
                neighbours.add(index_to_uid[source])

    optimized = _time(resolver.resolve)
    baseline = _time(naive_scan)

    _header("C. relation one-hop neighbourhood (E=%d edges)" % EDGES)
    print(_format_row("adjacency BFS vs full edge scan", baseline, optimized))
    print(f"  -> {len(resolver.resolve())} neighbours of {seed}")


def bench_relation_type(graph: SemanticGraph) -> None:
    wanted = ["MENTION_OF"]

    def naive_scan() -> None:
        graph.rx_graph  # noqa: B018 - keep the baseline honest about what it reads
        for edge_idx in graph.rx_graph.edge_indices():
            data = graph.rx_graph.get_edge_data_by_index(edge_idx)
            if data and data.get("type") in wanted:
                continue

    optimized = _time(lambda: graph.get_edges_by_relation_types(wanted))
    baseline = _time(naive_scan)

    _header("D. relation-type edge lookup")
    print(_format_row("type inverted index vs full edge scan", baseline, optimized))
    print(f"  -> {len(graph.get_edges_by_relation_types(wanted))} of {EDGES} edges")


def bench_modes(graph: SemanticGraph, top_k: int = TOP_K) -> None:
    planner = ConstraintAwarePlanner(graph)
    print(f"\nE. execution-mode accounting (top_k={top_k}, corpus={UNITS})")
    print(
        f"  {'selectivity':>11} {'candidates':>11} {'mode':>12} "
        f"{'pool':>7} {'backend units':>14} {'vs full scan':>13}"
    )
    print(f"  {'':-<11} {'':-<11} {'':-<12} {'':-<7} {'':-<14} {'':-<13}")
    for speakers in (1, 2, 4, 8, 16, 40):
        selected = [f"speaker_{index}" for index in range(speakers)]
        constraints = QueryConstraints(metadata_filters={"speaker": {"in": selected}})
        plan = planner.plan(constraints, top_k)
        candidates = len(plan.candidate_set)
        # Pre-filter restricts retrieval to the candidates; post-filter still
        # scans the corpus but asks for a larger pool to refill ``top_k``.
        backend_units = candidates if plan.mode == "pre_filter" else UNITS
        print(
            f"  {candidates / UNITS:>10.1%} {candidates:>11} {plan.mode:>12} "
            f"{plan.retrieval_top_k:>7} {backend_units:>14} "
            f"{UNITS / backend_units:>12.1f}x"
        )


def bench_pool_sizing(graph: SemanticGraph, top_k: int = 100) -> None:
    """Show when the post-filter pool grows past the baseline expansion."""
    base_pool = max(top_k * 3, 50)
    planner = ConstraintAwarePlanner(graph)
    print(f"\nF. post-filter pool sizing (top_k={top_k}, base pool={base_pool})")
    print(
        f"  {'selectivity':>11} {'candidates':>11} {'mode':>12} "
        f"{'pool':>7} {'pool/base':>10}"
    )
    print(f"  {'':-<11} {'':-<11} {'':-<12} {'':-<7} {'':-<10}")
    for speakers in (4, 12, 16, 40):
        selected = [f"speaker_{index}" for index in range(speakers)]
        constraints = QueryConstraints(metadata_filters={"speaker": {"in": selected}})
        plan = planner.plan(constraints, top_k)
        print(
            f"  {len(plan.candidate_set) / UNITS:>10.1%} {len(plan.candidate_set):>11} "
            f"{plan.mode:>12} {plan.retrieval_top_k:>7} "
            f"{plan.retrieval_top_k / base_pool:>9.2f}x"
        )
    print("  -> post-filter only scales the pool when top_k / selectivity exceeds it")


def bench_fallback(graph: SemanticGraph) -> None:
    """Compare the two fallback policies on an unsatisfiable intersection."""
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_7"}},
        time_range=TimeRange(start="2025-01-01T00:00:00", end="2025-12-31T23:59:59"),
    )

    print("\nG. degenerate-candidate fallback policy")
    for policy in (FallbackPolicy(), FallbackPolicy(on_degenerate=DROP_CONSTRAINT)):
        planner = ConstraintAwarePlanner(graph, fallback_policy=policy)
        captured: List[str] = []
        original_warning = planner_module.logger.warning
        planner_module.logger.warning = lambda message, *args: captured.append(
            message % args if args else message
        )
        try:
            plan = planner.plan(constraints, TOP_K)
        finally:
            planner_module.logger.warning = original_warning
        print(
            f"  {policy.on_degenerate:<14} -> {len(plan.candidate_set):>4} candidates, "
            f"mode={plan.mode}"
        )
        for message in captured:
            print(f"      warning: {message}")


def main() -> None:
    set_log_level(logging.ERROR)  # the corpus build logs thousands of INFO lines
    print(f"Building synthetic corpus: {UNITS} units, {EDGES} edges ...")
    started = time.perf_counter()
    graph = _build_corpus()
    print(f"Corpus ready in {time.perf_counter() - started:.1f}s")

    bench_metadata(graph)
    bench_time_range(graph)
    bench_relation(graph)
    bench_relation_type(graph)
    bench_modes(graph)
    bench_pool_sizing(graph)
    bench_fallback(graph)

    print(
        "\nNotes: baseline columns are the pre-optimization algorithms run "
        "in-process (linear scan / full edge scan); the timing excludes model\n"
        "encoding and vector search, which dominate end-to-end latency and are "
        "unchanged by these milestones."
    )


if __name__ == "__main__":
    main()
