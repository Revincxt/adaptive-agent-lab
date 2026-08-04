"""Deterministic orchestration for paired benchmark suites.

The orchestrator keeps scenario randomness separate from agent randomness.  A
condition therefore produces exactly one immutable :class:`Scenario`, shared
by every participating agent, while each agent receives its own stable policy
seed.  This is the central pairing guarantee used by the statistical report.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from typing import cast

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.benchmarking.runner import EpisodeMetrics, EpisodeResult, run_episode
from adaptive_agent_lab.benchmarking.statistics import compare_paired, summarize
from adaptive_agent_lab.environment.generator import (
    DynamicsName,
    SizeName,
    generate_scenario,
)
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.randomness import derive_seed
from adaptive_agent_lab.reporting.artifacts import (
    canonical_json,
    fingerprint,
    read_json,
    write_json_atomic,
)
from adaptive_agent_lab.version import __version__

AgentFactory = Callable[[], Agent]
MetricValue = int | float | None

_KNOWN_SCALES = frozenset(("tiny", "small", "medium"))
_KNOWN_DYNAMICS = frozenset(("static", "low", "medium", "high"))
_LOWER_IS_BETTER = frozenset(
    (
        "mean_lateness",
        "valid_movement_steps",
        "energy_per_completed_order",
        "constraint_violations",
        "decision_time_ms",
        "steps",
        "mean_tardiness_ticks",
        "movement_steps",
        "energy_used",
        "astar_nodes_expanded",
        "replan_count",
    )
)


@dataclass(frozen=True, slots=True)
class BenchmarkCondition:
    """One map/dynamics/load/event-tape cell in a benchmark block."""

    block_id: str
    classification: str
    map_scale: str
    dynamics: str
    order_load: str
    tape_index: int

    def __post_init__(self) -> None:
        _nonempty_string(self.block_id, "condition.block_id")
        _nonempty_string(self.classification, "condition.classification")
        _nonempty_string(self.map_scale, "condition.map_scale")
        _nonempty_string(self.dynamics, "condition.dynamics")
        _nonempty_string(self.order_load, "condition.order_load")
        _integer(self.tape_index, "condition.tape_index", minimum=0)

    @property
    def condition_id(self) -> str:
        """Stable human-readable key used to pair agents."""

        return (
            f"{self.block_id}/{self.map_scale}/{self.dynamics}/"
            f"{self.order_load}/tape-{self.tape_index:04d}"
        )

    @property
    def scenario_id(self) -> str:
        """Scenario key containing only fields allowed to influence generation."""

        return (
            f"benchmark/{self.map_scale}/{self.dynamics}/"
            f"{self.order_load}/tape-{self.tape_index:04d}"
        )

    @property
    def pairing_key(self) -> tuple[str, str, str, str, int]:
        return (
            self.block_id,
            self.map_scale,
            self.dynamics,
            self.order_load,
            self.tape_index,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "classification": self.classification,
            "condition_id": self.condition_id,
            "dynamics": self.dynamics,
            "map_scale": self.map_scale,
            "order_load": self.order_load,
            "tape_index": self.tape_index,
        }


@dataclass(frozen=True, slots=True)
class EvaluationBlock:
    """Validated cartesian-product declaration from the JSON suite config."""

    block_id: str
    classification: str
    map_scales: tuple[str, ...]
    dynamics: tuple[str, ...]
    order_loads: tuple[str, ...]
    event_tapes_per_cell: int
    tape_index_start: int

    def conditions(self) -> tuple[BenchmarkCondition, ...]:
        return tuple(
            BenchmarkCondition(
                block_id=self.block_id,
                classification=self.classification,
                map_scale=scale,
                dynamics=dynamics,
                order_load=load,
                tape_index=tape_index,
            )
            for scale in self.map_scales
            for dynamics in self.dynamics
            for load in self.order_loads
            for tape_index in range(
                self.tape_index_start,
                self.tape_index_start + self.event_tapes_per_cell,
            )
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """The executable subset of ``configs/benchmarks/main.json``."""

    schema_version: str
    suite_id: str
    test_root_seed: int
    agent_ids: tuple[str, ...]
    map_horizons: Mapping[str, int]
    dynamics_names: tuple[str, ...]
    order_counts: Mapping[str, int]
    evaluation_blocks: tuple[EvaluationBlock, ...]
    primary_metric: str
    planned_comparisons: tuple[tuple[str, str], ...]
    bootstrap_resamples: int
    bootstrap_seed: int
    confidence_level: float
    artifact_names: Mapping[str, str]
    raw: Mapping[str, object]

    @property
    def config_fingerprint(self) -> str:
        return fingerprint(self.raw)

    def conditions(
        self,
        *,
        block_ids: Iterable[str] | None = None,
    ) -> tuple[BenchmarkCondition, ...]:
        """Expand configured evaluation blocks in their declared order."""

        selected = None if block_ids is None else frozenset(block_ids)
        if selected is not None:
            known = {block.block_id for block in self.evaluation_blocks}
            unknown = selected - known
            if unknown:
                raise ValueError(f"unknown evaluation blocks: {sorted(unknown)!r}")
        return tuple(
            condition
            for block in self.evaluation_blocks
            if selected is None or block.block_id in selected
            for condition in block.conditions()
        )

    def validate_condition(self, condition: BenchmarkCondition) -> None:
        if condition.map_scale not in self.map_horizons:
            raise ValueError(f"unknown map scale in condition: {condition.map_scale!r}")
        if condition.dynamics not in self.dynamics_names:
            raise ValueError(f"unknown dynamics in condition: {condition.dynamics!r}")
        if condition.order_load not in self.order_counts:
            raise ValueError(f"unknown order load in condition: {condition.order_load!r}")

    def scenario_seed(self, condition: BenchmarkCondition, *, root_seed: int | None = None) -> int:
        """Derive the paired scenario seed; deliberately contains no agent label."""

        self.validate_condition(condition)
        resolved_root = self.test_root_seed if root_seed is None else root_seed
        _integer(resolved_root, "root_seed", minimum=0)
        return derive_scenario_seed(resolved_root, condition)

    def make_scenario(
        self,
        condition: BenchmarkCondition,
        *,
        root_seed: int | None = None,
    ) -> Scenario:
        """Generate the single canonical scenario for a condition."""

        scenario_seed = self.scenario_seed(condition, root_seed=root_seed)
        return generate_scenario(
            scenario_seed,
            size=cast(SizeName, condition.map_scale),
            dynamics=cast(DynamicsName, condition.dynamics),
            order_count=self.order_counts[condition.order_load],
            horizon=self.map_horizons[condition.map_scale],
            scenario_id=condition.scenario_id,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkEpisodeRecord:
    """One agent result plus the provenance needed for paired analysis."""

    condition: BenchmarkCondition
    agent_id: str
    execution_index: int
    scenario_seed: int
    scenario_fingerprint: str
    agent_seed: int
    result: EpisodeResult

    @property
    def metrics(self) -> EpisodeMetrics:
        return self.result.metrics

    def metric_values(self) -> dict[str, MetricValue]:
        """Return canonical metrics plus protocol-facing aliases."""

        metrics = {
            field.name: cast(MetricValue, getattr(self.result.metrics, field.name))
            for field in fields(EpisodeMetrics)
        }
        metrics.update(
            {
                "total_task_utility": self.result.metrics.total_reward,
                "mean_tardiness_ticks": self.result.metrics.mean_lateness,
                "movement_steps": self.result.metrics.valid_movement_steps,
                "energy_used": self.result.metrics.valid_movement_steps,
                "astar_nodes_expanded": self.result.diagnostics.expanded_nodes,
                "replan_count": self.result.diagnostics.planning_calls,
            }
        )
        return metrics

    def to_dict(self) -> dict[str, object]:
        """Encode a compact episode row; trajectories remain separate artifacts."""

        return {
            "agent": self.agent_id,
            "agent_reported_name": self.result.agent,
            "agent_seed": self.agent_seed,
            "condition": self.condition.to_dict(),
            "diagnostics": asdict(self.result.diagnostics),
            "execution_index": self.execution_index,
            "metrics": asdict(self.result.metrics),
            "scenario_fingerprint": self.scenario_fingerprint,
            "scenario_id": self.result.scenario_id,
            "scenario_seed": self.scenario_seed,
            "violation_counts": dict(sorted(self.result.violation_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """In-memory paired benchmark evidence and its serializable reports."""

    episodes: tuple[BenchmarkEpisodeRecord, ...]
    manifest: Mapping[str, object]
    summary: Mapping[str, object]
    artifact_names: Mapping[str, str]

    @property
    def episode_records(self) -> tuple[BenchmarkEpisodeRecord, ...]:
        return self.episodes

    @property
    def records(self) -> tuple[BenchmarkEpisodeRecord, ...]:
        return self.episodes

    @property
    def agent_summaries(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.summary["agents"])

    @property
    def paired_differences(self) -> Mapping[str, object]:
        return cast(Mapping[str, object], self.summary["paired_differences"])

    def write_artifacts(self, output_directory: str | Path) -> Mapping[str, Path]:
        return write_benchmark_artifacts(self, output_directory)


class BenchmarkSuite:
    """Convenience facade around a validated :class:`BenchmarkConfig`."""

    def __init__(self, config: BenchmarkConfig) -> None:
        if not isinstance(config, BenchmarkConfig):
            raise TypeError("config must be a BenchmarkConfig")
        self.config = config

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkSuite:
        return cls(load_benchmark_config(path))

    def conditions(
        self,
        *,
        block_ids: Iterable[str] | None = None,
    ) -> tuple[BenchmarkCondition, ...]:
        return self.config.conditions(block_ids=block_ids)

    def scenario_for(
        self,
        condition: BenchmarkCondition,
        *,
        root_seed: int | None = None,
    ) -> Scenario:
        return self.config.make_scenario(condition, root_seed=root_seed)

    def run(
        self,
        agent_factories: Mapping[str, AgentFactory],
        *,
        conditions: Iterable[BenchmarkCondition] | None = None,
        root_seed: int | None = None,
        measure_timing: bool = False,
    ) -> BenchmarkResult:
        return run_benchmark(
            self.config,
            agent_factories,
            conditions=conditions,
            root_seed=root_seed,
            measure_timing=measure_timing,
        )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate the executable fields of a benchmark JSON file."""

    raw_value = read_json(Path(path))
    raw = _mapping(raw_value, "config")
    schema_version = _string(raw, "schema_version", "config")
    suite_id = _string(raw, "suite_id", "config")

    splits = _child_mapping(raw, "splits", "config")
    test_split = _child_mapping(splits, "test", "config.splits")
    root_seed = _integer_value(test_split, "root_seed", "config.splits.test", minimum=0)

    raw_agents = _list(raw, "agents", "config")
    agent_ids = tuple(
        _string(_mapping(agent, f"config.agents[{index}]"), "id", f"config.agents[{index}]")
        for index, agent in enumerate(raw_agents)
    )
    _require_nonempty_unique(agent_ids, "config.agents[].id")

    raw_scales = _child_mapping(raw, "map_scales", "config")
    if not raw_scales:
        raise ValueError("config.map_scales must not be empty")
    map_horizons: dict[str, int] = {}
    for scale, value in raw_scales.items():
        if scale not in _KNOWN_SCALES:
            raise ValueError(f"config.map_scales contains unsupported scale: {scale!r}")
        data = _mapping(value, f"config.map_scales.{scale}")
        map_horizons[scale] = _integer_value(
            data,
            "horizon_ticks",
            f"config.map_scales.{scale}",
            minimum=4,
        )

    raw_dynamics = _child_mapping(raw, "dynamics", "config")
    if not raw_dynamics:
        raise ValueError("config.dynamics must not be empty")
    unknown_dynamics = set(raw_dynamics) - _KNOWN_DYNAMICS
    if unknown_dynamics:
        raise ValueError(
            "config.dynamics contains unsupported values: "
            f"{sorted(unknown_dynamics)!r}"
        )

    raw_loads = _child_mapping(raw, "order_loads", "config")
    if not raw_loads:
        raise ValueError("config.order_loads must not be empty")
    order_counts: dict[str, int] = {}
    for load, value in raw_loads.items():
        data = _mapping(value, f"config.order_loads.{load}")
        order_counts[load] = _integer_value(
            data,
            "initial_orders",
            f"config.order_loads.{load}",
            minimum=1,
        )

    raw_blocks = _list(raw, "evaluation_blocks", "config")
    blocks = tuple(
        _parse_evaluation_block(
            value,
            index=index,
            scales=frozenset(map_horizons),
            dynamics=frozenset(raw_dynamics),
            loads=frozenset(order_counts),
        )
        for index, value in enumerate(raw_blocks)
    )
    if not blocks:
        raise ValueError("config.evaluation_blocks must not be empty")
    _require_nonempty_unique(
        tuple(block.block_id for block in blocks),
        "config.evaluation_blocks[].id",
    )

    pairing = _child_mapping(raw, "pairing", "config")
    if not _boolean(pairing, "enabled", "config.pairing"):
        raise ValueError("config.pairing.enabled must be true")
    if not _boolean(pairing, "agent_label_excluded_from_event_seed", "config.pairing"):
        raise ValueError("paired scenarios must exclude the agent label from event seeds")
    execution_order = _string(pairing, "agent_execution_order", "config.pairing")
    if execution_order != "rotated_by_tape_index":
        raise ValueError("config.pairing.agent_execution_order must be 'rotated_by_tape_index'")

    metrics = _child_mapping(raw, "metrics", "config")
    primary_metric = _string(metrics, "primary", "config.metrics")
    available_metrics = {field.name for field in fields(EpisodeMetrics)}
    if primary_metric not in available_metrics:
        raise ValueError(f"unsupported primary metric: {primary_metric!r}")

    statistics = _child_mapping(raw, "statistics", "config")
    bootstrap_resamples = _integer_value(
        statistics,
        "bootstrap_resamples",
        "config.statistics",
        minimum=100,
    )
    bootstrap_seed = _integer_value(
        statistics,
        "bootstrap_seed",
        "config.statistics",
        minimum=0,
    )
    confidence_level = _number(statistics, "confidence_level", "config.statistics")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("config.statistics.confidence_level must be between zero and one")
    raw_comparisons = _list(statistics, "planned_comparisons", "config.statistics")
    comparisons = tuple(
        _comparison(value, index=index, agent_ids=frozenset(agent_ids))
        for index, value in enumerate(raw_comparisons)
    )

    artifacts = _child_mapping(raw, "artifacts", "config")
    artifact_names = {
        "manifest": _safe_filename(
            _string(artifacts, "manifest", "config.artifacts"),
            "config.artifacts.manifest",
        ),
        "episodes": _safe_filename(
            _string(artifacts, "episode_records", "config.artifacts"),
            "config.artifacts.episode_records",
        ),
        "summary": _safe_filename(
            _string(artifacts, "summary", "config.artifacts"),
            "config.artifacts.summary",
        ),
    }

    return BenchmarkConfig(
        schema_version=schema_version,
        suite_id=suite_id,
        test_root_seed=root_seed,
        agent_ids=agent_ids,
        map_horizons=map_horizons,
        dynamics_names=tuple(raw_dynamics),
        order_counts=order_counts,
        evaluation_blocks=blocks,
        primary_metric=primary_metric,
        planned_comparisons=comparisons,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        artifact_names=artifact_names,
        raw=dict(raw),
    )


def derive_scenario_seed(root_seed: int, condition: BenchmarkCondition) -> int:
    """Derive a scenario seed solely from the declared pairing dimensions."""

    _integer(root_seed, "root_seed", minimum=0)
    if not isinstance(condition, BenchmarkCondition):
        raise TypeError("condition must be a BenchmarkCondition")
    return derive_seed(
        root_seed,
        "test",
        condition.map_scale,
        condition.dynamics,
        condition.order_load,
        condition.tape_index,
    )


def run_benchmark(
    config: BenchmarkConfig | str | Path,
    agent_factories: Mapping[str, AgentFactory],
    *,
    conditions: Iterable[BenchmarkCondition] | None = None,
    root_seed: int | None = None,
    measure_timing: bool = False,
) -> BenchmarkResult:
    """Run a deterministic paired evaluation over the requested conditions."""

    resolved_config = (
        load_benchmark_config(config) if isinstance(config, str | Path) else config
    )
    if not isinstance(resolved_config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig or JSON path")
    if not isinstance(agent_factories, Mapping) or not agent_factories:
        raise ValueError("agent_factories must be a non-empty mapping")
    if not isinstance(measure_timing, bool):
        raise TypeError("measure_timing must be a boolean")

    unknown_agents = set(agent_factories) - set(resolved_config.agent_ids)
    if unknown_agents:
        raise ValueError(
            "agent factories are not declared by the suite: "
            f"{sorted(unknown_agents)!r}"
        )
    for agent_id, factory in agent_factories.items():
        _nonempty_string(agent_id, "agent factory key")
        if not callable(factory):
            raise TypeError(f"factory for {agent_id!r} must be callable")

    base_agent_order = tuple(
        agent_id for agent_id in resolved_config.agent_ids if agent_id in agent_factories
    )
    selected_conditions = tuple(
        resolved_config.conditions() if conditions is None else conditions
    )
    if not selected_conditions:
        raise ValueError("at least one benchmark condition is required")
    condition_ids: set[str] = set()
    for condition in selected_conditions:
        if not isinstance(condition, BenchmarkCondition):
            raise TypeError("conditions must contain only BenchmarkCondition values")
        resolved_config.validate_condition(condition)
        if condition.condition_id in condition_ids:
            raise ValueError(f"duplicate benchmark condition: {condition.condition_id!r}")
        condition_ids.add(condition.condition_id)

    resolved_root = resolved_config.test_root_seed if root_seed is None else root_seed
    _integer(resolved_root, "root_seed", minimum=0)
    records: list[BenchmarkEpisodeRecord] = []
    scenario_entries: list[dict[str, object]] = []

    for condition in selected_conditions:
        scenario_seed = resolved_config.scenario_seed(condition, root_seed=resolved_root)
        scenario = resolved_config.make_scenario(condition, root_seed=resolved_root)
        scenario_fingerprint = fingerprint(scenario.to_dict())
        execution_order = _rotated(base_agent_order, condition.tape_index)
        scenario_entries.append(
            {
                "condition_id": condition.condition_id,
                "execution_order": list(execution_order),
                "fingerprint": scenario_fingerprint,
                "scenario_seed": scenario_seed,
            }
        )
        for execution_index, agent_id in enumerate(execution_order):
            agent = agent_factories[agent_id]()
            if not isinstance(agent, Agent):
                raise TypeError(f"factory for {agent_id!r} did not return an Agent")
            agent_seed = derive_seed(
                resolved_root,
                "agent",
                agent_id,
                condition.map_scale,
                condition.dynamics,
                condition.order_load,
                condition.tape_index,
            )
            result = run_episode(
                agent,
                scenario,
                seed=agent_seed,
                explore=False,
                measure_timing=measure_timing,
            )
            records.append(
                BenchmarkEpisodeRecord(
                    condition=condition,
                    agent_id=agent_id,
                    execution_index=execution_index,
                    scenario_seed=scenario_seed,
                    scenario_fingerprint=scenario_fingerprint,
                    agent_seed=agent_seed,
                    result=result,
                )
            )

    summary = _summarize_records(resolved_config, tuple(records), base_agent_order)
    manifest: dict[str, object] = {
        "agents": list(base_agent_order),
        "condition_count": len(selected_conditions),
        "config_fingerprint": resolved_config.config_fingerprint,
        "episode_count": len(records),
        "kind": "paired_benchmark",
        "measure_timing": measure_timing,
        "package_version": __version__,
        "pairing_unit": "condition_event_tape",
        "root_seed": resolved_root,
        "scenarios": scenario_entries,
        "schema_version": resolved_config.schema_version,
        "suite_id": resolved_config.suite_id,
    }
    return BenchmarkResult(
        episodes=tuple(records),
        manifest=manifest,
        summary=summary,
        artifact_names=dict(resolved_config.artifact_names),
    )


def write_benchmark_artifacts(
    result: BenchmarkResult,
    output_directory: str | Path,
) -> Mapping[str, Path]:
    """Atomically write compact episode rows, summary, then completion manifest."""

    if not isinstance(result, BenchmarkResult):
        raise TypeError("result must be a BenchmarkResult")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        key: directory / _safe_filename(name, f"artifact_names.{key}")
        for key, name in result.artifact_names.items()
    }
    required = {"manifest", "episodes", "summary"}
    if set(paths) != required:
        raise ValueError(f"artifact_names must contain exactly {sorted(required)!r}")

    _write_jsonl_atomic(
        paths["episodes"],
        (record.to_dict() for record in result.episodes),
    )
    write_json_atomic(paths["summary"], result.summary)
    # Write the manifest last so its presence acts as the run-complete marker.
    write_json_atomic(paths["manifest"], result.manifest)
    return paths


def _summarize_records(
    config: BenchmarkConfig,
    records: tuple[BenchmarkEpisodeRecord, ...],
    agent_order: tuple[str, ...],
) -> dict[str, object]:
    by_agent: dict[str, list[BenchmarkEpisodeRecord]] = {agent: [] for agent in agent_order}
    for record in records:
        by_agent[record.agent_id].append(record)

    agent_summaries: dict[str, object] = {}
    for agent_id in agent_order:
        agent_records = by_agent[agent_id]
        metric_names = tuple(agent_records[0].metric_values())
        metrics: dict[str, object] = {}
        for metric_name in metric_names:
            values = [
                float(value)
                for record in agent_records
                if (value := record.metric_values()[metric_name]) is not None
            ]
            if not values:
                continue
            seed = derive_seed(config.bootstrap_seed, "summary", agent_id, metric_name)
            metrics[metric_name] = asdict(
                summarize(
                    values,
                    confidence=config.confidence_level,
                    resamples=config.bootstrap_resamples,
                    seed=seed,
                )
            )
        agent_summaries[agent_id] = {
            "episode_count": len(agent_records),
            "metrics": metrics,
        }

    by_agent_and_condition = {
        agent_id: {record.condition.condition_id: record for record in agent_records}
        for agent_id, agent_records in by_agent.items()
    }
    paired_summaries: dict[str, object] = {}
    for candidate, baseline in config.planned_comparisons:
        if candidate not in by_agent or baseline not in by_agent:
            continue
        candidate_records = by_agent_and_condition[candidate]
        baseline_records = by_agent_and_condition[baseline]
        shared_condition_ids = tuple(
            record.condition.condition_id
            for record in records
            if record.agent_id == candidate
            and record.condition.condition_id in baseline_records
        )
        metric_names = tuple(next(iter(candidate_records.values())).metric_values())
        metric_differences: dict[str, object] = {}
        for metric_name in metric_names:
            paired_values = [
                (
                    candidate_records[key].metric_values()[metric_name],
                    baseline_records[key].metric_values()[metric_name],
                )
                for key in shared_condition_ids
            ]
            candidate_values = [
                float(left)
                for left, right in paired_values
                if left is not None and right is not None
            ]
            baseline_values = [
                float(right)
                for left, right in paired_values
                if left is not None and right is not None
            ]
            if not candidate_values:
                continue
            higher_is_better = metric_name not in _LOWER_IS_BETTER
            seed = derive_seed(
                config.bootstrap_seed,
                "paired",
                candidate,
                baseline,
                metric_name,
            )
            metric_differences[metric_name] = {
                **asdict(
                    compare_paired(
                        candidate_values,
                        baseline_values,
                        higher_is_better=higher_is_better,
                        confidence=config.confidence_level,
                        resamples=config.bootstrap_resamples,
                        seed=seed,
                    )
                ),
                "higher_is_better": higher_is_better,
            }
        comparison_id = f"{candidate}_vs_{baseline}"
        paired_summaries[comparison_id] = {
            "baseline": baseline,
            "candidate": candidate,
            "condition_count": len(shared_condition_ids),
            "metrics": metric_differences,
        }

    return {
        "agents": agent_summaries,
        "paired_differences": paired_summaries,
        "primary_metric": config.primary_metric,
        "suite_id": config.suite_id,
    }


def _rotated(values: tuple[str, ...], tape_index: int) -> tuple[str, ...]:
    if not values:
        return ()
    offset = tape_index % len(values)
    return (*values[offset:], *values[:offset])


def _parse_evaluation_block(
    value: object,
    *,
    index: int,
    scales: frozenset[str],
    dynamics: frozenset[str],
    loads: frozenset[str],
) -> EvaluationBlock:
    path = f"config.evaluation_blocks[{index}]"
    data = _mapping(value, path)
    block_id = _string(data, "id", path)
    classification = _string(data, "classification", path)
    block_scales = _string_tuple(data, "map_scales", path)
    block_dynamics = _string_tuple(data, "dynamics", path)
    block_loads = _string_tuple(data, "order_loads", path)
    _validate_references(block_scales, scales, f"{path}.map_scales")
    _validate_references(block_dynamics, dynamics, f"{path}.dynamics")
    _validate_references(block_loads, loads, f"{path}.order_loads")
    return EvaluationBlock(
        block_id=block_id,
        classification=classification,
        map_scales=block_scales,
        dynamics=block_dynamics,
        order_loads=block_loads,
        event_tapes_per_cell=_integer_value(
            data,
            "event_tapes_per_cell",
            path,
            minimum=1,
        ),
        tape_index_start=_integer_value(
            data,
            "test_tape_index_start",
            path,
            minimum=0,
        ),
    )


def _comparison(
    value: object,
    *,
    index: int,
    agent_ids: frozenset[str],
) -> tuple[str, str]:
    path = f"config.statistics.planned_comparisons[{index}]"
    values = _sequence(value, path)
    if len(values) != 2:
        raise ValueError(f"{path} must contain candidate and baseline")
    candidate = _nonempty_string(values[0], f"{path}[0]")
    baseline = _nonempty_string(values[1], f"{path}[1]")
    if candidate == baseline:
        raise ValueError(f"{path} must compare two different agents")
    unknown = {candidate, baseline} - agent_ids
    if unknown:
        raise ValueError(f"{path} references unknown agents: {sorted(unknown)!r}")
    return candidate, baseline


def _write_jsonl_atomic(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical_json(row))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} must be a JSON object")
    return cast(Mapping[str, object], value)


def _child_mapping(data: Mapping[str, object], key: str, path: str) -> Mapping[str, object]:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    return _mapping(data[key], f"{path}.{key}")


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"{path} must be a JSON array")
    return cast(Sequence[object], value)


def _list(data: Mapping[str, object], key: str, path: str) -> Sequence[object]:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    return _sequence(data[key], f"{path}.{key}")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value.strip():
        raise ValueError(f"{path} must not be empty")
    return value


def _string(data: Mapping[str, object], key: str, path: str) -> str:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    return _nonempty_string(data[key], f"{path}.{key}")


def _string_tuple(data: Mapping[str, object], key: str, path: str) -> tuple[str, ...]:
    values = _list(data, key, path)
    result = tuple(
        _nonempty_string(value, f"{path}.{key}[{index}]")
        for index, value in enumerate(values)
    )
    _require_nonempty_unique(result, f"{path}.{key}")
    return result


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _integer_value(
    data: Mapping[str, object],
    key: str,
    path: str,
    *,
    minimum: int,
) -> int:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    return _integer(data[key], f"{path}.{key}", minimum=minimum)


def _number(data: Mapping[str, object], key: str, path: str) -> float:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{path}.{key} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{path}.{key} must be finite")
    return result


def _boolean(data: Mapping[str, object], key: str, path: str) -> bool:
    if key not in data:
        raise ValueError(f"{path}.{key} is required")
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{path}.{key} must be a boolean")
    return value


def _require_nonempty_unique(values: tuple[str, ...], path: str) -> None:
    if not values:
        raise ValueError(f"{path} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{path} must not contain duplicates")


def _validate_references(
    values: tuple[str, ...],
    allowed: frozenset[str],
    path: str,
) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{path} references unknown values: {sorted(unknown)!r}")


def _safe_filename(value: str, path: str) -> str:
    name = _nonempty_string(value, path)
    candidate = Path(name)
    if candidate.is_absolute() or candidate.name != name or name in {".", ".."}:
        raise ValueError(f"{path} must be a plain filename")
    return name


__all__ = [
    "AgentFactory",
    "BenchmarkCondition",
    "BenchmarkConfig",
    "BenchmarkEpisodeRecord",
    "BenchmarkResult",
    "BenchmarkSuite",
    "EvaluationBlock",
    "derive_scenario_seed",
    "load_benchmark_config",
    "run_benchmark",
    "write_benchmark_artifacts",
]
