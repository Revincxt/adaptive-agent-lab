"""Shared episode execution and operational metric extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import cast

from adaptive_agent_lab.agents.base import Agent, AgentDiagnostics
from adaptive_agent_lab.environment.contracts import OrderStatus, Transition
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


@dataclass(frozen=True, slots=True)
class StepRecord:
    time: int
    action: str
    position: tuple[int, int]
    battery: int
    reward: float
    cumulative_reward: float
    carried_order_id: str | None
    delivered_order_id: str | None
    violations: tuple[str, ...]
    event_count: int


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    total_reward: float
    weighted_completion_rate: float
    weighted_on_time_completion_rate: float
    completed_orders: int
    total_orders: int
    mean_lateness: float
    valid_movement_steps: int
    energy_per_completed_order: float | None
    constraint_violations: int
    decision_time_ms: float
    steps: int


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    scenario_id: str
    agent: str
    seed: int
    metrics: EpisodeMetrics
    diagnostics: AgentDiagnostics
    trace: tuple[StepRecord, ...]
    violation_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "agent": self.agent,
            "seed": self.seed,
            "metrics": asdict(self.metrics),
            "diagnostics": asdict(self.diagnostics),
            "trace": [asdict(record) for record in self.trace],
            "violation_counts": dict(sorted(self.violation_counts.items())),
        }


def run_episode(
    agent: Agent,
    scenario: Scenario,
    *,
    seed: int,
    explore: bool = False,
    learn: bool | None = None,
    measure_timing: bool = True,
) -> EpisodeResult:
    """Run one complete episode through the sole authoritative environment."""

    environment = WarehouseEnvironment(scenario)
    agent.set_learning_enabled(explore if learn is None else learn)
    agent.reset(environment.snapshot, seed=seed)
    records: list[StepRecord] = []
    decision_ns = 0

    while not environment.state.terminated:
        started = perf_counter_ns() if measure_timing else 0
        action = agent.act(environment.snapshot, explore=explore)
        if measure_timing:
            decision_ns += perf_counter_ns() - started
        environment.step(action)
        transition = environment.history[-1]
        agent.observe(transition)
        records.append(_step_record(transition))

    agent.end_episode(environment.snapshot)
    metrics = _episode_metrics(
        scenario,
        environment.history,
        decision_time_ms=decision_ns / 1_000_000,
    )
    counts = Counter(
        violation.code.value
        for transition in environment.history
        for violation in transition.violations
    )
    return EpisodeResult(
        scenario_id=scenario.scenario_id,
        agent=agent.name,
        seed=seed,
        metrics=metrics,
        diagnostics=agent.diagnostics,
        trace=tuple(records),
        violation_counts=dict(counts),
    )


def train_episodes(
    agent: Agent,
    scenarios: Iterable[Scenario] | Callable[[int], Scenario],
    *,
    episodes: int,
    root_seed: int,
) -> tuple[EpisodeResult, ...]:
    """Train through explicit episode seeds while retaining agent parameters."""

    if episodes < 1:
        raise ValueError("episodes must be positive")
    if not agent.trainable:
        raise ValueError(f"agent {agent.name!r} does not expose a training lifecycle")
    scenario_values: tuple[Scenario, ...] | None
    if callable(scenarios):
        scenario_values = None
    else:
        scenario_values = tuple(scenarios)
        if not scenario_values:
            raise ValueError("at least one training scenario is required")

    results: list[EpisodeResult] = []
    for episode in range(episodes):
        if callable(scenarios):
            scenario = scenarios(episode)
        else:
            assert scenario_values is not None
            scenario = scenario_values[episode % len(scenario_values)]
        results.append(
            run_episode(
                agent,
                scenario,
                seed=root_seed + episode,
                explore=True,
                measure_timing=False,
            )
        )
    return tuple(results)


def _step_record(transition: Transition) -> StepRecord:
    state = transition.next_state
    delivered = transition.info.get("delivered_order")
    return StepRecord(
        time=state.time,
        action=transition.action.value,
        position=(state.robot.position.x, state.robot.position.y),
        battery=state.robot.battery,
        reward=transition.reward,
        cumulative_reward=state.cumulative_reward,
        carried_order_id=state.robot.carried_order_id,
        delivered_order_id=delivered if isinstance(delivered, str) else None,
        violations=tuple(violation.code.value for violation in transition.violations),
        event_count=int(
            cast(str | int | float, transition.info.get("event_count", 0))
        ),
    )


def _episode_metrics(
    scenario: Scenario,
    transitions: tuple[Transition, ...],
    *,
    decision_time_ms: float,
) -> EpisodeMetrics:
    final_state = transitions[-1].next_state if transitions else scenario.initial_state()
    total_priority = sum(order.priority for order in scenario.orders)
    completed_ids = {
        order_id
        for order_id, status in final_state.order_status.items()
        if status is OrderStatus.DELIVERED
    }
    completion_times = {
        str(transition.info["delivered_order"]): transition.next_state.time
        for transition in transitions
        if isinstance(transition.info.get("delivered_order"), str)
    }
    completed_priority = sum(
        order.priority for order in scenario.orders if order.order_id in completed_ids
    )
    on_time_priority = sum(
        order.priority
        for order in scenario.orders
        if order.order_id in completed_ids
        and completion_times[order.order_id] <= order.deadline
    )
    lateness = [
        max(0, completion_times[order.order_id] - order.deadline)
        for order in scenario.orders
        if order.order_id in completed_ids
    ]
    valid_movements = sum(
        transition.action.is_movement and not transition.violations
        for transition in transitions
    )
    violation_count = sum(len(transition.violations) for transition in transitions)
    completed_count = len(completed_ids)
    return EpisodeMetrics(
        total_reward=final_state.cumulative_reward,
        weighted_completion_rate=(completed_priority / total_priority if total_priority else 1.0),
        weighted_on_time_completion_rate=(
            on_time_priority / total_priority if total_priority else 1.0
        ),
        completed_orders=completed_count,
        total_orders=len(scenario.orders),
        mean_lateness=(sum(lateness) / len(lateness) if lateness else 0.0),
        valid_movement_steps=int(valid_movements),
        energy_per_completed_order=(
            valid_movements / completed_count if completed_count else None
        ),
        constraint_violations=violation_count,
        decision_time_ms=decision_time_ms,
        steps=len(transitions),
    )


__all__ = [
    "EpisodeMetrics",
    "EpisodeResult",
    "StepRecord",
    "run_episode",
    "train_episodes",
]
