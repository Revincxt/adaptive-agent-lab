"""Build the browser replay artifact from real environment trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.dqn import DQNAgent, DQNConfig
from adaptive_agent_lab.agents.hybrid import HybridAgent, HybridConfig
from adaptive_agent_lab.agents.planning import OpenLoopPlanningAgent, ReplanningAgent
from adaptive_agent_lab.agents.tabular import DynaQAgent, QLearningAgent
from adaptive_agent_lab.benchmarking.runner import EpisodeResult, run_episode, train_episodes
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.reporting.artifacts import fingerprint, write_json_atomic

DEFAULT_TRAINING_EPISODES: Mapping[str, int] = {
    "q-learning": 120,
    "dyna-q": 80,
    "dqn": 80,
    "hybrid": 40,
}

AGENT_PRESENTATION: Mapping[str, tuple[str, str, str]] = {
    "planning": (
        "Open-loop A*",
        "Planning",
        "Plans once against the nominal map and deliberately does not repair disruptions.",
    ),
    "replanning": (
        "Event-triggered A*",
        "Replanning",
        "Repairs the current route when closures, arrivals, or service state invalidate it.",
    ),
    "q-learning": (
        "Q-learning",
        "Model-free RL",
        "Learns primitive action values from real interactions over an exact tabular state.",
    ),
    "dyna-q": (
        "Dyna-Q",
        "Model-based RL",
        "Adds seeded simulated updates from a learned deterministic transition model.",
    ),
    "dqn": (
        "NumPy DQN",
        "Deep RL",
        "Approximates masked primitive action values with a from-scratch multilayer network.",
    ),
    "hybrid": (
        "Learning + A*",
        "Hybrid",
        "Learns the next task option while A* supplies inspectable, constraint-aware routing.",
    ),
}

AGENT_COLORS: Mapping[str, str] = {
    "planning": "#727776",
    "replanning": "#1d6f68",
    "q-learning": "#c3983f",
    "dyna-q": "#8b6fb0",
    "dqn": "#397caf",
    "hybrid": "#dc735b",
}


def make_demo_agents() -> tuple[Agent, ...]:
    """Construct fresh agents with small, documented demonstration settings."""

    return (
        OpenLoopPlanningAgent(),
        ReplanningAgent(),
        QLearningAgent(alpha=0.15, epsilon_decay=0.999, epsilon_min=0.05),
        DynaQAgent(
            alpha=0.15,
            epsilon_decay=0.998,
            epsilon_min=0.05,
            planning_steps=20,
        ),
        DQNAgent(
            DQNConfig(
                hidden_sizes=(32, 32),
                batch_size=32,
                replay_capacity=5_000,
                warmup_steps=64,
                target_sync_interval=100,
                epsilon_decay_steps=5_000,
            )
        ),
        HybridAgent(HybridConfig(hidden=(32, 32), epsilon=0.20)),
    )


def build_demo_data(
    scenario: Scenario,
    *,
    root_seed: int = 42,
    training_episodes: Mapping[str, int] = DEFAULT_TRAINING_EPISODES,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Train configured learners, then evaluate every method on one paired tape."""

    results: list[EpisodeResult] = []
    training_counts: dict[str, int] = {}
    for index, agent in enumerate(make_demo_agents()):
        episodes = int(training_episodes.get(agent.name, 0)) if agent.trainable else 0
        if episodes < 0:
            raise ValueError("training episode counts must be non-negative")
        training_counts[agent.name] = episodes
        if episodes:
            train_episodes(
                agent,
                (scenario,),
                episodes=episodes,
                root_seed=root_seed * 10_000 + index * 1_000,
            )
        results.append(
            run_episode(
                agent,
                scenario,
                seed=root_seed,
                explore=False,
                learn=False,
                measure_timing=False,
            )
        )

    timestamp = (generated_at or datetime.now(UTC)).astimezone(UTC).isoformat()
    return {
        "generatedAt": timestamp,
        "verificationStatus": "DEMO · NON-CONFIRMATORY · PAIRED TAPE",
        "scenarioFingerprint": fingerprint(scenario.to_dict()),
        "scenario": _scenario_payload(scenario),
        "trainingEpisodes": training_counts,
        "agents": [_agent_payload(result) for result in results],
    }


def write_demo_data(
    path: Path,
    scenario: Scenario,
    *,
    root_seed: int = 42,
    training_episodes: Mapping[str, int] = DEFAULT_TRAINING_EPISODES,
) -> dict[str, object]:
    payload = build_demo_data(
        scenario,
        root_seed=root_seed,
        training_episodes=training_episodes,
    )
    write_json_atomic(path, payload)
    return payload


def _scenario_payload(scenario: Scenario) -> dict[str, object]:
    return {
        "id": scenario.scenario_id,
        "width": scenario.map.width,
        "height": scenario.map.height,
        "horizon": scenario.horizon,
        "batteryCapacity": scenario.battery_capacity,
        "initialRobot": {
            "x": scenario.initial_robot.position.x,
            "y": scenario.initial_robot.position.y,
        },
        "obstacles": [
            {"x": position.x, "y": position.y}
            for position in sorted(scenario.map.obstacles)
        ],
        "chargingStations": [
            {"x": position.x, "y": position.y}
            for position in sorted(scenario.map.charging_stations)
        ],
        "orders": [
            {
                "id": order.order_id,
                "pickup": {"x": order.pickup.x, "y": order.pickup.y},
                "dropoff": {"x": order.dropoff.x, "y": order.dropoff.y},
                "releaseTime": order.release_time,
                "deadline": order.deadline,
                "priority": order.priority,
            }
            for order in scenario.orders
        ],
        "events": [
            {
                "time": event.time,
                "kind": event.kind.value,
                **(
                    {"position": {"x": event.position.x, "y": event.position.y}}
                    if event.position is not None
                    else {}
                ),
                **({"orderId": event.order_id} if event.order_id is not None else {}),
            }
            for event in scenario.event_tape
        ],
    }


def _agent_payload(result: EpisodeResult) -> dict[str, object]:
    label, family, description = AGENT_PRESENTATION[result.agent]
    metrics = result.metrics
    return {
        "id": result.agent,
        "label": label,
        "family": family,
        "description": description,
        "color": AGENT_COLORS[result.agent],
        "metrics": {
            "weightedCompletionRate": metrics.weighted_completion_rate,
            "weightedOnTimeCompletionRate": metrics.weighted_on_time_completion_rate,
            "totalReward": metrics.total_reward,
            "completedOrders": metrics.completed_orders,
            "totalOrders": metrics.total_orders,
            "constraintViolations": metrics.constraint_violations,
            "decisionTimeMs": metrics.decision_time_ms,
            "steps": metrics.steps,
        },
        "planningCalls": result.diagnostics.planning_calls,
        "expandedNodes": result.diagnostics.expanded_nodes,
        "learningUpdates": result.diagnostics.learning_updates,
        "trace": [
            {
                "time": record.time,
                "action": record.action,
                "position": list(record.position),
                "battery": record.battery,
                "reward": record.reward,
                "cumulativeReward": record.cumulative_reward,
                "carriedOrderId": record.carried_order_id,
                "deliveredOrderId": record.delivered_order_id,
                "violations": list(record.violations),
                "eventCount": record.event_count,
            }
            for record in result.trace
        ],
    }


__all__ = [
    "DEFAULT_TRAINING_EPISODES",
    "build_demo_data",
    "make_demo_agents",
    "write_demo_data",
]
