from __future__ import annotations

from collections import deque
from pathlib import Path

from adaptive_agent_lab.agents.planning import OpenLoopPlanningAgent, ReplanningAgent
from adaptive_agent_lab.benchmarking.runner import run_episode
from adaptive_agent_lab.environment.contracts import Position
from adaptive_agent_lab.environment.events import EventKind
from adaptive_agent_lab.environment.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "medium" / "maze-warehouse.json"


def _reachable(
    scenario: Scenario,
    *,
    blocked: frozenset[Position] = frozenset(),
) -> set[Position]:
    reached = {scenario.initial_robot.position}
    queue = deque(reached)
    while queue:
        current = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = current.translated(dx, dy)
            if scenario.map.is_traversable(neighbor, blocked) and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


def test_maze_warehouse_fixture_is_canonical_and_operationally_structured() -> None:
    text = SCENARIO_PATH.read_text(encoding="utf-8")
    scenario = Scenario.from_json(text)

    assert text == scenario.to_json(indent=2) + "\n"
    assert (scenario.map.width, scenario.map.height) == (16, 12)
    assert len(scenario.map.obstacles) == 54
    assert len(scenario.map.charging_stations) == 3

    # Pick faces sit beside rack cells, while every delivery terminates in the
    # south-east packing and outbound staging area.
    for order in scenario.orders:
        rack_neighbors = {
            order.pickup.translated(dx, dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))
        }
        assert rack_neighbors & scenario.map.obstacles
        assert order.dropoff.x >= 13
        assert order.dropoff.y >= 10

    # The north-west receiving bay and south-west dispatch bay each include a
    # charger, and the robot starts at the dispatch charger.
    assert Position(1, 1) in scenario.map.charging_stations
    assert Position(1, 10) in scenario.map.charging_stations
    assert scenario.initial_robot.position == Position(1, 10)


def test_maze_closures_are_paired_and_leave_a_real_detour() -> None:
    scenario = Scenario.from_json(SCENARIO_PATH.read_text(encoding="utf-8"))
    blocked_events = [event for event in scenario.events if event.kind is EventKind.CELL_BLOCKED]
    unblocked_events = [
        event for event in scenario.events if event.kind is EventKind.CELL_UNBLOCKED
    ]

    assert [(event.time, event.position) for event in blocked_events] == [
        (6, Position(3, 5)),
        (55, Position(5, 10)),
    ]
    assert [(event.time, event.position) for event in unblocked_events] == [
        (20, Position(3, 5)),
        (70, Position(5, 10)),
    ]

    required = set(scenario.map.charging_stations)
    required.update(order.pickup for order in scenario.orders)
    required.update(order.dropoff for order in scenario.orders)
    for event in blocked_events:
        assert event.position is not None
        assert required <= _reachable(scenario, blocked=frozenset({event.position}))


def test_replanning_solves_the_paired_maze_tape_without_violations() -> None:
    scenario = Scenario.from_json(SCENARIO_PATH.read_text(encoding="utf-8"))
    open_loop = run_episode(
        OpenLoopPlanningAgent(),
        scenario,
        seed=42,
        explore=False,
        learn=False,
        measure_timing=False,
    )
    replanning = run_episode(
        ReplanningAgent(),
        scenario,
        seed=42,
        explore=False,
        learn=False,
        measure_timing=False,
    )

    assert replanning.metrics.completed_orders == len(scenario.orders)
    assert replanning.metrics.weighted_on_time_completion_rate == 1.0
    assert replanning.metrics.constraint_violations == 0
    assert open_loop.metrics.completed_orders < replanning.metrics.completed_orders
    assert open_loop.metrics.constraint_violations > 0
