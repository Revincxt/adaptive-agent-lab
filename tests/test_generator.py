from __future__ import annotations

import unittest
from collections import deque
from pathlib import Path

from adaptive_agent_lab.environment.contracts import Action, Position
from adaptive_agent_lab.environment.events import EventKind
from adaptive_agent_lab.environment.generator import generate_scenario
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment

ROOT = Path(__file__).resolve().parents[1]


def reachable_from(start: Position, scenario: Scenario) -> set[Position]:
    reached = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = current.translated(dx, dy)
            if scenario.map.is_traversable(neighbor) and neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    return reached


class ScenarioGeneratorTests(unittest.TestCase):
    def test_same_seed_has_identical_canonical_json(self) -> None:
        for size in ("tiny", "small", "medium"):
            for dynamics in ("static", "low", "medium", "high"):
                with self.subTest(size=size, dynamics=dynamics):
                    first = generate_scenario(
                        42,
                        size=size,  # type: ignore[arg-type]
                        dynamics=dynamics,  # type: ignore[arg-type]
                        order_count=5,
                        horizon=120,
                    )
                    second = generate_scenario(
                        42,
                        size=size,  # type: ignore[arg-type]
                        dynamics=dynamics,  # type: ignore[arg-type]
                        order_count=5,
                        horizon=120,
                    )
                    self.assertEqual(first.to_json(), second.to_json())

    def test_different_seeds_change_canonical_json(self) -> None:
        first = generate_scenario(7, size="small", dynamics="high", order_count=5)
        second = generate_scenario(8, size="small", dynamics="high", order_count=5)

        self.assertNotEqual(first.to_json(), second.to_json())

    def test_orders_and_charging_stations_are_reachable(self) -> None:
        for size in ("tiny", "small", "medium"):
            scenario = generate_scenario(
                91,
                size=size,  # type: ignore[arg-type]
                dynamics="high",
                order_count=8,
            )
            reachable = reachable_from(scenario.initial_robot.position, scenario)
            required = set(scenario.map.charging_stations)
            required.update(order.pickup for order in scenario.orders)
            required.update(order.dropoff for order in scenario.orders)
            self.assertLessEqual(required, reachable)

    def test_events_are_paired_and_arrivals_match_release_times(self) -> None:
        scenario = generate_scenario(
            2026,
            size="medium",
            dynamics="high",
            order_count=10,
            horizon=180,
        )
        arrivals: dict[str, int] = {}
        blocked: set[Position] = set()
        service_cells = {
            position
            for order in scenario.orders
            for position in (order.pickup, order.dropoff)
        }

        for event in scenario.event_tape:
            self.assertLess(event.time, scenario.horizon)
            if event.kind is EventKind.ORDER_ARRIVAL:
                assert event.order_id is not None
                arrivals[event.order_id] = event.time
            elif event.kind is EventKind.CELL_BLOCKED:
                assert event.position is not None
                self.assertNotIn(event.position, blocked)
                self.assertNotIn(event.position, service_cells)
                self.assertNotIn(event.position, scenario.map.charging_stations)
                blocked.add(event.position)
            else:
                assert event.position is not None
                self.assertIn(event.position, blocked)
                blocked.remove(event.position)

        self.assertFalse(blocked)
        self.assertEqual(
            arrivals,
            {
                order.order_id: order.release_time
                for order in scenario.orders
                if order.release_time > 0
            },
        )

    def test_static_scenario_has_no_events(self) -> None:
        scenario = generate_scenario(
            5,
            size="tiny",
            dynamics="static",
            order_count=3,
            horizon=40,
        )

        self.assertEqual(len(scenario.event_tape), 0)
        self.assertTrue(all(order.release_time == 0 for order in scenario.orders))

    def test_invalid_generation_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_scenario(-1)
        with self.assertRaises(ValueError):
            generate_scenario(1, size="large")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_scenario(1, dynamics="chaotic")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_scenario(1, order_count=-1)
        with self.assertRaises(ValueError):
            generate_scenario(1, horizon=3)

    def test_checked_in_scenarios_are_canonical_and_runnable(self) -> None:
        fixtures = (
            ("scenarios/tiny/static-demo.json", 11, "tiny", "static", 2, 40),
            ("scenarios/small/dynamic-demo.json", 22, "small", "medium", 4, 100),
            ("scenarios/medium/dynamic-demo.json", 33, "medium", "high", 6, 160),
        )
        for relative_path, seed, size, dynamics, order_count, horizon in fixtures:
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                text = path.read_text(encoding="utf-8")
                loaded = Scenario.from_json(text)
                expected = generate_scenario(
                    seed,
                    size=size,  # type: ignore[arg-type]
                    dynamics=dynamics,  # type: ignore[arg-type]
                    order_count=order_count,
                    horizon=horizon,
                    scenario_id=path.stem,
                )
                self.assertEqual(text, expected.to_json(indent=2) + "\n")
                environment = WarehouseEnvironment(loaded)
                result = environment.step(Action.WAIT)
                self.assertEqual(result.state.time, 1)


if __name__ == "__main__":
    unittest.main()
