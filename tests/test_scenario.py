from __future__ import annotations

import json
import unittest

from adaptive_agent_lab.environment.contracts import (
    Order,
    OrderStatus,
    Position,
    RobotState,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape
from adaptive_agent_lab.environment.scenario import Scenario


def make_scenario(*, reverse_inputs: bool = False) -> Scenario:
    orders = [
        Order("order-a", Position(0, 2), Position(4, 2), 0, 10, 2),
        Order("order-b", Position(0, 1), Position(4, 1), 3, 11),
    ]
    events = [
        DynamicEvent(3, EventKind.ORDER_ARRIVAL, order_id="order-b"),
        DynamicEvent(4, EventKind.CELL_BLOCKED, Position(2, 1)),
        DynamicEvent(7, EventKind.CELL_UNBLOCKED, Position(2, 1)),
    ]
    if reverse_inputs:
        orders.reverse()
        events.reverse()
    obstacles = [Position(1, 0), Position(3, 0)]
    if reverse_inputs:
        obstacles.reverse()
    return Scenario(
        scenario_id="dynamic-demo",
        warehouse_map=WarehouseMap(
            width=5,
            height=3,
            obstacles=frozenset(obstacles),
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=tuple(orders),
        initial_robot=RobotState(Position(0, 0), battery=12),
        event_tape=EventTape(tuple(events)),
        horizon=12,
        battery_capacity=12,
    )


class ScenarioTests(unittest.TestCase):
    def test_initial_state_uses_release_times(self) -> None:
        scenario = make_scenario(reverse_inputs=True)
        state = scenario.initial_state()

        self.assertEqual(tuple(order.order_id for order in scenario.orders), ("order-a", "order-b"))
        self.assertIs(state.status_for("order-a"), OrderStatus.AVAILABLE)
        self.assertIs(state.status_for("order-b"), OrderStatus.PENDING)
        snapshot = scenario.snapshot(state)
        self.assertEqual(snapshot.horizon, 12)
        self.assertEqual(snapshot.battery_capacity, 12)

    def test_json_round_trip_is_canonical_across_input_order(self) -> None:
        first = make_scenario()
        second = make_scenario(reverse_inputs=True)

        self.assertEqual(first, second)
        self.assertEqual(first.to_json(), second.to_json())
        restored = Scenario.from_json(first.to_json())
        self.assertEqual(restored, first)
        self.assertEqual(restored.to_json(), first.to_json())

    def test_pretty_json_also_round_trips(self) -> None:
        scenario = make_scenario()
        encoded = scenario.to_json(indent=2)

        self.assertIn("\n", encoded)
        self.assertEqual(Scenario.from_json(encoded), scenario)

    def test_arrivals_must_match_nonzero_release_times(self) -> None:
        original = make_scenario()
        missing_arrival = EventTape(
            event for event in original.event_tape if event.kind is not EventKind.ORDER_ARRIVAL
        )
        with self.assertRaises(ValueError):
            Scenario(
                original.scenario_id,
                original.warehouse_map,
                original.orders,
                original.initial_robot,
                missing_arrival,
                original.horizon,
                original.battery_capacity,
            )

    def test_cell_event_sequence_must_be_physically_consistent(self) -> None:
        order = Order("order-a", Position(0, 1), Position(2, 1), 0, 5)
        with self.assertRaises(ValueError):
            Scenario(
                "bad-unblock",
                WarehouseMap(3, 2),
                (order,),
                RobotState(Position(0, 0), 5),
                EventTape(
                    (DynamicEvent(2, EventKind.CELL_UNBLOCKED, Position(1, 1)),)
                ),
                horizon=6,
                battery_capacity=5,
            )

    def test_geometry_battery_and_horizon_are_validated(self) -> None:
        original = make_scenario()
        with self.assertRaises(ValueError):
            Scenario(
                original.scenario_id,
                original.warehouse_map,
                original.orders,
                RobotState(Position(0, 0), 13),
                original.event_tape,
                original.horizon,
                12,
            )
        with self.assertRaises(ValueError):
            Scenario(
                "blocked-robot",
                WarehouseMap(2, 2, obstacles={Position(0, 0)}),
                (),
                RobotState(Position(0, 0), 2),
                EventTape(),
                horizon=2,
                battery_capacity=2,
            )

    def test_json_rejects_unknown_fields_and_duplicate_cells(self) -> None:
        data = make_scenario().to_dict()
        data["unknown"] = "value"
        with self.assertRaises(ValueError):
            Scenario.from_json(json.dumps(data))

        data = make_scenario().to_dict()
        warehouse_map = data["warehouse_map"]
        assert isinstance(warehouse_map, dict)
        obstacles = warehouse_map["obstacles"]
        assert isinstance(obstacles, list)
        obstacles.append(dict(obstacles[0]))
        with self.assertRaises(ValueError):
            Scenario.from_json(json.dumps(data))


if __name__ == "__main__":
    unittest.main()
