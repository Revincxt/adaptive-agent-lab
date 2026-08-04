from __future__ import annotations

import unittest

from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    RobotState,
    ViolationCode,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


def dynamic_scenario(*, horizon: int = 20, battery: int = 10) -> Scenario:
    return Scenario(
        scenario_id="dynamic-test",
        warehouse_map=WarehouseMap(
            width=4,
            height=3,
            obstacles=frozenset(),
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(
            Order("order-0", Position(1, 0), Position(3, 0), 0, min(10, horizon)),
            Order("order-1", Position(0, 2), Position(3, 2), 3, min(18, horizon)),
        ),
        initial_robot=RobotState(Position(0, 0), battery),
        event_tape=EventTape(
            (
                DynamicEvent(2, EventKind.CELL_BLOCKED, position=Position(2, 0)),
                DynamicEvent(3, EventKind.ORDER_ARRIVAL, order_id="order-1"),
                DynamicEvent(4, EventKind.CELL_UNBLOCKED, position=Position(2, 0)),
            )
        ),
        horizon=horizon,
        battery_capacity=10,
    )


class SimulatorTests(unittest.TestCase):
    def test_reset_exposes_only_released_orders(self) -> None:
        snapshot = WarehouseEnvironment(dynamic_scenario()).snapshot
        self.assertEqual(snapshot.state.order_status["order-0"], OrderStatus.AVAILABLE)
        self.assertEqual(snapshot.state.order_status["order-1"], OrderStatus.PENDING)

    def test_events_are_applied_after_action_at_their_timestamp(self) -> None:
        environment = WarehouseEnvironment(dynamic_scenario())
        environment.step(Action.RIGHT)
        pickup = environment.step(Action.PICKUP)
        self.assertEqual(pickup.info["picked_order"], "order-0")
        self.assertIn(Position(2, 0), pickup.state.blocked_cells)

        blocked_move = environment.step(Action.RIGHT)
        self.assertFalse(blocked_move.info["valid"])
        self.assertEqual(blocked_move.violations[0].code, ViolationCode.DYNAMIC_BLOCKAGE)
        self.assertEqual(blocked_move.state.order_status["order-1"], OrderStatus.AVAILABLE)

        environment.step(Action.WAIT)
        self.assertNotIn(Position(2, 0), environment.state.blocked_cells)
        successful_move = environment.step(Action.RIGHT)
        self.assertTrue(successful_move.info["valid"])
        self.assertEqual(successful_move.state.robot.position, Position(2, 0))

    def test_pickup_route_and_delivery_update_status_and_reward(self) -> None:
        environment = WarehouseEnvironment(dynamic_scenario())
        for action in (
            Action.RIGHT,
            Action.PICKUP,
            Action.WAIT,
            Action.WAIT,
            Action.RIGHT,
            Action.RIGHT,
            Action.DROPOFF,
        ):
            result = environment.step(action)
        self.assertEqual(result.state.order_status["order-0"], OrderStatus.DELIVERED)
        self.assertIsNone(result.state.robot.carried_order_id)
        self.assertEqual(result.info["delivered_order"], "order-0")
        self.assertGreater(result.reward, 10.0)

    def test_identical_actions_produce_identical_transition_history(self) -> None:
        actions = (Action.RIGHT, Action.PICKUP, Action.WAIT, Action.WAIT, Action.RIGHT)
        first = WarehouseEnvironment(dynamic_scenario())
        second = WarehouseEnvironment(dynamic_scenario())
        for action in actions:
            first.step(action)
            second.step(action)
        self.assertEqual(first.history, second.history)

    def test_zero_battery_prevents_movement_but_charger_can_recover(self) -> None:
        environment = WarehouseEnvironment(dynamic_scenario(battery=0))
        failed = environment.step(Action.RIGHT)
        self.assertEqual(failed.violations[0].code, ViolationCode.BATTERY_DEPLETED)
        charged = environment.step(Action.CHARGE)
        self.assertEqual(charged.state.robot.battery, 2)

    def test_horizon_marks_unfinished_orders_expired(self) -> None:
        scenario = dynamic_scenario(horizon=5)
        environment = WarehouseEnvironment(scenario)
        for _ in range(5):
            result = environment.step(Action.WAIT)
        self.assertTrue(result.terminated)
        self.assertTrue(all(status.is_terminal for status in result.state.order_status.values()))


if __name__ == "__main__":
    unittest.main()
