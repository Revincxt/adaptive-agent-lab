from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    RobotState,
    StepResult,
    Transition,
    Violation,
    ViolationCode,
    WarehouseMap,
    WarehouseSnapshot,
    WarehouseState,
)


class PositionAndMapTests(unittest.TestCase):
    def test_position_supports_movement_and_distance(self) -> None:
        origin = Position(2, 3)

        self.assertEqual(origin.moved(Action.UP), Position(2, 2))
        self.assertEqual(origin.moved(Action.RIGHT), Position(3, 3))
        self.assertEqual(origin.moved(Action.WAIT), origin)
        self.assertEqual(origin.manhattan_distance(Position(5, 1)), 5)

    def test_values_and_collections_are_immutable(self) -> None:
        position = Position(1, 1)
        warehouse_map = WarehouseMap(3, 3, {Position(2, 2)}, {Position(0, 0)})

        with self.assertRaises(FrozenInstanceError):
            position.x = 2  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            warehouse_map.obstacles.add(Position(1, 2))  # type: ignore[attr-defined]

    def test_map_distinguishes_static_and_dynamic_blocks(self) -> None:
        warehouse_map = WarehouseMap(3, 2, {Position(1, 0)}, {Position(0, 0)})

        self.assertTrue(warehouse_map.is_traversable(Position(2, 1)))
        self.assertFalse(warehouse_map.is_traversable(Position(1, 0)))
        self.assertFalse(
            warehouse_map.is_traversable(Position(2, 1), {Position(2, 1)})
        )
        self.assertFalse(warehouse_map.contains(Position(-1, 0)))

    def test_map_rejects_invalid_geometry(self) -> None:
        with self.assertRaises(ValueError):
            WarehouseMap(2, 2, {Position(2, 0)})
        with self.assertRaises(ValueError):
            WarehouseMap(2, 2, {Position(0, 0)}, {Position(0, 0)})


class RuntimeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.order = Order(
            order_id="order-1",
            pickup=Position(0, 1),
            dropoff=Position(2, 1),
            release_time=0,
            deadline=8,
            priority=2,
        )
        self.warehouse_map = WarehouseMap(3, 2, charging_stations={Position(0, 0)})
        self.state = WarehouseState(
            time=0,
            robot=RobotState(Position(0, 0), battery=8),
            order_status={"order-1": OrderStatus.AVAILABLE},
        )

    def test_state_freezes_status_and_dynamic_cells(self) -> None:
        statuses = {"order-1": OrderStatus.AVAILABLE}
        blocked = {Position(1, 0)}
        state = WarehouseState(
            time=1,
            robot=RobotState(Position(0, 0), 7),
            order_status=statuses,
            blocked_cells=blocked,
        )
        statuses["order-1"] = OrderStatus.EXPIRED
        blocked.add(Position(2, 0))

        self.assertIs(state.status_for("order-1"), OrderStatus.AVAILABLE)
        self.assertEqual(state.blocked_cells, frozenset({Position(1, 0)}))
        with self.assertRaises(TypeError):
            state.order_status["order-1"] = OrderStatus.DELIVERED  # type: ignore[index]

    def test_carried_order_must_match_runtime_status(self) -> None:
        with self.assertRaises(ValueError):
            WarehouseState(
                time=1,
                robot=RobotState(Position(0, 0), 7, "order-1"),
                order_status={"order-1": OrderStatus.AVAILABLE},
            )

    def test_snapshot_is_read_only_and_allows_closure_under_robot(self) -> None:
        occupied_and_blocked = WarehouseState(
            time=2,
            robot=RobotState(Position(1, 1), 6),
            order_status={"order-1": OrderStatus.AVAILABLE},
            blocked_cells={Position(1, 1)},
        )
        snapshot = WarehouseSnapshot(
            warehouse_map=self.warehouse_map,
            state=occupied_and_blocked,
            orders=[self.order],
            horizon=10,
            battery_capacity=8,
        )

        self.assertIs(snapshot.map, self.warehouse_map)
        self.assertEqual(snapshot.order_by_id("order-1"), self.order)
        self.assertIsInstance(snapshot.orders, tuple)

    def test_snapshot_requires_exact_status_coverage(self) -> None:
        with self.assertRaises(ValueError):
            WarehouseSnapshot(
                self.warehouse_map,
                self.state,
                (),
                horizon=10,
                battery_capacity=8,
            )

    def test_transition_and_step_result_keep_typed_diagnostics(self) -> None:
        next_state = WarehouseState(
            time=1,
            robot=RobotState(Position(0, 0), 8),
            order_status={"order-1": OrderStatus.AVAILABLE},
            cumulative_reward=-1,
        )
        violation = Violation(
            ViolationCode.NOT_AT_CHARGER,
            "robot is not on a charging station",
            position=Position(0, 0),
        )
        transition = Transition(
            state=self.state,
            action=Action.CHARGE,
            next_state=next_state,
            reward=-1,
            terminated=False,
            violations=[violation],
            info={"valid": False, "decision_ms": 0.25},
        )

        result = transition.as_step_result()
        self.assertIsInstance(result, StepResult)
        self.assertEqual(result.action, Action.CHARGE)
        self.assertEqual(result.violations, (violation,))
        self.assertEqual(result.info["valid"], False)
        with self.assertRaises(TypeError):
            result.info["valid"] = True  # type: ignore[index]

    def test_result_rejects_non_finite_reward_and_flag_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            StepResult(self.state, Action.WAIT, float("nan"), False)
        with self.assertRaises(ValueError):
            StepResult(self.state, Action.WAIT, 0.0, True)


if __name__ == "__main__":
    unittest.main()
