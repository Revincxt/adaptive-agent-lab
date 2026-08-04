"""Validated, deterministic scenario serialization."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from adaptive_agent_lab.environment.contracts import (
    Order,
    OrderStatus,
    Position,
    RobotState,
    WarehouseMap,
    WarehouseSnapshot,
    WarehouseState,
)
from adaptive_agent_lab.environment.events import EventKind, EventTape

SCENARIO_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Scenario:
    """All immutable inputs needed to start a warehouse episode."""

    scenario_id: str
    warehouse_map: WarehouseMap
    orders: tuple[Order, ...]
    initial_robot: RobotState
    event_tape: EventTape
    horizon: int
    battery_capacity: int

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str):
            raise TypeError("scenario_id must be a string")
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not isinstance(self.warehouse_map, WarehouseMap):
            raise TypeError("warehouse_map must be a WarehouseMap")
        if not isinstance(self.initial_robot, RobotState):
            raise TypeError("initial_robot must be a RobotState")
        if not isinstance(self.event_tape, EventTape):
            raise TypeError("event_tape must be an EventTape")
        _expect_int(self.horizon, "horizon", minimum=1)
        _expect_int(self.battery_capacity, "battery_capacity", minimum=1)
        if self.initial_robot.battery > self.battery_capacity:
            raise ValueError("initial battery cannot exceed battery_capacity")

        orders = tuple(self.orders)
        if not all(isinstance(order, Order) for order in orders):
            raise TypeError("orders must contain only Order values")
        orders = tuple(sorted(orders, key=lambda order: order.order_id))
        object.__setattr__(self, "orders", orders)
        order_ids = tuple(order.order_id for order in orders)
        if len(set(order_ids)) != len(order_ids):
            raise ValueError("order IDs must be unique")

        if not self.warehouse_map.is_traversable(self.initial_robot.position):
            raise ValueError("initial robot position must be statically traversable")
        for order in orders:
            if not self.warehouse_map.is_traversable(order.pickup):
                raise ValueError(f"pickup for {order.order_id!r} must be traversable")
            if not self.warehouse_map.is_traversable(order.dropoff):
                raise ValueError(f"dropoff for {order.order_id!r} must be traversable")
            if order.release_time >= self.horizon:
                raise ValueError(f"release time for {order.order_id!r} must precede horizon")
            if order.deadline > self.horizon:
                raise ValueError(f"deadline for {order.order_id!r} cannot exceed horizon")

        carried = self.initial_robot.carried_order_id
        if carried is not None:
            if carried not in set(order_ids):
                raise ValueError("initial carried order is not part of the scenario")
            carried_order = self.order_by_id(carried)
            if carried_order.release_time != 0:
                raise ValueError("an initially carried order must be available at time zero")

        self.event_tape.validate_horizon(self.horizon)
        self._validate_events()

    @property
    def map(self) -> WarehouseMap:
        """Concise alias used by environment and planner code."""

        return self.warehouse_map

    @property
    def events(self) -> EventTape:
        """Concise alias for the replayable event tape."""

        return self.event_tape

    def order_by_id(self, order_id: str) -> Order:
        for order in self.orders:
            if order.order_id == order_id:
                return order
        raise KeyError(f"unknown order: {order_id}")

    def initial_state(self) -> WarehouseState:
        """Build the canonical runtime state at time zero."""

        statuses = {
            order.order_id: (
                OrderStatus.AVAILABLE if order.release_time == 0 else OrderStatus.PENDING
            )
            for order in self.orders
        }
        carried = self.initial_robot.carried_order_id
        if carried is not None:
            statuses[carried] = OrderStatus.PICKED_UP
        return WarehouseState(
            time=0,
            robot=self.initial_robot,
            order_status=statuses,
            blocked_cells=frozenset(),
            cumulative_reward=0.0,
            terminated=False,
        )

    def snapshot(self, state: WarehouseState) -> WarehouseSnapshot:
        """Pair a runtime state with this scenario's read-only context."""

        return WarehouseSnapshot(
            warehouse_map=self.warehouse_map,
            state=state,
            orders=self.orders,
            horizon=self.horizon,
            battery_capacity=self.battery_capacity,
        )

    def _validate_events(self) -> None:
        orders_by_id = {order.order_id: order for order in self.orders}
        arrivals: dict[str, int] = {}
        blocked: set[Position] = set()

        for event in self.event_tape:
            if event.kind is EventKind.ORDER_ARRIVAL:
                order_id = event.order_id
                if order_id is None:  # DynamicEvent already guarantees this.
                    raise AssertionError("ORDER_ARRIVAL missing order_id")
                if order_id not in orders_by_id:
                    raise ValueError(f"arrival references unknown order: {order_id!r}")
                if order_id in arrivals:
                    raise ValueError(f"order has more than one arrival: {order_id!r}")
                arrivals[order_id] = event.time
                continue

            position = event.position
            if position is None:  # DynamicEvent already guarantees this.
                raise AssertionError("cell event missing position")
            if not self.warehouse_map.is_traversable(position):
                raise ValueError("cell events must target statically traversable cells")
            if event.kind is EventKind.CELL_BLOCKED:
                if position in blocked:
                    raise ValueError(f"cell is already blocked: {position!r}")
                blocked.add(position)
            else:
                if position not in blocked:
                    raise ValueError(f"cell is not blocked: {position!r}")
                blocked.remove(position)

        expected_arrivals = {
            order.order_id: order.release_time for order in self.orders if order.release_time > 0
        }
        if arrivals != expected_arrivals:
            raise ValueError(
                "order-arrival events must exactly match non-zero release times; "
                f"expected={expected_arrivals!r}, actual={arrivals!r}"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-ready representation."""

        return {
            "battery_capacity": self.battery_capacity,
            "event_tape": self.event_tape.to_dict(),
            "horizon": self.horizon,
            "initial_robot": {
                "battery": self.initial_robot.battery,
                "carried_order_id": self.initial_robot.carried_order_id,
                "position": _position_to_dict(self.initial_robot.position),
            },
            "orders": [
                {
                    "deadline": order.deadline,
                    "dropoff": _position_to_dict(order.dropoff),
                    "order_id": order.order_id,
                    "pickup": _position_to_dict(order.pickup),
                    "priority": order.priority,
                    "release_time": order.release_time,
                }
                for order in self.orders
            ],
            "scenario_id": self.scenario_id,
            "schema_version": SCENARIO_SCHEMA_VERSION,
            "warehouse_map": {
                "charging_stations": [
                    _position_to_dict(position)
                    for position in sorted(self.warehouse_map.charging_stations)
                ],
                "height": self.warehouse_map.height,
                "obstacles": [
                    _position_to_dict(position) for position in sorted(self.warehouse_map.obstacles)
                ],
                "width": self.warehouse_map.width,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Scenario:
        _expect_keys(
            data,
            {
                "battery_capacity",
                "event_tape",
                "horizon",
                "initial_robot",
                "orders",
                "scenario_id",
                "schema_version",
                "warehouse_map",
            },
            "scenario",
        )
        version = _expect_int(data["schema_version"], "scenario.schema_version")
        if version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"unsupported scenario schema version: {version}")

        map_data = _expect_mapping(data["warehouse_map"], "scenario.warehouse_map")
        _expect_keys(
            map_data,
            {"charging_stations", "height", "obstacles", "width"},
            "scenario.warehouse_map",
        )
        obstacles = _position_list(map_data["obstacles"], "scenario.warehouse_map.obstacles")
        charging_stations = _position_list(
            map_data["charging_stations"], "scenario.warehouse_map.charging_stations"
        )
        warehouse_map = WarehouseMap(
            width=_expect_int(map_data["width"], "scenario.warehouse_map.width", minimum=1),
            height=_expect_int(map_data["height"], "scenario.warehouse_map.height", minimum=1),
            obstacles=frozenset(obstacles),
            charging_stations=frozenset(charging_stations),
        )

        robot_data = _expect_mapping(data["initial_robot"], "scenario.initial_robot")
        _expect_keys(
            robot_data,
            {"battery", "carried_order_id", "position"},
            "scenario.initial_robot",
        )
        carried_value = robot_data["carried_order_id"]
        carried_order_id = (
            None
            if carried_value is None
            else _expect_string(carried_value, "scenario.initial_robot.carried_order_id")
        )
        initial_robot = RobotState(
            position=_position_from_dict(
                robot_data["position"], "scenario.initial_robot.position"
            ),
            battery=_expect_int(robot_data["battery"], "scenario.initial_robot.battery", minimum=0),
            carried_order_id=carried_order_id,
        )

        raw_orders = data["orders"]
        if not isinstance(raw_orders, list):
            raise TypeError("scenario.orders must be a JSON array")
        orders = tuple(
            _order_from_dict(value, f"scenario.orders[{index}]")
            for index, value in enumerate(raw_orders)
        )

        event_tape = EventTape.from_dict(
            _expect_mapping(data["event_tape"], "scenario.event_tape")
        )
        return cls(
            scenario_id=_expect_string(data["scenario_id"], "scenario.scenario_id"),
            warehouse_map=warehouse_map,
            orders=orders,
            initial_robot=initial_robot,
            event_tape=event_tape,
            horizon=_expect_int(data["horizon"], "scenario.horizon", minimum=1),
            battery_capacity=_expect_int(
                data["battery_capacity"], "scenario.battery_capacity", minimum=1
            ),
        )

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize with stable key and collection ordering."""

        if indent is not None:
            _expect_int(indent, "indent", minimum=0)
        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str | bytes | bytearray) -> Scenario:
        raw: object = json.loads(text)
        return cls.from_dict(_expect_mapping(raw, "scenario"))


def _position_to_dict(position: Position) -> dict[str, int]:
    return {"x": position.x, "y": position.y}


def _position_from_dict(value: object, path: str) -> Position:
    data = _expect_mapping(value, path)
    _expect_keys(data, {"x", "y"}, path)
    return Position(_expect_int(data["x"], f"{path}.x"), _expect_int(data["y"], f"{path}.y"))


def _position_list(value: object, path: str) -> tuple[Position, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    positions = tuple(
        _position_from_dict(item, f"{path}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(positions)) != len(positions):
        raise ValueError(f"{path} must not contain duplicate positions")
    return positions


def _order_from_dict(value: object, path: str) -> Order:
    data = _expect_mapping(value, path)
    _expect_keys(
        data,
        {"deadline", "dropoff", "order_id", "pickup", "priority", "release_time"},
        path,
    )
    return Order(
        order_id=_expect_string(data["order_id"], f"{path}.order_id"),
        pickup=_position_from_dict(data["pickup"], f"{path}.pickup"),
        dropoff=_position_from_dict(data["dropoff"], f"{path}.dropoff"),
        release_time=_expect_int(data["release_time"], f"{path}.release_time", minimum=0),
        deadline=_expect_int(data["deadline"], f"{path}.deadline", minimum=0),
        priority=_expect_number(data["priority"], f"{path}.priority"),
    )


def _expect_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_keys(data: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {path} keys; missing={missing}, unknown={unknown}")


def _expect_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _expect_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return float(value)


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value.strip():
        raise ValueError(f"{path} must not be empty")
    return value


__all__ = ["SCENARIO_SCHEMA_VERSION", "Scenario"]
