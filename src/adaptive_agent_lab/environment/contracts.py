"""Immutable domain and runtime contracts for the warehouse environment.

The classes in this module deliberately contain no simulator policy.  They are
small, validated values that can be shared by planners, learning agents, the
environment, and benchmark/reporting code without exposing mutable state.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

InfoValue: TypeAlias = str | int | float | bool | None


def _require_int(name: str, value: int, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _freeze_positions(values: Iterable[Position]) -> frozenset[Position]:
    positions = frozenset(values)
    if not all(isinstance(position, Position) for position in positions):
        raise TypeError("all cells must be Position values")
    return positions


@dataclass(frozen=True, slots=True, order=True)
class Position:
    """An integer coordinate in a warehouse grid.

    Coordinates may be temporarily outside a map while an agent constructs a
    candidate move.  :class:`WarehouseMap` is the authority for map bounds.
    """

    x: int
    y: int

    def __post_init__(self) -> None:
        _require_int("x", self.x)
        _require_int("y", self.y)

    def translated(self, dx: int, dy: int) -> Position:
        """Return a new position translated by the supplied grid delta."""

        _require_int("dx", dx)
        _require_int("dy", dy)
        return Position(self.x + dx, self.y + dy)

    def moved(self, action: Action) -> Position:
        """Return the position reached by a movement action.

        Service actions have a zero delta, which is useful when transition code
        treats every action uniformly.
        """

        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        dx, dy = action.delta
        return self.translated(dx, dy)

    def manhattan_distance(self, other: Position) -> int:
        if not isinstance(other, Position):
            raise TypeError("other must be a Position")
        return abs(self.x - other.x) + abs(self.y - other.y)


class Action(StrEnum):
    """The complete discrete action space for the v0.1 environment."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    PICKUP = "pickup"
    DROPOFF = "dropoff"
    CHARGE = "charge"
    WAIT = "wait"

    @property
    def delta(self) -> tuple[int, int]:
        """Grid delta; service actions deliberately return ``(0, 0)``."""

        if self is Action.UP:
            return (0, -1)
        if self is Action.DOWN:
            return (0, 1)
        if self is Action.LEFT:
            return (-1, 0)
        if self is Action.RIGHT:
            return (1, 0)
        return (0, 0)

    @property
    def is_movement(self) -> bool:
        return self in {Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT}


class OrderStatus(StrEnum):
    """Lifecycle of an order in a single episode."""

    PENDING = "pending"
    AVAILABLE = "available"
    PICKED_UP = "picked_up"
    DELIVERED = "delivered"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in {OrderStatus.DELIVERED, OrderStatus.EXPIRED}


@dataclass(frozen=True, slots=True)
class Order:
    """A pickup-and-delivery request with a release time and deadline."""

    order_id: str
    pickup: Position
    dropoff: Position
    release_time: int
    deadline: int
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.order_id, str):
            raise TypeError("order_id must be a string")
        if not self.order_id.strip():
            raise ValueError("order_id must not be empty")
        if not isinstance(self.pickup, Position) or not isinstance(self.dropoff, Position):
            raise TypeError("pickup and dropoff must be Position values")
        _require_int("release_time", self.release_time, minimum=0)
        _require_int("deadline", self.deadline, minimum=0)
        if self.deadline < self.release_time:
            raise ValueError("deadline must not precede release_time")
        _require_finite("priority", self.priority)
        if self.priority <= 0:
            raise ValueError("priority must be positive")
        object.__setattr__(self, "priority", float(self.priority))


@dataclass(frozen=True, slots=True)
class RobotState:
    """The robot-specific portion of an environment state."""

    position: Position
    battery: int
    carried_order_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, Position):
            raise TypeError("position must be a Position")
        _require_int("battery", self.battery, minimum=0)
        if self.carried_order_id is not None:
            if not isinstance(self.carried_order_id, str):
                raise TypeError("carried_order_id must be a string or None")
            if not self.carried_order_id.strip():
                raise ValueError("carried_order_id must not be empty")


@dataclass(frozen=True, slots=True)
class WarehouseMap:
    """Static grid geometry.

    ``obstacles`` are permanently impassable.  Temporary closures live in
    :class:`WarehouseState` so a single map can be reused across event tapes.
    """

    width: int
    height: int
    obstacles: frozenset[Position] = field(default_factory=frozenset)
    charging_stations: frozenset[Position] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        _require_int("width", self.width, minimum=1)
        _require_int("height", self.height, minimum=1)
        obstacles = _freeze_positions(self.obstacles)
        charging_stations = _freeze_positions(self.charging_stations)
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "charging_stations", charging_stations)

        invalid = {cell for cell in obstacles | charging_stations if not self.contains(cell)}
        if invalid:
            raise ValueError(f"map cells outside bounds: {sorted(invalid)!r}")
        overlap = obstacles & charging_stations
        if overlap:
            raise ValueError(f"charging stations cannot be obstacles: {sorted(overlap)!r}")

    def contains(self, position: Position) -> bool:
        if not isinstance(position, Position):
            raise TypeError("position must be a Position")
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def is_traversable(
        self,
        position: Position,
        blocked_cells: Iterable[Position] = (),
    ) -> bool:
        """Return whether a position is in bounds and currently passable."""

        return (
            self.contains(position)
            and position not in self.obstacles
            and position not in blocked_cells
        )


def _freeze_order_status(
    values: Mapping[str, OrderStatus],
) -> Mapping[str, OrderStatus]:
    normalized: dict[str, OrderStatus] = {}
    for order_id, status in values.items():
        if not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order status keys must be non-empty strings")
        if not isinstance(status, OrderStatus):
            raise TypeError("order status values must be OrderStatus members")
        normalized[order_id] = status
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class WarehouseState:
    """Complete immutable runtime state at one discrete time step."""

    time: int
    robot: RobotState
    order_status: Mapping[str, OrderStatus] = field(default_factory=dict, hash=False)
    blocked_cells: frozenset[Position] = field(default_factory=frozenset)
    cumulative_reward: float = 0.0
    terminated: bool = False

    def __post_init__(self) -> None:
        _require_int("time", self.time, minimum=0)
        if not isinstance(self.robot, RobotState):
            raise TypeError("robot must be a RobotState")
        if not isinstance(self.order_status, Mapping):
            raise TypeError("order_status must be a mapping")
        object.__setattr__(self, "order_status", _freeze_order_status(self.order_status))
        object.__setattr__(self, "blocked_cells", _freeze_positions(self.blocked_cells))
        _require_finite("cumulative_reward", self.cumulative_reward)
        object.__setattr__(self, "cumulative_reward", float(self.cumulative_reward))
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be a boolean")

        carried = self.robot.carried_order_id
        if carried is not None and self.order_status.get(carried) is not OrderStatus.PICKED_UP:
            raise ValueError("the carried order must have PICKED_UP status")

    def status_for(self, order_id: str) -> OrderStatus:
        """Return an order status, raising a descriptive error for unknown IDs."""

        try:
            return self.order_status[order_id]
        except KeyError as error:
            raise KeyError(f"unknown order: {order_id}") from error


@dataclass(frozen=True, slots=True)
class WarehouseSnapshot:
    """Read-only observation passed to every agent implementation."""

    warehouse_map: WarehouseMap
    state: WarehouseState
    orders: tuple[Order, ...]
    horizon: int
    battery_capacity: int

    def __post_init__(self) -> None:
        if not isinstance(self.warehouse_map, WarehouseMap):
            raise TypeError("warehouse_map must be a WarehouseMap")
        if not isinstance(self.state, WarehouseState):
            raise TypeError("state must be a WarehouseState")
        orders = tuple(self.orders)
        if not all(isinstance(order, Order) for order in orders):
            raise TypeError("orders must contain only Order values")
        object.__setattr__(self, "orders", orders)
        _require_int("horizon", self.horizon, minimum=1)
        _require_int("battery_capacity", self.battery_capacity, minimum=1)

        order_ids = tuple(order.order_id for order in orders)
        if len(set(order_ids)) != len(order_ids):
            raise ValueError("order IDs must be unique")
        if set(self.state.order_status) != set(order_ids):
            raise ValueError("state must contain exactly one status for every order")
        if self.state.time > self.horizon:
            raise ValueError("state time cannot exceed the episode horizon")
        if self.state.robot.battery > self.battery_capacity:
            raise ValueError("robot battery cannot exceed battery_capacity")
        # A closure may be applied to the cell the robot currently occupies.
        # Dynamic blocks prevent entering a cell but do not trap a robot that is
        # already there; only static geometry is checked for a snapshot.
        if not self.warehouse_map.is_traversable(self.state.robot.position):
            raise ValueError("robot position must be statically traversable")

    @property
    def map(self) -> WarehouseMap:
        """Concise alias used by planning code."""

        return self.warehouse_map

    def order_by_id(self, order_id: str) -> Order:
        for order in self.orders:
            if order.order_id == order_id:
                return order
        raise KeyError(f"unknown order: {order_id}")


class ViolationCode(StrEnum):
    """Machine-readable reasons why an attempted transition was infeasible."""

    OUT_OF_BOUNDS = "out_of_bounds"
    STATIC_OBSTACLE = "static_obstacle"
    DYNAMIC_BLOCKAGE = "dynamic_blockage"
    BATTERY_DEPLETED = "battery_depleted"
    NO_ORDER_AT_PICKUP = "no_order_at_pickup"
    NOT_CARRYING_ORDER = "not_carrying_order"
    WRONG_DROPOFF = "wrong_dropoff"
    NOT_AT_CHARGER = "not_at_charger"
    EPISODE_TERMINATED = "episode_terminated"


@dataclass(frozen=True, slots=True)
class Violation:
    """A typed constraint violation independent of reward shaping."""

    code: ViolationCode
    message: str
    position: Position | None = None
    order_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, ViolationCode):
            raise TypeError("code must be a ViolationCode")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        if not self.message.strip():
            raise ValueError("message must not be empty")
        if self.position is not None and not isinstance(self.position, Position):
            raise TypeError("position must be a Position or None")
        if self.order_id is not None:
            if not isinstance(self.order_id, str):
                raise TypeError("order_id must be a string or None")
            if not self.order_id.strip():
                raise ValueError("order_id must not be empty")


def _freeze_info(values: Mapping[str, InfoValue]) -> Mapping[str, InfoValue]:
    normalized: dict[str, InfoValue] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("info keys must be non-empty strings")
        if type(value) not in {str, int, float, bool, type(None)}:
            raise TypeError("info values must be scalar JSON-compatible values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("floating-point info values must be finite")
        normalized[key] = value
    return MappingProxyType(dict(sorted(normalized.items())))


def _freeze_violations(values: Iterable[Violation]) -> tuple[Violation, ...]:
    violations = tuple(values)
    if not all(isinstance(violation, Violation) for violation in violations):
        raise TypeError("violations must contain only Violation values")
    return violations


@dataclass(frozen=True, slots=True)
class StepResult:
    """The result of applying one action to an environment."""

    state: WarehouseState
    action: Action
    reward: float
    terminated: bool
    violations: tuple[Violation, ...] = ()
    info: Mapping[str, InfoValue] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, WarehouseState):
            raise TypeError("state must be a WarehouseState")
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        _require_finite("reward", self.reward)
        object.__setattr__(self, "reward", float(self.reward))
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be a boolean")
        if self.terminated is not self.state.terminated:
            raise ValueError("terminated must match the returned state")
        object.__setattr__(self, "violations", _freeze_violations(self.violations))
        if not isinstance(self.info, Mapping):
            raise TypeError("info must be a mapping")
        object.__setattr__(self, "info", _freeze_info(self.info))


@dataclass(frozen=True, slots=True)
class Transition:
    """A replay-ready state/action/next-state transition."""

    state: WarehouseState
    action: Action
    next_state: WarehouseState
    reward: float
    terminated: bool
    violations: tuple[Violation, ...] = ()
    info: Mapping[str, InfoValue] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, WarehouseState):
            raise TypeError("state must be a WarehouseState")
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        if not isinstance(self.next_state, WarehouseState):
            raise TypeError("next_state must be a WarehouseState")
        _require_finite("reward", self.reward)
        object.__setattr__(self, "reward", float(self.reward))
        if not isinstance(self.terminated, bool):
            raise TypeError("terminated must be a boolean")
        if self.terminated is not self.next_state.terminated:
            raise ValueError("terminated must match next_state")
        if self.next_state.time < self.state.time:
            raise ValueError("next_state cannot move backwards in time")
        object.__setattr__(self, "violations", _freeze_violations(self.violations))
        if not isinstance(self.info, Mapping):
            raise TypeError("info must be a mapping")
        object.__setattr__(self, "info", _freeze_info(self.info))

    def as_step_result(self) -> StepResult:
        """Return the environment-facing view of this transition."""

        return StepResult(
            state=self.next_state,
            action=self.action,
            reward=self.reward,
            terminated=self.terminated,
            violations=self.violations,
            info=self.info,
        )


__all__ = [
    "Action",
    "InfoValue",
    "Order",
    "OrderStatus",
    "Position",
    "RobotState",
    "StepResult",
    "Transition",
    "Violation",
    "ViolationCode",
    "WarehouseMap",
    "WarehouseSnapshot",
    "WarehouseState",
]
