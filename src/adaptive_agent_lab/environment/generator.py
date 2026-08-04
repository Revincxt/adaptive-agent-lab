"""Fully seeded scenario generation for reproducible warehouse experiments."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Final, Literal

from adaptive_agent_lab.environment.contracts import (
    Order,
    Position,
    RobotState,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.randomness import derive_seed

SizeName = Literal["tiny", "small", "medium"]
DynamicsName = Literal["static", "low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class _SizeSpec:
    width: int
    height: int
    obstacle_fraction: float
    charging_station_count: int
    battery_capacity: int
    default_horizon: int


_SIZE_SPECS: Final[dict[str, _SizeSpec]] = {
    "tiny": _SizeSpec(6, 5, 0.10, 1, 24, 80),
    "small": _SizeSpec(10, 8, 0.15, 2, 40, 180),
    "medium": _SizeSpec(16, 12, 0.18, 3, 64, 320),
}

_CLOSURE_RATES: Final[dict[str, float]] = {
    "static": 0.0,
    "low": 0.010,
    "medium": 0.020,
    "high": 0.040,
}

_MINIMUM_CLOSURE_PAIRS: Final[dict[str, int]] = {
    "static": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

_RELEASE_WINDOWS: Final[dict[str, float]] = {
    "static": 0.0,
    "low": 0.35,
    "medium": 0.55,
    "high": 0.70,
}


def generate_scenario(
    seed: int,
    *,
    size: SizeName = "small",
    dynamics: DynamicsName = "medium",
    order_count: int = 4,
    horizon: int | None = None,
    scenario_id: str | None = None,
) -> Scenario:
    """Generate one canonical scenario from explicit inputs.

    Separate deterministic random streams are used for geometry, orders, and
    events.  Changing event-generation details therefore cannot silently change
    the underlying map for the same root seed and size.
    """

    _require_int("seed", seed, minimum=0)
    if not isinstance(size, str):
        raise TypeError("size must be a string")
    if size not in _SIZE_SPECS:
        raise ValueError(f"unknown size: {size!r}")
    if not isinstance(dynamics, str):
        raise TypeError("dynamics must be a string")
    if dynamics not in _CLOSURE_RATES:
        raise ValueError(f"unknown dynamics: {dynamics!r}")
    _require_int("order_count", order_count, minimum=0)

    spec = _SIZE_SPECS[size]
    if horizon is None:
        resolved_horizon = spec.default_horizon
    else:
        _require_int("horizon", horizon, minimum=4)
        resolved_horizon = horizon

    geometry_rng = random.Random(derive_seed(seed, "generator", "geometry", size))
    order_rng = random.Random(
        derive_seed(seed, "generator", "orders", size, dynamics, resolved_horizon)
    )
    event_rng = random.Random(
        derive_seed(seed, "generator", "events", size, dynamics, resolved_horizon)
    )

    warehouse_map, robot_position = _generate_map(spec, geometry_rng)
    orders = _generate_orders(
        warehouse_map=warehouse_map,
        robot_position=robot_position,
        count=order_count,
        horizon=resolved_horizon,
        dynamics=dynamics,
        rng=order_rng,
    )
    arrival_events = tuple(
        DynamicEvent(
            time=order.release_time,
            kind=EventKind.ORDER_ARRIVAL,
            order_id=order.order_id,
        )
        for order in orders
        if order.release_time > 0
    )
    closure_events = _generate_closure_events(
        warehouse_map=warehouse_map,
        robot_position=robot_position,
        orders=orders,
        dynamics=dynamics,
        horizon=resolved_horizon,
        rng=event_rng,
    )

    identifier = scenario_id
    if identifier is None:
        identifier = (
            f"generated-{size}-{dynamics}-s{seed}-o{order_count}-h{resolved_horizon}"
        )
    return Scenario(
        scenario_id=identifier,
        warehouse_map=warehouse_map,
        orders=orders,
        initial_robot=RobotState(robot_position, battery=spec.battery_capacity),
        event_tape=EventTape(arrival_events + closure_events),
        horizon=resolved_horizon,
        battery_capacity=spec.battery_capacity,
    )


def _generate_map(spec: _SizeSpec, rng: random.Random) -> tuple[WarehouseMap, Position]:
    all_cells = [
        Position(x, y) for y in range(spec.height) for x in range(spec.width)
    ]
    shuffled = list(all_cells)
    rng.shuffle(shuffled)
    robot_position = shuffled[0]

    charging_stations = {robot_position}
    for position in shuffled[1:]:
        if len(charging_stations) >= spec.charging_station_count:
            break
        charging_stations.add(position)

    obstacles: set[Position] = set()
    candidates = [cell for cell in shuffled if cell not in charging_stations]
    target_count = int(len(all_cells) * spec.obstacle_fraction)
    all_cell_set = set(all_cells)
    for candidate in candidates:
        if len(obstacles) >= target_count:
            break
        proposed = obstacles | {candidate}
        if _is_connected(all_cell_set - proposed):
            obstacles = proposed

    warehouse_map = WarehouseMap(
        width=spec.width,
        height=spec.height,
        obstacles=frozenset(obstacles),
        charging_stations=frozenset(charging_stations),
    )
    return warehouse_map, robot_position


def _generate_orders(
    *,
    warehouse_map: WarehouseMap,
    robot_position: Position,
    count: int,
    horizon: int,
    dynamics: str,
    rng: random.Random,
) -> tuple[Order, ...]:
    if count == 0:
        return ()

    traversable = _traversable_cells(warehouse_map)
    service_cells = sorted(
        traversable - warehouse_map.charging_stations - {robot_position}
    )
    if len(service_cells) < 2:
        raise ValueError("the generated map does not have two service cells")

    distance_cache: dict[Position, dict[Position, int]] = {}

    def distances(source: Position) -> dict[Position, int]:
        if source not in distance_cache:
            distance_cache[source] = _distances_from(source, traversable)
        return distance_cache[source]

    from_robot = distances(robot_position)
    feasible_pairs: list[tuple[Position, Position, int]] = []
    for pickup in service_cells:
        from_pickup = distances(pickup)
        for dropoff in service_cells:
            if pickup == dropoff:
                continue
            route_ticks = from_robot[pickup] + from_pickup[dropoff] + 2
            if route_ticks <= horizon:
                feasible_pairs.append((pickup, dropoff, route_ticks))
    if not feasible_pairs:
        raise ValueError("horizon is too short for a reachable pickup-and-delivery order")

    orders: list[Order] = []
    release_fraction = _RELEASE_WINDOWS[dynamics]
    for index in range(count):
        pickup, dropoff, route_ticks = rng.choice(feasible_pairs)
        latest_release = horizon - route_ticks
        if dynamics == "static" or index == 0 or latest_release < 1:
            release_time = 0
        else:
            release_ceiling = min(latest_release, max(1, int(horizon * release_fraction)))
            release_time = rng.randint(1, release_ceiling)
        earliest_deadline = release_time + route_ticks
        deadline = rng.randint(earliest_deadline, horizon)
        orders.append(
            Order(
                order_id=f"order-{index:03d}",
                pickup=pickup,
                dropoff=dropoff,
                release_time=release_time,
                deadline=deadline,
                priority=float(rng.choice((1, 2, 3))),
            )
        )
    return tuple(orders)


def _generate_closure_events(
    *,
    warehouse_map: WarehouseMap,
    robot_position: Position,
    orders: tuple[Order, ...],
    dynamics: str,
    horizon: int,
    rng: random.Random,
) -> tuple[DynamicEvent, ...]:
    if dynamics == "static":
        return ()

    traversable = _traversable_cells(warehouse_map)
    protected = set(warehouse_map.charging_stations) | {robot_position}
    for order in orders:
        protected.add(order.pickup)
        protected.add(order.dropoff)

    candidates = [
        cell
        for cell in sorted(traversable - protected)
        if _is_connected(traversable - {cell})
    ]
    rng.shuffle(candidates)

    desired = max(
        _MINIMUM_CLOSURE_PAIRS[dynamics],
        round(horizon * _CLOSURE_RATES[dynamics]),
    )
    pair_count = min(desired, len(candidates), (horizon - 1) // 2)
    if pair_count == 0:
        return ()

    timeline = sorted(rng.sample(range(1, horizon), pair_count * 2))
    events: list[DynamicEvent] = []
    for index, position in enumerate(candidates[:pair_count]):
        block_time = timeline[index * 2]
        unblock_time = timeline[index * 2 + 1]
        events.extend(
            (
                DynamicEvent(block_time, EventKind.CELL_BLOCKED, position=position),
                DynamicEvent(unblock_time, EventKind.CELL_UNBLOCKED, position=position),
            )
        )
    return tuple(events)


def _traversable_cells(warehouse_map: WarehouseMap) -> set[Position]:
    return {
        Position(x, y)
        for y in range(warehouse_map.height)
        for x in range(warehouse_map.width)
        if Position(x, y) not in warehouse_map.obstacles
    }


def _neighbors(position: Position) -> tuple[Position, ...]:
    return (
        position.translated(1, 0),
        position.translated(-1, 0),
        position.translated(0, 1),
        position.translated(0, -1),
    )


def _is_connected(cells: set[Position]) -> bool:
    if not cells:
        return False
    reached = set(_distances_from(min(cells), cells))
    return reached == cells


def _distances_from(source: Position, cells: set[Position]) -> dict[Position, int]:
    if source not in cells:
        raise ValueError("distance source must be traversable")
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in _neighbors(current):
            if neighbor in cells and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


__all__ = ["DynamicsName", "SizeName", "generate_scenario"]
