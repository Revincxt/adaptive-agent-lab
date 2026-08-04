"""A* search, open-loop planning, and event-triggered replanning agents."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import count

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    WarehouseMap,
    WarehouseSnapshot,
)
from adaptive_agent_lab.environment.observation import (
    ACTION_INDEX,
    ObservationEncoder,
    ObservationSpec,
)

MOVEMENT_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


@dataclass(frozen=True, slots=True)
class SearchResult:
    actions: tuple[Action, ...]
    cost: int
    expanded_nodes: int
    reached: bool


def astar_path(
    warehouse_map: WarehouseMap,
    start: Position,
    goal: Position,
    *,
    blocked_cells: frozenset[Position] = frozenset(),
) -> SearchResult:
    """Find one deterministic shortest path through current static geometry."""

    if not warehouse_map.contains(start) or start in warehouse_map.obstacles:
        raise ValueError("start must be statically traversable")
    if not warehouse_map.is_traversable(goal, blocked_cells):
        return SearchResult((), 0, 0, False)
    if start == goal:
        return SearchResult((), 0, 0, True)

    sequence = count()
    frontier: list[tuple[int, int, int, Position]] = []
    heapq.heappush(frontier, (start.manhattan_distance(goal), 0, next(sequence), start))
    costs = {start: 0}
    parent: dict[Position, tuple[Position, Action]] = {}
    expanded = 0

    while frontier:
        _, current_cost, _, current = heapq.heappop(frontier)
        if current_cost != costs[current]:
            continue
        expanded += 1
        if current == goal:
            actions: list[Action] = []
            cursor = goal
            while cursor != start:
                previous, action = parent[cursor]
                actions.append(action)
                cursor = previous
            actions.reverse()
            return SearchResult(tuple(actions), current_cost, expanded, True)

        for action in MOVEMENT_ACTIONS:
            neighbor = current.moved(action)
            if not warehouse_map.is_traversable(neighbor, blocked_cells):
                continue
            neighbor_cost = current_cost + 1
            if neighbor_cost >= costs.get(neighbor, 2**63 - 1):
                continue
            costs[neighbor] = neighbor_cost
            parent[neighbor] = (current, action)
            priority = neighbor_cost + neighbor.manhattan_distance(goal)
            heapq.heappush(
                frontier,
                (priority, neighbor_cost, next(sequence), neighbor),
            )
    return SearchResult((), 0, expanded, False)


def _ordered_available_orders(snapshot: WarehouseSnapshot, position: Position) -> list[Order]:
    orders = [
        order
        for order in snapshot.orders
        if snapshot.state.order_status[order.order_id] is OrderStatus.AVAILABLE
    ]
    return sorted(
        orders,
        key=lambda order: (
            order.deadline,
            -order.priority,
            position.manhattan_distance(order.pickup),
            order.order_id,
        ),
    )


class OpenLoopPlanningAgent(Agent):
    """Build a nominal plan at reset and never repair it during execution."""

    name = "planning"

    def __init__(self) -> None:
        super().__init__()
        self._plan: list[Action] = []

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        self._plan = []
        position = snapshot.state.robot.position
        blocked = snapshot.state.blocked_cells
        expanded = 0
        for order in _ordered_available_orders(snapshot, position):
            to_pickup = astar_path(snapshot.map, position, order.pickup, blocked_cells=blocked)
            if not to_pickup.reached:
                continue
            to_dropoff = astar_path(
                snapshot.map,
                order.pickup,
                order.dropoff,
                blocked_cells=blocked,
            )
            expanded += to_pickup.expanded_nodes + to_dropoff.expanded_nodes
            if not to_dropoff.reached:
                continue
            self._plan.extend(to_pickup.actions)
            self._plan.append(Action.PICKUP)
            self._plan.extend(to_dropoff.actions)
            self._plan.append(Action.DROPOFF)
            position = order.dropoff
        self._record_plan(expanded)

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        del snapshot, explore
        if self._plan:
            return self._plan.pop(0)
        return Action.WAIT


class ReplanningAgent(Agent):
    """Repair the current route after relevant events or service transitions."""

    name = "replanning"

    def __init__(self, *, battery_reserve: int = 1) -> None:
        super().__init__()
        if battery_reserve < 0:
            raise ValueError("battery_reserve must be non-negative")
        self.battery_reserve = battery_reserve
        self._plan: list[Action] = []
        self._context: tuple[object, ...] | None = None
        self._encoder: ObservationEncoder | None = None

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        self._plan = []
        self._context = None
        self._encoder = ObservationEncoder(ObservationSpec.from_snapshot(snapshot))

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        del explore
        context = self._planning_context(snapshot)
        if self._plan and context == self._context:
            candidate = self._plan[0]
            assert self._encoder is not None
            if self._encoder.action_mask(snapshot)[ACTION_INDEX[candidate]]:
                return self._plan.pop(0)
        self._plan = self._build_plan(snapshot)
        self._context = context
        if self._plan:
            return self._plan.pop(0)
        return Action.WAIT

    def _planning_context(self, snapshot: WarehouseSnapshot) -> tuple[object, ...]:
        return (
            snapshot.state.blocked_cells,
            tuple(snapshot.state.order_status.items()),
            snapshot.state.robot.carried_order_id,
        )

    def _build_plan(self, snapshot: WarehouseSnapshot) -> list[Action]:
        state = snapshot.state
        robot = state.robot
        if robot.carried_order_id is not None:
            order = snapshot.order_by_id(robot.carried_order_id)
            return self._route_with_battery(snapshot, order.dropoff, Action.DROPOFF)

        orders = _ordered_available_orders(snapshot, robot.position)
        if not orders:
            if (
                robot.position in snapshot.map.charging_stations
                and robot.battery < snapshot.battery_capacity
            ):
                return [Action.CHARGE]
            return []
        order = orders[0]
        return self._route_with_battery(snapshot, order.pickup, Action.PICKUP)

    def _route_with_battery(
        self,
        snapshot: WarehouseSnapshot,
        goal: Position,
        service_action: Action,
    ) -> list[Action]:
        robot = snapshot.state.robot
        route = astar_path(
            snapshot.map,
            robot.position,
            goal,
            blocked_cells=snapshot.state.blocked_cells,
        )
        self._record_plan(route.expanded_nodes)
        if not route.reached:
            return []
        required = route.cost + self.battery_reserve
        if robot.battery < required:
            if robot.position in snapshot.map.charging_stations:
                return [Action.CHARGE]
            charger_route = self._nearest_charger_route(snapshot)
            if charger_route is None or charger_route.cost > robot.battery:
                return []
            self._record_plan(charger_route.expanded_nodes)
            return list(charger_route.actions)
        return [*route.actions, service_action]

    def _nearest_charger_route(self, snapshot: WarehouseSnapshot) -> SearchResult | None:
        candidates: list[tuple[int, Position, SearchResult]] = []
        for charger in sorted(snapshot.map.charging_stations):
            route = astar_path(
                snapshot.map,
                snapshot.state.robot.position,
                charger,
                blocked_cells=snapshot.state.blocked_cells,
            )
            if route.reached:
                candidates.append((route.cost, charger, route))
        if not candidates:
            return None
        return min(candidates, key=lambda item: (item[0], item[1]))[2]


__all__ = [
    "MOVEMENT_ACTIONS",
    "OpenLoopPlanningAgent",
    "ReplanningAgent",
    "SearchResult",
    "astar_path",
]
