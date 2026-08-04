"""Authoritative discrete-time warehouse transition system."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from adaptive_agent_lab.environment.contracts import (
    Action,
    InfoValue,
    Order,
    OrderStatus,
    Position,
    RobotState,
    StepResult,
    Transition,
    Violation,
    ViolationCode,
    WarehouseSnapshot,
    WarehouseState,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind
from adaptive_agent_lab.environment.rewards import DEFAULT_REWARD_SCHEME, RewardScheme
from adaptive_agent_lab.environment.scenario import Scenario


class WarehouseEnvironment:
    """Mutable episode wrapper around immutable state values.

    Exogenous events are supplied entirely by the scenario's event tape.  The
    environment performs no random draws, which makes the same scenario exactly
    replayable across agent families.
    """

    def __init__(
        self,
        scenario: Scenario,
        *,
        rewards: RewardScheme = DEFAULT_REWARD_SCHEME,
        charge_rate: int = 2,
    ) -> None:
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if not isinstance(rewards, RewardScheme):
            raise TypeError("rewards must be a RewardScheme")
        if isinstance(charge_rate, bool) or not isinstance(charge_rate, int):
            raise TypeError("charge_rate must be an integer")
        if charge_rate < 1:
            raise ValueError("charge_rate must be positive")
        self.scenario = scenario
        self.rewards = rewards
        self.charge_rate = charge_rate
        self._state: WarehouseState
        self._history: list[Transition]
        self.reset()

    @property
    def state(self) -> WarehouseState:
        return self._state

    @property
    def history(self) -> tuple[Transition, ...]:
        return tuple(self._history)

    @property
    def snapshot(self) -> WarehouseSnapshot:
        return WarehouseSnapshot(
            warehouse_map=self.scenario.warehouse_map,
            state=self._state,
            orders=self.scenario.orders,
            horizon=self.scenario.horizon,
            battery_capacity=self.scenario.battery_capacity,
        )

    def reset(self) -> WarehouseSnapshot:
        """Return the canonical initial observation and clear prior history."""

        initial = self.scenario.initial_state()
        statuses = dict(initial.order_status)
        blocked: set[Position] = set()
        self._apply_events(
            events=self.scenario.event_tape.at(0),
            statuses=statuses,
            blocked=blocked,
        )
        self._state = WarehouseState(
            time=0,
            robot=initial.robot,
            order_status=statuses,
            blocked_cells=frozenset(blocked),
            cumulative_reward=0.0,
            terminated=False,
        )
        self._history = []
        return self.snapshot

    def step(self, action: Action) -> StepResult:
        """Apply one action, advance one tick, then expose scheduled events."""

        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        previous = self._state
        if previous.terminated:
            violation = Violation(
                ViolationCode.EPISODE_TERMINATED,
                "an action cannot advance a terminated episode",
                position=previous.robot.position,
            )
            result = StepResult(
                state=previous,
                action=action,
                reward=0.0,
                terminated=True,
                violations=(violation,),
                info={"valid": False, "event_count": 0},
            )
            return result

        statuses = dict(previous.order_status)
        blocked = set(previous.blocked_cells)
        robot = previous.robot
        violations: list[Violation] = []
        info: dict[str, InfoValue] = {
            "valid": True,
            "event_count": 0,
            "picked_order": None,
            "delivered_order": None,
        }
        reward = self.rewards.step_cost
        next_time = previous.time + 1

        if action.is_movement:
            destination = robot.position.moved(action)
            movement_violation = self._movement_violation(robot, destination, blocked)
            if movement_violation is None:
                robot = replace(robot, position=destination, battery=robot.battery - 1)
                reward += self.rewards.movement_cost
            else:
                violations.append(movement_violation)
        elif action is Action.PICKUP:
            order = self._available_order_at(robot.position, statuses)
            if robot.carried_order_id is not None or order is None:
                violations.append(
                    Violation(
                        ViolationCode.NO_ORDER_AT_PICKUP,
                        "no available order can be picked up at the robot position",
                        position=robot.position,
                    )
                )
            else:
                statuses[order.order_id] = OrderStatus.PICKED_UP
                robot = replace(robot, carried_order_id=order.order_id)
                reward += self.rewards.pickup_reward
                info["picked_order"] = order.order_id
        elif action is Action.DROPOFF:
            carried_id = robot.carried_order_id
            if carried_id is None:
                violations.append(
                    Violation(
                        ViolationCode.NOT_CARRYING_ORDER,
                        "the robot is not carrying an order",
                        position=robot.position,
                    )
                )
            else:
                order = self._order_by_id(carried_id)
                if robot.position != order.dropoff:
                    violations.append(
                        Violation(
                            ViolationCode.WRONG_DROPOFF,
                            "the carried order must be delivered at its drop-off cell",
                            position=robot.position,
                            order_id=carried_id,
                        )
                    )
                else:
                    statuses[carried_id] = OrderStatus.DELIVERED
                    robot = replace(robot, carried_order_id=None)
                    reward += self.rewards.delivery(
                        priority=order.priority,
                        completion_time=next_time,
                        deadline=order.deadline,
                    )
                    info["delivered_order"] = carried_id
        elif action is Action.CHARGE:
            if robot.position not in self.scenario.warehouse_map.charging_stations:
                violations.append(
                    Violation(
                        ViolationCode.NOT_AT_CHARGER,
                        "the robot can charge only at a charging station",
                        position=robot.position,
                    )
                )
            else:
                robot = replace(
                    robot,
                    battery=min(
                        self.scenario.battery_capacity,
                        robot.battery + self.charge_rate,
                    ),
                )

        if violations:
            reward += self.rewards.invalid_action_cost
            info["valid"] = False

        events = self.scenario.event_tape.at(next_time)
        self._apply_events(events=events, statuses=statuses, blocked=blocked)
        info["event_count"] = len(events)

        terminated = all(status is OrderStatus.DELIVERED for status in statuses.values())
        if next_time >= self.scenario.horizon:
            for order_id, status in tuple(statuses.items()):
                if not status.is_terminal:
                    statuses[order_id] = OrderStatus.EXPIRED
            robot = replace(robot, carried_order_id=None)
            terminated = True

        next_state = WarehouseState(
            time=next_time,
            robot=robot,
            order_status=statuses,
            blocked_cells=frozenset(blocked),
            cumulative_reward=previous.cumulative_reward + reward,
            terminated=terminated,
        )
        transition = Transition(
            state=previous,
            action=action,
            next_state=next_state,
            reward=reward,
            terminated=terminated,
            violations=tuple(violations),
            info=info,
        )
        self._state = next_state
        self._history.append(transition)
        return transition.as_step_result()

    def _movement_violation(
        self,
        robot: RobotState,
        destination: Position,
        blocked: set[Position],
    ) -> Violation | None:
        warehouse_map = self.scenario.warehouse_map
        if robot.battery <= 0:
            return Violation(
                ViolationCode.BATTERY_DEPLETED,
                "movement requires positive battery charge",
                position=robot.position,
            )
        if not warehouse_map.contains(destination):
            return Violation(
                ViolationCode.OUT_OF_BOUNDS,
                "movement would leave the warehouse map",
                position=destination,
            )
        if destination in warehouse_map.obstacles:
            return Violation(
                ViolationCode.STATIC_OBSTACLE,
                "movement would enter a permanent obstacle",
                position=destination,
            )
        if destination in blocked:
            return Violation(
                ViolationCode.DYNAMIC_BLOCKAGE,
                "movement would enter a temporarily blocked cell",
                position=destination,
            )
        return None

    def _available_order_at(
        self,
        position: Position,
        statuses: dict[str, OrderStatus],
    ) -> Order | None:
        candidates = (
            order
            for order in self.scenario.orders
            if order.pickup == position and statuses[order.order_id] is OrderStatus.AVAILABLE
        )
        return min(
            candidates,
            key=lambda order: (order.deadline, -order.priority, order.order_id),
            default=None,
        )

    def _order_by_id(self, order_id: str) -> Order:
        for order in self.scenario.orders:
            if order.order_id == order_id:
                return order
        raise KeyError(f"unknown order: {order_id}")

    def _apply_events(
        self,
        *,
        events: Iterable[DynamicEvent],
        statuses: dict[str, OrderStatus],
        blocked: set[Position],
    ) -> None:
        for event in events:
            if event.kind is EventKind.ORDER_ARRIVAL:
                assert event.order_id is not None
                if statuses[event.order_id] is OrderStatus.PENDING:
                    statuses[event.order_id] = OrderStatus.AVAILABLE
            elif event.kind is EventKind.CELL_BLOCKED:
                assert event.position is not None
                blocked.add(event.position)
            elif event.kind is EventKind.CELL_UNBLOCKED:
                assert event.position is not None
                blocked.discard(event.position)


__all__ = ["WarehouseEnvironment"]
