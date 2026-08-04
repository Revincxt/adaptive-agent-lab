"""Independent constraint auditing for states and recorded trajectories.

This module deliberately does not import :mod:`environment.simulator`.  It
derives invariants directly from immutable scenario and transition contracts,
providing a second implementation path that can detect corrupted run artifacts
instead of merely replaying them through the production environment.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    RobotState,
    Transition,
    ViolationCode,
    WarehouseState,
)
from adaptive_agent_lab.environment.events import EventKind
from adaptive_agent_lab.environment.scenario import Scenario


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One machine-readable inconsistency found by the independent auditor."""

    code: str
    message: str
    transition_index: int | None = None
    state_role: str | None = None
    time: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("validation issue code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("validation issue message must be a non-empty string")
        if self.transition_index is not None and self.transition_index < 0:
            raise ValueError("transition_index must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "state_role": self.state_role,
            "time": self.time,
            "transition_index": self.transition_index,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Structured result returned by every validation entry point."""

    valid: bool
    issues: tuple[ValidationIssue, ...]
    transition_count: int
    state_count: int
    violation_counts: Mapping[str, int]
    reported_violation_counts: Mapping[str, int]
    start_time: int | None = None
    end_time: int | None = None

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation suitable for run artifacts."""

        return {
            "end_time": self.end_time,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "reported_violation_counts": dict(self.reported_violation_counts),
            "start_time": self.start_time,
            "state_count": self.state_count,
            "transition_count": self.transition_count,
            "valid": self.valid,
            "violation_counts": dict(self.violation_counts),
        }


def validate_state(scenario: Scenario, state: WarehouseState) -> ValidationReport:
    """Audit one state against scenario geometry and lifecycle invariants."""

    _require_inputs(scenario, state=state)
    issues = _state_issues(scenario, state, state_role="state", transition_index=None)
    return _report(
        issues,
        transition_count=0,
        state_count=1,
        start_time=state.time,
        end_time=state.time,
    )


def validate_transition(scenario: Scenario, transition: Transition) -> ValidationReport:
    """Audit one transition without invoking the environment implementation."""

    _require_inputs(scenario, transition=transition)
    issues = _transition_issues(
        scenario,
        transition,
        transition_index=None,
        include_state_checks=True,
    )
    return _report(
        issues,
        transitions=(transition,),
        transition_count=1,
        state_count=2,
        start_time=transition.state.time,
        end_time=transition.next_state.time,
    )


def validate_episode(
    scenario: Scenario,
    transitions: Iterable[Transition],
) -> ValidationReport:
    """Audit a complete or partial episode and verify exact chain continuity."""

    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    history = tuple(transitions)
    if not all(isinstance(transition, Transition) for transition in history):
        raise TypeError("transitions must contain only Transition values")
    if not history:
        return _report((), transition_count=0, state_count=0)

    issues: list[ValidationIssue] = []
    first_state = history[0].state
    issues.extend(
        _state_issues(
            scenario,
            first_state,
            state_role="initial",
            transition_index=0,
        )
    )
    expected_initial = _canonical_initial_state(scenario)
    if first_state != expected_initial:
        issues.append(
            _issue(
                "episode.initial_state_mismatch",
                "the first transition does not start at the scenario's canonical initial state",
                transition_index=0,
                state_role="initial",
                time=first_state.time,
            )
        )

    for index, transition in enumerate(history):
        if index > 0 and history[index - 1].next_state != transition.state:
            issues.append(
                _issue(
                    "episode.chain_discontinuity",
                    "transition state does not equal the preceding next_state",
                    transition_index=index,
                    state_role="previous",
                    time=transition.state.time,
                )
            )
        issues.extend(
            _transition_issues(
                scenario,
                transition,
                transition_index=index,
                include_state_checks=False,
            )
        )
        issues.extend(
            _state_issues(
                scenario,
                transition.next_state,
                state_role="next",
                transition_index=index,
            )
        )

    return _report(
        issues,
        transitions=history,
        transition_count=len(history),
        state_count=len(history) + 1,
        start_time=first_state.time,
        end_time=history[-1].next_state.time,
    )


def _state_issues(
    scenario: Scenario,
    state: WarehouseState,
    *,
    state_role: str,
    transition_index: int | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(
            _issue(
                code,
                message,
                transition_index=transition_index,
                state_role=state_role,
                time=state.time,
            )
        )

    if state.time > scenario.horizon:
        add("state.time_out_of_range", "state time exceeds the scenario horizon")
    if state.robot.battery > scenario.battery_capacity:
        add("state.battery_out_of_range", "robot battery exceeds scenario capacity")

    warehouse_map = scenario.warehouse_map
    if not warehouse_map.contains(state.robot.position):
        add("state.robot_out_of_bounds", "robot position lies outside the warehouse map")
    elif state.robot.position in warehouse_map.obstacles:
        add("state.robot_on_obstacle", "robot position lies on a permanent obstacle")

    invalid_blocks = {
        cell
        for cell in state.blocked_cells
        if not warehouse_map.contains(cell) or cell in warehouse_map.obstacles
    }
    if invalid_blocks:
        add(
            "state.invalid_blocked_cells",
            f"dynamic blocked cells must be statically traversable: {sorted(invalid_blocks)!r}",
        )
    expected_blocks = _blocked_cells_at(scenario, state.time)
    if state.blocked_cells != expected_blocks:
        add(
            "state.blockage_tape_mismatch",
            "blocked cells do not match the scenario event tape at this time",
        )

    known_ids = {order.order_id for order in scenario.orders}
    actual_ids = set(state.order_status)
    if actual_ids != known_ids:
        missing = sorted(known_ids - actual_ids)
        unknown = sorted(actual_ids - known_ids)
        add(
            "state.order_coverage_mismatch",
            "order status coverage differs from the scenario; "
            f"missing={missing}, unknown={unknown}",
        )

    orders = {order.order_id: order for order in scenario.orders}
    for order_id in sorted(known_ids & actual_ids):
        order = orders[order_id]
        status = state.order_status[order_id]
        if state.time < order.release_time and status is not OrderStatus.PENDING:
            add(
                "state.order_released_early",
                f"order {order_id!r} has status {status.value!r} before its release time",
            )
        if state.time >= order.release_time and status is OrderStatus.PENDING:
            add(
                "state.order_still_pending",
                f"order {order_id!r} remains pending after its release time",
            )
        if status is OrderStatus.EXPIRED and state.time < scenario.horizon:
            add(
                "state.order_expired_early",
                f"order {order_id!r} expired before the episode horizon",
            )
        if state.time >= scenario.horizon and not status.is_terminal:
            add(
                "state.order_not_terminal_at_horizon",
                f"order {order_id!r} is non-terminal at the episode horizon",
            )

    picked_ids = {
        order_id
        for order_id, status in state.order_status.items()
        if status is OrderStatus.PICKED_UP
    }
    carried = state.robot.carried_order_id
    expected_picked = set() if carried is None else {carried}
    if carried is not None and carried not in known_ids:
        add("state.unknown_carried_order", "robot carries an order absent from the scenario")
    if picked_ids != expected_picked:
        add(
            "state.carry_status_mismatch",
            "PICKED_UP statuses must identify exactly the robot's carried order",
        )

    all_delivered = all(
        state.order_status.get(order.order_id) is OrderStatus.DELIVERED
        for order in scenario.orders
    )
    expected_terminated = state.time >= scenario.horizon or (
        state.time > 0 and all_delivered
    )
    if state.terminated != expected_terminated:
        add(
            "state.termination_mismatch",
            "terminated flag disagrees with horizon and order completion state",
        )
    return issues


def _transition_issues(
    scenario: Scenario,
    transition: Transition,
    *,
    transition_index: int | None,
    include_state_checks: bool,
) -> list[ValidationIssue]:
    previous = transition.state
    following = transition.next_state
    issues: list[ValidationIssue] = []
    if include_state_checks:
        issues.extend(
            _state_issues(
                scenario,
                previous,
                state_role="previous",
                transition_index=transition_index,
            )
        )
        issues.extend(
            _state_issues(
                scenario,
                following,
                state_role="next",
                transition_index=transition_index,
            )
        )

    def add(code: str, message: str, *, role: str | None = None) -> None:
        issues.append(
            _issue(
                code,
                message,
                transition_index=transition_index,
                state_role=role,
                time=following.time,
            )
        )

    expected_time = previous.time + 1
    if previous.terminated:
        add("transition.after_termination", "a recorded transition starts from a terminated state")
    if following.time != expected_time:
        add("transition.time_step", "a transition must advance time by exactly one tick")
    if expected_time > scenario.horizon:
        add("transition.past_horizon", "a transition advances beyond the episode horizon")

    statuses = dict(previous.order_status)
    blocked = set(previous.blocked_cells)
    expected_position = previous.robot.position
    expected_battery: int | None = previous.robot.battery
    expected_carried = previous.robot.carried_order_id
    expected_violation: ViolationCode | None = None
    picked_order: str | None = None
    delivered_order: str | None = None

    action = transition.action
    if action.is_movement:
        destination = previous.robot.position.moved(action)
        expected_violation = _movement_violation(scenario, previous, destination)
        if expected_violation is None:
            expected_position = destination
            expected_battery = previous.robot.battery - 1
    elif action is Action.PICKUP:
        order = _available_order_at(scenario.orders, previous.robot.position, statuses)
        if expected_carried is not None or order is None:
            expected_violation = ViolationCode.NO_ORDER_AT_PICKUP
        else:
            statuses[order.order_id] = OrderStatus.PICKED_UP
            expected_carried = order.order_id
            picked_order = order.order_id
    elif action is Action.DROPOFF:
        if expected_carried is None:
            expected_violation = ViolationCode.NOT_CARRYING_ORDER
        else:
            order = scenario.order_by_id(expected_carried)
            if previous.robot.position != order.dropoff:
                expected_violation = ViolationCode.WRONG_DROPOFF
            else:
                statuses[expected_carried] = OrderStatus.DELIVERED
                delivered_order = expected_carried
                expected_carried = None
    elif action is Action.CHARGE:
        if previous.robot.position not in scenario.warehouse_map.charging_stations:
            expected_violation = ViolationCode.NOT_AT_CHARGER
        else:
            # Scenario does not encode a charge rate.  The independent invariant
            # is therefore monotonic bounded charging, not a simulator constant.
            expected_battery = None

    if expected_violation is not None:
        add(
            f"action.{expected_violation.value}",
            f"action violates the {expected_violation.value} constraint",
        )

    _apply_events(scenario, expected_time, statuses, blocked)
    if expected_time >= scenario.horizon:
        statuses = {
            order_id: status if status.is_terminal else OrderStatus.EXPIRED
            for order_id, status in statuses.items()
        }
        expected_carried = None
        expected_terminated = True
    else:
        expected_terminated = all(
            status is OrderStatus.DELIVERED for status in statuses.values()
        )

    if following.robot.position != expected_position:
        add(
            "transition.position_mismatch",
            "next robot position is inconsistent with the action and map constraints",
            role="next",
        )
    if action is Action.CHARGE and expected_violation is None:
        old_battery = previous.robot.battery
        new_battery = following.robot.battery
        charge_valid = (
            new_battery == scenario.battery_capacity
            if old_battery == scenario.battery_capacity
            else old_battery < new_battery <= scenario.battery_capacity
        )
        if not charge_valid:
            add(
                "transition.battery_mismatch",
                "charging must increase battery toward capacity without exceeding it",
                role="next",
            )
    elif following.robot.battery != expected_battery:
        add(
            "transition.battery_mismatch",
            "next battery is inconsistent with movement or service action semantics",
            role="next",
        )
    if following.robot.carried_order_id != expected_carried:
        add(
            "transition.carried_order_mismatch",
            "next carried order is inconsistent with pickup/drop-off semantics",
            role="next",
        )
    if dict(following.order_status) != statuses:
        add(
            "transition.order_status_mismatch",
            "next order statuses are inconsistent with action, events, or horizon expiry",
            role="next",
        )
    if following.blocked_cells != frozenset(blocked):
        add(
            "transition.blockage_event_mismatch",
            "next blocked cells do not reflect scheduled events",
            role="next",
        )
    if following.terminated != expected_terminated or transition.terminated != expected_terminated:
        add(
            "transition.termination_mismatch",
            "transition termination does not match completion and horizon rules",
            role="next",
        )

    expected_cumulative = previous.cumulative_reward + transition.reward
    if not math.isclose(
        following.cumulative_reward,
        expected_cumulative,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        add(
            "transition.cumulative_reward_mismatch",
            "next cumulative reward must equal previous cumulative reward plus reward",
            role="next",
        )

    actual_violation_codes = tuple(violation.code for violation in transition.violations)
    expected_codes = () if expected_violation is None else (expected_violation,)
    if actual_violation_codes != expected_codes:
        add(
            "transition.violation_record_mismatch",
            "declared violation codes do not match independently derived constraints",
        )
    _validate_known_info(
        transition,
        expected_violation=expected_violation,
        event_count=len(scenario.event_tape.at(expected_time)) if expected_time >= 0 else 0,
        picked_order=picked_order,
        delivered_order=delivered_order,
        add=add,
    )
    return issues


def _validate_known_info(
    transition: Transition,
    *,
    expected_violation: ViolationCode | None,
    event_count: int,
    picked_order: str | None,
    delivered_order: str | None,
    add: Callable[[str, str], None],
) -> None:
    expected: dict[str, object] = {
        "valid": expected_violation is None,
        "event_count": event_count,
        "picked_order": picked_order,
        "delivered_order": delivered_order,
    }
    for key, value in expected.items():
        if key in transition.info and transition.info[key] != value:
            add(
                "transition.info_mismatch",
                f"transition info field {key!r} is inconsistent with audited effects",
            )


def _movement_violation(
    scenario: Scenario,
    state: WarehouseState,
    destination: Position,
) -> ViolationCode | None:
    if state.robot.battery <= 0:
        return ViolationCode.BATTERY_DEPLETED
    warehouse_map = scenario.warehouse_map
    if not warehouse_map.contains(destination):
        return ViolationCode.OUT_OF_BOUNDS
    if destination in warehouse_map.obstacles:
        return ViolationCode.STATIC_OBSTACLE
    if destination in state.blocked_cells:
        return ViolationCode.DYNAMIC_BLOCKAGE
    return None


def _available_order_at(
    orders: tuple[Order, ...],
    position: Position,
    statuses: Mapping[str, OrderStatus],
) -> Order | None:
    candidates = (
        order
        for order in orders
        if order.pickup == position and statuses.get(order.order_id) is OrderStatus.AVAILABLE
    )
    return min(
        candidates,
        key=lambda order: (order.deadline, -order.priority, order.order_id),
        default=None,
    )


def _apply_events(
    scenario: Scenario,
    time: int,
    statuses: dict[str, OrderStatus],
    blocked: set[Position],
) -> None:
    if time < 0:
        return
    for event in scenario.event_tape.at(time):
        if event.kind is EventKind.ORDER_ARRIVAL:
            assert event.order_id is not None
            if statuses.get(event.order_id) is OrderStatus.PENDING:
                statuses[event.order_id] = OrderStatus.AVAILABLE
        elif event.kind is EventKind.CELL_BLOCKED:
            assert event.position is not None
            blocked.add(event.position)
        elif event.kind is EventKind.CELL_UNBLOCKED:
            assert event.position is not None
            blocked.discard(event.position)


def _blocked_cells_at(scenario: Scenario, time: int) -> frozenset[Position]:
    blocked: set[Position] = set()
    for event in scenario.event_tape:
        if event.time > time:
            break
        if event.kind is EventKind.CELL_BLOCKED:
            assert event.position is not None
            blocked.add(event.position)
        elif event.kind is EventKind.CELL_UNBLOCKED:
            assert event.position is not None
            blocked.discard(event.position)
    return frozenset(blocked)


def _canonical_initial_state(scenario: Scenario) -> WarehouseState:
    initial = scenario.initial_state()
    statuses = dict(initial.order_status)
    blocked: set[Position] = set()
    _apply_events(scenario, 0, statuses, blocked)
    return WarehouseState(
        time=0,
        robot=RobotState(
            position=scenario.initial_robot.position,
            battery=scenario.initial_robot.battery,
            carried_order_id=scenario.initial_robot.carried_order_id,
        ),
        order_status=statuses,
        blocked_cells=frozenset(blocked),
        cumulative_reward=0.0,
        terminated=False,
    )


def _issue(
    code: str,
    message: str,
    *,
    transition_index: int | None,
    state_role: str | None,
    time: int | None,
) -> ValidationIssue:
    return ValidationIssue(code, message, transition_index, state_role, time)


def _report(
    issues: Iterable[ValidationIssue],
    *,
    transitions: Iterable[Transition] = (),
    transition_count: int,
    state_count: int,
    start_time: int | None = None,
    end_time: int | None = None,
) -> ValidationReport:
    issue_tuple = tuple(issues)
    counts = Counter(issue.code for issue in issue_tuple)
    reported = Counter(
        violation.code.value
        for transition in transitions
        for violation in transition.violations
    )
    return ValidationReport(
        valid=not issue_tuple,
        issues=issue_tuple,
        transition_count=transition_count,
        state_count=state_count,
        violation_counts=MappingProxyType(dict(sorted(counts.items()))),
        reported_violation_counts=MappingProxyType(dict(sorted(reported.items()))),
        start_time=start_time,
        end_time=end_time,
    )


def _require_inputs(
    scenario: Scenario,
    *,
    state: WarehouseState | None = None,
    transition: Transition | None = None,
) -> None:
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    if state is not None and not isinstance(state, WarehouseState):
        raise TypeError("state must be a WarehouseState")
    if transition is not None and not isinstance(transition, Transition):
        raise TypeError("transition must be a Transition")


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "validate_episode",
    "validate_state",
    "validate_transition",
]
