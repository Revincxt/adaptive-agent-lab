from __future__ import annotations

import json
import unittest
from dataclasses import replace

from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    RobotState,
    Transition,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment
from adaptive_agent_lab.environment.validation import (
    validate_episode,
    validate_state,
    validate_transition,
)


def audit_scenario() -> Scenario:
    return Scenario(
        scenario_id="validation-test",
        warehouse_map=WarehouseMap(
            width=4,
            height=2,
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(Order("order-0", Position(1, 0), Position(3, 0), 0, 7),),
        initial_robot=RobotState(Position(0, 0), 6),
        event_tape=EventTape(
            (
                DynamicEvent(2, EventKind.CELL_BLOCKED, position=Position(2, 1)),
                DynamicEvent(4, EventKind.CELL_UNBLOCKED, position=Position(2, 1)),
            )
        ),
        horizon=8,
        battery_capacity=6,
    )


def completed_history() -> tuple[Transition, ...]:
    environment = WarehouseEnvironment(audit_scenario())
    for action in (
        Action.RIGHT,
        Action.PICKUP,
        Action.RIGHT,
        Action.RIGHT,
        Action.DROPOFF,
    ):
        environment.step(action)
    return environment.history


class ValidationTests(unittest.TestCase):
    def test_valid_state_transition_and_episode(self) -> None:
        scenario = audit_scenario()
        history = completed_history()

        self.assertTrue(validate_state(scenario, history[2].next_state).valid)
        self.assertTrue(validate_transition(scenario, history[0]).valid)
        report = validate_episode(scenario, history)
        self.assertTrue(report.valid)
        self.assertEqual(report.transition_count, 5)
        self.assertEqual(report.state_count, 6)
        self.assertEqual(report.violation_counts, {})
        json.dumps(report.to_dict())

    def test_state_detects_event_tape_and_battery_tampering(self) -> None:
        scenario = audit_scenario()
        state = completed_history()[1].next_state
        forged = replace(state, blocked_cells=frozenset(), robot=replace(state.robot, battery=99))

        report = validate_state(scenario, forged)
        self.assertFalse(report.valid)
        self.assertIn("state.blockage_tape_mismatch", report.violation_counts)
        self.assertIn("state.battery_out_of_range", report.violation_counts)

    def test_transition_detects_teleport_and_battery_tampering(self) -> None:
        scenario = audit_scenario()
        original = completed_history()[0]
        forged_state = replace(
            original.next_state,
            robot=replace(original.next_state.robot, position=Position(3, 1), battery=6),
        )
        forged = Transition(
            state=original.state,
            action=original.action,
            next_state=forged_state,
            reward=original.reward,
            terminated=original.terminated,
            violations=original.violations,
            info=original.info,
        )

        report = validate_transition(scenario, forged)
        self.assertFalse(report.valid)
        self.assertIn("transition.position_mismatch", report.violation_counts)
        self.assertIn("transition.battery_mismatch", report.violation_counts)

    def test_transition_detects_forged_pickup_and_termination(self) -> None:
        scenario = audit_scenario()
        initial = scenario.initial_state()
        statuses = dict(initial.order_status)
        statuses["order-0"] = OrderStatus.PICKED_UP
        forged_state = replace(
            initial,
            time=1,
            robot=replace(initial.robot, carried_order_id="order-0"),
            order_status=statuses,
            terminated=True,
        )
        forged = Transition(
            state=initial,
            action=Action.PICKUP,
            next_state=forged_state,
            reward=0.0,
            terminated=True,
        )

        report = validate_transition(scenario, forged)
        self.assertFalse(report.valid)
        self.assertIn("action.no_order_at_pickup", report.violation_counts)
        self.assertIn("transition.order_status_mismatch", report.violation_counts)
        self.assertIn("transition.termination_mismatch", report.violation_counts)

    def test_episode_detects_chain_discontinuity(self) -> None:
        scenario = audit_scenario()
        history = list(completed_history())
        second = history[1]
        history[1] = Transition(
            state=history[0].state,
            action=second.action,
            next_state=second.next_state,
            reward=second.reward,
            terminated=second.terminated,
            violations=second.violations,
            info=second.info,
        )

        report = validate_episode(scenario, history)
        self.assertFalse(report.valid)
        self.assertIn("episode.chain_discontinuity", report.violation_counts)
        discontinuity = next(
            issue for issue in report.issues if issue.code == "episode.chain_discontinuity"
        )
        self.assertEqual(discontinuity.transition_index, 1)


if __name__ == "__main__":
    unittest.main()
