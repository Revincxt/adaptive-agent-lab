from __future__ import annotations

import unittest

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.environment.contracts import (
    Action,
    Position,
    RobotState,
    Transition,
    WarehouseMap,
    WarehouseSnapshot,
)
from adaptive_agent_lab.environment.events import EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


def lifecycle_environment() -> WarehouseEnvironment:
    return WarehouseEnvironment(
        Scenario(
            scenario_id="agent-lifecycle",
            warehouse_map=WarehouseMap(1, 1),
            orders=(),
            initial_robot=RobotState(Position(0, 0), 1),
            event_tape=EventTape(),
            horizon=1,
            battery_capacity=1,
        )
    )


class FixedAgent(Agent):
    name = "fixed"

    def __init__(self, result: object = Action.WAIT) -> None:
        super().__init__()
        self.result = result

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        del snapshot, explore
        return self.result  # type: ignore[return-value]


class AgentLifecycleTests(unittest.TestCase):
    def test_learning_toggle_is_strictly_boolean(self) -> None:
        agent = FixedAgent()
        self.assertTrue(agent.learning_enabled)
        agent.set_learning_enabled(False)
        self.assertFalse(agent.learning_enabled)
        with self.assertRaises(TypeError):
            agent.set_learning_enabled(1)  # type: ignore[arg-type]

    def test_reset_rejects_negative_seeds_and_clears_diagnostics(self) -> None:
        environment = lifecycle_environment()
        agent = FixedAgent()
        agent.reset(environment.snapshot, seed=3)
        self.assertIs(agent.act(environment.snapshot), Action.WAIT)
        self.assertEqual(agent.diagnostics.decisions, 1)

        agent.reset(environment.snapshot, seed=4)
        self.assertEqual(agent.diagnostics.decisions, 0)
        with self.assertRaises(ValueError):
            agent.reset(environment.snapshot, seed=-1)

    def test_terminated_snapshot_short_circuits_policy_and_diagnostics(self) -> None:
        environment = lifecycle_environment()
        environment.step(Action.WAIT)
        agent = FixedAgent(result="not-an-action")
        agent.reset(environment.snapshot, seed=0)

        self.assertIs(agent.act(environment.snapshot), Action.WAIT)
        self.assertEqual(agent.diagnostics.decisions, 0)

    def test_non_action_policy_result_is_rejected(self) -> None:
        environment = lifecycle_environment()
        agent = FixedAgent(result="not-an-action")
        agent.reset(environment.snapshot, seed=0)
        with self.assertRaises(TypeError):
            agent.act(environment.snapshot)

    def test_default_observation_hooks_are_safe_no_ops(self) -> None:
        environment = lifecycle_environment()
        agent = FixedAgent()
        agent.reset(environment.snapshot, seed=0)
        action = agent.act(environment.snapshot)
        environment.step(action)
        transition: Transition = environment.history[-1]

        agent.observe(transition)
        agent.end_episode(environment.snapshot)
        self.assertEqual(agent.diagnostics.decisions, 1)


if __name__ == "__main__":
    unittest.main()
