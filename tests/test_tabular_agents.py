from __future__ import annotations

import json
import unittest

from adaptive_agent_lab.agents.tabular import DynaQAgent, QLearningAgent
from adaptive_agent_lab.environment.contracts import (
    Action,
    Position,
    RobotState,
    Transition,
    WarehouseMap,
    WarehouseState,
)
from adaptive_agent_lab.environment.events import EventTape
from adaptive_agent_lab.environment.observation import ObservationEncoder, ObservationSpec
from adaptive_agent_lab.environment.scenario import Scenario


def empty_scenario(*, width: int = 3, start_x: int = 1) -> Scenario:
    return Scenario(
        scenario_id="tabular-test",
        warehouse_map=WarehouseMap(width=width, height=1),
        orders=(),
        initial_robot=RobotState(Position(start_x, 0), battery=4),
        event_tape=EventTape(),
        horizon=8,
        battery_capacity=4,
    )


def state_key(scenario: Scenario, state: WarehouseState) -> tuple[int, ...]:
    snapshot = scenario.snapshot(state)
    encoder = ObservationEncoder(ObservationSpec.from_snapshot(snapshot))
    return encoder.tabular(snapshot)


def next_state(
    scenario: Scenario,
    *,
    position: Position,
    reward: float,
    terminated: bool = False,
) -> WarehouseState:
    return WarehouseState(
        time=1,
        robot=RobotState(position, battery=3),
        order_status={},
        cumulative_reward=reward,
        terminated=terminated,
    )


class QLearningAgentTests(unittest.TestCase):
    def test_q_update_bootstraps_only_over_feasible_next_actions(self) -> None:
        scenario = empty_scenario()
        snapshot = scenario.snapshot(scenario.initial_state())
        following = next_state(scenario, position=Position(2, 0), reward=2.0)
        current_key = state_key(scenario, snapshot.state)
        next_key = state_key(scenario, following)
        agent = QLearningAgent(
            alpha=0.5,
            gamma=0.5,
            epsilon=0.8,
            epsilon_decay=0.5,
            epsilon_min=0.1,
        )
        agent.set_q_value(current_key, Action.RIGHT, 0.1)
        agent.set_q_value(next_key, Action.WAIT, 4.0)
        agent.set_q_value(next_key, Action.RIGHT, 100.0)  # Out of bounds and masked.
        agent.reset(snapshot, seed=7)

        action = agent.act(snapshot, explore=False)
        self.assertIs(action, Action.RIGHT)
        agent.observe(
            Transition(
                state=snapshot.state,
                action=action,
                next_state=following,
                reward=2.0,
                terminated=False,
            )
        )

        # 0.1 + 0.5 * ((2 + 0.5 * 4) - 0.1) = 2.05
        self.assertAlmostEqual(agent.q_value(current_key, Action.RIGHT), 2.05)
        self.assertAlmostEqual(agent.epsilon, 0.4)
        self.assertEqual(agent.diagnostics.learning_updates, 1)

    def test_action_selection_is_masked_and_greedy_without_exploration(self) -> None:
        scenario = empty_scenario(width=2, start_x=0)
        snapshot = scenario.snapshot(scenario.initial_state())
        key = state_key(scenario, snapshot.state)
        agent = QLearningAgent(epsilon=1.0, epsilon_min=0.0)
        agent.set_q_value(key, Action.UP, 1000.0)
        agent.set_q_value(key, Action.LEFT, 900.0)
        agent.set_q_value(key, Action.RIGHT, 2.0)
        agent.set_q_value(key, Action.WAIT, 1.0)
        agent.reset(snapshot, seed=11)

        self.assertIs(agent.act(snapshot, explore=False), Action.RIGHT)
        explored = {agent.act(snapshot, explore=True) for _ in range(100)}
        self.assertTrue(explored <= {Action.RIGHT, Action.WAIT})

    def test_seeded_exploration_is_reproducible(self) -> None:
        scenario = empty_scenario(width=3, start_x=1)
        snapshot = scenario.snapshot(scenario.initial_state())
        first = QLearningAgent(epsilon=1.0, epsilon_min=0.0)
        second = QLearningAgent(epsilon=1.0, epsilon_min=0.0)
        first.reset(snapshot, seed=2026)
        second.reset(snapshot, seed=2026)

        first_actions = [first.act(snapshot, explore=True) for _ in range(30)]
        second_actions = [second.act(snapshot, explore=True) for _ in range(30)]

        self.assertEqual(first_actions, second_actions)
        self.assertGreater(len(set(first_actions)), 1)

    def test_reset_keeps_learning_while_clear_discards_it(self) -> None:
        scenario = empty_scenario()
        snapshot = scenario.snapshot(scenario.initial_state())
        key = state_key(scenario, snapshot.state)
        agent = QLearningAgent(epsilon=0.8, epsilon_decay=0.5, epsilon_min=0.1)
        agent.set_q_value(key, Action.WAIT, 3.0)
        agent.reset(snapshot, seed=1)
        action = agent.act(snapshot, explore=False)
        following = next_state(
            scenario,
            position=snapshot.state.robot.position,
            reward=1.0,
            terminated=True,
        )
        agent.observe(
            Transition(snapshot.state, action, following, 1.0, terminated=True)
        )
        decayed_epsilon = agent.epsilon

        agent.reset(snapshot, seed=2)
        self.assertTrue(agent.q_table)
        self.assertEqual(agent.epsilon, decayed_epsilon)

        agent.clear()
        self.assertFalse(agent.q_table)
        self.assertEqual(agent.epsilon, 0.8)

    def test_q_state_dict_is_json_serializable_and_strictly_loadable(self) -> None:
        scenario = empty_scenario()
        snapshot = scenario.snapshot(scenario.initial_state())
        key = state_key(scenario, snapshot.state)
        agent = QLearningAgent(
            alpha=0.25,
            gamma=0.8,
            epsilon=0.6,
            epsilon_decay=0.9,
            epsilon_min=0.2,
        )
        agent.set_q_value(key, Action.LEFT, -1.25)
        encoded = json.dumps(agent.state_dict(), sort_keys=True)
        restored = QLearningAgent(
            alpha=0.25,
            gamma=0.8,
            epsilon=0.6,
            epsilon_decay=0.9,
            epsilon_min=0.2,
        )

        restored.load_state_dict(json.loads(encoded))

        self.assertEqual(restored.state_dict(), agent.state_dict())
        self.assertAlmostEqual(restored.q_value(key, Action.LEFT), -1.25)

    def test_observe_requires_the_transition_returned_for_the_cached_action(self) -> None:
        scenario = empty_scenario()
        snapshot = scenario.snapshot(scenario.initial_state())
        agent = QLearningAgent(epsilon=0.0, epsilon_min=0.0)
        agent.reset(snapshot, seed=0)
        following = next_state(scenario, position=Position(1, 0), reward=0.0)

        with self.assertRaises(RuntimeError):
            agent.observe(
                Transition(snapshot.state, Action.WAIT, following, 0.0, False)
            )
        chosen = agent.act(snapshot)
        wrong = Action.WAIT if chosen is not Action.WAIT else Action.LEFT
        with self.assertRaises(ValueError):
            agent.observe(Transition(snapshot.state, wrong, following, 0.0, False))


class DynaQAgentTests(unittest.TestCase):
    def test_each_real_transition_runs_configured_planning_updates(self) -> None:
        scenario = empty_scenario(width=1, start_x=0)
        snapshot = scenario.snapshot(scenario.initial_state())
        current_key = state_key(scenario, snapshot.state)
        agent = DynaQAgent(
            alpha=0.5,
            gamma=0.0,
            epsilon=0.0,
            epsilon_min=0.0,
            planning_steps=3,
        )
        agent.reset(snapshot, seed=5)
        action = agent.act(snapshot)
        following = next_state(
            scenario,
            position=Position(0, 0),
            reward=2.0,
            terminated=True,
        )

        agent.observe(
            Transition(snapshot.state, action, following, 2.0, terminated=True)
        )

        # One real update and three model updates: 0 -> 1 -> 1.5 -> 1.75 -> 1.875.
        self.assertAlmostEqual(agent.q_value(current_key, Action.WAIT), 1.875)
        self.assertEqual(agent.model_size, 1)
        self.assertEqual(agent.diagnostics.learning_updates, 4)

    def test_dyna_state_dict_round_trips_q_table_and_model(self) -> None:
        scenario = empty_scenario(width=1, start_x=0)
        snapshot = scenario.snapshot(scenario.initial_state())
        agent = DynaQAgent(
            alpha=0.5,
            gamma=0.0,
            epsilon=0.4,
            epsilon_decay=0.5,
            epsilon_min=0.1,
            planning_steps=2,
        )
        agent.reset(snapshot, seed=42)
        action = agent.act(snapshot)
        following = next_state(
            scenario,
            position=Position(0, 0),
            reward=1.0,
            terminated=True,
        )
        agent.observe(
            Transition(snapshot.state, action, following, 1.0, terminated=True)
        )
        encoded = json.dumps(agent.state_dict(), sort_keys=True)
        restored = DynaQAgent(
            alpha=0.5,
            gamma=0.0,
            epsilon=0.4,
            epsilon_decay=0.5,
            epsilon_min=0.1,
            planning_steps=2,
        )

        restored.load_state_dict(json.loads(encoded))

        self.assertEqual(restored.state_dict(), agent.state_dict())
        self.assertEqual(restored.model_size, 1)
        restored.clear()
        self.assertFalse(restored.q_table)
        self.assertEqual(restored.model_size, 0)


if __name__ == "__main__":
    unittest.main()
