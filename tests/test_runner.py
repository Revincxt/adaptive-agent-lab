from __future__ import annotations

import unittest

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.dqn import DQNAgent, DQNConfig
from adaptive_agent_lab.agents.planning import ReplanningAgent
from adaptive_agent_lab.benchmarking.runner import run_episode, train_episodes
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    Position,
    RobotState,
    WarehouseMap,
    WarehouseSnapshot,
)
from adaptive_agent_lab.environment.events import EventTape
from adaptive_agent_lab.environment.scenario import Scenario


def runner_scenario() -> Scenario:
    return Scenario(
        scenario_id="runner-test",
        warehouse_map=WarehouseMap(
            3,
            2,
            charging_stations=frozenset({Position(0, 1)}),
        ),
        orders=(Order("a", Position(1, 0), Position(2, 0), 0, 8),),
        initial_robot=RobotState(Position(0, 0), 8),
        event_tape=EventTape(),
        horizon=8,
        battery_capacity=8,
    )


class FixedActionAgent(Agent):
    name = "fixed-action"

    def __init__(self, action: Action) -> None:
        super().__init__()
        self.action = action

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        del snapshot, explore
        return self.action


class RunnerTests(unittest.TestCase):
    def test_episode_result_contains_operational_metrics_and_trace(self) -> None:
        result = run_episode(
            ReplanningAgent(battery_reserve=0),
            runner_scenario(),
            seed=42,
            measure_timing=False,
        )
        self.assertEqual(result.metrics.weighted_completion_rate, 1.0)
        self.assertEqual(result.metrics.weighted_on_time_completion_rate, 1.0)
        self.assertEqual(result.metrics.completed_orders, 1)
        self.assertEqual(result.metrics.constraint_violations, 0)
        self.assertEqual(result.trace[-1].delivered_order_id, "a")
        self.assertEqual(result.to_dict()["agent"], "replanning")

    def test_deterministic_run_has_identical_non_timing_artifact(self) -> None:
        first = run_episode(
            ReplanningAgent(), runner_scenario(), seed=7, measure_timing=False
        )
        second = run_episode(
            ReplanningAgent(), runner_scenario(), seed=7, measure_timing=False
        )
        self.assertEqual(first, second)

    def test_train_episodes_retains_learning_agent(self) -> None:
        agent = DQNAgent(
            DQNConfig(
                hidden_sizes=(8,),
                batch_size=1,
                warmup_steps=1,
                replay_capacity=32,
                target_sync_interval=2,
                epsilon_decay_steps=10,
            )
        )
        results = train_episodes(agent, (runner_scenario(),), episodes=2, root_seed=10)
        self.assertEqual(len(results), 2)
        self.assertGreater(agent.update_count, 0)
        self.assertGreater(agent.replay_size, 1)

    def test_evaluation_does_not_mutate_a_trained_dqn(self) -> None:
        agent = DQNAgent(
            DQNConfig(
                hidden_sizes=(8,),
                batch_size=1,
                warmup_steps=1,
                replay_capacity=32,
                target_sync_interval=2,
                epsilon_decay_steps=10,
            )
        )
        run_episode(
            agent,
            runner_scenario(),
            seed=21,
            explore=True,
            learn=True,
            measure_timing=False,
        )
        before = agent.state_dict()
        replay_size = agent.replay_size

        evaluation = run_episode(
            agent,
            runner_scenario(),
            seed=22,
            explore=False,
            learn=False,
            measure_timing=False,
        )

        self.assertEqual(agent.state_dict(), before)
        self.assertEqual(agent.replay_size, replay_size)
        self.assertEqual(evaluation.diagnostics.learning_updates, 0)

    def test_metrics_cover_late_delivery_and_zero_order_scenarios(self) -> None:
        late_scenario = Scenario(
            scenario_id="runner-late",
            warehouse_map=WarehouseMap(2, 1),
            orders=(Order("late", Position(0, 0), Position(1, 0), 0, 2),),
            initial_robot=RobotState(Position(0, 0), 3),
            event_tape=EventTape(),
            horizon=5,
            battery_capacity=3,
        )
        late = run_episode(
            ReplanningAgent(battery_reserve=0),
            late_scenario,
            seed=1,
            measure_timing=False,
        )
        self.assertEqual(late.metrics.weighted_completion_rate, 1.0)
        self.assertEqual(late.metrics.weighted_on_time_completion_rate, 0.0)
        self.assertEqual(late.metrics.mean_lateness, 1.0)
        self.assertEqual(late.metrics.energy_per_completed_order, 1.0)

        no_orders = Scenario(
            scenario_id="runner-empty",
            warehouse_map=WarehouseMap(1, 1),
            orders=(),
            initial_robot=RobotState(Position(0, 0), 1),
            event_tape=EventTape(),
            horizon=1,
            battery_capacity=1,
        )
        empty = run_episode(
            FixedActionAgent(Action.WAIT),
            no_orders,
            seed=1,
            measure_timing=False,
        )
        self.assertEqual(empty.metrics.weighted_completion_rate, 1.0)
        self.assertEqual(empty.metrics.weighted_on_time_completion_rate, 1.0)
        self.assertIsNone(empty.metrics.energy_per_completed_order)

    def test_violation_counts_and_timed_decisions_are_reported(self) -> None:
        scenario = Scenario(
            scenario_id="runner-violation",
            warehouse_map=WarehouseMap(1, 1),
            orders=(Order("a", Position(0, 0), Position(0, 0), 0, 1),),
            initial_robot=RobotState(Position(0, 0), 1),
            event_tape=EventTape(),
            horizon=1,
            battery_capacity=1,
        )
        result = run_episode(FixedActionAgent(Action.LEFT), scenario, seed=2)

        self.assertEqual(result.violation_counts, {"out_of_bounds": 1})
        self.assertEqual(result.trace[0].violations, ("out_of_bounds",))
        self.assertEqual(result.metrics.constraint_violations, 1)
        self.assertGreaterEqual(result.metrics.decision_time_ms, 0.0)

    def test_training_contract_validates_inputs_and_supports_scenario_factories(self) -> None:
        trainable = DQNAgent(
            DQNConfig(
                hidden_sizes=(4,),
                batch_size=1,
                warmup_steps=1,
                replay_capacity=32,
            )
        )
        with self.assertRaises(ValueError):
            train_episodes(trainable, (runner_scenario(),), episodes=0, root_seed=0)
        with self.assertRaises(ValueError):
            train_episodes(ReplanningAgent(), (runner_scenario(),), episodes=1, root_seed=0)
        with self.assertRaises(ValueError):
            train_episodes(trainable, (), episodes=1, root_seed=0)

        results = train_episodes(
            trainable,
            lambda episode: runner_scenario(),
            episodes=2,
            root_seed=30,
        )
        self.assertEqual([result.seed for result in results], [30, 31])


if __name__ == "__main__":
    unittest.main()
