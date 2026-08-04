from __future__ import annotations

import json
import unittest
from copy import deepcopy

import numpy as np

from adaptive_agent_lab.agents.hybrid import (
    HybridAgent,
    HybridConfig,
    config_from_mapping,
)
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


def masking_environment() -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="hybrid-mask",
        warehouse_map=WarehouseMap(
            4,
            2,
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(
            Order("available", Position(1, 0), Position(3, 0), 0, 10),
            Order("pending", Position(1, 1), Position(3, 1), 4, 11),
        ),
        initial_robot=RobotState(Position(0, 0), 5),
        event_tape=EventTape(
            (DynamicEvent(4, EventKind.ORDER_ARRIVAL, order_id="pending"),)
        ),
        horizon=12,
        battery_capacity=8,
    )
    return WarehouseEnvironment(scenario)


def detour_environment() -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="hybrid-detour",
        warehouse_map=WarehouseMap(
            5,
            3,
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(Order("a", Position(1, 0), Position(4, 0), 0, 15),),
        initial_robot=RobotState(Position(0, 0), 20),
        event_tape=EventTape(
            (
                DynamicEvent(3, EventKind.CELL_BLOCKED, position=Position(3, 0)),
                DynamicEvent(10, EventKind.CELL_UNBLOCKED, position=Position(3, 0)),
            )
        ),
        horizon=20,
        battery_capacity=20,
    )
    return WarehouseEnvironment(scenario)


def completion_environment() -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="hybrid-completion",
        warehouse_map=WarehouseMap(
            2,
            1,
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(Order("a", Position(0, 0), Position(1, 0), 0, 6),),
        initial_robot=RobotState(Position(0, 0), 5),
        event_tape=EventTape(),
        horizon=6,
        battery_capacity=5,
    )
    return WarehouseEnvironment(scenario)


def low_battery_environment(
    *,
    start: Position,
    chargers: frozenset[Position],
    obstacles: frozenset[Position] = frozenset(),
    battery: int = 1,
) -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="hybrid-low-battery",
        warehouse_map=WarehouseMap(
            5,
            2,
            obstacles=obstacles,
            charging_stations=chargers,
        ),
        orders=(Order("a", Position(4, 0), Position(4, 1), 0, 10),),
        initial_robot=RobotState(start, battery),
        event_tape=EventTape(),
        horizon=10,
        battery_capacity=5,
    )
    return WarehouseEnvironment(scenario)


def force_option(agent: HybridAgent, option_index: int) -> None:
    state = agent.state_dict()
    network = state["network"]
    assert isinstance(network, dict)
    weights = network["weights"]
    biases = network["biases"]
    assert isinstance(weights, list)
    assert isinstance(biases, list)
    network["weights"] = [np.zeros_like(value).tolist() for value in weights]
    zero_biases = [np.zeros_like(value).tolist() for value in biases]
    zero_biases[-1][option_index] = 10.0
    network["biases"] = zero_biases
    agent.load_state_dict(state)


def advance(
    agent: HybridAgent,
    environment: WarehouseEnvironment,
    *,
    explore: bool = False,
) -> Action:
    action = agent.act(environment.snapshot, explore=explore)
    environment.step(action)
    agent.observe(environment.history[-1])
    return action


class HybridAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HybridConfig(
            hidden=(8,),
            gamma=0.9,
            lr=0.01,
            epsilon=0.0,
            battery_reserve=0,
            max_grad_norm=10.0,
        )

    def test_high_level_option_mask(self) -> None:
        environment = masking_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=1)

        mask = agent.option_mask(environment.snapshot)

        self.assertEqual(agent.option_labels(environment.snapshot), (
            "order:available",
            "order:pending",
            "charge",
            "wait",
        ))
        self.assertEqual(mask.tolist(), [True, False, True, True])

    def test_dynamic_blockage_replans_without_changing_order_option(self) -> None:
        environment = detour_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=2)
        force_option(agent, 0)

        self.assertIs(advance(agent, environment), Action.RIGHT)
        self.assertIs(advance(agent, environment), Action.PICKUP)
        self.assertEqual(agent.current_option, "order:a")
        self.assertIs(advance(agent, environment), Action.RIGHT)
        self.assertIn(Position(3, 0), environment.state.blocked_cells)

        detour_action = agent.act(environment.snapshot)
        self.assertIsNot(detour_action, Action.RIGHT)
        self.assertEqual(agent.current_option, "order:a")
        environment.step(detour_action)
        agent.observe(environment.history[-1])

        while not environment.state.terminated:
            advance(agent, environment)
        self.assertIs(environment.state.order_status["a"], OrderStatus.DELIVERED)
        self.assertFalse(any(step.violations for step in environment.history))
        self.assertGreaterEqual(agent.diagnostics.planning_calls, 3)

    def test_completed_option_uses_discounted_semi_mdp_update(self) -> None:
        environment = completion_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=3)
        force_option(agent, 0)

        while not environment.state.terminated:
            advance(agent, environment)

        rewards = [transition.reward for transition in environment.history]
        expected_return = sum(
            self.config.gamma**index * reward for index, reward in enumerate(rewards)
        )
        self.assertEqual(agent.update_count, 1)
        self.assertEqual(agent.last_option_duration, 3)
        self.assertAlmostEqual(agent.last_discount or 0.0, self.config.gamma**3)
        self.assertAlmostEqual(agent.last_td_target or 0.0, expected_return)
        self.assertIsNotNone(agent.last_loss)
        self.assertEqual(agent.last_outcome, "terminal")

    def test_same_seed_produces_deterministic_option_and_action_stream(self) -> None:
        config = HybridConfig(
            hidden=(8,),
            gamma=0.9,
            lr=0.01,
            epsilon=1.0,
            battery_reserve=0,
        )
        first_environment = masking_environment()
        second_environment = masking_environment()
        first = HybridAgent(config)
        second = HybridAgent(config)
        first.reset(first_environment.snapshot, seed=55)
        second.reset(second_environment.snapshot, seed=55)

        first_actions: list[Action] = []
        second_actions: list[Action] = []
        for _ in range(5):
            first_actions.append(advance(first, first_environment, explore=True))
            second_actions.append(advance(second, second_environment, explore=True))

        self.assertEqual(first_actions, second_actions)
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_network_persists_across_reset_and_state_round_trip(self) -> None:
        environment = masking_environment()
        original = HybridAgent(self.config)
        original.reset(environment.snapshot, seed=8)
        force_option(original, 0)
        before_reset = original.q_values(environment.snapshot)

        original.reset(environment.snapshot, seed=999)
        self.assertTrue(np.array_equal(before_reset, original.q_values(environment.snapshot)))
        encoded = json.loads(json.dumps(original.state_dict()))

        restored = HybridAgent(self.config)
        restored.load_state_dict(encoded)
        restored.reset(environment.snapshot, seed=123)
        self.assertTrue(
            np.array_equal(
                original.q_values(environment.snapshot),
                restored.q_values(environment.snapshot),
            )
        )
        self.assertEqual(original.update_count, restored.update_count)

    def test_evaluation_episode_does_not_update_network_or_counters(self) -> None:
        environment = completion_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=5)
        force_option(agent, 0)
        before = deepcopy(agent.state_dict())
        agent.set_learning_enabled(False)

        while not environment.state.terminated:
            advance(agent, environment)

        self.assertEqual(agent.state_dict(), before)
        self.assertEqual(agent.update_count, 0)
        self.assertEqual(agent.environment_steps, 0)
        self.assertEqual(agent.diagnostics.learning_updates, 0)
        self.assertEqual(agent.last_outcome, "terminal")

    def test_repeated_act_before_observe_returns_the_same_primitive_action(self) -> None:
        environment = completion_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=2)
        force_option(agent, 0)

        first = agent.act(environment.snapshot)
        second = agent.act(environment.snapshot)

        self.assertIs(first, Action.PICKUP)
        self.assertIs(second, first)

    def test_low_battery_order_charges_or_routes_to_a_reachable_charger(self) -> None:
        at_charger = low_battery_environment(
            start=Position(0, 0),
            chargers=frozenset({Position(0, 0)}),
        )
        charging_agent = HybridAgent(self.config)
        charging_agent.reset(at_charger.snapshot, seed=3)
        force_option(charging_agent, 0)
        self.assertIs(charging_agent.act(at_charger.snapshot), Action.CHARGE)

        near_charger = low_battery_environment(
            start=Position(1, 0),
            chargers=frozenset({Position(0, 0)}),
        )
        routing_agent = HybridAgent(self.config)
        routing_agent.reset(near_charger.snapshot, seed=3)
        force_option(routing_agent, 0)
        self.assertIs(routing_agent.act(near_charger.snapshot), Action.LEFT)

    def test_low_battery_order_waits_when_no_charger_is_reachable(self) -> None:
        environment = low_battery_environment(
            start=Position(1, 0),
            chargers=frozenset(),
        )
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=4)
        force_option(agent, 0)
        self.assertIs(agent.act(environment.snapshot), Action.WAIT)

    def test_unreachable_order_option_fails_safely_with_wait(self) -> None:
        environment = low_battery_environment(
            start=Position(0, 0),
            chargers=frozenset({Position(0, 0)}),
            obstacles=frozenset({Position(1, 0), Position(1, 1)}),
            battery=5,
        )
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=4)
        force_option(agent, 0)
        self.assertIs(agent.act(environment.snapshot), Action.WAIT)

    def test_charge_and_wait_options_finish_at_the_option_boundary(self) -> None:
        charge_environment = WarehouseEnvironment(
            Scenario(
                scenario_id="hybrid-charge",
                warehouse_map=WarehouseMap(
                    2,
                    1,
                    charging_stations=frozenset({Position(0, 0)}),
                ),
                orders=(Order("a", Position(1, 0), Position(0, 0), 0, 5),),
                initial_robot=RobotState(Position(0, 0), 3),
                event_tape=EventTape(),
                horizon=5,
                battery_capacity=5,
            )
        )
        charge_agent = HybridAgent(self.config)
        charge_agent.reset(charge_environment.snapshot, seed=5)
        force_option(charge_agent, 1)
        self.assertIs(advance(charge_agent, charge_environment), Action.CHARGE)
        self.assertEqual(charge_agent.last_outcome, "complete")
        self.assertIsNone(charge_agent.current_option)

        wait_environment = masking_environment()
        wait_agent = HybridAgent(self.config)
        wait_agent.reset(wait_environment.snapshot, seed=6)
        force_option(wait_agent, len(wait_environment.scenario.orders) + 1)
        self.assertIs(advance(wait_agent, wait_environment), Action.WAIT)
        self.assertEqual(wait_agent.last_outcome, "complete")

    def test_end_episode_closes_an_in_progress_option(self) -> None:
        environment = completion_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=7)
        force_option(agent, 0)
        self.assertIs(advance(agent, environment), Action.PICKUP)

        agent.end_episode(environment.snapshot)

        self.assertEqual(agent.last_outcome, "terminal")
        self.assertEqual(agent.last_option_duration, 1)
        self.assertIsNone(agent.current_option)

    def test_observe_validates_type_order_and_elapsed_time(self) -> None:
        environment = completion_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=8)
        with self.assertRaises(TypeError):
            agent.observe(object())  # type: ignore[arg-type]

        environment.step(Action.WAIT)
        with self.assertRaises(RuntimeError):
            agent.observe(environment.history[-1])

        environment = completion_environment()
        agent.reset(environment.snapshot, seed=9)
        force_option(agent, 0)
        selected = agent.act(environment.snapshot)
        environment.step(Action.WAIT)
        with self.assertRaises(ValueError):
            agent.observe(environment.history[-1])

        environment = completion_environment()
        agent.reset(environment.snapshot, seed=10)
        force_option(agent, 0)
        selected = agent.act(environment.snapshot)
        stationary = Transition(
            state=environment.state,
            action=selected,
            next_state=environment.state,
            reward=0.0,
            terminated=False,
        )
        with self.assertRaises(ValueError):
            agent.observe(stationary)

    def test_terminated_snapshot_has_no_available_options(self) -> None:
        environment = completion_environment()
        environment.step(Action.PICKUP)
        environment.step(Action.RIGHT)
        environment.step(Action.DROPOFF)
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=11)

        self.assertEqual(agent.option_mask(environment.snapshot).tolist(), [False, False, False])
        self.assertIs(agent.act(environment.snapshot), Action.WAIT)

    def test_reset_and_loader_reject_incompatible_model_shapes(self) -> None:
        environment = masking_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=12)
        state = agent.state_dict()

        incompatible = completion_environment()
        with self.assertRaises(ValueError):
            agent.reset(incompatible.snapshot, seed=13)

        with self.assertRaises(ValueError):
            HybridAgent(self.config).load_state_dict(
                {"format_version": 2, "agent": "hybrid"}
            )
        with self.assertRaises(ValueError):
            HybridAgent(self.config).load_state_dict(
                {"format_version": 1, "agent": "hybrid", "network": []}
            )
        with self.assertRaises(ValueError):
            HybridAgent(HybridConfig(hidden=(4,))).load_state_dict(state)

        for key, invalid in (("environment_steps", -1), ("updates", True)):
            malformed = deepcopy(state)
            malformed[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                HybridAgent(self.config).load_state_dict(malformed)

    def test_clear_learning_discards_the_model_and_diagnostics(self) -> None:
        environment = masking_environment()
        agent = HybridAgent(self.config)
        agent.reset(environment.snapshot, seed=14)
        force_option(agent, len(environment.scenario.orders) + 1)
        advance(agent, environment)
        self.assertGreater(agent.update_count, 0)

        agent.clear_learning()

        self.assertEqual(agent.update_count, 0)
        self.assertEqual(agent.environment_steps, 0)
        self.assertIsNone(agent.last_loss)
        with self.assertRaises(RuntimeError):
            agent.q_values(environment.snapshot)

        agent.reset(environment.snapshot, seed=15)
        self.assertEqual(agent.q_values(environment.snapshot).shape, (agent.option_count,))


class HybridConfigTests(unittest.TestCase):
    def test_config_rejects_invalid_architecture_and_numeric_bounds(self) -> None:
        invalid_arguments = (
            {"hidden": ()},
            {"hidden": (8, 8, 8)},
            {"hidden": (0,)},
            {"hidden": (True,)},
            {"gamma": "0.9"},
            {"gamma": float("inf")},
            {"gamma": -0.1},
            {"lr": 0.0},
            {"epsilon": 1.1},
            {"max_grad_norm": 0.0},
            {"battery_reserve": True},
            {"battery_reserve": -1},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                HybridConfig(**arguments)  # type: ignore[arg-type]

    def test_config_mapping_supports_compatibility_names(self) -> None:
        config = config_from_mapping(
            {
                "hidden_sizes": [16],
                "learning_rate": "0.02",
                "gamma": "0.8",
                "battery_reserve": "3",
            }
        )
        self.assertEqual(config.hidden_sizes, (16,))
        self.assertEqual(config.learning_rate, 0.02)
        self.assertEqual(config.gamma, 0.8)
        self.assertEqual(config.battery_reserve, 3)
        with self.assertRaises(TypeError):
            config_from_mapping({"hidden": "16,16"})

    def test_constructor_and_uninitialized_properties_fail_descriptively(self) -> None:
        with self.assertRaises(TypeError):
            HybridAgent({})  # type: ignore[arg-type]
        agent = HybridAgent()
        with self.assertRaises(RuntimeError):
            _ = agent.option_count
        with self.assertRaises(RuntimeError):
            agent.state_dict()


if __name__ == "__main__":
    unittest.main()
