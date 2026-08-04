from __future__ import annotations

import unittest

import numpy as np

from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    Position,
    RobotState,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import EventTape
from adaptive_agent_lab.environment.observation import (
    ACTION_INDEX,
    ObservationEncoder,
    ObservationSpec,
)
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


def observation_environment() -> WarehouseEnvironment:
    scenario = Scenario(
        scenario_id="observation",
        warehouse_map=WarehouseMap(
            width=3,
            height=2,
            obstacles=frozenset({Position(1, 1)}),
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(Order("a", Position(0, 0), Position(2, 1), 0, 10, 2.0),),
        initial_robot=RobotState(Position(0, 0), 4),
        event_tape=EventTape(),
        horizon=10,
        battery_capacity=5,
    )
    return WarehouseEnvironment(scenario)


class ObservationTests(unittest.TestCase):
    def test_vector_shape_values_and_determinism(self) -> None:
        environment = observation_environment()
        spec = ObservationSpec.from_snapshot(environment.snapshot, max_orders=2)
        encoder = ObservationEncoder(spec)
        first = encoder.vector(environment.snapshot)
        second = encoder.vector(environment.snapshot)
        self.assertEqual(first.shape, (spec.vector_size,))
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))

    def test_tabular_state_changes_with_time(self) -> None:
        environment = observation_environment()
        encoder = ObservationEncoder(ObservationSpec.from_snapshot(environment.snapshot))
        before = encoder.tabular(environment.snapshot)
        environment.step(Action.WAIT)
        after = encoder.tabular(environment.snapshot)
        self.assertNotEqual(before, after)

    def test_action_mask_exposes_only_immediately_feasible_actions(self) -> None:
        environment = observation_environment()
        encoder = ObservationEncoder(ObservationSpec.from_snapshot(environment.snapshot))
        mask = encoder.action_mask(environment.snapshot)
        self.assertTrue(mask[ACTION_INDEX[Action.RIGHT]])
        self.assertFalse(mask[ACTION_INDEX[Action.LEFT]])
        self.assertTrue(mask[ACTION_INDEX[Action.PICKUP]])
        self.assertTrue(mask[ACTION_INDEX[Action.CHARGE]])
        self.assertFalse(mask[ACTION_INDEX[Action.DROPOFF]])

        environment.step(Action.PICKUP)
        picked_mask = encoder.action_mask(environment.snapshot)
        self.assertFalse(picked_mask[ACTION_INDEX[Action.PICKUP]])
        self.assertFalse(picked_mask[ACTION_INDEX[Action.DROPOFF]])


if __name__ == "__main__":
    unittest.main()
