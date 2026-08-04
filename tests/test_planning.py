from __future__ import annotations

import unittest

from adaptive_agent_lab.agents.planning import (
    OpenLoopPlanningAgent,
    ReplanningAgent,
    astar_path,
)
from adaptive_agent_lab.environment.contracts import (
    Action,
    Order,
    OrderStatus,
    Position,
    RobotState,
    WarehouseMap,
)
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment


def planning_scenario() -> Scenario:
    return Scenario(
        scenario_id="planning-test",
        warehouse_map=WarehouseMap(
            width=4,
            height=3,
            charging_stations=frozenset({Position(0, 0)}),
        ),
        orders=(Order("a", Position(1, 0), Position(3, 0), 0, 15),),
        initial_robot=RobotState(Position(0, 0), 12),
        event_tape=EventTape(
            (
                DynamicEvent(2, EventKind.CELL_BLOCKED, position=Position(2, 0)),
                DynamicEvent(7, EventKind.CELL_UNBLOCKED, position=Position(2, 0)),
            )
        ),
        horizon=20,
        battery_capacity=12,
    )


def run_agent(agent: OpenLoopPlanningAgent | ReplanningAgent) -> WarehouseEnvironment:
    environment = WarehouseEnvironment(planning_scenario())
    agent.reset(environment.snapshot, seed=42)
    while not environment.state.terminated:
        before = environment.state
        action = agent.act(environment.snapshot)
        environment.step(action)
        agent.observe(environment.history[-1])
        if environment.state.time >= environment.scenario.horizon:
            break
        self_check = environment.history[-1]
        assert self_check.state == before
    agent.end_episode(environment.snapshot)
    return environment


class AStarTests(unittest.TestCase):
    def test_astar_finds_deterministic_shortest_detour(self) -> None:
        warehouse_map = WarehouseMap(
            4,
            3,
            obstacles=frozenset({Position(1, 0), Position(1, 1)}),
        )
        result = astar_path(warehouse_map, Position(0, 0), Position(3, 0))
        self.assertTrue(result.reached)
        self.assertEqual(result.cost, 7)
        position = Position(0, 0)
        for action in result.actions:
            position = position.moved(action)
        self.assertEqual(position, Position(3, 0))

    def test_astar_reports_unreachable_goal(self) -> None:
        warehouse_map = WarehouseMap(2, 1, obstacles=frozenset({Position(1, 0)}))
        result = astar_path(warehouse_map, Position(0, 0), Position(1, 0))
        self.assertFalse(result.reached)

    def test_astar_reports_a_traversable_but_disconnected_goal(self) -> None:
        warehouse_map = WarehouseMap(
            3,
            3,
            obstacles=frozenset({Position(1, 0), Position(1, 1), Position(1, 2)}),
        )
        result = astar_path(warehouse_map, Position(0, 1), Position(2, 1))
        self.assertFalse(result.reached)
        self.assertGreater(result.expanded_nodes, 0)

    def test_astar_handles_identity_and_dynamic_goal_blockage(self) -> None:
        warehouse_map = WarehouseMap(2, 1)
        identity = astar_path(warehouse_map, Position(0, 0), Position(0, 0))
        blocked = astar_path(
            warehouse_map,
            Position(0, 0),
            Position(1, 0),
            blocked_cells=frozenset({Position(1, 0)}),
        )
        self.assertEqual(identity.actions, ())
        self.assertTrue(identity.reached)
        self.assertFalse(blocked.reached)

    def test_astar_rejects_an_invalid_start(self) -> None:
        warehouse_map = WarehouseMap(2, 1, obstacles=frozenset({Position(1, 0)}))
        for start in (Position(-1, 0), Position(1, 0)):
            with self.subTest(start=start), self.assertRaises(ValueError):
                astar_path(warehouse_map, start, Position(0, 0))


class PlanningAgentTests(unittest.TestCase):
    def test_open_loop_plan_is_disrupted_by_new_blockage(self) -> None:
        environment = run_agent(OpenLoopPlanningAgent())
        self.assertEqual(environment.state.order_status["a"], OrderStatus.EXPIRED)
        self.assertGreater(
            sum(bool(transition.violations) for transition in environment.history),
            0,
        )

    def test_replanning_agent_repairs_route_and_delivers(self) -> None:
        agent = ReplanningAgent()
        environment = run_agent(agent)
        self.assertEqual(environment.state.order_status["a"], OrderStatus.DELIVERED)
        self.assertGreaterEqual(agent.diagnostics.planning_calls, 2)
        self.assertGreater(agent.diagnostics.expanded_nodes, 0)

    def test_open_loop_skips_an_order_with_a_dynamically_blocked_service_cell(self) -> None:
        for blocked in (Position(1, 0), Position(2, 0)):
            with self.subTest(blocked=blocked):
                environment = WarehouseEnvironment(
                    Scenario(
                        scenario_id="open-loop-blocked-service",
                        warehouse_map=WarehouseMap(3, 2),
                        orders=(Order("a", Position(1, 0), Position(2, 0), 0, 5),),
                        initial_robot=RobotState(Position(0, 0), 5),
                        event_tape=EventTape(
                            (DynamicEvent(0, EventKind.CELL_BLOCKED, position=blocked),)
                        ),
                        horizon=5,
                        battery_capacity=5,
                    )
                )
                agent = OpenLoopPlanningAgent()
                agent.reset(environment.snapshot, seed=0)

                self.assertIs(agent.act(environment.snapshot), Action.WAIT)
                self.assertEqual(agent.diagnostics.planning_calls, 1)

    def test_replanning_charges_at_station_when_the_goal_exceeds_battery(self) -> None:
        environment = WarehouseEnvironment(
            Scenario(
                scenario_id="replanning-charge",
                warehouse_map=WarehouseMap(
                    5,
                    1,
                    charging_stations=frozenset({Position(0, 0)}),
                ),
                orders=(Order("a", Position(4, 0), Position(3, 0), 0, 8),),
                initial_robot=RobotState(Position(0, 0), 1),
                event_tape=EventTape(),
                horizon=8,
                battery_capacity=5,
            )
        )
        agent = ReplanningAgent(battery_reserve=0)
        agent.reset(environment.snapshot, seed=0)

        self.assertIs(agent.act(environment.snapshot), Action.CHARGE)

    def test_replanning_routes_to_the_nearest_reachable_charger(self) -> None:
        environment = WarehouseEnvironment(
            Scenario(
                scenario_id="replanning-nearest-charge",
                warehouse_map=WarehouseMap(
                    5,
                    1,
                    charging_stations=frozenset({Position(0, 0), Position(2, 0)}),
                ),
                orders=(Order("a", Position(4, 0), Position(3, 0), 0, 8),),
                initial_robot=RobotState(Position(1, 0), 1),
                event_tape=EventTape(),
                horizon=8,
                battery_capacity=5,
            )
        )
        agent = ReplanningAgent(battery_reserve=0)
        agent.reset(environment.snapshot, seed=0)

        self.assertIs(agent.act(environment.snapshot), Action.LEFT)
        self.assertEqual(agent.diagnostics.planning_calls, 2)

    def test_replanning_waits_when_order_or_charger_is_unreachable(self) -> None:
        scenarios = (
            Scenario(
                scenario_id="replanning-no-charger",
                warehouse_map=WarehouseMap(5, 1),
                orders=(Order("a", Position(4, 0), Position(3, 0), 0, 8),),
                initial_robot=RobotState(Position(1, 0), 1),
                event_tape=EventTape(),
                horizon=8,
                battery_capacity=5,
            ),
            Scenario(
                scenario_id="replanning-disconnected",
                warehouse_map=WarehouseMap(
                    3,
                    3,
                    obstacles=frozenset(
                        {Position(1, 0), Position(1, 1), Position(1, 2)}
                    ),
                ),
                orders=(Order("a", Position(2, 1), Position(2, 2), 0, 8),),
                initial_robot=RobotState(Position(0, 1), 5),
                event_tape=EventTape(),
                horizon=8,
                battery_capacity=5,
            ),
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario.scenario_id):
                environment = WarehouseEnvironment(scenario)
                agent = ReplanningAgent(battery_reserve=0)
                agent.reset(environment.snapshot, seed=0)
                self.assertIs(agent.act(environment.snapshot), Action.WAIT)

    def test_replanning_handles_no_orders_and_initially_carried_orders(self) -> None:
        at_charger = WarehouseEnvironment(
            Scenario(
                scenario_id="replanning-no-orders",
                warehouse_map=WarehouseMap(
                    1,
                    1,
                    charging_stations=frozenset({Position(0, 0)}),
                ),
                orders=(),
                initial_robot=RobotState(Position(0, 0), 1),
                event_tape=EventTape(),
                horizon=3,
                battery_capacity=3,
            )
        )
        charging_agent = ReplanningAgent()
        charging_agent.reset(at_charger.snapshot, seed=0)
        self.assertIs(charging_agent.act(at_charger.snapshot), Action.CHARGE)

        carrying = WarehouseEnvironment(
            Scenario(
                scenario_id="replanning-carrying",
                warehouse_map=WarehouseMap(3, 1),
                orders=(Order("a", Position(0, 0), Position(2, 0), 0, 5),),
                initial_robot=RobotState(Position(0, 0), 3, carried_order_id="a"),
                event_tape=EventTape(),
                horizon=5,
                battery_capacity=3,
            )
        )
        carrying_agent = ReplanningAgent(battery_reserve=0)
        carrying_agent.reset(carrying.snapshot, seed=0)
        self.assertIs(carrying_agent.act(carrying.snapshot), Action.RIGHT)

    def test_replanning_configuration_rejects_negative_battery_reserve(self) -> None:
        with self.assertRaises(ValueError):
            ReplanningAgent(battery_reserve=-1)


if __name__ == "__main__":
    unittest.main()
