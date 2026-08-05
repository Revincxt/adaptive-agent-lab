from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from adaptive_agent_lab.environment.contracts import Action
from adaptive_agent_lab.environment.generator import generate_scenario
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.environment.simulator import WarehouseEnvironment
from adaptive_agent_lab.reporting.artifacts import fingerprint
from adaptive_agent_lab.reporting.demo import build_demo_data

ROOT = Path(__file__).resolve().parents[1]


class DemoReportTests(unittest.TestCase):
    def test_demo_contains_six_real_agent_trajectories(self) -> None:
        scenario = generate_scenario(
            5,
            size="tiny",
            dynamics="low",
            order_count=1,
            horizon=30,
        )
        payload = build_demo_data(
            scenario,
            training_episodes={},
            generated_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
        agents = payload["agents"]
        self.assertIsInstance(agents, list)
        self.assertEqual(len(agents), 6)
        self.assertEqual(payload["generatedAt"], "2026-08-05T00:00:00+00:00")
        self.assertEqual(payload["rootSeed"], 42)
        self.assertTrue(all(agent["trace"] for agent in agents))
        self.assertTrue(all(agent["metrics"]["decisionTimeMs"] is None for agent in agents))
        self.assertTrue(str(payload["scenarioFingerprint"]).startswith("sha256:"))

    def test_checked_in_gallery_traces_replay_through_the_real_environment(self) -> None:
        scenario_paths = {
            "rack-maze": ROOT / "scenarios/medium/maze-warehouse.json",
            "parallel-aisles": ROOT / "scenarios/medium/parallel-aisles.json",
            "cross-dock": ROOT / "scenarios/medium/cross-dock.json",
            "serpentine": ROOT / "scenarios/medium/serpentine.json",
        }
        payload = json.loads((ROOT / "web/public/demo-data.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["schemaVersion"], 2)
        self.assertEqual(payload["rootSeed"], 42)
        self.assertEqual(payload["defaultCaseId"], "rack-maze")
        self.assertEqual(
            {demo_case["caseId"] for demo_case in payload["cases"]},
            set(scenario_paths),
        )

        for demo_case in payload["cases"]:
            scenario = Scenario.from_json(
                scenario_paths[demo_case["caseId"]].read_text(encoding="utf-8")
            )
            self.assertEqual(demo_case["scenario"]["id"], scenario.scenario_id)
            self.assertEqual(demo_case["scenarioFingerprint"], fingerprint(scenario.to_dict()))
            self.assertEqual(
                {agent["id"] for agent in demo_case["agents"]},
                {"planning", "replanning", "q-learning", "dyna-q", "dqn", "hybrid"},
            )

            for agent in demo_case["agents"]:
                with self.subTest(case=demo_case["caseId"], agent=agent["id"]):
                    environment = WarehouseEnvironment(scenario)
                    for record in agent["trace"]:
                        result = environment.step(Action(record["action"]))
                        state = result.state
                        self.assertEqual(record["time"], state.time)
                        self.assertEqual(
                            record["position"],
                            [state.robot.position.x, state.robot.position.y],
                        )
                        self.assertEqual(record["battery"], state.robot.battery)
                        self.assertAlmostEqual(record["reward"], result.reward)
                        self.assertAlmostEqual(
                            record["cumulativeReward"], state.cumulative_reward
                        )
                        self.assertEqual(record["carriedOrderId"], state.robot.carried_order_id)
                        self.assertEqual(
                            record["deliveredOrderId"], result.info["delivered_order"]
                        )
                        self.assertEqual(
                            record["violations"],
                            [violation.code.value for violation in result.violations],
                        )
                        self.assertEqual(record["eventCount"], result.info["event_count"])
                    self.assertTrue(environment.state.terminated)


if __name__ == "__main__":
    unittest.main()
