from __future__ import annotations

import unittest
from datetime import UTC, datetime

from adaptive_agent_lab.environment.generator import generate_scenario
from adaptive_agent_lab.reporting.demo import build_demo_data


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
        self.assertTrue(all(agent["trace"] for agent in agents))
        self.assertTrue(str(payload["scenarioFingerprint"]).startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
