from __future__ import annotations

import unittest

from adaptive_agent_lab.environment.rewards import RewardScheme


class RewardTests(unittest.TestCase):
    def test_on_time_delivery_receives_priority_scaled_bonus(self) -> None:
        rewards = RewardScheme()
        self.assertEqual(rewards.delivery(priority=2.0, completion_time=8, deadline=8), 30.0)

    def test_late_delivery_uses_elapsed_lateness_not_priority(self) -> None:
        rewards = RewardScheme()
        self.assertEqual(rewards.delivery(priority=2.0, completion_time=13, deadline=10), 19.4)

    def test_invalid_priority_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RewardScheme().delivery(priority=0.0, completion_time=0, deadline=0)


if __name__ == "__main__":
    unittest.main()
