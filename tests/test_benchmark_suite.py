from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.benchmarking.suite import (
    AgentFactory,
    BenchmarkCondition,
    load_benchmark_config,
    run_benchmark,
)
from adaptive_agent_lab.environment.contracts import Action, WarehouseSnapshot
from adaptive_agent_lab.reporting.artifacts import read_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _WaitAgent(Agent):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        del snapshot, explore
        return Action.WAIT


def _factories(*names: str) -> dict[str, AgentFactory]:
    return {name: (lambda name=name: _WaitAgent(name)) for name in names}


class BenchmarkSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_benchmark_config(PROJECT_ROOT / "configs" / "benchmarks" / "main.json")

    def test_main_config_loads_and_expands_declared_conditions(self) -> None:
        self.assertEqual(self.config.suite_id, "main-v0.1")
        self.assertEqual(self.config.test_root_seed, 2026080503)
        self.assertEqual(len(self.config.conditions()), 1080)
        self.assertEqual(
            self.config.planned_comparisons[0],
            ("replanning", "planning"),
        )

    def test_each_agent_in_a_condition_receives_the_same_scenario(self) -> None:
        condition = BenchmarkCondition(
            "confirmatory-small",
            "confirmatory",
            "small",
            "low",
            "light",
            0,
        )
        result = run_benchmark(
            self.config,
            _factories("planning", "replanning"),
            conditions=(condition,),
            measure_timing=False,
        )
        self.assertEqual(len(result.episodes), 2)
        self.assertEqual(
            {record.scenario_fingerprint for record in result.episodes},
            {result.episodes[0].scenario_fingerprint},
        )
        self.assertEqual(
            {record.scenario_seed for record in result.episodes},
            {result.episodes[0].scenario_seed},
        )

        same_pairing_dimensions = BenchmarkCondition(
            "another-block",
            "exploratory",
            "small",
            "low",
            "light",
            0,
        )
        self.assertEqual(
            self.config.scenario_seed(condition),
            self.config.scenario_seed(same_pairing_dimensions),
        )
        self.assertEqual(
            self.config.make_scenario(condition).to_json(),
            self.config.make_scenario(same_pairing_dimensions).to_json(),
        )

    def test_timing_disabled_run_is_exactly_repeatable(self) -> None:
        conditions = (
            BenchmarkCondition(
                "confirmatory-small",
                "confirmatory",
                "small",
                "static",
                "light",
                0,
            ),
            BenchmarkCondition(
                "confirmatory-small",
                "confirmatory",
                "small",
                "static",
                "light",
                1,
            ),
        )
        first = run_benchmark(
            self.config,
            _factories("planning", "replanning"),
            conditions=conditions,
            measure_timing=False,
        )
        second = run_benchmark(
            self.config,
            _factories("planning", "replanning"),
            conditions=conditions,
            measure_timing=False,
        )
        self.assertEqual(first, second)

    def test_factory_mapping_order_does_not_change_scenarios_or_rotation(self) -> None:
        condition = BenchmarkCondition(
            "confirmatory-small",
            "confirmatory",
            "small",
            "static",
            "light",
            1,
        )
        forward = run_benchmark(
            self.config,
            _factories("planning", "replanning"),
            conditions=(condition,),
        )
        reverse = run_benchmark(
            self.config,
            _factories("replanning", "planning"),
            conditions=(condition,),
        )
        self.assertEqual(forward.manifest["scenarios"], reverse.manifest["scenarios"])
        self.assertEqual(
            [record.agent_id for record in forward.episodes],
            ["replanning", "planning"],
        )
        self.assertEqual(
            [record.agent_id for record in reverse.episodes],
            ["replanning", "planning"],
        )

    def test_summary_and_atomic_artifacts_are_correct(self) -> None:
        conditions = tuple(
            BenchmarkCondition(
                "confirmatory-small",
                "confirmatory",
                "small",
                "static",
                "light",
                tape_index,
            )
            for tape_index in range(2)
        )
        result = run_benchmark(
            self.config,
            _factories("planning", "replanning"),
            conditions=conditions,
            measure_timing=False,
        )
        agents = result.summary["agents"]
        assert isinstance(agents, dict)
        planning = agents["planning"]
        assert isinstance(planning, dict)
        self.assertEqual(planning["episode_count"], 2)
        metrics = planning["metrics"]
        assert isinstance(metrics, dict)
        total_reward = metrics["total_reward"]
        assert isinstance(total_reward, dict)
        self.assertEqual(total_reward["count"], 2)

        comparisons = result.summary["paired_differences"]
        assert isinstance(comparisons, dict)
        comparison = comparisons["replanning_vs_planning"]
        assert isinstance(comparison, dict)
        comparison_metrics = comparison["metrics"]
        assert isinstance(comparison_metrics, dict)
        reward_difference = comparison_metrics["total_reward"]
        assert isinstance(reward_difference, dict)
        self.assertEqual(reward_difference["count"], 2)
        self.assertEqual(reward_difference["mean_difference"], 0.0)
        self.assertEqual(reward_difference["improvement_probability"], 0.5)

        with tempfile.TemporaryDirectory() as directory:
            paths = result.write_artifacts(directory)
            self.assertEqual(read_json(paths["manifest"]), result.manifest)
            self.assertEqual(read_json(paths["summary"]), result.summary)
            lines = paths["episodes"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 4)
            self.assertEqual(json.loads(lines[0]), result.episodes[0].to_dict())
            self.assertFalse(any(Path(directory).glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
