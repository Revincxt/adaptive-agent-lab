from __future__ import annotations

import unittest

from adaptive_agent_lab.benchmarking.statistics import (
    bootstrap_mean_interval,
    compare_paired,
    summarize,
)


class StatisticsTests(unittest.TestCase):
    def test_summary_is_deterministic(self) -> None:
        first = summarize([1.0, 2.0, 3.0], resamples=200, seed=42)
        second = summarize([1.0, 2.0, 3.0], resamples=200, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first.count, 3)
        self.assertEqual(first.mean, 2.0)
        self.assertLessEqual(first.mean_interval.low, first.mean)
        self.assertGreaterEqual(first.mean_interval.high, first.mean)

    def test_paired_comparison_preserves_direction(self) -> None:
        comparison = compare_paired(
            [3.0, 3.0, 5.0],
            [1.0, 3.0, 2.0],
            resamples=200,
            seed=7,
        )
        self.assertEqual(comparison.mean_difference, 5.0 / 3.0)
        self.assertEqual(comparison.improvement_probability, 2.5 / 3.0)

    def test_paired_comparison_rejects_unmatched_seeds(self) -> None:
        with self.assertRaises(ValueError):
            compare_paired([1.0], [1.0, 2.0], resamples=200)

    def test_non_finite_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize([1.0, float("nan")], resamples=200)

    def test_empty_and_multidimensional_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize([], resamples=200)
        with self.assertRaises(ValueError):
            summarize([[1.0], [2.0]], resamples=200)  # type: ignore[list-item]

    def test_bootstrap_contract_validates_confidence_and_resample_count(self) -> None:
        for confidence in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(confidence=confidence), self.assertRaises(ValueError):
                bootstrap_mean_interval([1.0], confidence=confidence, resamples=100)
        with self.assertRaises(ValueError):
            bootstrap_mean_interval([1.0], resamples=99)

    def test_single_value_summary_has_zero_sampling_error(self) -> None:
        result = summarize([4.5], resamples=100, seed=3)
        self.assertEqual(result.standard_deviation, 0.0)
        self.assertEqual(result.standard_error, 0.0)
        self.assertEqual((result.mean_interval.low, result.mean_interval.high), (4.5, 4.5))

    def test_lower_is_better_orients_improvement_without_flipping_raw_difference(self) -> None:
        comparison = compare_paired(
            [1.0, 1.0],
            [2.0, 1.0],
            higher_is_better=False,
            resamples=100,
            seed=4,
        )
        self.assertEqual(comparison.mean_difference, -0.5)
        self.assertEqual(comparison.median_difference, -0.5)
        self.assertEqual(comparison.improvement_probability, 0.75)


if __name__ == "__main__":
    unittest.main()
