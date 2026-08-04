from __future__ import annotations

import json
import unittest

import numpy as np

from adaptive_agent_lab.learning.network import MLPQNetwork


class MLPQNetworkTests(unittest.TestCase):
    def test_predict_preserves_single_and_batch_shapes(self) -> None:
        network = MLPQNetwork(3, 2, (5,), seed=7)
        self.assertEqual(network.predict([1.0, 2.0, 3.0]).shape, (2,))
        self.assertEqual(network.predict([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]).shape, (2, 2))

    def test_mse_training_reduces_loss(self) -> None:
        network = MLPQNetwork(2, 2, (8,), seed=3)
        states = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        targets = np.asarray([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.5]])
        initial = float(np.mean((network.predict(states) - targets) ** 2))
        for _ in range(200):
            network.train_batch(states, targets, learning_rate=0.02, max_grad_norm=5.0)
        final = float(np.mean((network.predict(states) - targets) ** 2))
        self.assertLess(final, initial * 0.25)

    def test_td_training_and_action_validation(self) -> None:
        network = MLPQNetwork(2, 3, (6, 4), seed=9)
        states = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        before = network.predict(states)
        targets = before[[0, 1], [1, 2]] + np.asarray([1.0, -1.0])
        loss = network.train_td_batch(states, [1, 2], targets, learning_rate=0.01)
        self.assertGreater(loss, 0.0)
        with self.assertRaises(ValueError):
            network.train_td_batch(states, [1, 3], targets)

    def test_copy_soft_update_and_json_round_trip(self) -> None:
        source = MLPQNetwork(2, 2, (4,), seed=1)
        target = MLPQNetwork(2, 2, (4,), seed=2)
        probe = np.asarray([[0.25, -0.5]])
        target_before = target.state_dict()
        source_state = source.state_dict()
        source_values = source.predict(probe)
        target.soft_update_from(source, 0.25)
        updated = target.state_dict()
        for old, origin, blended in zip(
            target_before["weights"],
            source_state["weights"],
            updated["weights"],
            strict=True,
        ):
            np.testing.assert_allclose(
                blended, 0.75 * np.asarray(old) + 0.25 * np.asarray(origin), atol=1e-12
            )

        state = source.state_dict()
        json.dumps(state)
        restored = MLPQNetwork.from_state_dict(state)
        np.testing.assert_allclose(restored.predict(probe), source_values)

        clone = source.copy()
        clone.train_td_batch(probe, [0], [2.0], learning_rate=0.1)
        np.testing.assert_allclose(source.predict(probe), source_values)

    def test_invalid_shapes_and_non_finite_inputs_are_rejected(self) -> None:
        network = MLPQNetwork(2, 2, (4,), seed=0)
        with self.assertRaises(ValueError):
            network.predict([1.0])
        with self.assertRaises(ValueError):
            network.predict([1.0, np.nan])
        with self.assertRaises(ValueError):
            MLPQNetwork(2, 2, (), seed=0)


if __name__ == "__main__":
    unittest.main()
