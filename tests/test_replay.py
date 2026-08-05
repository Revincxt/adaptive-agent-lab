from __future__ import annotations

import unittest

import numpy as np

from adaptive_agent_lab.learning.replay import ReplayBuffer


class CopyTracked:
    copies = 0

    def __init__(self, value: int) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, object]) -> CopyTracked:
        del memo
        type(self).copies += 1
        return type(self)(self.value)


class ReplayBufferTests(unittest.TestCase):
    def test_capacity_keeps_newest_transitions_in_fifo_order(self) -> None:
        buffer: ReplayBuffer[np.ndarray, int] = ReplayBuffer(3, seed=1)
        for value in range(5):
            buffer.add(np.asarray([value]), value, float(value), np.asarray([value + 1]), False)
        self.assertEqual(len(buffer), 3)
        self.assertTrue(buffer.is_full)
        self.assertEqual([item.action for item in buffer.transitions()], [2, 3, 4])
        self.assertEqual({item.action for item in buffer.sample(3)}, {2, 3, 4})

    def test_same_seed_produces_same_sampling_sequence(self) -> None:
        first: ReplayBuffer[tuple[int, ...], int] = ReplayBuffer(10, seed=44)
        second: ReplayBuffer[tuple[int, ...], int] = ReplayBuffer(10, seed=44)
        for value in range(8):
            first.add((value,), value, float(value), (value + 1,), value == 7)
            second.add((value,), value, float(value), (value + 1,), value == 7)
        for _ in range(3):
            self.assertEqual(
                [item.action for item in first.sample(4)],
                [item.action for item in second.sample(4)],
            )

    def test_values_are_defensively_copied(self) -> None:
        state = np.asarray([1.0, 2.0])
        buffer: ReplayBuffer[np.ndarray, int] = ReplayBuffer(2, seed=0)
        buffer.add(state, 0, 1.0, state + 1.0, False)
        state[0] = 99.0
        sampled = buffer.sample(1)[0]
        self.assertEqual(float(sampled.state[0]), 1.0)
        sampled.state[0] = -5.0
        self.assertEqual(float(buffer.transitions()[0].state[0]), 1.0)

    def test_numeric_batch_shapes_and_dtypes(self) -> None:
        buffer: ReplayBuffer[np.ndarray, int] = ReplayBuffer(4, seed=5)
        for value in range(4):
            buffer.add(
                np.asarray([value, value + 1], dtype=np.float32),
                value % 2,
                value / 2,
                np.asarray([value + 1, value + 2], dtype=np.float32),
                value == 3,
            )
        batch = buffer.sample_batch(3)
        self.assertEqual(batch.states.shape, (3, 2))
        self.assertEqual(batch.next_states.shape, (3, 2))
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.rewards.dtype, np.float64)
        self.assertEqual(batch.dones.dtype, np.bool_)

    def test_sampling_copies_only_selected_transitions(self) -> None:
        buffer: ReplayBuffer[CopyTracked, int] = ReplayBuffer(10, seed=5)
        for value in range(10):
            buffer.add(CopyTracked(value), value, 0.0, CopyTracked(value + 1), False)

        CopyTracked.copies = 0
        sampled = buffer.sample(2)

        self.assertEqual(len(sampled), 2)
        self.assertEqual(CopyTracked.copies, 4)

    def test_invalid_sample_sizes_are_rejected(self) -> None:
        buffer: ReplayBuffer[int, int] = ReplayBuffer(2, seed=0)
        with self.assertRaises(ValueError):
            buffer.sample(1)
        buffer.add(0, 0, 0.0, 1, False)
        with self.assertRaises(ValueError):
            buffer.sample(2)
        with self.assertRaises(ValueError):
            buffer.sample(0)


if __name__ == "__main__":
    unittest.main()
