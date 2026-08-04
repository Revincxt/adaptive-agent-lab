from __future__ import annotations

import unittest

from adaptive_agent_lab.randomness import SeedBook, derive_seed


class RandomnessTests(unittest.TestCase):
    def test_seed_derivation_is_stable_and_namespaced(self) -> None:
        self.assertEqual(derive_seed(42, "scenario", 1), derive_seed(42, "scenario", 1))
        self.assertNotEqual(derive_seed(42, "scenario", 1), derive_seed(42, "events", 1))

    def test_negative_root_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            derive_seed(-1, "agent")

    def test_seed_book_separates_agent_and_event_randomness(self) -> None:
        book = SeedBook(2026)
        self.assertNotEqual(book.for_agent("dqn"), book.for_events(0))
        self.assertNotEqual(book.for_agent("dqn", 0), book.for_agent("dqn", 1))


if __name__ == "__main__":
    unittest.main()
