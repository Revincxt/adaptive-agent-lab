from __future__ import annotations

import json
import unittest

from adaptive_agent_lab.environment.contracts import Position
from adaptive_agent_lab.environment.events import DynamicEvent, EventKind, EventTape


class DynamicEventTests(unittest.TestCase):
    def test_event_payload_is_kind_specific(self) -> None:
        arrival = DynamicEvent(3, EventKind.ORDER_ARRIVAL, order_id="订单-1")
        closure = DynamicEvent(4, EventKind.CELL_BLOCKED, position=Position(2, 1))

        self.assertEqual(arrival.order_id, "订单-1")
        self.assertEqual(closure.position, Position(2, 1))
        with self.assertRaises(ValueError):
            DynamicEvent(3, EventKind.ORDER_ARRIVAL, position=Position(0, 0))
        with self.assertRaises(ValueError):
            DynamicEvent(3, EventKind.CELL_UNBLOCKED, order_id="order-1")

    def test_negative_event_time_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DynamicEvent(-1, EventKind.ORDER_ARRIVAL, order_id="order-1")


class EventTapeTests(unittest.TestCase):
    def test_events_are_canonically_sorted_and_queryable(self) -> None:
        unblock = DynamicEvent(6, EventKind.CELL_UNBLOCKED, Position(2, 1))
        arrival = DynamicEvent(2, EventKind.ORDER_ARRIVAL, order_id="order-2")
        block = DynamicEvent(4, EventKind.CELL_BLOCKED, Position(2, 1))
        tape = EventTape([unblock, block, arrival])

        self.assertEqual(tape.events, (arrival, block, unblock))
        self.assertEqual(tape.at(4), (block,))
        self.assertEqual(tape.between(2, 6), (arrival, block))

    def test_conflicting_events_for_one_target_and_time_are_rejected(self) -> None:
        position = Position(1, 1)
        with self.assertRaises(ValueError):
            EventTape(
                (
                    DynamicEvent(2, EventKind.CELL_BLOCKED, position),
                    DynamicEvent(2, EventKind.CELL_UNBLOCKED, position),
                )
            )

    def test_json_round_trip_is_byte_deterministic(self) -> None:
        tape = EventTape(
            (
                DynamicEvent(5, EventKind.CELL_BLOCKED, Position(3, 2)),
                DynamicEvent(1, EventKind.ORDER_ARRIVAL, order_id="订单-1"),
            )
        )

        encoded = tape.to_json()
        decoded = EventTape.from_json(encoded)

        self.assertEqual(decoded, tape)
        self.assertEqual(decoded.to_json(), encoded)
        self.assertIn("订单-1", encoded)
        self.assertNotIn(" ", encoded)

    def test_json_schema_is_strict(self) -> None:
        raw = EventTape().to_dict()
        raw["unexpected"] = True
        with self.assertRaises(ValueError):
            EventTape.from_json(json.dumps(raw))

        unknown_kind = {
            "schema_version": 1,
            "events": [
                {"kind": "earthquake", "order_id": None, "position": None, "time": 1}
            ],
        }
        with self.assertRaises(ValueError):
            EventTape.from_json(json.dumps(unknown_kind))

    def test_horizon_validation_uses_half_open_episode_times(self) -> None:
        tape = EventTape((DynamicEvent(5, EventKind.ORDER_ARRIVAL, order_id="late"),))
        tape.validate_horizon(6)
        with self.assertRaises(ValueError):
            tape.validate_horizon(5)


if __name__ == "__main__":
    unittest.main()
