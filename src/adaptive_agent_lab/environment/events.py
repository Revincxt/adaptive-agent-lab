"""Replayable dynamic events for paired warehouse experiments."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from adaptive_agent_lab.environment.contracts import Position

EVENT_TAPE_SCHEMA_VERSION = 1


class EventKind(StrEnum):
    """External changes supported by the v0.1 environment."""

    ORDER_ARRIVAL = "order_arrival"
    CELL_BLOCKED = "cell_blocked"
    CELL_UNBLOCKED = "cell_unblocked"


@dataclass(frozen=True, slots=True)
class DynamicEvent:
    """One exogenous change at a discrete environment time."""

    time: int
    kind: EventKind
    position: Position | None = None
    order_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            raise TypeError("time must be an integer")
        if self.time < 0:
            raise ValueError("time must be non-negative")
        if not isinstance(self.kind, EventKind):
            raise TypeError("kind must be an EventKind")
        if self.position is not None and not isinstance(self.position, Position):
            raise TypeError("position must be a Position or None")
        if self.order_id is not None:
            if not isinstance(self.order_id, str):
                raise TypeError("order_id must be a string or None")
            if not self.order_id.strip():
                raise ValueError("order_id must not be empty")

        if self.kind is EventKind.ORDER_ARRIVAL:
            if self.order_id is None or self.position is not None:
                raise ValueError("ORDER_ARRIVAL requires order_id and no position")
        elif self.position is None or self.order_id is not None:
            raise ValueError("cell events require position and no order_id")

    @property
    def sort_key(self) -> tuple[int, str, str, int, int]:
        """Canonical ordering used in tapes and serialized scenarios."""

        position = self.position
        return (
            self.time,
            self.kind.value,
            self.order_id or "",
            position.x if position is not None else -1,
            position.y if position is not None else -1,
        )

    def to_dict(self) -> dict[str, object]:
        position: dict[str, int] | None = None
        if self.position is not None:
            position = {"x": self.position.x, "y": self.position.y}
        return {
            "kind": self.kind.value,
            "order_id": self.order_id,
            "position": position,
            "time": self.time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> DynamicEvent:
        _expect_keys(data, {"kind", "order_id", "position", "time"}, "event")
        time = _expect_int(data["time"], "event.time", minimum=0)
        kind_value = _expect_string(data["kind"], "event.kind")
        try:
            kind = EventKind(kind_value)
        except ValueError as error:
            raise ValueError(f"unknown event kind: {kind_value!r}") from error

        order_id_value = data["order_id"]
        order_id: str | None
        if order_id_value is None:
            order_id = None
        else:
            order_id = _expect_string(order_id_value, "event.order_id")

        position_value = data["position"]
        position: Position | None
        if position_value is None:
            position = None
        else:
            position_data = _expect_mapping(position_value, "event.position")
            _expect_keys(position_data, {"x", "y"}, "event.position")
            position = Position(
                _expect_int(position_data["x"], "event.position.x"),
                _expect_int(position_data["y"], "event.position.y"),
            )
        return cls(time=time, kind=kind, position=position, order_id=order_id)


@dataclass(frozen=True, slots=True)
class EventTape:
    """A canonical, immutable sequence of exogenous events.

    Construction sorts events by a total key, so logically identical tapes
    serialize to exactly the same bytes regardless of input iteration order.
    """

    events: tuple[DynamicEvent, ...] = ()

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if not all(isinstance(event, DynamicEvent) for event in events):
            raise TypeError("events must contain only DynamicEvent values")
        if len(set(events)) != len(events):
            raise ValueError("an event tape cannot contain duplicate events")
        events = tuple(sorted(events, key=lambda event: event.sort_key))
        object.__setattr__(self, "events", events)

        targets: set[tuple[int, str, object]] = set()
        for event in events:
            target: tuple[int, str, object]
            if event.kind is EventKind.ORDER_ARRIVAL:
                target = (event.time, "order", event.order_id)
            else:
                target = (event.time, "cell", event.position)
            if target in targets:
                raise ValueError("one target cannot receive two events at the same time")
            targets.add(target)

    def __iter__(self) -> Iterator[DynamicEvent]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    def at(self, time: int) -> tuple[DynamicEvent, ...]:
        """Return all events scheduled for an exact time."""

        if isinstance(time, bool) or not isinstance(time, int):
            raise TypeError("time must be an integer")
        if time < 0:
            raise ValueError("time must be non-negative")
        return tuple(event for event in self.events if event.time == time)

    def between(self, start: int, stop: int) -> tuple[DynamicEvent, ...]:
        """Return events in the half-open interval ``[start, stop)``."""

        start = _expect_int(start, "start", minimum=0)
        stop = _expect_int(stop, "stop", minimum=0)
        if stop < start:
            raise ValueError("stop must not precede start")
        return tuple(event for event in self.events if start <= event.time < stop)

    def validate_horizon(self, horizon: int) -> None:
        horizon = _expect_int(horizon, "horizon", minimum=1)
        outside = tuple(event for event in self.events if event.time >= horizon)
        if outside:
            raise ValueError("all events must occur before the episode horizon")

    def to_dict(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self.events],
            "schema_version": EVENT_TAPE_SCHEMA_VERSION,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EventTape:
        _expect_keys(data, {"events", "schema_version"}, "event tape")
        version = _expect_int(data["schema_version"], "event_tape.schema_version")
        if version != EVENT_TAPE_SCHEMA_VERSION:
            raise ValueError(f"unsupported event tape schema version: {version}")
        raw_events = data["events"]
        if not isinstance(raw_events, list):
            raise TypeError("event_tape.events must be a JSON array")
        events = tuple(
            DynamicEvent.from_dict(_expect_mapping(value, f"event_tape.events[{index}]"))
            for index, value in enumerate(raw_events)
        )
        return cls(events)

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the tape using stable key and collection ordering."""

        if indent is not None:
            _expect_int(indent, "indent", minimum=0)
        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            separators=separators,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str | bytes | bytearray) -> EventTape:
        raw: object = json.loads(text)
        return cls.from_dict(_expect_mapping(raw, "event tape"))


def _expect_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{path} keys must be strings")
    return cast(Mapping[str, object], value)


def _expect_keys(data: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"invalid {path} keys; missing={missing}, unknown={unknown}")


def _expect_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}")
    return value


def _expect_string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    if not value.strip():
        raise ValueError(f"{path} must not be empty")
    return value


__all__ = ["EVENT_TAPE_SCHEMA_VERSION", "DynamicEvent", "EventKind", "EventTape"]
