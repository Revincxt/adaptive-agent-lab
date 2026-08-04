"""Shared agent lifecycle and diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np

from adaptive_agent_lab.environment.contracts import Action, Transition, WarehouseSnapshot


@dataclass(frozen=True, slots=True)
class AgentDiagnostics:
    decisions: int = 0
    planning_calls: int = 0
    expanded_nodes: int = 0
    learning_updates: int = 0


class Agent(ABC):
    """One policy operating through the common snapshot/action contract."""

    name: str
    trainable: bool = False

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self._diagnostics = AgentDiagnostics()
        self._learning_enabled = True

    @property
    def diagnostics(self) -> AgentDiagnostics:
        return self._diagnostics

    @property
    def learning_enabled(self) -> bool:
        return self._learning_enabled

    def set_learning_enabled(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self._learning_enabled = enabled

    def reset(self, snapshot: WarehouseSnapshot, *, seed: int) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self._rng = np.random.default_rng(seed)
        self._diagnostics = AgentDiagnostics()
        self._reset(snapshot)

    def _reset(self, snapshot: WarehouseSnapshot) -> None:
        del snapshot

    def act(self, snapshot: WarehouseSnapshot, *, explore: bool = False) -> Action:
        if snapshot.state.terminated:
            return Action.WAIT
        action = self._act(snapshot, explore=explore)
        if not isinstance(action, Action):
            raise TypeError("agent policies must return an Action")
        self._diagnostics = replace(
            self._diagnostics,
            decisions=self._diagnostics.decisions + 1,
        )
        return action

    @abstractmethod
    def _act(self, snapshot: WarehouseSnapshot, *, explore: bool) -> Action:
        raise NotImplementedError

    def observe(self, transition: Transition) -> None:
        del transition

    def end_episode(self, snapshot: WarehouseSnapshot) -> None:
        del snapshot

    def _record_plan(self, expanded_nodes: int) -> None:
        self._diagnostics = replace(
            self._diagnostics,
            planning_calls=self._diagnostics.planning_calls + 1,
            expanded_nodes=self._diagnostics.expanded_nodes + expanded_nodes,
        )

    def _record_learning_update(self, count: int = 1) -> None:
        self._diagnostics = replace(
            self._diagnostics,
            learning_updates=self._diagnostics.learning_updates + count,
        )


__all__ = ["Agent", "AgentDiagnostics"]
