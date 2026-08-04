"""Versioned reward terms shared by every learning agent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RewardScheme:
    """Scalar training reward with separately reportable components.

    Benchmark conclusions use operational metrics such as weighted on-time
    completion rather than this shaped scalar alone.
    """

    step_cost: float = -0.05
    movement_cost: float = -0.02
    invalid_action_cost: float = -1.0
    pickup_reward: float = 0.25
    delivery_reward_per_priority: float = 10.0
    on_time_bonus_per_priority: float = 5.0
    lateness_cost_per_step: float = -0.20
    stranded_cost: float = -10.0

    def as_dict(self) -> Mapping[str, float]:
        return asdict(self)

    def delivery(self, *, priority: float, completion_time: int, deadline: int) -> float:
        if priority <= 0.0:
            raise ValueError("priority must be positive")
        if completion_time < 0 or deadline < 0:
            raise ValueError("times must be non-negative")
        value = self.delivery_reward_per_priority * priority
        if completion_time <= deadline:
            value += self.on_time_bonus_per_priority * priority
        else:
            value += self.lateness_cost_per_step * (completion_time - deadline)
        return value


DEFAULT_REWARD_SCHEME = RewardScheme()
