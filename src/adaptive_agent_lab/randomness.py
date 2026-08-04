"""Deterministic seed derivation for paired experiments.

Python's built-in ``hash`` is deliberately randomized between interpreter
processes.  Experiment seeds therefore use a stable BLAKE2 digest instead.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

MAX_SEED = 2**63 - 1


def derive_seed(root_seed: int, *labels: object) -> int:
    """Derive a stable non-negative seed from a root seed and semantic labels."""

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    digest = hashlib.blake2b(digest_size=8, person=b"aal-seed")
    digest.update(str(root_seed).encode("utf-8"))
    for label in labels:
        digest.update(b"\x00")
        digest.update(str(label).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") & MAX_SEED


@dataclass(frozen=True, slots=True)
class SeedBook:
    """Named seed namespaces for one experiment root."""

    root: int

    def for_scenario(self, scenario_index: int) -> int:
        return derive_seed(self.root, "scenario", scenario_index)

    def for_events(self, scenario_index: int) -> int:
        return derive_seed(self.root, "events", scenario_index)

    def for_agent(self, agent_name: str, replicate: int = 0) -> int:
        return derive_seed(self.root, "agent", agent_name, replicate)

    def for_replay(self, agent_name: str, replicate: int = 0) -> int:
        return derive_seed(self.root, "replay", agent_name, replicate)
