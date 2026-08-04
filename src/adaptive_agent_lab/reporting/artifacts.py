"""Canonical machine-readable artifacts and reproducibility fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def canonical_json(value: object) -> str:
    """Encode a value with stable ordering and no presentation whitespace."""

    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: object, *, prefix: str = "sha256") -> str:
    """Return a versionable content fingerprint."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def write_json_atomic(path: Path, value: object, *, indent: int = 2) -> None:
    """Write JSON without exposing a partially written artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=indent,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Minimal provenance attached to every training or evaluation run."""

    run_id: str
    kind: str
    agent: str
    root_seed: int
    scenario_fingerprint: str
    config_fingerprint: str
    package_version: str
    created_at: str
    metadata: Mapping[str, str]

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        kind: str,
        agent: str,
        root_seed: int,
        scenario: object,
        config: object,
        package_version: str,
        metadata: Mapping[str, str] | None = None,
        created_at: datetime | None = None,
    ) -> RunManifest:
        timestamp = (created_at or datetime.now(UTC)).astimezone(UTC).isoformat()
        return cls(
            run_id=run_id,
            kind=kind,
            agent=agent,
            root_seed=root_seed,
            scenario_fingerprint=fingerprint(scenario),
            config_fingerprint=fingerprint(config),
            package_version=package_version,
            created_at=timestamp,
            metadata=dict(metadata or {}),
        )


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))
