from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from adaptive_agent_lab.environment.contracts import Action, Position
from adaptive_agent_lab.reporting.artifacts import (
    RunManifest,
    canonical_json,
    fingerprint,
    read_json,
    write_json_atomic,
)


class ArtifactTests(unittest.TestCase):
    def test_canonical_json_and_fingerprint_ignore_mapping_order(self) -> None:
        left = {"b": 2, "a": 1}
        right = {"a": 1, "b": 2}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(fingerprint(left), fingerprint(right))

    def test_atomic_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.json"
            write_json_atomic(path, {"seed": 42, "agent": "dyna-q"})
            self.assertEqual(read_json(path), {"agent": "dyna-q", "seed": 42})
            self.assertFalse(path.with_name(".artifact.json.tmp").exists())

    def test_manifest_has_stable_input_fingerprints(self) -> None:
        manifest = RunManifest.create(
            run_id="test-42",
            kind="evaluation",
            agent="replanning",
            root_seed=42,
            scenario={"name": "tiny"},
            config={"episodes": 1},
            package_version="0.1.0",
            created_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
        encoded = asdict(manifest)
        self.assertEqual(encoded["created_at"], "2026-08-05T00:00:00+00:00")
        self.assertTrue(encoded["scenario_fingerprint"].startswith("sha256:"))

    def test_default_encoder_supports_domain_values_and_paths(self) -> None:
        encoded = canonical_json(
            {
                "action": Action.WAIT,
                "coordinate": Position(2, 3),
                "path": Path("runs/example.json"),
                "sequence": (1, 2),
                "tags": frozenset({"evaluation"}),
            }
        )
        self.assertEqual(
            encoded,
            '{"action":"wait","coordinate":{"x":2,"y":3},'
            '"path":"runs/example.json","sequence":[1,2],"tags":["evaluation"]}',
        )

    def test_default_encoder_rejects_unknown_objects_and_non_finite_values(self) -> None:
        with self.assertRaises(TypeError):
            canonical_json(object())
        with self.assertRaises(ValueError):
            canonical_json({"loss": float("nan")})

    def test_manifest_defaults_to_utc_timestamp_and_empty_metadata(self) -> None:
        manifest = RunManifest.create(
            run_id="test-now",
            kind="training",
            agent="dqn",
            root_seed=1,
            scenario={},
            config={},
            package_version="0.1.0",
        )
        self.assertEqual(manifest.metadata, {})
        self.assertTrue(manifest.created_at.endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
