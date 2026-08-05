from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptive_agent_lab import cli
from adaptive_agent_lab.environment.generator import generate_scenario
from adaptive_agent_lab.reporting import demo

ROOT = Path(__file__).resolve().parents[1]
TRAINING_EPISODES_ZERO = {
    "q-learning": 0,
    "dyna-q": 0,
    "dqn": 0,
    "hybrid": 0,
}


def _write_scenario(path: Path, *, seed: int, scenario_id: str) -> None:
    scenario = generate_scenario(
        seed,
        size="tiny",
        dynamics="static",
        order_count=0,
        horizon=4,
        scenario_id=scenario_id,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario.to_json(indent=2) + "\n", encoding="utf-8")


def test_export_gallery_writes_v2_cases_relative_to_config_with_fresh_agents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_directory = tmp_path / "nested" / "configs"
    scenario_directory = tmp_path / "nested" / "scenarios"
    first_scenario = scenario_directory / "first.json"
    second_scenario = scenario_directory / "second.json"
    _write_scenario(first_scenario, seed=1, scenario_id="first-scenario")
    _write_scenario(second_scenario, seed=2, scenario_id="second-scenario")

    config = {
        "defaultCaseId": "first",
        "trainingEpisodes": TRAINING_EPISODES_ZERO,
        "cases": [
            {
                "caseId": "first",
                "mapId": "first-map",
                "label": "First map",
                "description": "A tiny first map.",
                "tags": ["tiny", "static"],
                "display": {"topology": "Open grid", "difficulty": "Easy"},
                "scenario": "../scenarios/first.json",
            },
            {
                "caseId": "second",
                "mapId": "second-map",
                "label": "Second map",
                "description": "A tiny second map.",
                "tags": ["tiny", "alternate"],
                "display": {"topology": "Compact grid", "difficulty": "Easy"},
                "scenario": "../scenarios/second.json",
            },
        ],
    }
    config_path = config_directory / "gallery.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config), encoding="utf-8")

    constructed_agents: list[tuple[object, ...]] = []
    original_make_demo_agents = demo.make_demo_agents

    def tracked_make_demo_agents() -> tuple[object, ...]:
        agents = original_make_demo_agents()
        constructed_agents.append(agents)
        return agents

    monkeypatch.setattr(demo, "make_demo_agents", tracked_make_demo_agents)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    output = tmp_path / "gallery-output.json"

    assert (
        cli.main(
            [
                "export-gallery",
                "--config",
                str(config_path),
                "--output",
                str(output),
                "--seed",
                "17",
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == 2
    assert payload["rootSeed"] == 17
    assert payload["verificationStatus"] == demo.DEMO_VERIFICATION_STATUS
    assert payload["defaultCaseId"] == "first"
    assert isinstance(payload["generatedAt"], str)
    assert [case["caseId"] for case in payload["cases"]] == ["first", "second"]
    assert [case["scenario"]["id"] for case in payload["cases"]] == [
        "first-scenario",
        "second-scenario",
    ]
    assert all(case["display"]["topology"] for case in payload["cases"])
    assert all(case["display"]["difficulty"] for case in payload["cases"])
    assert all(len(case["agents"]) == 6 for case in payload["cases"])
    assert all(case["scenarioFingerprint"].startswith("sha256:") for case in payload["cases"])
    assert all(
        set(case["trainingEpisodes"].values()) == {0} for case in payload["cases"]
    )

    assert len(constructed_agents) == 2
    assert all(
        first is not second
        for first, second in zip(constructed_agents[0], constructed_agents[1], strict=True)
    )
    assert "gallery cases=2 | default=first | seed=17" in capsys.readouterr().out


def test_gallery_config_requires_display_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "gallery.json"
    config_path.write_text(
        json.dumps(
            {
                "defaultCaseId": "missing-display-field",
                "trainingEpisodes": {},
                "cases": [
                    {
                        "caseId": "missing-display-field",
                        "mapId": "missing-display-field",
                        "label": "Missing metadata",
                        "description": "Invalid on purpose.",
                        "tags": [],
                        "display": {"topology": "Open grid"},
                        "scenario": "unused.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field 'difficulty'"):
        demo.build_demo_gallery(config_path)


def test_gallery_validates_every_case_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario_path = tmp_path / "scenario.json"
    _write_scenario(scenario_path, seed=1, scenario_id="validation-scenario")
    config_path = tmp_path / "gallery.json"
    config_path.write_text(
        json.dumps(
            {
                "defaultCaseId": "not-a-case",
                "trainingEpisodes": TRAINING_EPISODES_ZERO,
                "cases": [
                    {
                        "caseId": "only-case",
                        "mapId": "only-map",
                        "label": "Only map",
                        "description": "Valid case with an invalid gallery default.",
                        "tags": ["tiny"],
                        "display": {"topology": "Open grid", "difficulty": "Easy"},
                        "scenario": "scenario.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fail_if_trained(*_args: object, **_kwargs: object) -> dict[str, object]:
        pytest.fail("gallery generation began before config validation finished")

    monkeypatch.setattr(demo, "build_demo_data", fail_if_trained)
    with pytest.raises(ValueError, match="defaultCaseId must match"):
        demo.build_demo_gallery(config_path)


def test_checked_in_gallery_config_declares_four_maps() -> None:
    config = json.loads((ROOT / "configs/demo-gallery.json").read_text(encoding="utf-8"))

    assert config["defaultCaseId"] == "rack-maze"
    assert [case["caseId"] for case in config["cases"]] == [
        "rack-maze",
        "parallel-aisles",
        "cross-dock",
        "serpentine",
    ]
    assert all(case["display"]["topology"] for case in config["cases"])
    assert all(case["display"]["difficulty"] for case in config["cases"])
    assert all(not Path(case["scenario"]).is_absolute() for case in config["cases"])
