from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from adaptive_agent_lab import cli
from adaptive_agent_lab.environment.generator import generate_scenario


@pytest.fixture
def scenario_path(tmp_path: Path) -> Path:
    scenario = generate_scenario(
        17,
        size="tiny",
        dynamics="static",
        order_count=0,
        horizon=4,
        scenario_id="cli-fixture",
    )
    path = tmp_path / "scenario.json"
    path.write_text(scenario.to_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_parser_exposes_commands_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    parser = cli.build_parser()
    parsed = parser.parse_args(
        [
            "scenario",
            "generate",
            "out.json",
            "--size",
            "tiny",
            "--dynamics",
            "static",
            "--orders",
            "0",
        ]
    )
    assert parsed.command == "scenario"
    assert parsed.scenario_command == "generate"
    assert parsed.handler is cli._handle_scenario_generate

    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("aal 0.1.0")


def test_scenario_generate_and_show_in_text_and_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "nested" / "generated.json"
    assert (
        cli.main(
            [
                "scenario",
                "generate",
                str(output),
                "--seed",
                "9",
                "--size",
                "tiny",
                "--dynamics",
                "static",
                "--orders",
                "0",
                "--horizon",
                "4",
                "--id",
                "cli-generated",
            ]
        )
        == 0
    )
    generated_message = capsys.readouterr().out
    assert "id=cli-generated" in generated_message
    assert "fingerprint=sha256:" in generated_message
    assert output.exists()

    assert cli.main(["scenario", "show", str(output)]) == 0
    text_output = capsys.readouterr().out
    assert "scenario: cli-generated" in text_output
    assert "map: 6x5" in text_output
    assert "orders=0 | events=0 | horizon=4 | battery=24" in text_output

    assert cli.main(["scenario", "show", str(output), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["scenario_id"] == "cli-generated"
    assert summary["map"]["width"] == 6
    assert summary["orders"] == 0


def test_run_writes_json_trace_and_prints_human_summary(
    scenario_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "traces" / "planning.json"
    assert (
        cli.main(
            [
                "run",
                "--agent",
                "planning",
                "--scenario",
                str(scenario_path),
                "--seed",
                "31",
                "--trace-output",
                str(trace_path),
                "--json",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(trace_path.read_text(encoding="utf-8"))
    assert printed == persisted
    assert printed["agent"] == "planning"
    assert printed["seed"] == 31

    assert (
        cli.main(
            [
                "run",
                "--agent",
                "replanning",
                "--scenario",
                str(scenario_path),
                "--trace-output",
                str(trace_path),
            ]
        )
        == 0
    )
    human_output = capsys.readouterr().out
    assert "agent=replanning | scenario=cli-fixture | seed=42" in human_output
    assert "completed=0/0 | WOTCR=1.000" in human_output
    assert f"trace={trace_path}" in human_output


def test_train_checkpoint_round_trip(
    scenario_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "models" / "q-learning.json"
    assert (
        cli.main(
            [
                "train",
                "--agent",
                "q-learning",
                "--scenario",
                str(scenario_path),
                "--episodes",
                "1",
                "--seed",
                "73",
                "--output",
                str(checkpoint_path),
            ]
        )
        == 0
    )
    assert "trained q-learning for 1 episodes" in capsys.readouterr().out
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "0.1"
    assert checkpoint["agent"] == "q-learning"
    assert checkpoint["root_seed"] == 73
    assert checkpoint["episodes"] == 1
    assert checkpoint["scenario_fingerprints"][0].startswith("sha256:")
    assert isinstance(checkpoint["state"], dict)

    assert (
        cli.main(
            [
                "run",
                "--agent",
                "q-learning",
                "--scenario",
                str(scenario_path),
                "--checkpoint",
                str(checkpoint_path),
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["agent"] == "q-learning"
    assert result["scenario_id"] == "cli-fixture"


@pytest.mark.parametrize(
    ("agent_name", "checkpoint", "message"),
    [
        ("q-learning", [], "checkpoint must be a JSON object"),
        (
            "q-learning",
            {"agent": "dyna-q", "state": {}},
            "checkpoint agent does not match --agent",
        ),
        (
            "q-learning",
            {"agent": "q-learning", "state": []},
            "checkpoint.state must be a JSON object",
        ),
        (
            "planning",
            {"agent": "planning", "state": {}},
            "agent 'planning' cannot load checkpoints",
        ),
    ],
)
def test_checkpoint_validation_errors_exit_cleanly(
    agent_name: str,
    checkpoint: object,
    message: str,
    scenario_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / f"{agent_name}.json"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "run",
                "--agent",
                agent_name,
                "--scenario",
                str(scenario_path),
                "--checkpoint",
                str(checkpoint_path),
            ]
        )
    assert raised.value.code == 2
    assert message in capsys.readouterr().err


def test_benchmark_quick_selection_artifacts_and_learning_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResult:
        episodes = (object(), object())

        def write_artifacts(self, output: Path) -> dict[str, Path]:
            captured["output"] = output
            return {
                "summary": output / "summary.json",
                "episodes": output / "episodes.jsonl",
            }

    class FakeSuite:
        @classmethod
        def from_file(cls, path: Path) -> FakeSuite:
            captured["config"] = path
            return cls()

        def conditions(self) -> tuple[str, ...]:
            return ("first", "second", "third")

        def run(
            self,
            factories: object,
            *,
            conditions: object,
            root_seed: int | None,
            measure_timing: bool,
        ) -> FakeResult:
            captured["factory_names"] = tuple(factories)  # type: ignore[arg-type]
            captured["conditions"] = conditions
            captured["root_seed"] = root_seed
            captured["measure_timing"] = measure_timing
            return FakeResult()

    monkeypatch.setattr(cli, "BenchmarkSuite", FakeSuite)
    output = tmp_path / "benchmark"
    assert (
        cli.main(
            [
                "benchmark",
                "--config",
                str(tmp_path / "config.json"),
                "--agents",
                "q-learning,planning",
                "--quick",
                "1",
                "--seed",
                "81",
                "--output",
                str(output),
                "--measure-timing",
            ]
        )
        == 0
    )
    streams = capsys.readouterr()
    assert "fresh, untrained learning agents selected" in streams.err
    assert "non-confirmatory smoke test (q-learning)" in streams.err
    assert "completed 2 paired episodes across 1 conditions" in streams.out
    assert "episodes=" in streams.out
    assert "summary=" in streams.out
    assert captured == {
        "config": tmp_path / "config.json",
        "factory_names": ("q-learning", "planning"),
        "conditions": ("first",),
        "root_seed": 81,
        "measure_timing": True,
        "output": output,
    }


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (",  ,", "--agents must name at least one agent"),
        ("planning,missing", "unknown agents: ['missing']"),
        ("planning,planning", "--agents must not contain duplicates"),
    ],
)
def test_benchmark_agent_validation(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        cli._parse_agent_names(value)
    assert cli._parse_agent_names(" planning, replanning ") == (
        "planning",
        "replanning",
    )


def test_export_demo_uses_default_and_override_episode_counts(
    scenario_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, int, object]] = []

    def fake_write_demo_data(
        output: Path,
        scenario: object,
        *,
        root_seed: int,
        training_episodes: object,
    ) -> dict[str, object]:
        calls.append((output, root_seed, training_episodes))
        return {"agents": [{"id": "planning"}, {"id": "hybrid"}]}

    monkeypatch.setattr(cli, "write_demo_data", fake_write_demo_data)
    first_output = tmp_path / "default.json"
    assert (
        cli.main(
            [
                "export-demo",
                "--scenario",
                str(scenario_path),
                "--output",
                str(first_output),
                "--seed",
                "12",
            ]
        )
        == 0
    )
    assert "paired agents=2 | scenario=cli-fixture" in capsys.readouterr().out
    assert calls[-1] == (first_output, 12, cli.DEFAULT_TRAINING_EPISODES)

    second_output = tmp_path / "fast.json"
    assert (
        cli.main(
            [
                "export-demo",
                "--scenario",
                str(scenario_path),
                "--output",
                str(second_output),
                "--training-episodes",
                "0",
            ]
        )
        == 0
    )
    assert "paired agents=2" in capsys.readouterr().out
    assert calls[-1] == (
        second_output,
        42,
        {name: 0 for name in cli.TRAINABLE_AGENT_NAMES},
    )


def test_main_reports_os_and_missing_handler_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as missing_file:
        cli.main(["scenario", "show", str(tmp_path / "missing.json")])
    assert missing_file.value.code == 2
    assert "aal: error:" in capsys.readouterr().err

    parser = Mock()
    parser.parse_args.return_value = SimpleNamespace(handler=None)
    parser.exit.side_effect = SystemExit(2)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    with pytest.raises(SystemExit) as missing_handler:
        cli.main([])
    assert missing_handler.value.code == 2
    parser.exit.assert_called_once_with(2, "aal: error: selected command has no handler\n")


def test_agent_factory_and_integer_parsers() -> None:
    assert tuple(cli.make_agent(name).name for name in cli.AGENT_NAMES) == cli.AGENT_NAMES
    with pytest.raises(ValueError, match="unknown agent"):
        cli.make_agent("missing")

    assert cli._positive_int("5") == 5
    assert cli._nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        cli._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
        cli._nonnegative_int("-1")
    with pytest.raises(ValueError):
        cli._positive_int("not-an-integer")
