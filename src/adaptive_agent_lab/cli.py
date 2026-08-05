"""Command-line interface for reproducible Adaptive Agent Lab workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from adaptive_agent_lab.agents.base import Agent
from adaptive_agent_lab.agents.dqn import DQNAgent
from adaptive_agent_lab.agents.hybrid import HybridAgent
from adaptive_agent_lab.agents.planning import OpenLoopPlanningAgent, ReplanningAgent
from adaptive_agent_lab.agents.tabular import DynaQAgent, QLearningAgent
from adaptive_agent_lab.benchmarking.runner import run_episode, train_episodes
from adaptive_agent_lab.benchmarking.suite import BenchmarkSuite
from adaptive_agent_lab.environment.generator import generate_scenario
from adaptive_agent_lab.environment.scenario import Scenario
from adaptive_agent_lab.reporting.artifacts import fingerprint, read_json, write_json_atomic
from adaptive_agent_lab.reporting.demo import (
    DEFAULT_TRAINING_EPISODES,
    write_demo_data,
    write_demo_gallery,
)
from adaptive_agent_lab.version import __version__

AgentFactory = Callable[[], Agent]
AGENT_NAMES = (
    "planning",
    "replanning",
    "q-learning",
    "dyna-q",
    "dqn",
    "hybrid",
)
TRAINABLE_AGENT_NAMES = ("q-learning", "dyna-q", "dqn", "hybrid")


def _agent_factories() -> Mapping[str, AgentFactory]:
    return {
        "planning": OpenLoopPlanningAgent,
        "replanning": ReplanningAgent,
        "q-learning": QLearningAgent,
        "dyna-q": DynaQAgent,
        "dqn": DQNAgent,
        "hybrid": HybridAgent,
    }


def make_agent(name: str) -> Agent:
    """Construct one agent by its stable benchmark identifier."""

    try:
        factory = _agent_factories()[name]
    except KeyError as error:
        raise ValueError(f"unknown agent: {name!r}") from error
    return factory()


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario JSON document."""

    resolved = Path(path)
    return Scenario.from_json(resolved.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aal",
        description=(
            "Reproducible planning, reinforcement-learning, and hybrid-agent experiments."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scenario_parser = subparsers.add_parser("scenario", help="generate or inspect scenarios")
    scenario_subparsers = scenario_parser.add_subparsers(
        dest="scenario_command", required=True
    )
    generate_parser = scenario_subparsers.add_parser(
        "generate", help="generate a seeded scenario JSON file"
    )
    generate_parser.add_argument("output", type=Path)
    generate_parser.add_argument("--seed", type=_nonnegative_int, default=42)
    generate_parser.add_argument("--size", choices=("tiny", "small", "medium"), default="small")
    generate_parser.add_argument(
        "--dynamics", choices=("static", "low", "medium", "high"), default="medium"
    )
    generate_parser.add_argument("--orders", type=_nonnegative_int, default=4)
    generate_parser.add_argument("--horizon", type=_positive_int)
    generate_parser.add_argument("--id", dest="scenario_id")
    generate_parser.set_defaults(handler=_handle_scenario_generate)

    show_parser = scenario_subparsers.add_parser(
        "show", help="validate and summarize a scenario"
    )
    show_parser.add_argument("path", type=Path)
    show_parser.add_argument("--json", action="store_true", dest="as_json")
    show_parser.set_defaults(handler=_handle_scenario_show)

    run_parser = subparsers.add_parser("run", help="evaluate one agent on one event tape")
    run_parser.add_argument("--agent", choices=AGENT_NAMES, required=True)
    run_parser.add_argument("--scenario", type=Path, required=True)
    run_parser.add_argument("--seed", type=_nonnegative_int, default=42)
    run_parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint produced by `aal train` (learning agents only)",
    )
    run_parser.add_argument("--trace-output", type=Path)
    run_parser.add_argument("--json", action="store_true", dest="as_json")
    run_parser.set_defaults(handler=_handle_run)

    train_parser = subparsers.add_parser(
        "train", help="train one learning agent and write a JSON checkpoint"
    )
    train_parser.add_argument("--agent", choices=TRAINABLE_AGENT_NAMES, required=True)
    train_parser.add_argument("--scenario", type=Path, action="append", required=True)
    train_parser.add_argument("--episodes", type=_positive_int, default=100)
    train_parser.add_argument("--seed", type=_nonnegative_int, default=42)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.set_defaults(handler=_handle_train)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="run a paired, seeded benchmark suite"
    )
    benchmark_parser.add_argument("--config", type=Path, required=True)
    benchmark_parser.add_argument(
        "--agents",
        default="planning,replanning",
        help="comma-separated agent IDs (default: planning,replanning)",
    )
    benchmark_parser.add_argument(
        "--quick",
        type=_positive_int,
        metavar="CONDITIONS",
        help="run only the first N configured conditions",
    )
    benchmark_parser.add_argument("--seed", type=_nonnegative_int)
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--measure-timing", action="store_true")
    benchmark_parser.set_defaults(handler=_handle_benchmark)

    demo_parser = subparsers.add_parser(
        "export-demo", help="train demo agents and export real replay data"
    )
    demo_parser.add_argument("--scenario", type=Path, required=True)
    demo_parser.add_argument("--output", type=Path, required=True)
    demo_parser.add_argument("--seed", type=_nonnegative_int, default=42)
    demo_parser.add_argument(
        "--training-episodes",
        type=_nonnegative_int,
        help="override the episode count for each learning agent",
    )
    demo_parser.set_defaults(handler=_handle_export_demo)

    gallery_parser = subparsers.add_parser(
        "export-gallery", help="train fresh demo agents for each configured gallery case"
    )
    gallery_parser.add_argument("--config", type=Path, required=True)
    gallery_parser.add_argument("--output", type=Path, required=True)
    gallery_parser.add_argument("--seed", type=_nonnegative_int, default=42)
    gallery_parser.set_defaults(handler=_handle_export_gallery)
    return parser


def _handle_scenario_generate(args: argparse.Namespace) -> int:
    scenario = generate_scenario(
        args.seed,
        size=args.size,
        dynamics=args.dynamics,
        order_count=args.orders,
        horizon=args.horizon,
        scenario_id=args.scenario_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(scenario.to_json(indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output} | id={scenario.scenario_id} | "
        f"fingerprint={fingerprint(scenario.to_dict())}"
    )
    return 0


def _scenario_summary(scenario: Scenario) -> dict[str, object]:
    return {
        "scenario_id": scenario.scenario_id,
        "fingerprint": fingerprint(scenario.to_dict()),
        "map": {
            "width": scenario.map.width,
            "height": scenario.map.height,
            "obstacles": len(scenario.map.obstacles),
            "charging_stations": len(scenario.map.charging_stations),
        },
        "orders": len(scenario.orders),
        "events": len(scenario.event_tape),
        "horizon": scenario.horizon,
        "battery_capacity": scenario.battery_capacity,
    }


def _handle_scenario_show(args: argparse.Namespace) -> int:
    summary = _scenario_summary(load_scenario(args.path))
    if args.as_json:
        _print_json(summary)
    else:
        map_summary = summary["map"]
        assert isinstance(map_summary, dict)
        print(f"scenario: {summary['scenario_id']}")
        print(f"fingerprint: {summary['fingerprint']}")
        print(
            "map: "
            f"{map_summary['width']}x{map_summary['height']} | "
            f"obstacles={map_summary['obstacles']} | "
            f"chargers={map_summary['charging_stations']}"
        )
        print(
            f"orders={summary['orders']} | events={summary['events']} | "
            f"horizon={summary['horizon']} | battery={summary['battery_capacity']}"
        )
    return 0


def _handle_run(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    agent = make_agent(args.agent)
    if args.checkpoint is not None:
        checkpoint = read_json(args.checkpoint)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("checkpoint must be a JSON object")
        if checkpoint.get("agent") != agent.name:
            raise ValueError("checkpoint agent does not match --agent")
        raw_state = checkpoint.get("state")
        if not isinstance(raw_state, Mapping):
            raise TypeError("checkpoint.state must be a JSON object")
        load_state_dict = getattr(agent, "load_state_dict", None)
        if not callable(load_state_dict):
            raise TypeError(f"agent {agent.name!r} cannot load checkpoints")
        load_state_dict(raw_state)
    result = run_episode(
        agent,
        scenario,
        seed=args.seed,
        explore=False,
        learn=False,
    )
    if args.trace_output is not None:
        write_json_atomic(args.trace_output, result.to_dict())
    if args.as_json:
        _print_json(result.to_dict())
    else:
        metrics = result.metrics
        print(f"agent={result.agent} | scenario={result.scenario_id} | seed={result.seed}")
        print(
            f"completed={metrics.completed_orders}/{metrics.total_orders} | "
            f"WOTCR={metrics.weighted_on_time_completion_rate:.3f} | "
            f"reward={metrics.total_reward:.2f} | "
            f"violations={metrics.constraint_violations} | steps={metrics.steps}"
        )
        if args.trace_output is not None:
            print(f"trace={args.trace_output}")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    scenarios = tuple(load_scenario(path) for path in args.scenario)
    agent = make_agent(args.agent)
    results = train_episodes(
        agent,
        scenarios,
        episodes=args.episodes,
        root_seed=args.seed,
    )
    state_dict = getattr(agent, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"agent {agent.name!r} does not expose state_dict()")
    checkpoint = {
        "schema_version": "0.1",
        "agent": agent.name,
        "root_seed": args.seed,
        "episodes": args.episodes,
        "scenario_fingerprints": [fingerprint(scenario.to_dict()) for scenario in scenarios],
        "last_episode_metrics": asdict(results[-1].metrics),
        "state": state_dict(),
    }
    write_json_atomic(args.output, checkpoint)
    final = results[-1].metrics
    print(
        f"trained {agent.name} for {args.episodes} episodes | "
        f"last WOTCR={final.weighted_on_time_completion_rate:.3f} | "
        f"checkpoint={args.output}"
    )
    return 0


def _parse_agent_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise ValueError("--agents must name at least one agent")
    unknown = sorted(set(names) - set(AGENT_NAMES))
    if unknown:
        raise ValueError(f"unknown agents: {unknown!r}")
    if len(set(names)) != len(names):
        raise ValueError("--agents must not contain duplicates")
    return names


def _handle_benchmark(args: argparse.Namespace) -> int:
    suite = BenchmarkSuite.from_file(args.config)
    names = _parse_agent_names(args.agents)
    untrained = tuple(name for name in names if name in TRAINABLE_AGENT_NAMES)
    if untrained:
        print(
            "warning: fresh, untrained learning agents selected; this run is a "
            f"non-confirmatory smoke test ({','.join(untrained)})",
            file=sys.stderr,
        )
    available = _agent_factories()
    factories = {name: available[name] for name in names}
    conditions = suite.conditions()
    if args.quick is not None:
        conditions = conditions[: args.quick]
    result = suite.run(
        factories,
        conditions=conditions,
        root_seed=args.seed,
        measure_timing=args.measure_timing,
    )
    paths = result.write_artifacts(args.output)
    print(
        f"completed {len(result.episodes)} paired episodes across "
        f"{len(conditions)} conditions | output={args.output}"
    )
    for name, path in sorted(paths.items()):
        print(f"{name}={path}")
    return 0


def _handle_export_demo(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario)
    episodes = (
        DEFAULT_TRAINING_EPISODES
        if args.training_episodes is None
        else {
            name: args.training_episodes
            for name in TRAINABLE_AGENT_NAMES
        }
    )
    payload = write_demo_data(
        args.output,
        scenario,
        root_seed=args.seed,
        training_episodes=episodes,
    )
    agents = payload["agents"]
    assert isinstance(agents, list)
    print(
        f"wrote {args.output} | paired agents={len(agents)} | "
        f"scenario={scenario.scenario_id}"
    )
    return 0


def _handle_export_gallery(args: argparse.Namespace) -> int:
    payload = write_demo_gallery(
        args.output,
        args.config,
        root_seed=args.seed,
    )
    cases = payload["cases"]
    default_case_id = payload["defaultCaseId"]
    root_seed = payload["rootSeed"]
    assert isinstance(cases, list)
    assert isinstance(default_case_id, str)
    assert isinstance(root_seed, int)
    print(
        f"wrote {args.output} | gallery cases={len(cases)} | "
        f"default={default_case_id} | seed={root_seed}"
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute one command, and report actionable errors."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler = args.handler
        if not callable(handler):
            raise TypeError("selected command has no handler")
        return int(handler(args))
    except (OSError, TypeError, ValueError) as error:
        parser.exit(2, f"aal: error: {error}\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
