# Experiment protocol

## Material Passport

- Origin skill: `experiment-agent`
- Origin mode: `plan`
- Origin date: 2026-08-05
- Verification status: **PRE-SPECIFIED; NOT YET EXECUTABLE AS A CONFIRMATORY STUDY**
- Version label: `experiment-protocol-v0.1`

This protocol defines the intended release experiment. It is not a report of
results. Current CLI smoke runs and browser demonstrations do not satisfy the
training, checkpoint, split-isolation, or aggregation requirements below.

## Research questions and hypotheses

- **RQ1:** How does increasing cell-blockage dynamics change weighted on-time
  completion and online computation for each treatment?
- **RQ2:** At a fixed real-environment interaction budget, does Dyna-Q learn
  more sample-efficiently than exact tabular Q-learning?
- **RQ3:** Does learned high-level option selection plus A* improve task
  performance over replanning without increasing validity failures?

Pre-specified directional hypotheses are:

- **H1:** Open-loop planning degrades more sharply than replanning as blockage
  frequency increases.
- **H2:** Dyna-Q has a higher validation learning-curve area than Q-learning at
  equal real-transition budgets; simulated updates do not count as real data.
- **H3:** The hybrid agent improves WOTCR over replanning while keeping its
  constraint-violation rate no worse.

Failure to support a hypothesis is a valid result. Episode return alone cannot
declare an overall winner.

## Treatments

The six treatment IDs are `planning`, `replanning`, `q-learning`, `dyna-q`,
`dqn`, and `hybrid`. Their implemented behavior is specified in
[`problem-formulation.md`](problem-formulation.md).

Key representation and optimization facts are frozen as follows:

- Q-learning and Dyna-Q use the exact tuple of time, robot cell, battery,
  carried-order index, all order status codes, and the complete dynamic-blockage
  bitmask.
- DQN uses the flat four-grid-channel plus globals plus padded per-order vector.
- The NumPy DQN uses a ReLU MLP, replay, selected-action MSE, plain SGD with
  gradient clipping, and a periodically hard-copied target network. It is not
  Double DQN and does not use Adam or Huber loss.
- The hybrid uses the same vector and MLP/SGD implementation, but learns
  semi-MDP values for per-order, charge, and wait options and has no target
  network.

The shared primitive action order is `up`, `down`, `left`, `right`, `pickup`,
`dropoff`, `charge`, `wait`.

## Planned factorial design

The release benchmark varies:

| Factor | Levels |
| --- | --- |
| map scale | `tiny`, `small`, `medium` |
| dynamics | `static`, `low`, `medium`, `high` |
| order load | `light` (1), `normal` (2), `heavy` (3) orders |

`configs/benchmarks/main.json` declares 30 paired scenario indices per cell.
The planned `small` block contains 12 cells and is confirmatory. The `tiny` and
`medium` transfer block contains 24 cells and is exploratory. This expands to
360 planned confirmatory and 720 planned exploratory scenario conditions.

The config selects horizons of 150, 300, and 500 ticks for tiny, small, and
medium. `BenchmarkSuite` passes the scale, dynamics label, order count, horizon,
and derived seed to `generate_scenario`. Dynamics generation itself currently
uses the versioned constants in `environment/generator.py`; the matching values
are repeated in the JSON as auditable metadata.

Thirty scenarios per cell is an initial Monte Carlo budget, not a power
guarantee. It must not be extended selectively because an interval is near a
preferred threshold. A later protocol version may change the budget based on
the interval widths from a complete earlier run.

## Split and leakage policy

Train, validation, and test use distinct root seeds:

- training fits Q tables, model entries, replay-dependent networks, and option
  values;
- validation selects hyperparameters and checkpoints; and
- test is opened only once for a versioned release evaluation.

The same complete scenario—including map, orders, and event tape—is shared by
all agents in one paired condition. Scenario seed derivation deliberately
excludes agent identity. Agent initialization, exploration, replay sampling,
and Dyna-Q model sampling use agent-specific seeded streams.

Pending order definitions and release times are observable by design; future
blockage events are not. This information boundary cannot be changed after
validation begins.

Test scenarios or summaries must not influence the reward, encoder,
hyperparameters, checkpoint selection, stopping rule, or semantics. A semantic
bug discovered after opening test data voids the run and requires a version
increment and complete rerun.

## Planned training protocol

Q-learning, Dyna-Q, DQN, and hybrid each use five independent training seeds.
The flat learners receive at most 1,000,000 real environment transitions under
the same scenario distribution, reward, action mask, validation scenarios, and
validation frequency. The hybrid must receive an explicitly recorded real-step
budget as well; its option updates are counted separately from primitive
environment transitions.

Dyna-Q performs 20 simulated model updates per real step in the frozen
configuration. Simulated updates and compute time are reported separately.
DQN starts updates only after replay warm-up and hard-copies its online network
to the target network at the configured update interval. Exploration is disabled
during validation and test.

For each training seed, validation selects one checkpoint using mean validation
WOTCR. Reports include real steps on the x-axis, individual and aggregate
learning curves, validation area under the curve, selected training step,
wall-clock time, and learning-update counts.

`configs/training/*.json` are implementation-aligned single-scenario reference
configs, not complete release-training configs. They are not consumed by the
current CLI. `aal train` currently trains one selected learner on one supplied
scenario for an episode count and writes one JSON state; it does not implement
the multi-seed, validation, or checkpoint-selection procedure above.

The current DQN and hybrid agents derive `ObservationSpec.max_orders` from the
scenario supplied at reset. They can retain a network only when vector size is
unchanged; the hybrid additionally requires the same option count. A release
trainer spanning different order loads therefore needs to persist and reuse a
fixed maximum-order spec (or train explicitly separated compatible models)
before this protocol can be executed.

## Required confirmatory evaluation procedure

For each frozen test condition:

1. Resolve and fingerprint the benchmark config and complete generated
   scenario.
2. Load the validation-selected state for each independent learned-policy seed.
3. Disable exploration and parameter updates.
4. Run every treatment against the identical scenario, rotating execution order
   by scenario index.
5. Store one record per agent, learned-policy seed, and scenario.
6. Reject the entire paired block if scenario fingerprints differ or any
   required checkpoint/provenance record is missing.

Planning treatments run once per paired scenario. For each learning treatment,
first average the five independently trained policy results within scenario.
The scenario—not repeated policy evaluations—is the paired inference unit.
Between-training-seed variability is also reported separately.

Decision and episode timeouts, once implemented, are operational failure
outcomes rather than missing data. They must be recorded without automatic
retry.

## Outcomes

The primary planned outcome is weighted on-time completion rate (WOTCR),
defined in [`problem-formulation.md`](problem-formulation.md). Report its mean,
median, standard deviation, and 95% interval for every agent and condition.

Secondary implemented episode metrics are:

- total reward and weighted completion rate;
- completed and total orders;
- mean lateness among delivered orders;
- valid movement steps and valid movement steps per completed order;
- constraint violations;
- aggregate online decision time; and
- episode steps.

Agent diagnostics add planning calls, A* node expansions, and learning updates.
Metrics unavailable to an algorithm should be absent or `null`, never
misrepresented as zero. Reward return remains a learning diagnostic, not the
primary cross-family claim.

## Planned statistical analysis

The four planned comparisons are:

1. replanning minus planning;
2. Dyna-Q minus Q-learning;
3. DQN minus Q-learning; and
4. hybrid minus replanning.

For each confirmatory condition, compute the paired scenario-level WOTCR
difference after within-scenario aggregation across learned-policy seeds. Report
the absolute percentage-point effect and a paired bootstrap 95% confidence
interval with 10,000 resamples. If p-values are reported, adjust the four
planned comparisons with Holm's method at familywise alpha 0.05. A two-point
absolute WOTCR difference is an interpretation reference, not a pass/fail rule.

RQ1 uses per-dynamics effects and their trend. RQ2 uses validation learning-curve
area at equal real-transition budgets. RQ3 requires both the WOTCR comparison
and the violation-rate guardrail. Tiny and medium transfer results remain
exploratory and are not pooled into confirmatory claims.

The current benchmark summary implementation computes bootstrap summaries and
paired differences over the selected condition records. It does not load
checkpoints, aggregate five policy seeds within scenario, stratify the release
analysis by every confirmatory cell, or apply Holm correction. Those features
are release blockers rather than assumptions.

## Current smoke and demo policy

The current `aal benchmark` uses fresh factories and does not load the
`training_config` references in `main.json`.

- Its default `planning,replanning` execution is a paired deterministic smoke
  benchmark.
- Explicitly selecting any learner evaluates an untrained agent and must carry
  an `UNTRAINED SMOKE` label.
- `--quick` truncates the condition set for development and cannot produce a
  release result.
- Browser demo data uses small convenience training budgets on the displayed
  scenario. Those trajectories are real simulator outputs but are
  non-confirmatory, are not held out, and cannot support rankings.

Neither smoke nor demo outputs may be described as hypothesis tests, validation
results, or confirmatory evidence.

## Release integrity gates

A result becomes eligible for confirmatory language only when all of the
following are true:

- every JSON config parses and its canonical fingerprint is stored;
- config-driven multi-seed training and validation checkpoint selection exist;
- neural checkpoints persist a fixed observation and option specification that
  is compatible with every scenario in their declared evaluation block;
- the benchmark loads and fingerprints frozen checkpoints;
- train, validation, and test seed namespaces are enforced by code;
- all paired agents record an identical complete-scenario fingerprint;
- independent learned seeds are aggregated within scenario before inference;
- deterministic reruns reproduce planning trajectories exactly;
- learned evaluation is reproducible after state and seed load;
- non-finite metrics, incomplete pairs, and provenance gaps fail closed;
- the planned per-cell and multiple-comparison analysis is implemented; and
- the full automated test, lint, and type-check suite passes.

## Threats and interpretation limits

The hand-designed reward and generator may favor some treatments. Exact
tabulation still scales exponentially. A* and hard action masks provide domain
knowledge unavailable to unrestricted flat exploration. Pending order metadata
is known in advance. Medium-map evaluation is out of the small training
distribution. Current latency is aggregate episode decision time, not a full
hardware-normalized profile.

Results therefore describe this simulator and protocol only. They are not
evidence about real warehouse safety or a universal ranking of planning and
learning algorithms.

## Planned release artifacts

The current benchmark writer emits `manifest.json`, `episodes.jsonl`, and
`summary.json`. A confirmatory release additionally requires resolved training
configs, checkpoint fingerprints, per-seed learning curves, compressed replay
trajectories, anomaly/exclusion records, environment and dependency provenance,
and a report linking every table and figure to exact run and config IDs.
