# Contributing

Adaptive Agent Lab treats experimental comparability as part of the public API.
Changes to transitions, reward, observations, scenario generation, or metrics must
therefore include tests and a note in the experiment protocol.

## Local checks

```bash
python -m pip install -e '.[dev]'
make check
```

## Pull requests

- Keep the environment independent from individual agents.
- Never give one agent information unavailable to the others without documenting
  the representation difference as an explicit experimental condition.
- Add a regression test for every corrected transition or serialization bug.
- Do not commit generated model checkpoints or full run directories.
- Keep committed benchmark fixtures small enough for CI.

## Reproducibility

Every stochastic component accepts an explicit seed. Evaluation additions should
reuse shared scenario seeds and event tapes so paired comparisons remain valid.
