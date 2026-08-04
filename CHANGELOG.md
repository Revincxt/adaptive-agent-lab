# Changelog

All notable changes to this project are documented here.

## [0.1.0] - 2026-08-05

### Added

- Initial package, experiment contract, CI, and repository structure.
- Implementation-aligned reference configurations for Q-learning, Dyna-Q, DQN,
  and the hybrid option learner.

### Documentation

- Corrected the neural observation contract to four flattened grid channels,
  three global features, and padded per-order features.
- Corrected the tabular representation to the exact runtime-state tuple and the
  primitive action names to `up`, `down`, `left`, `right`, `pickup`, `dropoff`,
  `charge`, and `wait`.
- Documented the implemented reward coefficients and clarified that
  `stranded_cost` is currently declared but unused.
- Documented the NumPy DQN as ReLU MLP + selected-action MSE + plain SGD + hard
  target copies; removed unsupported Adam, Huber, and Double-DQN claims.
- Labeled browser data and learner benchmark runs as non-confirmatory, and
  recorded checkpoint-aware evaluation as a release blocker.
