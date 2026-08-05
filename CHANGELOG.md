# Changelog

All notable changes to this project are documented here.

## [0.2.0] - 2026-08-05

### Added

- Added a versioned four-case replay gallery with rack-maze, parallel-aisle,
  cross-dock, and serpentine warehouse topologies.
- Added `aal export-gallery` for generating all browser cases from validated
  configuration, with fresh agents for every case.
- Added synchronized A/B controller replay, event seeking, playback speed and
  step controls, bounded future-path previews, and explicit terminal states.
- Added root-seed provenance while retaining a per-case scenario fingerprint
  throughout the versioned gallery data.

### Changed

- Reworked the browser demo into a restrained research-oriented experiment
  explorer with improved mobile map readability and accessible controls.
- Published the interactive artifact through GitHub Pages and the project
  README while preserving its non-confirmatory evidence label.
- Represented intentionally unmeasured decision timing as `null` instead of a
  misleading zero-valued measurement.
- Validated the complete gallery configuration before beginning agent training.

### Performance

- Changed replay-buffer sampling to copy only selected transitions rather than
  the entire stored buffer on every DQN update, substantially reducing gallery
  export time.

### Verification

- Added scenario reachability, gallery-schema, trace-replay, root-deployment,
  and GitHub Pages integrity tests.
- Gated GitHub Pages deployment on simulator trace and provenance verification.

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
