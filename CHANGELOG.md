# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

Nothing

## [1.0.0] - 2024-07-14

Initial stable release.

### Added

- Documentation at [https://synergy.readthedocs.io/](https://synergy.readthedocs.io/).
- Type hints and improved docstrings.

### Changed

- API calls to drug models have been made more consistent.

### Fixed

- Dose-dependence of n-dimensional MuSyC models (#10).

### Removed

- Plotting functionality that was built into individual models. Use `synergy.utils.plots` directly instead, or your own custom plotting code.
- `synergy.utils.dose_utils.remove_replicates()`