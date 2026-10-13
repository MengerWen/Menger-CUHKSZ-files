# Models and Research Module

## Purpose

Compare simple statistical baselines against sequence models that attempt to predict whether funding-rate dislocations remain exploitable after costs.

## Current Research Assets

- a working canonical hourly market dataset pipeline
- a configurable feature-engineering pipeline under `src/funding_arb/features/`
- a cost-aware label-generation and supervised-dataset pipeline under `src/funding_arb/labels/`
- a baseline training/evaluation pipeline under `src/funding_arb/models/baselines.py`
- a compact LSTM/GRU/TCN/Transformer sequence-model zoo under `src/funding_arb/models/deep_learning.py`
- a deep-learning comparison workflow under `src/funding_arb/models/deep_learning_experiments.py`
- reproducible feature, label, baseline, single-model DL, and comparison configs under `configs/features/default.yaml`, `configs/labels/default.yaml`, `configs/models/`, and `configs/experiments/dl/`
- documented feature definitions in `docs/features.md`
- documented target definitions in `docs/labels.md`
- documented baseline definitions in `docs/baselines.md`
- documented deep-learning design in `docs/models.md`

## Planned Model Layers

- threshold and z-score baselines
- rolling mean-reversion signals
- simple regression or classification baselines
- LSTM, GRU, TCN, and TransformerEncoder sequence models
- optional Transformer-style sequence experiments if time permits

## Research Rules

- benchmark deep models against strong simple baselines
- use time-based train/validation/test splits
- document feature leakage controls
- report post-cost performance, not just raw prediction accuracy
- fail fast when validation/test degenerates into no tradable threshold-selection path unless fallback is explicitly enabled for diagnostics

## Current Files

- `src/funding_arb/features/pipeline.py`
- `src/funding_arb/features/builders.py`
- `src/funding_arb/features/transforms.py`
- `src/funding_arb/labels/generator.py`
- `src/funding_arb/labels/pipeline.py`
- `src/funding_arb/models/baselines.py`
- `src/funding_arb/models/deep_learning.py`

## Caveats

Baseline training and the sequence-model zoo are now implemented, but post-cost positive labels are intentionally sparse under the current assumptions, so benchmark interpretation should emphasize signal-return diagnostics and the new `status` / `reason` fields in manifests and reports, not just raw numeric metrics. The comparison workflow is intentionally compact and report-oriented rather than a full experiment-tracking platform.
