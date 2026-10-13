# Config Conventions

This repository keeps configuration centralized under `configs/` using YAML files.

- `configs/data/`: source selection, symbols, date range, sampling frequency
- `configs/features/`: feature windows, transformations, and feature-table output assumptions
- `configs/labels/`: label horizons, execution alignment, cost assumptions, and time-series splits
- `configs/models/`: baseline model configs and deep-learning experiment parameters
- `configs/experiments/`: named multi-run experiment bundles, including deep-learning model comparisons
- `configs/backtests/`: fees, slippage, funding mode, hedge mode, leverage guard, primary split, and portfolio rules
- `configs/demo/`: frontend/demo artifact settings plus the end-to-end demo workflow config
- `configs/integration/`: mock operator/oracle-style bridge settings for vault sync demos
- `configs/reports/`: exploratory reporting, figure generation, markdown summaries, and robustness experiments

Use the existing Python environment at `d:\MG\anaconda3\python.exe`, which already includes `PyYAML`, when running scripts that load these configs. Assumptions that materially affect outputs should live in config before they are hardcoded into scripts or notebooks.
