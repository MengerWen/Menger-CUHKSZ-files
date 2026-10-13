from __future__ import annotations

from funding_arb.cli import build_parser


def test_cli_parser_accepts_fetch_data_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["fetch-data"])
    assert args.command == "fetch-data"
    assert args.config.endswith("configs\\data\\default.yaml") or args.config.endswith(
        "configs/data/default.yaml"
    )


def test_cli_parser_accepts_report_data_quality_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["report-data-quality"])
    assert args.command == "report-data-quality"
    assert args.config.endswith(
        "configs\\reports\\data_quality.yaml"
    ) or args.config.endswith("configs/reports/data_quality.yaml")


def test_cli_parser_accepts_build_labels_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["build-labels"])
    assert args.command == "build-labels"
    assert args.config.endswith(
        "configs\\labels\\default.yaml"
    ) or args.config.endswith("configs/labels/default.yaml")


def test_cli_parser_accepts_build_exploratory_dataset_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["build-exploratory-dl-dataset"])
    assert args.command == "build-exploratory-dl-dataset"
    assert args.config.endswith(
        "configs\\models\\exploratory_dl\\dataset.yaml"
    ) or args.config.endswith("configs/models/exploratory_dl/dataset.yaml")


def test_cli_parser_accepts_generate_signals_with_source_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate-signals", "--source", "dl"])
    assert args.command == "generate-signals"
    assert args.source == "dl"
    assert args.config.endswith(
        "configs\\signals\\default.yaml"
    ) or args.config.endswith("configs/signals/default.yaml")


def test_cli_parser_accepts_evaluate_baseline_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["evaluate-baseline"])
    assert args.command == "evaluate-baseline"
    assert args.config.endswith(
        "configs\\models\\baseline.yaml"
    ) or args.config.endswith("configs/models/baseline.yaml")


def test_cli_parser_accepts_backtest_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["backtest"])
    assert args.command == "backtest"
    assert args.config.endswith(
        "configs\\backtests\\default.yaml"
    ) or args.config.endswith("configs/backtests/default.yaml")


def test_cli_parser_accepts_robustness_report_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["robustness-report"])
    assert args.command == "robustness-report"
    assert args.config.endswith(
        "configs\\reports\\robustness.yaml"
    ) or args.config.endswith("configs/reports/robustness.yaml")


def test_cli_parser_accepts_generate_final_report_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate-final-report"])
    assert args.command == "generate-final-report"
    assert args.config.endswith(
        "configs\\reports\\final_report.yaml"
    ) or args.config.endswith("configs/reports/final_report.yaml")


def test_cli_parser_accepts_train_dl_command_with_log_level() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-dl", "--log-level", "DEBUG"])
    assert args.command == "train-dl"
    assert args.log_level == "DEBUG"


def test_cli_parser_accepts_compare_dl_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["compare-dl"])
    assert args.command == "compare-dl"
    assert args.config.endswith(
        "configs\\experiments\\dl\\regression_all.yaml"
    ) or args.config.endswith("configs/experiments/dl/regression_all.yaml")


def test_cli_parser_accepts_generate_exploratory_signal_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate-exploratory-dl-signals"])
    assert args.command == "generate-exploratory-dl-signals"
    assert args.config.endswith(
        "configs\\signals\\exploratory_dl\\default.yaml"
    ) or args.config.endswith("configs/signals/exploratory_dl/default.yaml")


def test_cli_parser_accepts_generate_exploratory_report_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["generate-exploratory-dl-report"])
    assert args.command == "generate-exploratory-dl-report"
    assert args.config.endswith(
        "configs\\reports\\exploratory_dl\\showcase.yaml"
    ) or args.config.endswith("configs/reports/exploratory_dl/showcase.yaml")


def test_cli_parser_accepts_sync_vault_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["sync-vault"])
    assert args.command == "sync-vault"
    assert args.config.endswith(
        "configs\\integration\\default.yaml"
    ) or args.config.endswith("configs/integration/default.yaml")


def test_cli_parser_accepts_run_demo_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-demo"])
    assert args.command == "run-demo"
    assert args.config.endswith(
        "configs\\demo\\workflow.yaml"
    ) or args.config.endswith("configs/demo/workflow.yaml")


def test_cli_parser_accepts_run_exploratory_demo_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["run-exploratory-dl-demo"])
    assert args.command == "run-exploratory-dl-demo"
    assert args.config.endswith(
        "configs\\demo\\exploratory_workflow.yaml"
    ) or args.config.endswith("configs/demo/exploratory_workflow.yaml")


def test_cli_parser_accepts_build_demo_showcase_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["build-demo-showcase"])
    assert args.command == "build-demo-showcase"
    assert args.config.endswith(
        "configs\\demo\\showcase.yaml"
    ) or args.config.endswith("configs/demo/showcase.yaml")
