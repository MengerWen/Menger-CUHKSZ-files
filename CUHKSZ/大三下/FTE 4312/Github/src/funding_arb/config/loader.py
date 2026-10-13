"""Helpers for loading command settings from repository config files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from funding_arb.config.models import (
    BacktestSettings,
    BaselineSettings,
    DataQualityReportSettings,
    DataSettings,
    DemoWorkflowSettings,
    ExploratoryDLDatasetSettings,
    ExploratoryDLReportSettings,
    ExploratoryDLSignalSettings,
    DeepLearningComparisonSettings,
    DeepLearningSettings,
    FeatureSettings,
    FinalReportSettings,
    IntegrationSettings,
    LabelPipelineSettings,
    RobustnessReportSettings,
    SignalSettings,
)
from funding_arb.demo_showcase import DemoShowcaseSettings
from funding_arb.utils.config import load_config
from funding_arb.utils.paths import repo_path

SettingsModel = TypeVar("SettingsModel", bound=BaseModel)


@dataclass(frozen=True)
class CommandSettings:
    """Command metadata for config resolution."""

    command_name: str
    default_config_path: Path
    config_model: type[BaseModel]


COMMAND_SETTINGS: dict[str, CommandSettings] = {
    "fetch-data": CommandSettings(
        command_name="fetch-data",
        default_config_path=repo_path("configs", "data", "default.yaml"),
        config_model=DataSettings,
    ),
    "report-data-quality": CommandSettings(
        command_name="report-data-quality",
        default_config_path=repo_path("configs", "reports", "data_quality.yaml"),
        config_model=DataQualityReportSettings,
    ),
    "build-features": CommandSettings(
        command_name="build-features",
        default_config_path=repo_path("configs", "features", "default.yaml"),
        config_model=FeatureSettings,
    ),
    "build-labels": CommandSettings(
        command_name="build-labels",
        default_config_path=repo_path("configs", "labels", "default.yaml"),
        config_model=LabelPipelineSettings,
    ),
    "build-exploratory-dl-dataset": CommandSettings(
        command_name="build-exploratory-dl-dataset",
        default_config_path=repo_path("configs", "models", "exploratory_dl", "dataset.yaml"),
        config_model=ExploratoryDLDatasetSettings,
    ),
    "train-baseline": CommandSettings(
        command_name="train-baseline",
        default_config_path=repo_path("configs", "models", "baseline.yaml"),
        config_model=BaselineSettings,
    ),
    "evaluate-baseline": CommandSettings(
        command_name="evaluate-baseline",
        default_config_path=repo_path("configs", "models", "baseline.yaml"),
        config_model=BaselineSettings,
    ),
    "train-dl": CommandSettings(
        command_name="train-dl",
        default_config_path=repo_path("configs", "models", "lstm.yaml"),
        config_model=DeepLearningSettings,
    ),
    "compare-dl": CommandSettings(
        command_name="compare-dl",
        default_config_path=repo_path(
            "configs", "experiments", "dl", "regression_all.yaml"
        ),
        config_model=DeepLearningComparisonSettings,
    ),
    "generate-signals": CommandSettings(
        command_name="generate-signals",
        default_config_path=repo_path("configs", "signals", "default.yaml"),
        config_model=SignalSettings,
    ),
    "generate-exploratory-dl-signals": CommandSettings(
        command_name="generate-exploratory-dl-signals",
        default_config_path=repo_path("configs", "signals", "exploratory_dl", "default.yaml"),
        config_model=ExploratoryDLSignalSettings,
    ),
    "backtest": CommandSettings(
        command_name="backtest",
        default_config_path=repo_path("configs", "backtests", "default.yaml"),
        config_model=BacktestSettings,
    ),
    "robustness-report": CommandSettings(
        command_name="robustness-report",
        default_config_path=repo_path("configs", "reports", "robustness.yaml"),
        config_model=RobustnessReportSettings,
    ),
    "generate-final-report": CommandSettings(
        command_name="generate-final-report",
        default_config_path=repo_path("configs", "reports", "final_report.yaml"),
        config_model=FinalReportSettings,
    ),
    "generate-exploratory-dl-report": CommandSettings(
        command_name="generate-exploratory-dl-report",
        default_config_path=repo_path("configs", "reports", "exploratory_dl", "showcase.yaml"),
        config_model=ExploratoryDLReportSettings,
    ),
    "sync-vault": CommandSettings(
        command_name="sync-vault",
        default_config_path=repo_path("configs", "integration", "default.yaml"),
        config_model=IntegrationSettings,
    ),
    "run-demo": CommandSettings(
        command_name="run-demo",
        default_config_path=repo_path("configs", "demo", "workflow.yaml"),
        config_model=DemoWorkflowSettings,
    ),
    "run-exploratory-dl-demo": CommandSettings(
        command_name="run-exploratory-dl-demo",
        default_config_path=repo_path("configs", "demo", "exploratory_workflow.yaml"),
        config_model=DemoWorkflowSettings,
    ),
    "build-demo-showcase": CommandSettings(
        command_name="build-demo-showcase",
        default_config_path=repo_path("configs", "demo", "showcase.yaml"),
        config_model=DemoShowcaseSettings,
    ),
}


def get_command_settings(command_name: str) -> CommandSettings:
    """Return metadata for a supported CLI command."""
    try:
        return COMMAND_SETTINGS[command_name]
    except KeyError as exc:
        available = ", ".join(sorted(COMMAND_SETTINGS))
        raise ValueError(
            f"Unknown command '{command_name}'. Available commands: {available}"
        ) from exc


def load_settings(path: str | Path, model_type: type[SettingsModel]) -> SettingsModel:
    """Load and validate a config file into a typed settings model."""
    raw = load_config(path)
    return model_type.model_validate(raw)


def load_command_settings(
    command_name: str, config_path: str | Path | None = None
) -> BaseModel:
    """Load the typed config model for a named CLI command."""
    settings = get_command_settings(command_name)
    resolved_path = (
        Path(config_path) if config_path is not None else settings.default_config_path
    )
    return load_settings(resolved_path, settings.config_model)
