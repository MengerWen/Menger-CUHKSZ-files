"""Baseline strategy and predictive-model pipeline for the funding-rate arbitrage project.

This baseline layer stays lightweight, but it treats the task as a low-signal
time-series trading problem instead of a generic IID prediction benchmark.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from funding_arb.config.models import (
    BaselineSettings,
    ClassificationModelVariantSettings,
    RegressionModelVariantSettings,
    RuleBaselineSpec,
    TreeBaselineSettings,
)
from funding_arb.utils.degeneracy import (
    DegenerateExperimentDiagnostics,
    DegenerateExperimentError,
    infer_horizon_label,
    infer_profitable_column,
    infer_tradeable_column,
    label_split_diagnostics,
    signal_split_diagnostics,
    summarize_threshold_search,
    warn_on_degenerate_experiment,
)
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class BaselineArtifacts:
    """Paths produced by the baseline training/evaluation pipeline."""

    output_dir: str
    manifest_path: str
    metrics_path: str
    metrics_csv_path: str | None
    leaderboard_path: str
    leaderboard_csv_path: str | None
    predictions_path: str
    predictions_csv_path: str | None
    report_path: str | None
    feature_columns_path: str
    model_paths: dict[str, str]
    diagnostic_paths: dict[str, str]


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Outcome of selecting a decision threshold on the validation split."""

    selected_threshold: float
    objective_value: float | None
    search_frame: pd.DataFrame
    fallback_used: bool = False
    fallback_reason: str | None = None


def describe_baseline_job(config: BaselineSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the baseline training job."""
    settings = (
        config
        if isinstance(config, BaselineSettings)
        else BaselineSettings.model_validate(config)
    )
    rule_count = sum(1 for rule in settings.rules if rule.enabled)
    predictive_names: list[str] = []
    if settings.predictive.classification.enabled:
        predictive_names.append(settings.predictive.classification.name)
        predictive_names.extend(
            model.name
            for model in settings.predictive.classification.additional_models
            if model.enabled
        )
    if settings.predictive.regression.enabled:
        predictive_names.append(settings.predictive.regression.name)
        predictive_names.extend(
            model.name
            for model in settings.predictive.regression.additional_models
            if model.enabled
        )
    if settings.predictive.tree.enabled:
        predictive_names.extend(
            [
                settings.predictive.tree.classifier_name,
                settings.predictive.tree.regressor_name,
            ]
        )
    tuning_mode = settings.tuning.mode if settings.tuning.enabled else "disabled"
    prediction_mode = settings.prediction.mode
    return (
        f"Baseline training ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, using {rule_count} rule models and {len(predictive_names)} predictive models. "
        f"Time-series tuning is {tuning_mode}; prediction mode is {prediction_mode}. "
        f"Artifacts will be written under {settings.output.model_dir}/{settings.input.provider}/"
        f"{settings.input.symbol.lower()}/{settings.input.frequency}/{settings.output.run_name}."
    )


def describe_baseline_evaluation_job(config: BaselineSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the baseline evaluation job."""
    settings = (
        config
        if isinstance(config, BaselineSettings)
        else BaselineSettings.model_validate(config)
    )
    return (
        f"Baseline evaluation ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, loading trained artifacts from {settings.output.model_dir}/{settings.input.provider}/"
        f"{settings.input.symbol.lower()}/{settings.input.frequency}/{settings.output.run_name}. "
        f"Prediction mode is {settings.prediction.mode}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _load_supervised_dataset(settings: BaselineSettings) -> pd.DataFrame:
    dataset_path = _resolve_path(settings.input.dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported supervised dataset format: {dataset_path.suffix}")
    timestamp_column = settings.target.timestamp_column
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    frame = frame.sort_values(timestamp_column).reset_index(drop=True)
    ready_column = settings.target.ready_column
    split_column = settings.target.split_column
    frame = frame[
        (frame[ready_column] == 1)
        & frame[split_column].isin(["train", "validation", "test"])
    ].copy()
    if frame.empty:
        raise ValueError("No supervised-ready rows were found for baseline modeling.")
    return frame


def select_feature_columns(frame: pd.DataFrame, settings: BaselineSettings) -> list[str]:
    """Select leakage-safe numeric feature columns for predictive baselines."""
    feature_settings = settings.feature_selection
    if feature_settings.include_columns:
        columns = [
            column for column in feature_settings.include_columns if column in frame.columns
        ]
    else:
        columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded_columns = set(feature_settings.exclude_columns)
    columns = [
        column
        for column in columns
        if column not in excluded_columns
        and not any(
            column.startswith(prefix) for prefix in feature_settings.exclude_prefixes
        )
    ]
    if not columns:
        raise ValueError("Feature selection removed every candidate feature column.")

    candidate_frame = frame[columns].replace([np.inf, -np.inf], np.nan)
    missing_fraction = candidate_frame.isna().mean()
    columns = [
        column
        for column in columns
        if float(missing_fraction[column])
        <= float(feature_settings.max_missing_fraction)
    ]
    if feature_settings.drop_constant_features:
        non_constant: list[str] = []
        filtered_frame = candidate_frame[columns]
        for column in columns:
            unique_count = filtered_frame[column].nunique(dropna=True)
            if int(unique_count) > 1:
                non_constant.append(column)
        columns = non_constant
    if not columns:
        raise ValueError(
            "No usable feature columns remain after missing-value and variance filtering."
        )
    return sorted(columns)


def _output_dir(settings: BaselineSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.model_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / settings.output.run_name
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _mean_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _safe_float(series.mean())


def _median_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _safe_float(series.median())


def _sum_or_none(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return _safe_float(series.sum())


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=True, sort_keys=True)


def _feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].replace([np.inf, -np.inf], np.nan)


def _matches_prefix(column: str, prefixes: list[str]) -> bool:
    return any(column.startswith(prefix) for prefix in prefixes)


def _build_modeling_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: BaselineSettings,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Build a modeling frame with leakage-safe forward-fill and optional indicators."""

    feature_frame = frame[feature_columns].replace([np.inf, -np.inf], np.nan).copy()
    original_missing = feature_frame.isna()
    imputation_settings = settings.imputation
    designated_ffill_columns = {
        column
        for column in feature_columns
        if column in set(imputation_settings.forward_fill_columns)
        or _matches_prefix(column, imputation_settings.forward_fill_prefixes)
    }
    for column in sorted(designated_ffill_columns):
        feature_frame[column] = feature_frame[column].ffill()

    indicator_columns: list[str] = []
    if imputation_settings.add_missing_indicators:
        if imputation_settings.indicator_columns:
            candidate_indicator_columns = [
                column
                for column in imputation_settings.indicator_columns
                if column in feature_frame.columns
            ]
        else:
            candidate_indicator_columns = feature_columns
        for column in candidate_indicator_columns:
            if bool(original_missing[column].any()):
                indicator_name = f"{column}__missing"
                feature_frame[indicator_name] = original_missing[column].astype(int)
                indicator_columns.append(indicator_name)

    non_feature_columns = [column for column in frame.columns if column not in feature_columns]
    modeling_frame = pd.concat(
        [frame[non_feature_columns].copy(), feature_frame.copy()],
        axis=1,
    )
    final_feature_columns = feature_columns + indicator_columns
    metadata = {
        "forward_fill_columns": sorted(designated_ffill_columns),
        "missing_indicator_columns": indicator_columns,
        "remaining_imputation_strategy": imputation_settings.remaining_strategy,
    }
    return modeling_frame, final_feature_columns, metadata


def make_time_series_folds(
    n_samples: int,
    *,
    n_splits: int,
    gap: int,
    mode: str,
    min_train_size: int,
    rolling_window_size: int | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding or rolling chronological folds for tuning/calibration."""

    if n_samples <= 0 or n_splits < 2:
        return []
    available = n_samples - min_train_size - gap
    if available <= 0:
        return []
    fold_size = max(1, available // n_splits)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for split_index in range(n_splits):
        train_end = min_train_size + split_index * fold_size
        validation_start = train_end + gap
        if validation_start >= n_samples:
            break
        validation_end = (
            n_samples
            if split_index == n_splits - 1
            else min(n_samples, validation_start + fold_size)
        )
        if validation_start >= validation_end:
            continue
        train_start = (
            max(0, train_end - int(rolling_window_size or train_end))
            if mode == "rolling"
            else 0
        )
        train_index = np.arange(train_start, train_end, dtype=int)
        validation_index = np.arange(validation_start, validation_end, dtype=int)
        if len(train_index) == 0 or len(validation_index) == 0:
            continue
        folds.append((train_index, validation_index))
    return folds


def _tuning_folds(frame: pd.DataFrame, settings: BaselineSettings) -> list[tuple[np.ndarray, np.ndarray]]:
    return make_time_series_folds(
        len(frame),
        n_splits=settings.tuning.n_splits,
        gap=settings.tuning.gap,
        mode=settings.tuning.mode,
        min_train_size=settings.tuning.min_train_size,
        rolling_window_size=settings.tuning.rolling_window_size,
    )


def _base_classifier_params(
    model: ClassificationModelVariantSettings,
) -> dict[str, Any]:
    return {
        "estimator": model.estimator,
        "c": model.c,
        "max_iter": model.max_iter,
        "class_weight": model.class_weight,
        "random_state": model.random_state,
        "penalty": model.penalty,
        "solver": model.solver,
        "l1_ratio": model.l1_ratio,
    }


def _base_regression_params(
    model: RegressionModelVariantSettings,
) -> dict[str, Any]:
    return {
        "estimator": model.estimator,
        "alpha": model.alpha,
        "l1_ratio": model.l1_ratio,
        "random_state": model.random_state,
    }


def _tree_classifier_base_params(settings: TreeBaselineSettings) -> dict[str, Any]:
    return {
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "min_samples_leaf": settings.min_samples_leaf,
        "random_state": settings.random_state,
    }


def _tree_regressor_base_params(settings: TreeBaselineSettings) -> dict[str, Any]:
    return {
        "n_estimators": settings.n_estimators,
        "max_depth": settings.max_depth,
        "min_samples_leaf": settings.min_samples_leaf,
        "random_state": settings.random_state,
    }


def _grid_candidates(
    base_params: dict[str, Any],
    param_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    usable_grid = {
        key: values for key, values in param_grid.items() if isinstance(values, list) and values
    }
    if not usable_grid:
        return [base_params]
    keys = sorted(usable_grid)
    values = [usable_grid[key] for key in keys]
    candidates: list[dict[str, Any]] = []
    for combo in product(*values):
        candidate = dict(base_params)
        candidate.update(dict(zip(keys, combo, strict=False)))
        candidates.append(candidate)
    return candidates


def _build_classifier_pipeline(
    settings: BaselineSettings,
    model: ClassificationModelVariantSettings,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    combined = dict(_base_classifier_params(model))
    if params:
        combined.update(params)
    estimator_name = str(combined.get("estimator", model.estimator))
    penalty = str(combined.get("penalty") or model.penalty)
    solver = combined.get("solver")
    l1_ratio = combined.get("l1_ratio")
    if estimator_name == "logistic_regression":
        penalty = "l2"
        solver = solver or "lbfgs"
        l1_ratio = None
    elif estimator_name == "logistic_l1":
        penalty = "l1"
        solver = solver or "liblinear"
        l1_ratio = None
    elif estimator_name == "logistic_elastic_net":
        penalty = "elasticnet"
        solver = solver or "saga"
        if l1_ratio is None:
            l1_ratio = 0.5
    else:
        raise ValueError(f"Unsupported classifier estimator '{estimator_name}'.")

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=settings.imputation.remaining_strategy))
    ]
    if model.standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "model",
            LogisticRegression(
                C=float(combined.get("c", model.c)),
                class_weight=combined.get("class_weight", model.class_weight),
                max_iter=int(combined.get("max_iter", model.max_iter)),
                random_state=int(combined.get("random_state", model.random_state)),
                penalty=penalty,
                solver=str(solver),
                l1_ratio=None if l1_ratio is None else float(l1_ratio),
            ),
        )
    )
    return Pipeline(steps)


def _build_regression_pipeline(
    settings: BaselineSettings,
    model: RegressionModelVariantSettings,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    combined = dict(_base_regression_params(model))
    if params:
        combined.update(params)
    estimator_name = str(combined.get("estimator", model.estimator))
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=settings.imputation.remaining_strategy))
    ]
    if model.standardize:
        steps.append(("scaler", StandardScaler()))
    if estimator_name == "ridge":
        estimator = Ridge(alpha=float(combined.get("alpha", model.alpha)))
    elif estimator_name == "elastic_net":
        estimator = ElasticNet(
            alpha=float(combined.get("alpha", model.alpha)),
            l1_ratio=float(combined.get("l1_ratio", model.l1_ratio)),
            max_iter=5000,
            random_state=int(combined.get("random_state", model.random_state)),
        )
    else:
        raise ValueError(f"Unsupported regression estimator '{estimator_name}'.")
    steps.append(("model", estimator))
    return Pipeline(steps)


def _build_tree_classifier(
    settings: BaselineSettings,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    combined = dict(_tree_classifier_base_params(settings.predictive.tree))
    if params:
        combined.update(params)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=settings.imputation.remaining_strategy)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=int(combined["n_estimators"]),
                    max_depth=(
                        None
                        if combined.get("max_depth") is None
                        else int(combined["max_depth"])
                    ),
                    min_samples_leaf=int(combined["min_samples_leaf"]),
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=int(combined["random_state"]),
                ),
            ),
        ]
    )


def _build_tree_regressor(
    settings: BaselineSettings,
    params: dict[str, Any] | None = None,
) -> Pipeline:
    combined = dict(_tree_regressor_base_params(settings.predictive.tree))
    if params:
        combined.update(params)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy=settings.imputation.remaining_strategy)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=int(combined["n_estimators"]),
                    max_depth=(
                        None
                        if combined.get("max_depth") is None
                        else int(combined["max_depth"])
                    ),
                    min_samples_leaf=int(combined["min_samples_leaf"]),
                    n_jobs=-1,
                    random_state=int(combined["random_state"]),
                ),
            ),
        ]
    )


def _classification_cv_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    metric: str,
) -> float | None:
    if y_true.nunique(dropna=True) < 2:
        return None
    metric_name = metric.lower()
    if metric_name == "average_precision":
        return _safe_float(average_precision_score(y_true, probabilities))
    if metric_name == "roc_auc":
        return _safe_float(roc_auc_score(y_true, probabilities))
    if metric_name == "f1":
        return _safe_float(f1_score(y_true, predicted_labels, zero_division=0))
    if metric_name == "precision":
        return _safe_float(precision_score(y_true, predicted_labels, zero_division=0))
    if metric_name == "brier_neg":
        return _safe_float(-brier_score_loss(y_true, probabilities))
    raise ValueError(f"Unsupported classification tuning metric '{metric}'.")


def _regression_cv_score(
    y_true: pd.Series,
    predicted_values: np.ndarray,
    metric: str,
) -> float | None:
    metric_name = metric.lower()
    if metric_name == "neg_rmse":
        return _safe_float(-math.sqrt(mean_squared_error(y_true, predicted_values)))
    if metric_name == "neg_mae":
        return _safe_float(-mean_absolute_error(y_true, predicted_values))
    if metric_name == "pearson_corr":
        if float(y_true.std(ddof=0)) == 0.0 or float(np.std(predicted_values)) == 0.0:
            return None
        return _safe_float(np.corrcoef(y_true, predicted_values)[0, 1])
    if metric_name == "r2":
        return _safe_float(r2_score(y_true, predicted_values))
    raise ValueError(f"Unsupported regression tuning metric '{metric}'.")


def _tune_classifier_params(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model: ClassificationModelVariantSettings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_params = _base_classifier_params(model)
    candidate_params = _grid_candidates(base_params, model.param_grid)
    if not settings.tuning.enabled or len(candidate_params) == 1:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    X = _feature_matrix(train_frame, feature_columns)
    y = pd.to_numeric(train_frame[target_column], errors="coerce").astype(int)
    folds = _tuning_folds(train_frame, settings)
    if not folds:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    rows: list[dict[str, Any]] = []
    for params in candidate_params:
        fold_scores: list[float] = []
        for inner_train_index, inner_validation_index in folds:
            y_inner_train = y.iloc[inner_train_index]
            y_inner_validation = y.iloc[inner_validation_index]
            if y_inner_train.nunique(dropna=True) < 2 or y_inner_validation.nunique(
                dropna=True
            ) < 2:
                continue
            estimator = _build_classifier_pipeline(settings, model, params)
            estimator.fit(X.iloc[inner_train_index], y_inner_train)
            probabilities = estimator.predict_proba(X.iloc[inner_validation_index])[:, 1]
            predicted_labels = (probabilities >= 0.5).astype(int)
            score = _classification_cv_score(
                y_inner_validation,
                probabilities,
                predicted_labels,
                settings.tuning.classification_metric,
            )
            if score is not None:
                fold_scores.append(float(score))
        rows.append(
            {
                "params_json": _json_dumps(params),
                "mean_cv_score": _safe_float(np.mean(fold_scores))
                if fold_scores
                else None,
                "std_cv_score": _safe_float(np.std(fold_scores, ddof=0))
                if fold_scores
                else None,
                "fold_count": len(fold_scores),
            }
        )
    results = pd.DataFrame(rows)
    ranked = results.copy()
    ranked["_rank_score"] = ranked["mean_cv_score"].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_rank_score", "fold_count"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = json.loads(ranked.iloc[0]["params_json"])
    return selected, results.drop(columns="_rank_score", errors="ignore")


def _tune_regression_params(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model: RegressionModelVariantSettings,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_params = _base_regression_params(model)
    candidate_params = _grid_candidates(base_params, model.param_grid)
    if not settings.tuning.enabled or len(candidate_params) == 1:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    X = _feature_matrix(train_frame, feature_columns)
    y = pd.to_numeric(train_frame[target_column], errors="coerce").astype(float)
    folds = _tuning_folds(train_frame, settings)
    if not folds:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    rows: list[dict[str, Any]] = []
    for params in candidate_params:
        fold_scores: list[float] = []
        for inner_train_index, inner_validation_index in folds:
            estimator = _build_regression_pipeline(settings, model, params)
            estimator.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
            predicted_values = estimator.predict(X.iloc[inner_validation_index])
            score = _regression_cv_score(
                y.iloc[inner_validation_index],
                predicted_values,
                settings.tuning.regression_metric,
            )
            if score is not None:
                fold_scores.append(float(score))
        rows.append(
            {
                "params_json": _json_dumps(params),
                "mean_cv_score": _safe_float(np.mean(fold_scores))
                if fold_scores
                else None,
                "std_cv_score": _safe_float(np.std(fold_scores, ddof=0))
                if fold_scores
                else None,
                "fold_count": len(fold_scores),
            }
        )
    results = pd.DataFrame(rows)
    ranked = results.copy()
    ranked["_rank_score"] = ranked["mean_cv_score"].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_rank_score", "fold_count"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = json.loads(ranked.iloc[0]["params_json"])
    return selected, results.drop(columns="_rank_score", errors="ignore")


def _tune_tree_classifier_params(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_params = _tree_classifier_base_params(settings.predictive.tree)
    candidate_params = _grid_candidates(
        base_params, settings.predictive.tree.classifier_param_grid
    )
    if not settings.tuning.enabled or len(candidate_params) == 1:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    X = _feature_matrix(train_frame, feature_columns)
    y = pd.to_numeric(train_frame[target_column], errors="coerce").astype(int)
    folds = _tuning_folds(train_frame, settings)
    if not folds:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    rows: list[dict[str, Any]] = []
    for params in candidate_params:
        fold_scores: list[float] = []
        for inner_train_index, inner_validation_index in folds:
            y_inner_train = y.iloc[inner_train_index]
            y_inner_validation = y.iloc[inner_validation_index]
            if y_inner_train.nunique(dropna=True) < 2 or y_inner_validation.nunique(
                dropna=True
            ) < 2:
                continue
            estimator = _build_tree_classifier(settings, params)
            estimator.fit(X.iloc[inner_train_index], y_inner_train)
            probabilities = estimator.predict_proba(X.iloc[inner_validation_index])[:, 1]
            predicted_labels = (probabilities >= 0.5).astype(int)
            score = _classification_cv_score(
                y_inner_validation,
                probabilities,
                predicted_labels,
                settings.tuning.classification_metric,
            )
            if score is not None:
                fold_scores.append(float(score))
        rows.append(
            {
                "params_json": _json_dumps(params),
                "mean_cv_score": _safe_float(np.mean(fold_scores))
                if fold_scores
                else None,
                "std_cv_score": _safe_float(np.std(fold_scores, ddof=0))
                if fold_scores
                else None,
                "fold_count": len(fold_scores),
            }
        )
    results = pd.DataFrame(rows)
    ranked = results.copy()
    ranked["_rank_score"] = ranked["mean_cv_score"].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_rank_score", "fold_count"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = json.loads(ranked.iloc[0]["params_json"])
    return selected, results.drop(columns="_rank_score", errors="ignore")


def _tune_tree_regressor_params(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_params = _tree_regressor_base_params(settings.predictive.tree)
    candidate_params = _grid_candidates(
        base_params, settings.predictive.tree.regressor_param_grid
    )
    if not settings.tuning.enabled or len(candidate_params) == 1:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    X = _feature_matrix(train_frame, feature_columns)
    y = pd.to_numeric(train_frame[target_column], errors="coerce").astype(float)
    folds = _tuning_folds(train_frame, settings)
    if not folds:
        return base_params, pd.DataFrame(
            [{"params_json": _json_dumps(base_params), "mean_cv_score": None}]
        )

    rows: list[dict[str, Any]] = []
    for params in candidate_params:
        fold_scores: list[float] = []
        for inner_train_index, inner_validation_index in folds:
            estimator = _build_tree_regressor(settings, params)
            estimator.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
            predicted_values = estimator.predict(X.iloc[inner_validation_index])
            score = _regression_cv_score(
                y.iloc[inner_validation_index],
                predicted_values,
                settings.tuning.regression_metric,
            )
            if score is not None:
                fold_scores.append(float(score))
        rows.append(
            {
                "params_json": _json_dumps(params),
                "mean_cv_score": _safe_float(np.mean(fold_scores))
                if fold_scores
                else None,
                "std_cv_score": _safe_float(np.std(fold_scores, ddof=0))
                if fold_scores
                else None,
                "fold_count": len(fold_scores),
            }
        )
    results = pd.DataFrame(rows)
    ranked = results.copy()
    ranked["_rank_score"] = ranked["mean_cv_score"].fillna(-np.inf)
    ranked = ranked.sort_values(
        ["_rank_score", "fold_count"], ascending=[False, False]
    ).reset_index(drop=True)
    selected = json.loads(ranked.iloc[0]["params_json"])
    return selected, results.drop(columns="_rank_score", errors="ignore")


def _fit_classifier_with_calibration(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_spec: ClassificationModelVariantSettings,
    selected_params: dict[str, Any],
    *,
    tree_mode: bool = False,
) -> tuple[Any, Pipeline, str]:
    X_train = _feature_matrix(train_frame, feature_columns)
    y_train = pd.to_numeric(train_frame[target_column], errors="coerce").astype(int)
    if tree_mode:
        base_builder: Callable[[dict[str, Any]], Pipeline] = (
            lambda params: _build_tree_classifier(settings, params)
        )
        calibration_method = settings.predictive.tree.calibration_method
        calibration_cv_splits = settings.predictive.tree.calibration_cv_splits
        calibration_ensemble = settings.predictive.tree.calibration_ensemble
    else:
        base_builder = lambda params: _build_classifier_pipeline(settings, model_spec, params)
        calibration_method = model_spec.calibration_method
        calibration_cv_splits = model_spec.calibration_cv_splits
        calibration_ensemble = model_spec.calibration_ensemble

    diagnostic_model = base_builder(selected_params)
    diagnostic_model.fit(X_train, y_train)
    if calibration_method == "none" or y_train.nunique(dropna=True) < 2:
        return diagnostic_model, diagnostic_model, "none"

    calibration_folds = make_time_series_folds(
        len(train_frame),
        n_splits=calibration_cv_splits,
        gap=settings.tuning.gap,
        mode=settings.tuning.mode,
        min_train_size=max(
            10,
            min(
                settings.tuning.min_train_size,
                max(2, len(train_frame) // max(calibration_cv_splits + 1, 2)),
            ),
        ),
        rolling_window_size=settings.tuning.rolling_window_size,
    )
    if not calibration_folds:
        return diagnostic_model, diagnostic_model, "none"

    calibrated_model = CalibratedClassifierCV(
        estimator=base_builder(selected_params),
        method=calibration_method,
        cv=calibration_folds,
        ensemble=calibration_ensemble,
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, diagnostic_model, calibration_method


def _fit_regressor(
    settings: BaselineSettings,
    train_frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    model_spec: RegressionModelVariantSettings,
    selected_params: dict[str, Any],
    *,
    tree_mode: bool = False,
) -> Pipeline:
    X_train = _feature_matrix(train_frame, feature_columns)
    y_train = pd.to_numeric(train_frame[target_column], errors="coerce").astype(float)
    estimator = (
        _build_tree_regressor(settings, selected_params)
        if tree_mode
        else _build_regression_pipeline(settings, model_spec, selected_params)
    )
    estimator.fit(X_train, y_train)
    return estimator


def _score_column(predictions: pd.DataFrame) -> pd.Series:
    if "predicted_probability" in predictions.columns and not predictions[
        "predicted_probability"
    ].isna().all():
        return pd.to_numeric(predictions["predicted_probability"], errors="coerce")
    if "predicted_return_bps" in predictions.columns and not predictions[
        "predicted_return_bps"
    ].isna().all():
        return pd.to_numeric(predictions["predicted_return_bps"], errors="coerce")
    return pd.to_numeric(predictions["decision_score"], errors="coerce")


def _top_quantile_returns(
    predictions: pd.DataFrame,
    *,
    top_quantile: float,
) -> pd.Series:
    usable = predictions.dropna(subset=["actual_return_bps"]).copy()
    if usable.empty:
        return pd.Series(dtype=float)
    usable["_score"] = _score_column(usable)
    usable = usable.dropna(subset=["_score"]).sort_values("_score", ascending=False)
    if usable.empty:
        return pd.Series(dtype=float)
    top_count = max(1, int(math.ceil(len(usable) * float(top_quantile))))
    return pd.to_numeric(
        usable["actual_return_bps"].head(top_count), errors="coerce"
    ).dropna()


def _signal_return_summary(signaled_returns: pd.Series) -> dict[str, Any]:
    if signaled_returns.empty:
        return {
            "avg_signal_return_bps": None,
            "median_signal_return_bps": None,
            "cumulative_signal_return_bps": None,
            "signal_hit_rate": None,
            "signal_return_std_bps": None,
            "signal_sharpe_like": None,
        }
    std = signaled_returns.std(ddof=0)
    return {
        "avg_signal_return_bps": _mean_or_none(signaled_returns),
        "median_signal_return_bps": _median_or_none(signaled_returns),
        "cumulative_signal_return_bps": _sum_or_none(signaled_returns),
        "signal_hit_rate": _safe_float((signaled_returns > 0.0).mean()),
        "signal_return_std_bps": _safe_float(std),
        "signal_sharpe_like": (
            _safe_float(signaled_returns.mean() / std) if std and std > 0.0 else None
        ),
    }


def _classification_metrics(
    predictions: pd.DataFrame,
    *,
    top_quantile: float = 0.1,
) -> dict[str, Any]:
    usable = predictions.dropna(subset=["actual_label"]).copy()
    if usable.empty:
        return {}
    y_true = usable["actual_label"].astype(int)
    y_pred = usable["predicted_label"].astype(int)
    y_score = _score_column(usable)

    signaled = usable[usable["signal"] == 1].copy()
    signaled_returns = pd.to_numeric(signaled["actual_return_bps"], errors="coerce").dropna()
    top_returns = _top_quantile_returns(usable, top_quantile=top_quantile)

    metrics = {
        "row_count": int(len(usable)),
        "actual_positive_count": int(y_true.sum()),
        "predicted_positive_count": int(y_pred.sum()),
        "actual_positive_rate": _safe_float(y_true.mean()),
        "predicted_positive_rate": _safe_float(y_pred.mean()),
        "accuracy": _safe_float(accuracy_score(y_true, y_pred)),
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "signal_count": int((usable["signal"] == 1).sum()),
        "signal_rate": _safe_float((usable["signal"] == 1).mean()),
        "precision_among_signaled": _safe_float(signaled["actual_label"].mean())
        if not signaled.empty
        else None,
        "top_quantile_count": int(len(top_returns)),
        "top_quantile_avg_return_bps": _mean_or_none(top_returns),
        "top_quantile_cumulative_return_bps": _sum_or_none(top_returns),
    }
    metrics.update(_signal_return_summary(signaled_returns))
    if y_true.nunique() >= 2 and y_score.nunique(dropna=True) >= 2:
        metrics["roc_auc"] = _safe_float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = _safe_float(
            average_precision_score(y_true, y_score)
        )
        metrics["brier_score"] = (
            _safe_float(
                brier_score_loss(
                    y_true,
                    pd.to_numeric(usable["predicted_probability"], errors="coerce"),
                )
            )
            if not usable["predicted_probability"].isna().all()
            else None
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["brier_score"] = None
    return metrics


def _regression_metrics(
    predictions: pd.DataFrame,
    *,
    top_quantile: float = 0.1,
) -> dict[str, Any]:
    usable = predictions.dropna(
        subset=["actual_return_bps", "predicted_return_bps"]
    ).copy()
    if usable.empty:
        return {}
    y_true = usable["actual_return_bps"].astype(float)
    y_pred = usable["predicted_return_bps"].astype(float)
    signaled = usable[usable["signal"] == 1].copy()
    signaled_returns = pd.to_numeric(signaled["actual_return_bps"], errors="coerce").dropna()
    top_returns = _top_quantile_returns(usable, top_quantile=top_quantile)
    metrics = {
        "row_count": int(len(usable)),
        "mae": _safe_float(mean_absolute_error(y_true, y_pred)),
        "rmse": _safe_float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": _safe_float(r2_score(y_true, y_pred)) if len(usable) > 1 else None,
        "directional_accuracy": _safe_float(((y_true >= 0.0) == (y_pred >= 0.0)).mean()),
        "predicted_positive_rate": _safe_float((y_pred >= 0.0).mean()),
        "signal_count": int((usable["signal"] == 1).sum()),
        "signal_rate": _safe_float((usable["signal"] == 1).mean()),
        "precision_among_signaled": _safe_float(signaled["actual_label"].mean())
        if not signaled.empty
        else None,
        "top_quantile_count": int(len(top_returns)),
        "top_quantile_avg_return_bps": _mean_or_none(top_returns),
        "top_quantile_cumulative_return_bps": _sum_or_none(top_returns),
    }
    metrics.update(_signal_return_summary(signaled_returns))
    metrics["pearson_corr"] = (
        _safe_float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.std(ddof=0) > 0.0 and y_pred.std(ddof=0) > 0.0
        else None
    )
    return metrics


def evaluate_prediction_table(
    predictions: pd.DataFrame,
    *,
    top_quantile: float = 0.1,
) -> pd.DataFrame:
    """Aggregate split-aware metrics from a combined prediction table."""

    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(["model_name", "model_family", "task", "split"], sort=True)
    for (model_name, model_family, task, split), group in grouped:
        metrics = (
            _regression_metrics(group, top_quantile=top_quantile)
            if task == "regression"
            else _classification_metrics(group, top_quantile=top_quantile)
        )
        signal_count = int(metrics.get("signal_count", 0) or 0)
        row = {
            "model_name": model_name,
            "model_family": model_family,
            "task": task,
            "split": split,
            "status": "ok" if signal_count > 0 else "no_tradable_signals",
            "diagnostic_reason": (
                None if signal_count > 0 else f"signal_count == 0 for split '{split}'."
            ),
            "degenerate_experiment": bool(signal_count == 0),
        }
        row.update(metrics)
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["task", "model_family", "model_name", "split"])
        .reset_index(drop=True)
    )


def _build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    leaderboard = metrics[metrics["split"].isin(["validation", "test"])].copy()
    leaderboard["has_signals"] = leaderboard["signal_count"].fillna(0).astype(float) > 0.0
    sort_keys = [
        "has_signals",
        "task",
        "split",
        "avg_signal_return_bps",
        "cumulative_signal_return_bps",
        "signal_sharpe_like",
        "top_quantile_avg_return_bps",
        "f1",
        "pearson_corr",
    ]
    available_keys = [key for key in sort_keys if key in leaderboard.columns]
    if available_keys:
        leaderboard = leaderboard.sort_values(
            available_keys,
            ascending=[False] * len(available_keys),
            na_position="last",
        )
    return leaderboard.reset_index(drop=True)


def _table_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


def _rule_candidate_specs(rule: RuleBaselineSpec) -> list[RuleBaselineSpec]:
    if rule.kind == "funding_threshold":
        grid = rule.funding_threshold_grid_bps or [rule.funding_threshold_bps]
        return [
            rule.model_copy(update={"funding_threshold_bps": float(threshold)})
            for threshold in grid
        ]
    if rule.kind == "spread_zscore_threshold":
        grid = rule.spread_threshold_grid or [rule.spread_threshold]
        return [
            rule.model_copy(update={"spread_threshold": float(threshold)})
            for threshold in grid
        ]
    if rule.kind == "combined_threshold":
        funding_grid = rule.funding_threshold_grid_bps or [rule.funding_threshold_bps]
        spread_grid = rule.spread_threshold_grid or [rule.spread_threshold]
        return [
            rule.model_copy(
                update={
                    "funding_threshold_bps": float(funding_threshold),
                    "spread_threshold": float(spread_threshold),
                }
            )
            for funding_threshold, spread_threshold in product(funding_grid, spread_grid)
        ]
    raise ValueError(f"Unsupported rule baseline kind: {rule.kind}")


def _apply_rule_logic(
    frame: pd.DataFrame,
    rule: RuleBaselineSpec,
) -> tuple[pd.Series, pd.Series, float]:
    if rule.kind == "funding_threshold":
        decision_score = pd.to_numeric(frame[rule.funding_column], errors="coerce").astype(float)
        signal_threshold = float(rule.funding_threshold_bps)
        signal = decision_score >= signal_threshold
    elif rule.kind == "spread_zscore_threshold":
        decision_score = pd.to_numeric(frame[rule.spread_column], errors="coerce").astype(float)
        signal_threshold = float(rule.spread_threshold)
        signal = decision_score >= signal_threshold
    elif rule.kind == "combined_threshold":
        funding_margin = pd.to_numeric(frame[rule.funding_column], errors="coerce").astype(float) - float(
            rule.funding_threshold_bps
        )
        spread_margin = pd.to_numeric(frame[rule.spread_column], errors="coerce").astype(float) - float(
            rule.spread_threshold
        )
        decision_score = funding_margin.add(spread_margin, fill_value=0.0)
        signal_threshold = 0.0
        signal = (funding_margin >= 0.0) & (spread_margin >= 0.0)
    else:
        raise ValueError(f"Unsupported rule baseline kind: {rule.kind}")
    if rule.regime_column is not None:
        signal = signal & frame[rule.regime_column].eq(rule.regime_value)
    return signal.astype(int), decision_score.astype(float), float(signal_threshold)


def _rule_prediction_frame(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    rule: RuleBaselineSpec,
) -> pd.DataFrame:
    signal, decision_score, signal_threshold = _apply_rule_logic(frame, rule)
    return pd.DataFrame(
        {
            "timestamp": frame[settings.target.timestamp_column],
            "split": frame[settings.target.split_column],
            "model_name": rule.name,
            "model_family": "rule_based",
            "task": "classification",
            "signal_direction": "short_perp_long_spot",
            "signal": signal.astype(int),
            "decision_score": decision_score.astype(float),
            "signal_threshold": float(signal_threshold),
            "signal_strength": decision_score.astype(float) - float(signal_threshold),
            "predicted_probability": np.nan,
            "predicted_return_bps": np.nan,
            "predicted_label": signal.astype(int),
            "actual_label": pd.to_numeric(
                frame[settings.target.classification_column], errors="coerce"
            ),
            "actual_return_bps": pd.to_numeric(
                frame[settings.target.regression_column], errors="coerce"
            ),
            "selected_hyperparameters_json": _json_dumps({}),
            "selected_threshold_objective": None,
            "calibration_method": "none",
            "feature_importance_method": "not_applicable",
            "prediction_mode": "static",
        }
    )


def _metric_value_for_threshold(
    predictions: pd.DataFrame,
    *,
    objective: str,
    top_quantile: float,
) -> float:
    metrics = evaluate_prediction_table(predictions, top_quantile=top_quantile)
    if metrics.empty:
        return -np.inf
    value = metrics.iloc[0].get(objective)
    safe_value = _safe_float(value)
    return -np.inf if safe_value is None else float(safe_value)


def _select_rule_spec(
    validation_frame: pd.DataFrame,
    settings: BaselineSettings,
    rule: RuleBaselineSpec,
) -> tuple[RuleBaselineSpec, pd.DataFrame]:
    candidates = _rule_candidate_specs(rule)
    if (
        not settings.threshold_search.enabled
        or not settings.threshold_search.rule_search_enabled
        or validation_frame.empty
        or len(candidates) == 1
    ):
        return rule, pd.DataFrame(
            [
                {
                    "candidate_name": rule.name,
                    "params_json": _json_dumps(rule.model_dump()),
                    "objective_value": None,
                    "selected": True,
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    best_candidate = rule
    best_score = -np.inf
    for candidate in candidates:
        candidate_predictions = _rule_prediction_frame(validation_frame, settings, candidate)
        score = _metric_value_for_threshold(
            candidate_predictions,
            objective=settings.threshold_search.objective,
            top_quantile=settings.threshold_search.top_quantile,
        )
        rows.append(
            {
                "candidate_name": candidate.name,
                "params_json": _json_dumps(
                    {
                        "funding_threshold_bps": candidate.funding_threshold_bps,
                        "spread_threshold": candidate.spread_threshold,
                    }
                ),
                "objective_value": _safe_float(score),
            }
        )
        if score > best_score:
            best_score = score
            best_candidate = candidate
    selected_json = _json_dumps(
        {
            "funding_threshold_bps": best_candidate.funding_threshold_bps,
            "spread_threshold": best_candidate.spread_threshold,
        }
    )
    for row in rows:
        row["selected"] = row["params_json"] == selected_json
    return best_candidate, pd.DataFrame(rows).sort_values(
        "objective_value", ascending=False, na_position="last"
    )


def _base_prediction_columns(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    model_name: str,
    model_family: str,
    task: str,
    calibration_method: str,
    prediction_mode: str,
    selected_params: dict[str, Any],
    threshold_objective: str | None,
    feature_importance_method: str,
) -> dict[str, Any]:
    return {
        "timestamp": frame[settings.target.timestamp_column],
        "split": frame[settings.target.split_column],
        "model_name": model_name,
        "model_family": model_family,
        "task": task,
        "signal_direction": "short_perp_long_spot",
        "actual_label": pd.to_numeric(
            frame[settings.target.classification_column], errors="coerce"
        ),
        "actual_return_bps": pd.to_numeric(
            frame[settings.target.regression_column], errors="coerce"
        ),
        "selected_hyperparameters_json": _json_dumps(selected_params),
        "selected_threshold_objective": threshold_objective,
        "calibration_method": calibration_method,
        "feature_importance_method": feature_importance_method,
        "prediction_mode": prediction_mode,
    }


def _classifier_score_frame(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    model_name: str,
    model_family: str,
    probabilities: np.ndarray,
    calibration_method: str,
    prediction_mode: str,
    selected_params: dict[str, Any],
    threshold_objective: str | None,
    feature_importance_method: str,
) -> pd.DataFrame:
    base = _base_prediction_columns(
        frame,
        settings,
        model_name=model_name,
        model_family=model_family,
        task="classification",
        calibration_method=calibration_method,
        prediction_mode=prediction_mode,
        selected_params=selected_params,
        threshold_objective=threshold_objective,
        feature_importance_method=feature_importance_method,
    )
    return pd.DataFrame(
        {
            **base,
            "decision_score": probabilities.astype(float),
            "predicted_probability": probabilities.astype(float),
            "predicted_return_bps": np.nan,
        }
    )


def _apply_classifier_threshold(
    score_frame: pd.DataFrame,
    probability_threshold: float,
) -> pd.DataFrame:
    predictions = score_frame.copy()
    predicted_label = (
        pd.to_numeric(predictions["predicted_probability"], errors="coerce")
        >= float(probability_threshold)
    ).astype(int)
    predictions["signal"] = predicted_label.astype(int)
    predictions["predicted_label"] = predicted_label.astype(int)
    predictions["signal_threshold"] = float(probability_threshold)
    predictions["signal_strength"] = (
        pd.to_numeric(predictions["decision_score"], errors="coerce")
        - float(probability_threshold)
    )
    return predictions


def _regression_score_frame(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    model_name: str,
    model_family: str,
    predicted_return: np.ndarray,
    prediction_mode: str,
    selected_params: dict[str, Any],
    threshold_objective: str | None,
    feature_importance_method: str,
) -> pd.DataFrame:
    base = _base_prediction_columns(
        frame,
        settings,
        model_name=model_name,
        model_family=model_family,
        task="regression",
        calibration_method="none",
        prediction_mode=prediction_mode,
        selected_params=selected_params,
        threshold_objective=threshold_objective,
        feature_importance_method=feature_importance_method,
    )
    return pd.DataFrame(
        {
            **base,
            "decision_score": predicted_return.astype(float),
            "predicted_probability": np.nan,
            "predicted_return_bps": predicted_return.astype(float),
        }
    )


def _apply_regression_threshold(
    score_frame: pd.DataFrame,
    trade_threshold_bps: float,
) -> pd.DataFrame:
    predictions = score_frame.copy()
    signal = (
        pd.to_numeric(predictions["predicted_return_bps"], errors="coerce")
        >= float(trade_threshold_bps)
    ).astype(int)
    predictions["signal"] = signal.astype(int)
    predictions["predicted_label"] = signal.astype(int)
    predictions["signal_threshold"] = float(trade_threshold_bps)
    predictions["signal_strength"] = (
        pd.to_numeric(predictions["decision_score"], errors="coerce")
        - float(trade_threshold_bps)
    )
    return predictions


def _default_probability_grid() -> list[float]:
    return [round(value, 2) for value in np.linspace(0.45, 0.85, 9)]


def _default_regression_threshold_grid() -> list[float]:
    return [-5.0, 0.0, 2.5, 5.0, 7.5, 10.0]


def _split_metric_map(
    diagnostics_by_split: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    return {
        split_name: diagnostics.get(key)
        for split_name, diagnostics in diagnostics_by_split.items()
    }


def _return_summary_map(
    diagnostics_by_split: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    return {
        split_name: diagnostics.get("future_net_return_bps", {})
        for split_name, diagnostics in diagnostics_by_split.items()
    }


def _label_diagnostics_for_modeling_frame(
    modeling_frame: pd.DataFrame,
    settings: BaselineSettings,
) -> dict[str, dict[str, Any]]:
    regression_column = settings.target.regression_column
    tradeable_threshold_bps = 5.0
    profitable_threshold_bps = 0.0
    if settings.input.manifest_path:
        manifest_path = _resolve_path(settings.input.manifest_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_config = manifest.get("target", {})
            tradeable_threshold_bps = float(
                target_config.get("min_expected_edge_bps", tradeable_threshold_bps)
            )
            profitable_threshold_bps = float(
                target_config.get("positive_return_threshold_bps", profitable_threshold_bps)
            )
    return label_split_diagnostics(
        modeling_frame,
        split_column=settings.target.split_column,
        net_return_column=regression_column,
        tradeable_column=infer_tradeable_column(regression_column),
        profitable_column=infer_profitable_column(regression_column),
        tradeable_threshold_bps=tradeable_threshold_bps,
        profitable_threshold_bps=profitable_threshold_bps,
    )


def _finalize_threshold_search_frame(
    search_frame: pd.DataFrame,
    *,
    selected_threshold: float,
    status: str,
    reason: str | None,
    fallback_used: bool,
    fallback_reason: str | None,
) -> pd.DataFrame:
    frame = search_frame.copy()
    if frame.empty:
        frame = pd.DataFrame([{"threshold": float(selected_threshold), "objective_value": None}])
    if "valid_candidate" not in frame.columns:
        frame["valid_candidate"] = pd.to_numeric(frame["objective_value"], errors="coerce").notna()
    threshold_series = pd.to_numeric(frame["threshold"], errors="coerce")
    frame["selected"] = threshold_series.eq(float(selected_threshold))
    frame["status"] = status
    frame["reason"] = reason
    frame["fallback_used"] = bool(fallback_used)
    frame["fallback_reason"] = fallback_reason
    return frame


def _threshold_search_failure_result(
    *,
    settings: BaselineSettings,
    model_name: str,
    source: str,
    reason: str,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    signal_count_by_split: dict[str, int],
    default_threshold: float,
    search_frame: pd.DataFrame,
) -> ThresholdSelectionResult:
    summary = summarize_threshold_search(search_frame)
    diagnostics = DegenerateExperimentDiagnostics(
        stage="threshold_search",
        reason=reason,
        status="threshold_selection_failed",
        model_name=model_name,
        source=source,
        target_horizon=infer_horizon_label(settings.target.regression_column),
        split="validation",
        selected_threshold=float(default_threshold),
        candidate_threshold_count=summary.get("candidate_count"),
        valid_candidate_count=summary.get("valid_candidate_count"),
        signal_count_by_split=signal_count_by_split,
        tradeable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        profitable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        future_net_return_bps_by_split=_return_summary_map(label_diagnostics_by_split),
        fallback_used=bool(settings.threshold_search.allow_degenerate_fallback),
        fallback_reason=reason if settings.threshold_search.allow_degenerate_fallback else None,
        extra={"threshold_search_summary": summary},
    )
    if not settings.threshold_search.allow_degenerate_fallback:
        raise DegenerateExperimentError(diagnostics)
    warn_on_degenerate_experiment(diagnostics, stacklevel=3)
    finalized = _finalize_threshold_search_frame(
        search_frame,
        selected_threshold=float(default_threshold),
        status="degenerate_fallback",
        reason=reason,
        fallback_used=True,
        fallback_reason=reason,
    )
    return ThresholdSelectionResult(
        selected_threshold=float(default_threshold),
        objective_value=None,
        search_frame=finalized,
        fallback_used=True,
        fallback_reason=reason,
    )


def _threshold_search_success_result(
    selected_threshold: float,
    objective_value: float | None,
    search_frame: pd.DataFrame,
) -> ThresholdSelectionResult:
    return ThresholdSelectionResult(
        selected_threshold=float(selected_threshold),
        objective_value=_safe_float(objective_value),
        search_frame=_finalize_threshold_search_frame(
            search_frame,
            selected_threshold=float(selected_threshold),
            status="ok",
            reason=None,
            fallback_used=False,
            fallback_reason=None,
        ),
        fallback_used=False,
        fallback_reason=None,
    )


def _warn_on_zero_signal_splits(
    *,
    model_name: str,
    source: str,
    settings: BaselineSettings,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    signal_diagnostics_by_split: dict[str, dict[str, Any]],
) -> DegenerateExperimentDiagnostics | None:
    flagged = [
        (split_name, diagnostics)
        for split_name, diagnostics in signal_diagnostics_by_split.items()
        if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
    ]
    if not flagged:
        return None
    reason = "; ".join(
        f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
        for split_name, diagnostics in flagged
    )
    diagnostics = DegenerateExperimentDiagnostics(
        stage="signal_generation",
        reason=reason,
        status="no_tradable_signals",
        model_name=model_name,
        source=source,
        target_horizon=infer_horizon_label(settings.target.regression_column),
        signal_count_by_split={
            split_name: int(split_diagnostics.get("signal_count", 0))
            for split_name, split_diagnostics in signal_diagnostics_by_split.items()
        },
        tradeable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        profitable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        future_net_return_bps_by_split=_return_summary_map(label_diagnostics_by_split),
    )
    warn_on_degenerate_experiment(diagnostics, stacklevel=3)
    return diagnostics


def _model_diagnostics_payload(
    *,
    model_name: str,
    source: str,
    settings: BaselineSettings,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    threshold_search_frame: pd.DataFrame | None = None,
    threshold_selection: ThresholdSelectionResult | None = None,
) -> dict[str, Any]:
    signal_diagnostics_by_split = signal_split_diagnostics(
        predictions,
        signal_column="signal",
        split_names=predictions["split"].dropna().astype(str).unique().tolist(),
    )
    signal_warning = _warn_on_zero_signal_splits(
        model_name=model_name,
        source=source,
        settings=settings,
        label_diagnostics_by_split=label_diagnostics_by_split,
        signal_diagnostics_by_split=signal_diagnostics_by_split,
    )
    threshold_summary = (
        summarize_threshold_search(threshold_search_frame)
        if threshold_search_frame is not None
        else None
    )
    status = "ok"
    reason = None
    degenerate_stage = None
    degenerate_experiment = False
    if signal_warning is not None:
        status = signal_warning.status
        reason = signal_warning.reason
        degenerate_stage = signal_warning.stage
        degenerate_experiment = True
    if threshold_selection is not None and threshold_selection.fallback_used:
        status = "degenerate_fallback"
        reason = threshold_selection.fallback_reason
        degenerate_stage = "threshold_search"
        degenerate_experiment = True
    return {
        "status": status,
        "reason": reason,
        "degenerate_experiment": bool(degenerate_experiment),
        "degenerate_stage": degenerate_stage,
        "fallback_used": bool(threshold_selection.fallback_used) if threshold_selection is not None else False,
        "fallback_reason": (
            threshold_selection.fallback_reason if threshold_selection is not None else None
        ),
        "signal_count_by_split": {
            split_name: int(split_diagnostics.get("signal_count", 0))
            for split_name, split_diagnostics in signal_diagnostics_by_split.items()
        },
        "tradeable_rate_by_split": _split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        "profitable_rate_by_split": _split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        "future_net_return_bps_by_split": _return_summary_map(label_diagnostics_by_split),
        "threshold_search_summary": threshold_summary,
    }


def _select_classifier_threshold(
    validation_score_frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    model_name: str,
    source: str,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    default_threshold: float,
    threshold_grid: list[float],
) -> ThresholdSelectionResult:
    candidates = sorted(
        {
            float(value)
            for value in (
                threshold_grid
                or settings.threshold_search.probability_grid
                or _default_probability_grid()
            )
        }
        | {float(default_threshold)}
    )
    if not settings.threshold_search.enabled:
        return _threshold_search_success_result(
            float(default_threshold),
            None,
            pd.DataFrame(
                [
                    {
                        "threshold": float(default_threshold),
                        "objective_value": None,
                    }
                ]
            ),
        )
    validation_support = label_diagnostics_by_split.get("validation", {})
    if validation_score_frame.empty:
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason="Validation split produced no score rows, so threshold selection cannot run.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )
    if not bool(validation_support.get("supports_threshold_selection", True)):
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason=(
                "Validation split cannot support threshold selection because label diagnostics are "
                f"degenerate: {validation_support.get('reason') or validation_support.get('status')}."
            ),
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )
    rows: list[dict[str, Any]] = []
    best_threshold = float(default_threshold)
    best_score = -np.inf
    for threshold in candidates:
        candidate_frame = _apply_classifier_threshold(validation_score_frame, threshold)
        objective_value = _metric_value_for_threshold(
            candidate_frame,
            objective=settings.threshold_search.objective,
            top_quantile=settings.threshold_search.top_quantile,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "objective_value": _safe_float(objective_value),
                "valid_candidate": _safe_float(objective_value) is not None,
            }
        )
        if objective_value > best_score:
            best_score = objective_value
            best_threshold = float(threshold)
    search_frame = pd.DataFrame(rows)
    signal_diagnostics = signal_split_diagnostics(
        _apply_classifier_threshold(validation_score_frame, best_threshold),
        signal_column="signal",
        split_names=["validation"],
    )
    if int(search_frame["valid_candidate"].sum()) == 0:
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason="Threshold search found no valid classifier candidate on the validation split.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={
                split_name: int(diagnostics.get("signal_count", 0))
                for split_name, diagnostics in signal_diagnostics.items()
            },
            default_threshold=float(default_threshold),
            search_frame=search_frame,
        )
    return _threshold_search_success_result(best_threshold, best_score, search_frame)


def _select_regression_threshold(
    validation_score_frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    model_name: str,
    source: str,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    default_threshold: float,
    threshold_grid: list[float],
) -> ThresholdSelectionResult:
    candidates = sorted(
        {
            float(value)
            for value in (
                threshold_grid
                or settings.threshold_search.regression_threshold_grid_bps
                or _default_regression_threshold_grid()
            )
        }
        | {float(default_threshold)}
    )
    if not settings.threshold_search.enabled:
        return _threshold_search_success_result(
            float(default_threshold),
            None,
            pd.DataFrame(
                [
                    {
                        "threshold": float(default_threshold),
                        "objective_value": None,
                    }
                ]
            ),
        )
    validation_support = label_diagnostics_by_split.get("validation", {})
    if validation_score_frame.empty:
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason="Validation split produced no score rows, so threshold selection cannot run.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )
    if not bool(validation_support.get("supports_threshold_selection", True)):
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason=(
                "Validation split cannot support threshold selection because label diagnostics are "
                f"degenerate: {validation_support.get('reason') or validation_support.get('status')}."
            ),
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )
    rows: list[dict[str, Any]] = []
    best_threshold = float(default_threshold)
    best_score = -np.inf
    for threshold in candidates:
        candidate_frame = _apply_regression_threshold(validation_score_frame, threshold)
        objective_value = _metric_value_for_threshold(
            candidate_frame,
            objective=settings.threshold_search.objective,
            top_quantile=settings.threshold_search.top_quantile,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "objective_value": _safe_float(objective_value),
                "valid_candidate": _safe_float(objective_value) is not None,
            }
        )
        if objective_value > best_score:
            best_score = objective_value
            best_threshold = float(threshold)
    search_frame = pd.DataFrame(rows)
    signal_diagnostics = signal_split_diagnostics(
        _apply_regression_threshold(validation_score_frame, best_threshold),
        signal_column="signal",
        split_names=["validation"],
    )
    if int(search_frame["valid_candidate"].sum()) == 0:
        return _threshold_search_failure_result(
            settings=settings,
            model_name=model_name,
            source=source,
            reason="Threshold search found no valid regression candidate on the validation split.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={
                split_name: int(diagnostics.get("signal_count", 0))
                for split_name, diagnostics in signal_diagnostics.items()
            },
            default_threshold=float(default_threshold),
            search_frame=search_frame,
        )
    return _threshold_search_success_result(best_threshold, best_score, search_frame)


def _walk_forward_history(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    cutoff_timestamp: pd.Timestamp,
    target_column: str,
    current_split: str,
) -> pd.DataFrame:
    history = frame[
        (frame[settings.target.timestamp_column] < cutoff_timestamp)
        & frame[target_column].notna()
    ].copy()
    if current_split == "test" and not settings.prediction.use_validation_history_for_test:
        history = history[history[settings.target.split_column] != "validation"].copy()
    if settings.prediction.mode == "rolling" and settings.prediction.rolling_window_size:
        history = history.tail(settings.prediction.rolling_window_size).copy()
    return history


def _generate_classifier_scores(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    prediction_model: Any,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    model_family: str,
    selected_params: dict[str, Any],
    calibration_method: str,
    threshold_objective: str | None,
    tree_mode: bool = False,
    model_spec: ClassificationModelVariantSettings | None = None,
) -> pd.DataFrame:
    relevant = frame.dropna(subset=[target_column]).copy()
    if relevant.empty:
        return pd.DataFrame()
    if settings.prediction.mode == "static":
        probabilities = prediction_model.predict_proba(
            _feature_matrix(relevant, feature_columns)
        )[:, 1]
        return _classifier_score_frame(
            relevant,
            settings,
            model_name=model_name,
            model_family=model_family,
            probabilities=probabilities,
            calibration_method=calibration_method,
            prediction_mode="static",
            selected_params=selected_params,
            threshold_objective=threshold_objective,
            feature_importance_method="permutation_validation",
        )

    train_mask = relevant[settings.target.split_column].eq("train")
    train_rows = relevant[train_mask].copy()
    non_train_rows = relevant[~train_mask].copy().sort_values(settings.target.timestamp_column)
    score_frames: list[pd.DataFrame] = []
    if not train_rows.empty:
        train_probabilities = prediction_model.predict_proba(
            _feature_matrix(train_rows, feature_columns)
        )[:, 1]
        score_frames.append(
            _classifier_score_frame(
                train_rows,
                settings,
                model_name=model_name,
                model_family=model_family,
                probabilities=train_probabilities,
                calibration_method=calibration_method,
                prediction_mode="train_fit",
                selected_params=selected_params,
                threshold_objective=threshold_objective,
                feature_importance_method="permutation_validation",
            )
        )

    chunk_size = int(settings.prediction.refit_every_n_periods)
    for chunk_start in range(0, len(non_train_rows), chunk_size):
        chunk = non_train_rows.iloc[chunk_start : chunk_start + chunk_size].copy()
        history = _walk_forward_history(
            relevant,
            settings,
            cutoff_timestamp=chunk[settings.target.timestamp_column].iloc[0],
            target_column=target_column,
            current_split=str(chunk[settings.target.split_column].iloc[0]),
        )
        if history.empty or history[target_column].nunique(dropna=True) < 2:
            chunk_model = prediction_model
            chunk_calibration = calibration_method
        else:
            spec_to_use = model_spec or settings.predictive.classification
            chunk_model, _, chunk_calibration = _fit_classifier_with_calibration(
                settings,
                history,
                feature_columns,
                target_column,
                spec_to_use,
                selected_params,
                tree_mode=tree_mode,
            )
        probabilities = chunk_model.predict_proba(_feature_matrix(chunk, feature_columns))[:, 1]
        score_frames.append(
            _classifier_score_frame(
                chunk,
                settings,
                model_name=model_name,
                model_family=model_family,
                probabilities=probabilities,
                calibration_method=chunk_calibration,
                prediction_mode=settings.prediction.mode,
                selected_params=selected_params,
                threshold_objective=threshold_objective,
                feature_importance_method="permutation_validation",
            )
        )
    return (
        pd.concat(score_frames, ignore_index=True)
        .sort_values(["timestamp", "split"])
        .reset_index(drop=True)
    )


def _generate_regression_scores(
    frame: pd.DataFrame,
    settings: BaselineSettings,
    *,
    prediction_model: Pipeline,
    feature_columns: list[str],
    target_column: str,
    model_name: str,
    model_family: str,
    selected_params: dict[str, Any],
    threshold_objective: str | None,
    tree_mode: bool = False,
    model_spec: RegressionModelVariantSettings | None = None,
) -> pd.DataFrame:
    relevant = frame.dropna(subset=[target_column]).copy()
    if relevant.empty:
        return pd.DataFrame()
    if settings.prediction.mode == "static":
        predicted_return = prediction_model.predict(_feature_matrix(relevant, feature_columns))
        return _regression_score_frame(
            relevant,
            settings,
            model_name=model_name,
            model_family=model_family,
            predicted_return=predicted_return,
            prediction_mode="static",
            selected_params=selected_params,
            threshold_objective=threshold_objective,
            feature_importance_method="permutation_validation",
        )

    train_mask = relevant[settings.target.split_column].eq("train")
    train_rows = relevant[train_mask].copy()
    non_train_rows = relevant[~train_mask].copy().sort_values(settings.target.timestamp_column)
    score_frames: list[pd.DataFrame] = []
    if not train_rows.empty:
        train_predictions = prediction_model.predict(_feature_matrix(train_rows, feature_columns))
        score_frames.append(
            _regression_score_frame(
                train_rows,
                settings,
                model_name=model_name,
                model_family=model_family,
                predicted_return=train_predictions,
                prediction_mode="train_fit",
                selected_params=selected_params,
                threshold_objective=threshold_objective,
                feature_importance_method="permutation_validation",
            )
        )

    chunk_size = int(settings.prediction.refit_every_n_periods)
    for chunk_start in range(0, len(non_train_rows), chunk_size):
        chunk = non_train_rows.iloc[chunk_start : chunk_start + chunk_size].copy()
        history = _walk_forward_history(
            relevant,
            settings,
            cutoff_timestamp=chunk[settings.target.timestamp_column].iloc[0],
            target_column=target_column,
            current_split=str(chunk[settings.target.split_column].iloc[0]),
        )
        if history.empty:
            chunk_model = prediction_model
        else:
            spec_to_use = model_spec or settings.predictive.regression
            chunk_model = _fit_regressor(
                settings,
                history,
                feature_columns,
                target_column,
                spec_to_use,
                selected_params,
                tree_mode=tree_mode,
            )
        predicted_return = chunk_model.predict(_feature_matrix(chunk, feature_columns))
        score_frames.append(
            _regression_score_frame(
                chunk,
                settings,
                model_name=model_name,
                model_family=model_family,
                predicted_return=predicted_return,
                prediction_mode=settings.prediction.mode,
                selected_params=selected_params,
                threshold_objective=threshold_objective,
                feature_importance_method="permutation_validation",
            )
        )
    return (
        pd.concat(score_frames, ignore_index=True)
        .sort_values(["timestamp", "split"])
        .reset_index(drop=True)
    )


def _extract_linear_coefficients(model: Any, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else None
    if estimator is None or not hasattr(estimator, "coef_"):
        return None
    coefficients = np.asarray(estimator.coef_)
    if coefficients.ndim > 1:
        coefficients = coefficients[0]
    frame = pd.DataFrame({"feature": feature_columns, "coefficient": coefficients.astype(float)})
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def _extract_tree_importances(model: Any, feature_columns: list[str]) -> pd.DataFrame | None:
    estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else None
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None
    frame = pd.DataFrame(
        {"feature": feature_columns, "importance": np.asarray(estimator.feature_importances_, dtype=float)}
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def _permutation_importance_frame(
    model: Any,
    evaluation_frame: pd.DataFrame,
    feature_columns: list[str],
    settings: BaselineSettings,
    *,
    task: str,
) -> pd.DataFrame | None:
    if evaluation_frame.empty:
        return None
    X_eval = _feature_matrix(evaluation_frame, feature_columns)
    if task == "classification":
        y_eval = pd.to_numeric(
            evaluation_frame[settings.target.classification_column], errors="coerce"
        ).astype(int)
        if y_eval.nunique(dropna=True) < 2:
            return None
        scoring = "average_precision"
    else:
        y_eval = pd.to_numeric(
            evaluation_frame[settings.target.regression_column], errors="coerce"
        ).astype(float)
        scoring = "neg_mean_squared_error"
    importance = permutation_importance(
        model,
        X_eval,
        y_eval,
        n_repeats=5,
        random_state=42,
        scoring=scoring,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean.astype(float),
            "importance_std": importance.importances_std.astype(float),
        }
    )
    frame["abs_importance_mean"] = frame["importance_mean"].abs()
    return frame.sort_values("abs_importance_mean", ascending=False).reset_index(drop=True)


def _calibration_table(
    predictions: pd.DataFrame,
    *,
    split_name: str,
    n_bins: int = 10,
) -> pd.DataFrame | None:
    subset = predictions[
        (predictions["split"] == split_name)
        & predictions["predicted_probability"].notna()
        & predictions["actual_label"].notna()
    ].copy()
    if subset.empty or subset["actual_label"].nunique(dropna=True) < 2:
        return None
    try:
        subset["probability_bin"] = pd.qcut(
            subset["predicted_probability"],
            q=min(n_bins, subset["predicted_probability"].nunique(dropna=True)),
            duplicates="drop",
        )
    except ValueError:
        return None
    grouped = subset.groupby("probability_bin", observed=False)
    return (
        grouped.agg(
            count=("actual_label", "size"),
            avg_predicted_probability=("predicted_probability", "mean"),
            observed_positive_rate=("actual_label", "mean"),
            avg_realized_return_bps=("actual_return_bps", "mean"),
        )
        .reset_index()
        .sort_values("avg_predicted_probability")
        .reset_index(drop=True)
    )


def _write_diagnostic_frame(
    diagnostics_dir: Path,
    model_name: str,
    suffix: str,
    frame: pd.DataFrame | None,
) -> str | None:
    if frame is None or frame.empty:
        return None
    path = diagnostics_dir / f"{model_name}_{suffix}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _legacy_bundle(obj: Any) -> dict[str, Any]:
    return {
        "prediction_model": obj,
        "diagnostic_model": obj,
        "selected_params": {},
        "selected_threshold": None,
        "threshold_objective": None,
        "threshold_objective_value": None,
        "calibration_method": "none",
        "prediction_mode": "static",
        "tuning_metric": None,
        "legacy_artifact": True,
    }


def _save_model_bundle(path: Path, bundle: dict[str, Any]) -> None:
    dump(bundle, path)


def _load_model_bundle(path: Path) -> dict[str, Any]:
    loaded = load(path)
    if isinstance(loaded, dict):
        return loaded
    return _legacy_bundle(loaded)


def _build_model_report_summary(model_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not model_records:
        return pd.DataFrame()
    return pd.DataFrame(model_records).sort_values(
        ["task", "model_family", "model_name"]
    ).reset_index(drop=True)


def _write_markdown_report(
    settings: BaselineSettings,
    feature_columns: list[str],
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    model_summary: pd.DataFrame,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    output_dir: Path,
) -> str | None:
    if not settings.output.write_markdown_report:
        return None
    report_path = output_dir / "baseline_report.md"
    classification_columns = [
        "model_name",
        "split",
        "signal_count",
        "signal_rate",
        "avg_signal_return_bps",
        "signal_hit_rate",
        "precision_among_signaled",
        "top_quantile_avg_return_bps",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
        "status",
        "diagnostic_reason",
    ]
    regression_columns = [
        "model_name",
        "split",
        "signal_count",
        "signal_rate",
        "avg_signal_return_bps",
        "signal_hit_rate",
        "precision_among_signaled",
        "top_quantile_avg_return_bps",
        "mae",
        "rmse",
        "r2",
        "pearson_corr",
        "directional_accuracy",
        "status",
        "diagnostic_reason",
    ]
    summary_columns = [
        "model_name",
        "task",
        "model_family",
        "prediction_mode",
        "calibration_method",
        "feature_importance_method",
        "selected_threshold",
        "threshold_objective",
        "threshold_objective_value",
        "tuning_metric",
        "status",
        "reason",
        "fallback_used",
        "selected_params_json",
    ]
    degenerate_experiment = any(
        diagnostics.get("status") != "ok"
        for split_name, diagnostics in label_diagnostics_by_split.items()
        if split_name in {"validation", "test"}
    ) or (
        bool(model_summary["degenerate_experiment"].fillna(False).astype(bool).any())
        if "degenerate_experiment" in model_summary.columns and not model_summary.empty
        else False
    )
    lines = [
        "# Baseline Models Report",
        "",
        f"- Dataset: `{settings.input.dataset_path}`",
        f"- Classification target: `{settings.target.classification_column}`",
        f"- Regression target: `{settings.target.regression_column}`",
        f"- Feature count after indicators: `{len(feature_columns)}`",
        f"- Time-series tuning mode: `{settings.tuning.mode if settings.tuning.enabled else 'disabled'}`",
        f"- Prediction mode: `{settings.prediction.mode}`",
        f"- Threshold objective: `{settings.threshold_search.objective}`",
        f"- Remaining imputation strategy: `{settings.imputation.remaining_strategy}`",
        f"- Degenerate experiment: `{degenerate_experiment}`",
        f"- Tradeable rate by split: `{_json_dumps(_split_metric_map(label_diagnostics_by_split, 'tradeable_rate'))}`",
        f"- Profitable rate by split: `{_json_dumps(_split_metric_map(label_diagnostics_by_split, 'profitable_rate'))}`",
        f"- Signal count by split is reported per model in the configuration summary.",
        f"- Output directory: `{output_dir}`",
        "",
        "## Validation/Test Leaderboard",
        "",
        _table_to_markdown(leaderboard),
        "",
        "## Label Diagnostics",
        "",
        _table_to_markdown(pd.DataFrame(label_diagnostics_by_split).T.reset_index().rename(columns={"index": "split"})),
        "",
        "## Model Configuration Summary",
        "",
        _table_to_markdown(
            model_summary[[column for column in summary_columns if column in model_summary.columns]]
        ),
        "",
        "## Classification and Rule Baselines",
        "",
        _table_to_markdown(
            metrics[metrics["task"] == "classification"][
                [column for column in classification_columns if column in metrics.columns]
            ]
        ),
        "",
        "## Regression Baselines",
        "",
        _table_to_markdown(
            metrics[metrics["task"] == "regression"][
                [column for column in regression_columns if column in metrics.columns]
            ]
        ),
        "",
        "## Notes",
        "",
        "- Predictive models are tuned only on the train split using chronological inner folds.",
        "- Thresholds are selected on validation using a trading-oriented objective, then applied unchanged to the full prediction table.",
        "- Permutation importance on held-out data is the preferred importance diagnostic for final interpretation.",
        "- Optional walk-forward mode re-fits on chronological history and is intended as a lighter realism upgrade before the dedicated backtest layer.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def run_baseline_pipeline(
    settings: BaselineSettings,
    train_models: bool = True,
) -> BaselineArtifacts:
    """Train or evaluate baseline strategies and predictive models."""

    frame = _load_supervised_dataset(settings)
    raw_feature_columns = select_feature_columns(frame, settings)
    modeling_frame, feature_columns, preprocessing_metadata = _build_modeling_frame(
        frame, raw_feature_columns, settings
    )

    output_dir = _output_dir(settings)
    models_dir = ensure_directory(output_dir / "models")
    diagnostics_dir = ensure_directory(output_dir / "diagnostics")

    feature_columns_path = output_dir / "feature_columns.json"
    feature_columns_path.write_text(
        json.dumps(feature_columns, indent=2), encoding="utf-8"
    )

    prediction_frames: list[pd.DataFrame] = []
    model_paths: dict[str, str] = {}
    diagnostic_paths: dict[str, str] = {}
    model_records: list[dict[str, Any]] = []

    split_column = settings.target.split_column
    train_split = modeling_frame[modeling_frame[split_column] == "train"].copy()
    validation_split = modeling_frame[modeling_frame[split_column] == "validation"].copy()
    validation_available = not validation_split.empty
    label_diagnostics_by_split = _label_diagnostics_for_modeling_frame(modeling_frame, settings)

    for rule in settings.rules:
        if not rule.enabled:
            continue
        selected_rule, threshold_search_frame = _select_rule_spec(
            validation_split, settings, rule
        )
        threshold_search_path = _write_diagnostic_frame(
            diagnostics_dir,
            selected_rule.name,
            "rule_threshold_search",
            threshold_search_frame,
        )
        if threshold_search_path is not None:
            diagnostic_paths[f"{selected_rule.name}:threshold_search"] = threshold_search_path
        rule_predictions = _rule_prediction_frame(modeling_frame, settings, selected_rule)
        if settings.threshold_search.enabled:
            rule_predictions["selected_threshold_objective"] = (
                settings.threshold_search.objective
            )
        prediction_frames.append(rule_predictions)
        rule_diagnostics = _model_diagnostics_payload(
            model_name=selected_rule.name,
            source="baseline",
            settings=settings,
            label_diagnostics_by_split=label_diagnostics_by_split,
            predictions=rule_predictions,
            threshold_search_frame=threshold_search_frame,
        )
        model_records.append(
            {
                "model_name": selected_rule.name,
                "task": "classification",
                "model_family": "rule_based",
                "prediction_mode": "static",
                "calibration_method": "none",
                "feature_importance_method": "not_applicable",
                "selected_threshold": (
                    float(selected_rule.funding_threshold_bps)
                    if selected_rule.kind == "funding_threshold"
                    else (
                        float(selected_rule.spread_threshold)
                        if selected_rule.kind == "spread_zscore_threshold"
                        else 0.0
                    )
                ),
                "threshold_objective": settings.threshold_search.objective
                if settings.threshold_search.enabled
                else None,
                "threshold_objective_value": None,
                "tuning_metric": None,
                "selected_params_json": _json_dumps(
                    {
                        "funding_threshold_bps": selected_rule.funding_threshold_bps,
                        "spread_threshold": selected_rule.spread_threshold,
                        "regime_column": selected_rule.regime_column,
                        "regime_value": selected_rule.regime_value,
                    }
                ),
                **rule_diagnostics,
            }
        )

    classification_models = [settings.predictive.classification] + [
        model
        for model in settings.predictive.classification.additional_models
        if model.enabled
    ]
    regression_models = [settings.predictive.regression] + [
        model
        for model in settings.predictive.regression.additional_models
        if model.enabled
    ]

    if settings.predictive.classification.enabled:
        target_column = settings.target.classification_column
        classifier_train = train_split.dropna(subset=[target_column]).copy()
        for model_spec in classification_models:
            if not model_spec.enabled:
                continue
            model_path = models_dir / f"{model_spec.name}.joblib"
            cv_results_path = diagnostics_dir / f"{model_spec.name}_cv_results.csv"
            threshold_path = diagnostics_dir / f"{model_spec.name}_threshold_search.csv"
            if train_models:
                selected_params, cv_results = _tune_classifier_params(
                    settings,
                    classifier_train,
                    feature_columns,
                    target_column,
                    model_spec,
                )
                cv_results.to_csv(cv_results_path, index=False)
                prediction_model, diagnostic_model, calibration_used = _fit_classifier_with_calibration(
                    settings,
                    classifier_train,
                    feature_columns,
                    target_column,
                    model_spec,
                    selected_params,
                )
                validation_scores = _generate_classifier_scores(
                    validation_split,
                    settings,
                    prediction_model=prediction_model,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    model_name=model_spec.name,
                    model_family="linear",
                    selected_params=selected_params,
                    calibration_method=calibration_used,
                    threshold_objective=settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    model_spec=model_spec,
                )
                threshold_selection = _select_classifier_threshold(
                    validation_scores,
                    settings,
                    model_name=model_spec.name,
                    source="baseline",
                    label_diagnostics_by_split=label_diagnostics_by_split,
                    default_threshold=model_spec.probability_threshold,
                    threshold_grid=model_spec.probability_threshold_grid,
                )
                threshold_selection.search_frame.to_csv(threshold_path, index=False)
                bundle = {
                    "prediction_model": prediction_model,
                    "diagnostic_model": diagnostic_model,
                    "selected_params": selected_params,
                    "selected_threshold": threshold_selection.selected_threshold,
                    "threshold_objective": settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    "threshold_objective_value": threshold_selection.objective_value,
                    "calibration_method": calibration_used,
                    "prediction_mode": settings.prediction.mode,
                    "tuning_metric": settings.tuning.classification_metric
                    if settings.tuning.enabled
                    else None,
                    "fallback_used": threshold_selection.fallback_used,
                    "fallback_reason": threshold_selection.fallback_reason,
                }
                _save_model_bundle(model_path, bundle)
            else:
                bundle = _load_model_bundle(model_path)
                selected_params = dict(bundle.get("selected_params", {}))
                prediction_model = bundle["prediction_model"]
                diagnostic_model = bundle.get("diagnostic_model", prediction_model)
                calibration_used = str(bundle.get("calibration_method", "none"))
                selected_threshold = float(
                    bundle.get("selected_threshold", model_spec.probability_threshold)
                    or model_spec.probability_threshold
                )
                threshold_score = _safe_float(bundle.get("threshold_objective_value"))
                threshold_selection = ThresholdSelectionResult(
                    selected_threshold=float(selected_threshold),
                    objective_value=threshold_score,
                    search_frame=(
                        pd.read_csv(threshold_path)
                        if threshold_path.exists()
                        else pd.DataFrame()
                    ),
                    fallback_used=bool(bundle.get("fallback_used", False)),
                    fallback_reason=bundle.get("fallback_reason"),
                )
            selected_threshold = float(threshold_selection.selected_threshold)
            threshold_score = threshold_selection.objective_value

            model_paths[model_spec.name] = str(model_path)
            if cv_results_path.exists():
                diagnostic_paths[f"{model_spec.name}:cv_results"] = str(cv_results_path)
            if threshold_path.exists():
                diagnostic_paths[f"{model_spec.name}:threshold_search"] = str(threshold_path)

            full_scores = _generate_classifier_scores(
                modeling_frame,
                settings,
                prediction_model=prediction_model,
                feature_columns=feature_columns,
                target_column=target_column,
                model_name=model_spec.name,
                model_family="linear",
                selected_params=selected_params,
                calibration_method=calibration_used,
                threshold_objective=bundle.get("threshold_objective"),
                model_spec=model_spec,
            )
            predictions = _apply_classifier_threshold(full_scores, selected_threshold)
            prediction_frames.append(predictions)
            model_diagnostics = _model_diagnostics_payload(
                model_name=model_spec.name,
                source="baseline",
                settings=settings,
                label_diagnostics_by_split=label_diagnostics_by_split,
                predictions=predictions,
                threshold_search_frame=threshold_selection.search_frame,
                threshold_selection=threshold_selection,
            )

            coefficient_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "coefficients",
                _extract_linear_coefficients(diagnostic_model, feature_columns),
            )
            if coefficient_path is not None:
                diagnostic_paths[f"{model_spec.name}:coefficients"] = coefficient_path
            permutation_source = (
                validation_split
                if validation_available
                else modeling_frame[modeling_frame[split_column] == "test"].copy()
            )
            permutation_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "permutation_importance",
                _permutation_importance_frame(
                    prediction_model,
                    permutation_source.dropna(subset=[target_column]),
                    feature_columns,
                    settings,
                    task="classification",
                ),
            )
            if permutation_path is not None:
                diagnostic_paths[f"{model_spec.name}:permutation"] = permutation_path
            calibration_validation_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "calibration_validation",
                _calibration_table(predictions, split_name="validation"),
            )
            if calibration_validation_path is not None:
                diagnostic_paths[f"{model_spec.name}:calibration_validation"] = calibration_validation_path
            calibration_test_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "calibration_test",
                _calibration_table(predictions, split_name="test"),
            )
            if calibration_test_path is not None:
                diagnostic_paths[f"{model_spec.name}:calibration_test"] = calibration_test_path

            model_records.append(
                {
                    "model_name": model_spec.name,
                    "task": "classification",
                    "model_family": "linear",
                    "prediction_mode": settings.prediction.mode,
                    "calibration_method": calibration_used,
                    "feature_importance_method": "permutation_validation",
                    "selected_threshold": _safe_float(selected_threshold),
                    "threshold_objective": bundle.get("threshold_objective"),
                    "threshold_objective_value": threshold_score,
                    "tuning_metric": bundle.get("tuning_metric"),
                    "selected_params_json": _json_dumps(selected_params),
                    **model_diagnostics,
                }
            )

    if settings.predictive.regression.enabled:
        target_column = settings.target.regression_column
        regressor_train = train_split.dropna(subset=[target_column]).copy()
        for model_spec in regression_models:
            if not model_spec.enabled:
                continue
            model_path = models_dir / f"{model_spec.name}.joblib"
            cv_results_path = diagnostics_dir / f"{model_spec.name}_cv_results.csv"
            threshold_path = diagnostics_dir / f"{model_spec.name}_threshold_search.csv"
            if train_models:
                selected_params, cv_results = _tune_regression_params(
                    settings,
                    regressor_train,
                    feature_columns,
                    target_column,
                    model_spec,
                )
                cv_results.to_csv(cv_results_path, index=False)
                prediction_model = _fit_regressor(
                    settings,
                    regressor_train,
                    feature_columns,
                    target_column,
                    model_spec,
                    selected_params,
                )
                validation_scores = _generate_regression_scores(
                    validation_split,
                    settings,
                    prediction_model=prediction_model,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    model_name=model_spec.name,
                    model_family="linear",
                    selected_params=selected_params,
                    threshold_objective=settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    model_spec=model_spec,
                )
                threshold_selection = _select_regression_threshold(
                    validation_scores,
                    settings,
                    model_name=model_spec.name,
                    source="baseline",
                    label_diagnostics_by_split=label_diagnostics_by_split,
                    default_threshold=model_spec.trade_threshold_bps,
                    threshold_grid=model_spec.trade_threshold_grid_bps,
                )
                threshold_selection.search_frame.to_csv(threshold_path, index=False)
                bundle = {
                    "prediction_model": prediction_model,
                    "diagnostic_model": prediction_model,
                    "selected_params": selected_params,
                    "selected_threshold": threshold_selection.selected_threshold,
                    "threshold_objective": settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    "threshold_objective_value": threshold_selection.objective_value,
                    "calibration_method": "none",
                    "prediction_mode": settings.prediction.mode,
                    "tuning_metric": settings.tuning.regression_metric
                    if settings.tuning.enabled
                    else None,
                    "fallback_used": threshold_selection.fallback_used,
                    "fallback_reason": threshold_selection.fallback_reason,
                }
                _save_model_bundle(model_path, bundle)
            else:
                bundle = _load_model_bundle(model_path)
                selected_params = dict(bundle.get("selected_params", {}))
                prediction_model = bundle["prediction_model"]
                selected_threshold = float(
                    bundle.get("selected_threshold", model_spec.trade_threshold_bps)
                    or model_spec.trade_threshold_bps
                )
                threshold_score = _safe_float(bundle.get("threshold_objective_value"))
                threshold_selection = ThresholdSelectionResult(
                    selected_threshold=float(selected_threshold),
                    objective_value=threshold_score,
                    search_frame=(
                        pd.read_csv(threshold_path)
                        if threshold_path.exists()
                        else pd.DataFrame()
                    ),
                    fallback_used=bool(bundle.get("fallback_used", False)),
                    fallback_reason=bundle.get("fallback_reason"),
                )
            selected_threshold = float(threshold_selection.selected_threshold)
            threshold_score = threshold_selection.objective_value

            model_paths[model_spec.name] = str(model_path)
            if cv_results_path.exists():
                diagnostic_paths[f"{model_spec.name}:cv_results"] = str(cv_results_path)
            if threshold_path.exists():
                diagnostic_paths[f"{model_spec.name}:threshold_search"] = str(threshold_path)

            full_scores = _generate_regression_scores(
                modeling_frame,
                settings,
                prediction_model=prediction_model,
                feature_columns=feature_columns,
                target_column=target_column,
                model_name=model_spec.name,
                model_family="linear",
                selected_params=selected_params,
                threshold_objective=bundle.get("threshold_objective"),
                model_spec=model_spec,
            )
            predictions = _apply_regression_threshold(full_scores, selected_threshold)
            prediction_frames.append(predictions)
            model_diagnostics = _model_diagnostics_payload(
                model_name=model_spec.name,
                source="baseline",
                settings=settings,
                label_diagnostics_by_split=label_diagnostics_by_split,
                predictions=predictions,
                threshold_search_frame=threshold_selection.search_frame,
                threshold_selection=threshold_selection,
            )

            coefficient_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "coefficients",
                _extract_linear_coefficients(prediction_model, feature_columns),
            )
            if coefficient_path is not None:
                diagnostic_paths[f"{model_spec.name}:coefficients"] = coefficient_path
            permutation_source = (
                validation_split
                if validation_available
                else modeling_frame[modeling_frame[split_column] == "test"].copy()
            )
            permutation_path = _write_diagnostic_frame(
                diagnostics_dir,
                model_spec.name,
                "permutation_importance",
                _permutation_importance_frame(
                    prediction_model,
                    permutation_source.dropna(subset=[target_column]),
                    feature_columns,
                    settings,
                    task="regression",
                ),
            )
            if permutation_path is not None:
                diagnostic_paths[f"{model_spec.name}:permutation"] = permutation_path

            model_records.append(
                {
                    "model_name": model_spec.name,
                    "task": "regression",
                    "model_family": "linear",
                    "prediction_mode": settings.prediction.mode,
                    "calibration_method": "none",
                    "feature_importance_method": "permutation_validation",
                    "selected_threshold": _safe_float(selected_threshold),
                    "threshold_objective": bundle.get("threshold_objective"),
                    "threshold_objective_value": threshold_score,
                    "tuning_metric": bundle.get("tuning_metric"),
                    "selected_params_json": _json_dumps(selected_params),
                    **model_diagnostics,
                }
            )

    if settings.predictive.tree.enabled:
        class_target = settings.target.classification_column
        tree_classifier_train = train_split.dropna(subset=[class_target]).copy()
        if tree_classifier_train[class_target].nunique(dropna=True) >= 2:
            classifier_name = settings.predictive.tree.classifier_name
            classifier_path = models_dir / f"{classifier_name}.joblib"
            classifier_cv_path = diagnostics_dir / f"{classifier_name}_cv_results.csv"
            classifier_threshold_path = diagnostics_dir / f"{classifier_name}_threshold_search.csv"
            if train_models:
                selected_params, cv_results = _tune_tree_classifier_params(
                    settings,
                    tree_classifier_train,
                    feature_columns,
                    class_target,
                )
                cv_results.to_csv(classifier_cv_path, index=False)
                prediction_model, diagnostic_model, calibration_used = _fit_classifier_with_calibration(
                    settings,
                    tree_classifier_train,
                    feature_columns,
                    class_target,
                    settings.predictive.classification,
                    selected_params,
                    tree_mode=True,
                )
                validation_scores = _generate_classifier_scores(
                    validation_split,
                    settings,
                    prediction_model=prediction_model,
                    feature_columns=feature_columns,
                    target_column=class_target,
                    model_name=classifier_name,
                    model_family="tree",
                    selected_params=selected_params,
                    calibration_method=calibration_used,
                    threshold_objective=settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    tree_mode=True,
                    model_spec=settings.predictive.classification,
                )
                threshold_selection = _select_classifier_threshold(
                    validation_scores,
                    settings,
                    model_name=classifier_name,
                    source="baseline",
                    label_diagnostics_by_split=label_diagnostics_by_split,
                    default_threshold=settings.predictive.tree.classification_probability_threshold,
                    threshold_grid=settings.predictive.tree.classification_probability_threshold_grid,
                )
                threshold_selection.search_frame.to_csv(classifier_threshold_path, index=False)
                bundle = {
                    "prediction_model": prediction_model,
                    "diagnostic_model": diagnostic_model,
                    "selected_params": selected_params,
                    "selected_threshold": threshold_selection.selected_threshold,
                    "threshold_objective": settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    "threshold_objective_value": threshold_selection.objective_value,
                    "calibration_method": calibration_used,
                    "prediction_mode": settings.prediction.mode,
                    "tuning_metric": settings.tuning.classification_metric
                    if settings.tuning.enabled
                    else None,
                    "fallback_used": threshold_selection.fallback_used,
                    "fallback_reason": threshold_selection.fallback_reason,
                }
                _save_model_bundle(classifier_path, bundle)
            else:
                bundle = _load_model_bundle(classifier_path)
                selected_params = dict(bundle.get("selected_params", {}))
                prediction_model = bundle["prediction_model"]
                diagnostic_model = bundle.get("diagnostic_model", prediction_model)
                calibration_used = str(bundle.get("calibration_method", "none"))
                selected_threshold = float(
                    bundle.get(
                        "selected_threshold",
                        settings.predictive.tree.classification_probability_threshold,
                    )
                    or settings.predictive.tree.classification_probability_threshold
                )
                threshold_score = _safe_float(bundle.get("threshold_objective_value"))
                threshold_selection = ThresholdSelectionResult(
                    selected_threshold=float(selected_threshold),
                    objective_value=threshold_score,
                    search_frame=(
                        pd.read_csv(classifier_threshold_path)
                        if classifier_threshold_path.exists()
                        else pd.DataFrame()
                    ),
                    fallback_used=bool(bundle.get("fallback_used", False)),
                    fallback_reason=bundle.get("fallback_reason"),
                )
            selected_threshold = float(threshold_selection.selected_threshold)
            threshold_score = threshold_selection.objective_value

            model_paths[classifier_name] = str(classifier_path)
            if classifier_cv_path.exists():
                diagnostic_paths[f"{classifier_name}:cv_results"] = str(classifier_cv_path)
            if classifier_threshold_path.exists():
                diagnostic_paths[f"{classifier_name}:threshold_search"] = str(classifier_threshold_path)

            full_scores = _generate_classifier_scores(
                modeling_frame,
                settings,
                prediction_model=prediction_model,
                feature_columns=feature_columns,
                target_column=class_target,
                model_name=classifier_name,
                model_family="tree",
                selected_params=selected_params,
                calibration_method=calibration_used,
                threshold_objective=bundle.get("threshold_objective"),
                tree_mode=True,
                model_spec=settings.predictive.classification,
            )
            predictions = _apply_classifier_threshold(full_scores, selected_threshold)
            prediction_frames.append(predictions)
            model_diagnostics = _model_diagnostics_payload(
                model_name=classifier_name,
                source="baseline",
                settings=settings,
                label_diagnostics_by_split=label_diagnostics_by_split,
                predictions=predictions,
                threshold_search_frame=threshold_selection.search_frame,
                threshold_selection=threshold_selection,
            )

            impurity_path = _write_diagnostic_frame(
                diagnostics_dir,
                classifier_name,
                "impurity_importance",
                _extract_tree_importances(diagnostic_model, feature_columns),
            )
            if impurity_path is not None:
                diagnostic_paths[f"{classifier_name}:impurity"] = impurity_path
            permutation_source = (
                validation_split
                if validation_available
                else modeling_frame[modeling_frame[split_column] == "test"].copy()
            )
            permutation_path = _write_diagnostic_frame(
                diagnostics_dir,
                classifier_name,
                "permutation_importance",
                _permutation_importance_frame(
                    prediction_model,
                    permutation_source.dropna(subset=[class_target]),
                    feature_columns,
                    settings,
                    task="classification",
                ),
            )
            if permutation_path is not None:
                diagnostic_paths[f"{classifier_name}:permutation"] = permutation_path
            calibration_validation_path = _write_diagnostic_frame(
                diagnostics_dir,
                classifier_name,
                "calibration_validation",
                _calibration_table(predictions, split_name="validation"),
            )
            if calibration_validation_path is not None:
                diagnostic_paths[f"{classifier_name}:calibration_validation"] = calibration_validation_path
            calibration_test_path = _write_diagnostic_frame(
                diagnostics_dir,
                classifier_name,
                "calibration_test",
                _calibration_table(predictions, split_name="test"),
            )
            if calibration_test_path is not None:
                diagnostic_paths[f"{classifier_name}:calibration_test"] = calibration_test_path

            model_records.append(
                {
                    "model_name": classifier_name,
                    "task": "classification",
                    "model_family": "tree",
                    "prediction_mode": settings.prediction.mode,
                    "calibration_method": calibration_used,
                    "feature_importance_method": "permutation_validation",
                    "selected_threshold": _safe_float(selected_threshold),
                    "threshold_objective": bundle.get("threshold_objective"),
                    "threshold_objective_value": threshold_score,
                    "tuning_metric": bundle.get("tuning_metric"),
                    "selected_params_json": _json_dumps(selected_params),
                    **model_diagnostics,
                }
            )

        reg_target = settings.target.regression_column
        tree_regressor_train = train_split.dropna(subset=[reg_target]).copy()
        if not tree_regressor_train.empty:
            regressor_name = settings.predictive.tree.regressor_name
            regressor_path = models_dir / f"{regressor_name}.joblib"
            regressor_cv_path = diagnostics_dir / f"{regressor_name}_cv_results.csv"
            regressor_threshold_path = diagnostics_dir / f"{regressor_name}_threshold_search.csv"
            if train_models:
                selected_params, cv_results = _tune_tree_regressor_params(
                    settings,
                    tree_regressor_train,
                    feature_columns,
                    reg_target,
                )
                cv_results.to_csv(regressor_cv_path, index=False)
                prediction_model = _fit_regressor(
                    settings,
                    tree_regressor_train,
                    feature_columns,
                    reg_target,
                    settings.predictive.regression,
                    selected_params,
                    tree_mode=True,
                )
                validation_scores = _generate_regression_scores(
                    validation_split,
                    settings,
                    prediction_model=prediction_model,
                    feature_columns=feature_columns,
                    target_column=reg_target,
                    model_name=regressor_name,
                    model_family="tree",
                    selected_params=selected_params,
                    threshold_objective=settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    tree_mode=True,
                    model_spec=settings.predictive.regression,
                )
                threshold_selection = _select_regression_threshold(
                    validation_scores,
                    settings,
                    model_name=regressor_name,
                    source="baseline",
                    label_diagnostics_by_split=label_diagnostics_by_split,
                    default_threshold=settings.predictive.tree.regression_trade_threshold_bps,
                    threshold_grid=settings.predictive.tree.regression_trade_threshold_grid_bps,
                )
                threshold_selection.search_frame.to_csv(regressor_threshold_path, index=False)
                bundle = {
                    "prediction_model": prediction_model,
                    "diagnostic_model": prediction_model,
                    "selected_params": selected_params,
                    "selected_threshold": threshold_selection.selected_threshold,
                    "threshold_objective": settings.threshold_search.objective
                    if settings.threshold_search.enabled
                    else None,
                    "threshold_objective_value": threshold_selection.objective_value,
                    "calibration_method": "none",
                    "prediction_mode": settings.prediction.mode,
                    "tuning_metric": settings.tuning.regression_metric
                    if settings.tuning.enabled
                    else None,
                    "fallback_used": threshold_selection.fallback_used,
                    "fallback_reason": threshold_selection.fallback_reason,
                }
                _save_model_bundle(regressor_path, bundle)
            else:
                bundle = _load_model_bundle(regressor_path)
                selected_params = dict(bundle.get("selected_params", {}))
                prediction_model = bundle["prediction_model"]
                selected_threshold = float(
                    bundle.get(
                        "selected_threshold",
                        settings.predictive.tree.regression_trade_threshold_bps,
                    )
                    or settings.predictive.tree.regression_trade_threshold_bps
                )
                threshold_score = _safe_float(bundle.get("threshold_objective_value"))
                threshold_selection = ThresholdSelectionResult(
                    selected_threshold=float(selected_threshold),
                    objective_value=threshold_score,
                    search_frame=(
                        pd.read_csv(regressor_threshold_path)
                        if regressor_threshold_path.exists()
                        else pd.DataFrame()
                    ),
                    fallback_used=bool(bundle.get("fallback_used", False)),
                    fallback_reason=bundle.get("fallback_reason"),
                )
            selected_threshold = float(threshold_selection.selected_threshold)
            threshold_score = threshold_selection.objective_value

            model_paths[regressor_name] = str(regressor_path)
            if regressor_cv_path.exists():
                diagnostic_paths[f"{regressor_name}:cv_results"] = str(regressor_cv_path)
            if regressor_threshold_path.exists():
                diagnostic_paths[f"{regressor_name}:threshold_search"] = str(regressor_threshold_path)

            full_scores = _generate_regression_scores(
                modeling_frame,
                settings,
                prediction_model=prediction_model,
                feature_columns=feature_columns,
                target_column=reg_target,
                model_name=regressor_name,
                model_family="tree",
                selected_params=selected_params,
                threshold_objective=bundle.get("threshold_objective"),
                tree_mode=True,
                model_spec=settings.predictive.regression,
            )
            predictions = _apply_regression_threshold(full_scores, selected_threshold)
            prediction_frames.append(predictions)
            model_diagnostics = _model_diagnostics_payload(
                model_name=regressor_name,
                source="baseline",
                settings=settings,
                label_diagnostics_by_split=label_diagnostics_by_split,
                predictions=predictions,
                threshold_search_frame=threshold_selection.search_frame,
                threshold_selection=threshold_selection,
            )

            impurity_path = _write_diagnostic_frame(
                diagnostics_dir,
                regressor_name,
                "impurity_importance",
                _extract_tree_importances(prediction_model, feature_columns),
            )
            if impurity_path is not None:
                diagnostic_paths[f"{regressor_name}:impurity"] = impurity_path
            permutation_source = (
                validation_split
                if validation_available
                else modeling_frame[modeling_frame[split_column] == "test"].copy()
            )
            permutation_path = _write_diagnostic_frame(
                diagnostics_dir,
                regressor_name,
                "permutation_importance",
                _permutation_importance_frame(
                    prediction_model,
                    permutation_source.dropna(subset=[reg_target]),
                    feature_columns,
                    settings,
                    task="regression",
                ),
            )
            if permutation_path is not None:
                diagnostic_paths[f"{regressor_name}:permutation"] = permutation_path

            model_records.append(
                {
                    "model_name": regressor_name,
                    "task": "regression",
                    "model_family": "tree",
                    "prediction_mode": settings.prediction.mode,
                    "calibration_method": "none",
                    "feature_importance_method": "permutation_validation",
                    "selected_threshold": _safe_float(selected_threshold),
                    "threshold_objective": bundle.get("threshold_objective"),
                    "threshold_objective_value": threshold_score,
                    "tuning_metric": bundle.get("tuning_metric"),
                    "selected_params_json": _json_dumps(selected_params),
                    **model_diagnostics,
                }
            )

    if not prediction_frames:
        raise ValueError("No baseline models were enabled or trainable under the current config.")

    predictions = pd.concat(prediction_frames, ignore_index=True, sort=False)
    predictions = predictions.sort_values(["model_name", "timestamp"]).reset_index(drop=True)
    metrics = evaluate_prediction_table(
        predictions, top_quantile=settings.threshold_search.top_quantile
    )
    leaderboard = _build_leaderboard(metrics)
    model_summary = _build_model_report_summary(model_records)

    predictions_path = output_dir / "baseline_predictions.parquet"
    metrics_path = output_dir / "baseline_metrics.parquet"
    leaderboard_path = output_dir / "baseline_leaderboard.parquet"

    predictions_primary_path = _write_frame(predictions, predictions_path)
    metrics_primary_path = _write_frame(metrics, metrics_path)
    leaderboard_primary_path = _write_frame(leaderboard, leaderboard_path)

    predictions_csv_path: str | None = None
    metrics_csv_path: str | None = None
    leaderboard_csv_path: str | None = None
    if settings.output.write_csv:
        predictions_csv_path = _write_frame(predictions, predictions_path.with_suffix(".csv"))
        metrics_csv_path = _write_frame(metrics, metrics_path.with_suffix(".csv"))
        leaderboard_csv_path = _write_frame(leaderboard, leaderboard_path.with_suffix(".csv"))

    report_path = _write_markdown_report(
        settings,
        feature_columns,
        metrics,
        leaderboard,
        model_summary,
        label_diagnostics_by_split,
        output_dir,
    )

    split_counts = {
        split_name: int((frame[split_column] == split_name).sum())
        for split_name in ["train", "validation", "test"]
    }
    degenerate_models = [
        record["model_name"]
        for record in model_records
        if bool(record.get("degenerate_experiment"))
    ]
    fallback_models = [
        record["model_name"]
        for record in model_records
        if bool(record.get("fallback_used"))
    ]
    label_degenerate_reasons = [
        f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
        for split_name, diagnostics in label_diagnostics_by_split.items()
        if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
    ]
    manifest = {
        "input": settings.input.model_dump(),
        "target": settings.target.model_dump(),
        "feature_selection": settings.feature_selection.model_dump(),
        "tuning": settings.tuning.model_dump(),
        "threshold_search": settings.threshold_search.model_dump(),
        "imputation": settings.imputation.model_dump(),
        "prediction": settings.prediction.model_dump(),
        "rules": [rule.model_dump() for rule in settings.rules if rule.enabled],
        "predictive": settings.predictive.model_dump(),
        "train_models": train_models,
        "row_count": int(len(frame)),
        "split_counts": split_counts,
        "feature_count": len(feature_columns),
        "label_diagnostics_by_split": label_diagnostics_by_split,
        "tradeable_rate_by_split": _split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        "profitable_rate_by_split": _split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        "degenerate_experiment": bool(label_degenerate_reasons or degenerate_models),
        "status": (
            "ok"
            if not (label_degenerate_reasons or degenerate_models)
            else "warning"
        ),
        "degenerate_stage": (
            "labels"
            if label_degenerate_reasons
            else next(
                (
                    record.get("degenerate_stage")
                    for record in model_records
                    if record.get("degenerate_stage")
                ),
                None,
            )
        ),
        "degenerate_reason": (
            "; ".join(label_degenerate_reasons)
            if label_degenerate_reasons
            else next(
                (
                    record.get("reason")
                    for record in model_records
                    if record.get("reason")
                ),
                None,
            )
        ),
        "reason": (
            "; ".join(label_degenerate_reasons)
            if label_degenerate_reasons
            else next(
                (
                    record.get("reason")
                    for record in model_records
                    if record.get("reason")
                ),
                None,
            )
        ),
        "fallback_used": bool(fallback_models),
        "fallback_reason": next(
            (
                record.get("fallback_reason")
                for record in model_records
                if record.get("fallback_reason")
            ),
            None,
        ),
        "degenerate_models": degenerate_models,
        "fallback_models": fallback_models,
        "feature_columns_path": str(feature_columns_path),
        "preprocessing": preprocessing_metadata,
        "predictions_path": predictions_primary_path,
        "predictions_csv_path": predictions_csv_path,
        "metrics_path": metrics_primary_path,
        "metrics_csv_path": metrics_csv_path,
        "leaderboard_path": leaderboard_primary_path,
        "leaderboard_csv_path": leaderboard_csv_path,
        "report_path": report_path,
        "model_paths": model_paths,
        "diagnostic_paths": diagnostic_paths,
        "model_summary": _to_jsonable(model_records),
    }
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return BaselineArtifacts(
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        metrics_path=metrics_primary_path,
        metrics_csv_path=metrics_csv_path,
        leaderboard_path=leaderboard_primary_path,
        leaderboard_csv_path=leaderboard_csv_path,
        predictions_path=predictions_primary_path,
        predictions_csv_path=predictions_csv_path,
        report_path=report_path,
        feature_columns_path=str(feature_columns_path),
        model_paths=model_paths,
        diagnostic_paths=diagnostic_paths,
    )
