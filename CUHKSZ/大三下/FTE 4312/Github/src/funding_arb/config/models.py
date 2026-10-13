"""Typed config models for the Python-side pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SettingsBase(BaseModel):
    """Base settings model with permissive extra handling for easy extension."""

    model_config = ConfigDict(extra="allow")


class DataProviderSettings(SettingsBase):
    provider: str = "binance"
    timeout_seconds: int = 30
    limit_per_request: int = 1000


class DataDatasetSettings(SettingsBase):
    symbol: str
    venue: str
    reference_source: str
    frequency: str
    start: str
    end: str
    perpetual_symbol: str | None = None
    spot_symbol: str | None = None


class DataSourceSettings(SettingsBase):
    enabled: bool = True
    endpoint: str


class DataCleaningSettings(SettingsBase):
    timezone: str = "UTC"
    drop_duplicates: bool = True
    sort_ascending: bool = True
    max_forward_fill_hours: int = 6
    fill_price_method: str = "ffill"
    fill_volume_value: float = 0.0
    fill_funding_value: float = 0.0
    fill_open_interest_method: str = "ffill"
    validate_frequency: bool = True


class DataOutputSettings(SettingsBase):
    format: str = "parquet"
    write_csv: bool = True
    raw_subdir: str = "data/raw"
    interim_subdir: str = "data/interim"
    processed_subdir: str = "data/processed"


class DataSettings(SettingsBase):
    source: DataProviderSettings = Field(default_factory=DataProviderSettings)
    dataset: DataDatasetSettings
    sources: dict[str, DataSourceSettings]
    cleaning: DataCleaningSettings = Field(default_factory=DataCleaningSettings)
    output: DataOutputSettings = Field(default_factory=DataOutputSettings)
    notes: dict[str, Any] = Field(default_factory=dict)


class FeatureInputSettings(SettingsBase):
    dataset_path: str = "data/processed/binance/btcusdt/1h/hourly_market_data.parquet"
    manifest_path: str | None = "data/processed/binance/btcusdt/1h/manifest.json"
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class FeatureSetSettings(SettingsBase):
    rolling_windows: list[int] = Field(default_factory=list)
    volatility_window: int
    zscore_window: int
    funding_mean_window: int
    basis_mean_window: int
    shock_window: int = 24
    liquidity_window: int = 24
    regime_window: int = 168
    funding_interval_hours: int = 8
    annualization_factor_hours: int = 24 * 365


class FeatureLabelSettings(SettingsBase):
    forward_horizon_hours: int
    min_expected_edge_bps: float
    use_post_cost_target: bool = True


class FeatureOutputSettings(SettingsBase):
    processed_dir: str = "data/processed/features"
    artifact_name: str = "feature_set.parquet"
    manifest_name: str = "feature_manifest.json"
    write_csv: bool = True


class FeatureSettings(SettingsBase):
    input: FeatureInputSettings = Field(default_factory=FeatureInputSettings)
    feature_set: FeatureSetSettings
    labels: FeatureLabelSettings
    output: FeatureOutputSettings


class ModelSplitSettings(SettingsBase):
    train_end: str
    validation_end: str
    test_end: str


class LabelInputSettings(SettingsBase):
    feature_table_path: str = (
        "data/processed/features/binance/btcusdt/1h/btcusdt_feature_set.parquet"
    )
    feature_manifest_path: str | None = (
        "data/processed/features/binance/btcusdt/1h/btcusdt_feature_manifest.json"
    )
    market_dataset_path: str = (
        "data/processed/binance/btcusdt/1h/hourly_market_data.parquet"
    )
    market_manifest_path: str | None = "data/processed/binance/btcusdt/1h/manifest.json"
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class LabelTargetSettings(SettingsBase):
    direction: str = "short_perp_long_spot"
    holding_windows_hours: list[int] = Field(default_factory=lambda: [8])
    primary_horizon_hours: int = 8
    execution_delay_bars: int = 1
    execution_price_field: str = "open"
    min_expected_edge_bps: float = 5.0
    positive_return_threshold_bps: float = 0.0
    use_post_cost_target: bool = True


class LabelCostSettings(SettingsBase):
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    slippage_bps: float = 3.0
    gas_cost_usd: float = 2.0
    position_notional_usd: float = 10000.0
    other_friction_bps: float = 0.0
    borrow_cost_bps_per_hour: float = 0.0


class LabelOutputSettings(SettingsBase):
    output_dir: str = "data/processed/supervised"
    artifact_name: str = "supervised_dataset.parquet"
    label_table_name: str = "label_table.parquet"
    manifest_name: str = "supervised_manifest.json"
    write_csv: bool = True
    save_split_files: bool = True


class LabelPipelineSettings(SettingsBase):
    input: LabelInputSettings = Field(default_factory=LabelInputSettings)
    target: LabelTargetSettings = Field(default_factory=LabelTargetSettings)
    costs: LabelCostSettings = Field(default_factory=LabelCostSettings)
    split: ModelSplitSettings
    output: LabelOutputSettings = Field(default_factory=LabelOutputSettings)
    notes: dict[str, Any] = Field(default_factory=dict)


class ExploratoryDLDatasetInputSettings(SettingsBase):
    source_dataset_path: str = (
        "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_dataset.parquet"
    )
    source_manifest_path: str | None = (
        "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_manifest.json"
    )
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class ExploratoryDLDatasetTargetSettings(SettingsBase):
    horizon_hours: int = 24
    gross_return_column: str = "target_future_gross_return_bps_24h"
    split_column: str = "split"
    ready_column: str = "supervised_ready"
    short_direction_label: str = "short_perp_long_spot"
    long_direction_label: str = "long_perp_short_spot"
    short_direction_gross_column: str = (
        "target_future_short_perp_long_spot_gross_return_bps_24h"
    )
    long_direction_gross_column: str = (
        "target_future_long_perp_short_spot_gross_return_bps_24h"
    )
    signed_opportunity_column: str = "target_future_signed_opportunity_bps_24h"
    absolute_opportunity_column: str = "target_future_absolute_opportunity_bps_24h"
    direction_classification_column: str = (
        "target_best_direction_is_short_perp_long_spot_24h"
    )
    direction_label_column: str = "target_best_direction_label_24h"


class ExploratoryDLDatasetOutputSettings(SettingsBase):
    output_dir: str = "data/processed/exploratory_dl"
    artifact_name: str = "exploratory_dataset.parquet"
    manifest_name: str = "exploratory_manifest.json"
    write_csv: bool = True


class ExploratoryDLDatasetSettings(SettingsBase):
    input: ExploratoryDLDatasetInputSettings = Field(
        default_factory=ExploratoryDLDatasetInputSettings
    )
    target: ExploratoryDLDatasetTargetSettings = Field(
        default_factory=ExploratoryDLDatasetTargetSettings
    )
    output: ExploratoryDLDatasetOutputSettings = Field(
        default_factory=ExploratoryDLDatasetOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class BaselineInputSettings(SettingsBase):
    dataset_path: str = (
        "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_dataset.parquet"
    )
    manifest_path: str | None = (
        "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_manifest.json"
    )
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class BaselineTargetSettings(SettingsBase):
    timestamp_column: str = "timestamp"
    split_column: str = "split"
    ready_column: str = "supervised_ready"
    classification_column: str = "target_is_profitable_24h"
    regression_column: str = "target_future_net_return_bps_24h"


class BaselineFeatureSelectionSettings(SettingsBase):
    include_columns: list[str] = Field(default_factory=list)
    exclude_columns: list[str] = Field(
        default_factory=lambda: [
            "timestamp",
            "symbol",
            "venue",
            "frequency",
            "split",
            "feature_ready",
            "supervised_ready",
        ]
    )
    exclude_prefixes: list[str] = Field(default_factory=lambda: ["target_"])
    max_missing_fraction: float = 0.25
    drop_constant_features: bool = True


class BaselineTimeSeriesCVSettings(SettingsBase):
    enabled: bool = True
    n_splits: int = 4
    gap: int = 0
    mode: str = "expanding"
    min_train_size: int = 2000
    rolling_window_size: int | None = None
    classification_metric: str = "average_precision"
    regression_metric: str = "neg_rmse"

    @model_validator(mode="after")
    def validate_time_series_cv(self) -> "BaselineTimeSeriesCVSettings":
        valid_modes = {"expanding", "rolling"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Baseline tuning mode must be one of {sorted(valid_modes)}, got '{self.mode}'."
            )
        if self.n_splits < 2:
            raise ValueError("Baseline tuning requires at least 2 splits.")
        if self.gap < 0:
            raise ValueError("Baseline tuning gap must be non-negative.")
        if self.min_train_size <= 0:
            raise ValueError("Baseline tuning min_train_size must be positive.")
        if self.mode == "rolling" and self.rolling_window_size is None:
            raise ValueError(
                "Baseline tuning rolling mode requires rolling_window_size."
            )
        return self


class BaselineThresholdSearchSettings(SettingsBase):
    enabled: bool = True
    objective: str = "avg_signal_return_bps"
    probability_grid: list[float] = Field(default_factory=list)
    regression_threshold_grid_bps: list[float] = Field(default_factory=list)
    top_quantile: float = 0.1
    rule_search_enabled: bool = True
    allow_degenerate_fallback: bool = False

    @model_validator(mode="after")
    def validate_threshold_search(self) -> "BaselineThresholdSearchSettings":
        if not 0.0 < self.top_quantile <= 0.5:
            raise ValueError("Baseline threshold_search.top_quantile must be in (0, 0.5].")
        return self


class BaselineImputationSettings(SettingsBase):
    remaining_strategy: str = "median"
    forward_fill_columns: list[str] = Field(default_factory=list)
    forward_fill_prefixes: list[str] = Field(default_factory=list)
    add_missing_indicators: bool = True
    indicator_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_imputation(self) -> "BaselineImputationSettings":
        valid_strategies = {"median"}
        if self.remaining_strategy not in valid_strategies:
            raise ValueError(
                "Baseline imputation remaining_strategy currently supports only 'median'."
            )
        return self


class BaselineWalkForwardSettings(SettingsBase):
    mode: str = "static"
    refit_every_n_periods: int = 168
    rolling_window_size: int | None = None
    expanding_window_start: int = 2000
    use_validation_history_for_test: bool = True

    @model_validator(mode="after")
    def validate_walk_forward(self) -> "BaselineWalkForwardSettings":
        valid_modes = {"static", "expanding", "rolling"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Baseline walk-forward mode must be one of {sorted(valid_modes)}, got '{self.mode}'."
            )
        if self.refit_every_n_periods <= 0:
            raise ValueError("Baseline walk-forward refit_every_n_periods must be positive.")
        if self.expanding_window_start <= 0:
            raise ValueError("Baseline walk-forward expanding_window_start must be positive.")
        if self.mode == "rolling" and self.rolling_window_size is None:
            raise ValueError(
                "Baseline walk-forward rolling mode requires rolling_window_size."
            )
        return self


class RuleBaselineSpec(SettingsBase):
    name: str
    kind: str
    enabled: bool = True
    funding_column: str = "funding_rate_bps"
    funding_threshold_bps: float = 0.0
    spread_column: str = "spread_zscore_72h"
    spread_threshold: float = 0.0
    regime_column: str | None = None
    regime_value: float | int | None = 1
    funding_threshold_grid_bps: list[float] = Field(default_factory=list)
    spread_threshold_grid: list[float] = Field(default_factory=list)


class ClassificationModelVariantSettings(SettingsBase):
    enabled: bool = True
    name: str = "logistic_regression"
    estimator: str = "logistic_regression"
    probability_threshold: float = 0.5
    probability_threshold_grid: list[float] = Field(default_factory=list)
    standardize: bool = True
    max_iter: int = 1500
    c: float = 1.0
    penalty: str = "l2"
    solver: str | None = None
    l1_ratio: float | None = None
    class_weight: str | None = "balanced"
    random_state: int = 42
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    calibration_method: str = "none"
    calibration_cv_splits: int = 3
    calibration_ensemble: bool = True

    @model_validator(mode="after")
    def validate_classification_model(self) -> "ClassificationModelVariantSettings":
        valid_estimators = {
            "logistic_regression",
            "logistic_l1",
            "logistic_elastic_net",
        }
        if self.estimator not in valid_estimators:
            raise ValueError(
                f"Unsupported classification estimator '{self.estimator}'. Expected one of {sorted(valid_estimators)}."
            )
        valid_calibration = {"none", "sigmoid", "isotonic"}
        if self.calibration_method not in valid_calibration:
            raise ValueError(
                f"Unsupported calibration_method '{self.calibration_method}'. Expected one of {sorted(valid_calibration)}."
            )
        return self


class ClassificationBaselineSettings(ClassificationModelVariantSettings):
    additional_models: list[ClassificationModelVariantSettings] = Field(
        default_factory=list
    )


class RegressionModelVariantSettings(SettingsBase):
    enabled: bool = True
    name: str = "ridge_regression"
    estimator: str = "ridge"
    standardize: bool = True
    alpha: float = 1.0
    l1_ratio: float = 0.5
    trade_threshold_bps: float = 0.0
    trade_threshold_grid_bps: list[float] = Field(default_factory=list)
    random_state: int = 42
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_regression_model(self) -> "RegressionModelVariantSettings":
        valid_estimators = {"ridge", "elastic_net"}
        if self.estimator not in valid_estimators:
            raise ValueError(
                f"Unsupported regression estimator '{self.estimator}'. Expected one of {sorted(valid_estimators)}."
            )
        return self


class RegressionBaselineSettings(RegressionModelVariantSettings):
    additional_models: list[RegressionModelVariantSettings] = Field(
        default_factory=list
    )


class TreeBaselineSettings(SettingsBase):
    enabled: bool = False
    classifier_name: str = "random_forest_classifier"
    regressor_name: str = "random_forest_regressor"
    n_estimators: int = 200
    max_depth: int | None = 6
    min_samples_leaf: int = 50
    classification_probability_threshold: float = 0.5
    classification_probability_threshold_grid: list[float] = Field(default_factory=list)
    regression_trade_threshold_bps: float = 0.0
    regression_trade_threshold_grid_bps: list[float] = Field(default_factory=list)
    random_state: int = 42
    classifier_param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    regressor_param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    calibration_method: str = "none"
    calibration_cv_splits: int = 3
    calibration_ensemble: bool = True


class BaselinePredictiveSettings(SettingsBase):
    classification: ClassificationBaselineSettings = Field(
        default_factory=ClassificationBaselineSettings
    )
    regression: RegressionBaselineSettings = Field(
        default_factory=RegressionBaselineSettings
    )
    tree: TreeBaselineSettings = Field(default_factory=TreeBaselineSettings)


class BaselineOutputSettings(SettingsBase):
    model_dir: str = "data/artifacts/models/baselines"
    run_name: str = "default"
    write_csv: bool = True
    write_markdown_report: bool = True


class BaselineSettings(SettingsBase):
    input: BaselineInputSettings = Field(default_factory=BaselineInputSettings)
    target: BaselineTargetSettings = Field(default_factory=BaselineTargetSettings)
    feature_selection: BaselineFeatureSelectionSettings = Field(
        default_factory=BaselineFeatureSelectionSettings
    )
    tuning: BaselineTimeSeriesCVSettings = Field(
        default_factory=BaselineTimeSeriesCVSettings
    )
    threshold_search: BaselineThresholdSearchSettings = Field(
        default_factory=BaselineThresholdSearchSettings
    )
    imputation: BaselineImputationSettings = Field(
        default_factory=BaselineImputationSettings
    )
    prediction: BaselineWalkForwardSettings = Field(
        default_factory=BaselineWalkForwardSettings
    )
    rules: list[RuleBaselineSpec] = Field(default_factory=list)
    predictive: BaselinePredictiveSettings = Field(
        default_factory=BaselinePredictiveSettings
    )
    output: BaselineOutputSettings = Field(default_factory=BaselineOutputSettings)
    notes: dict[str, Any] = Field(default_factory=dict)


class ModelTrainingSettings(SettingsBase):
    split: ModelSplitSettings | None = None
    batch_size: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    seed: int | None = None


class ModelOutputSettings(SettingsBase):
    model_dir: str = "data/artifacts/models"


class DeepLearningTargetSettings(SettingsBase):
    task: str = "regression"
    column: str = "target_future_net_return_bps_24h"
    classification_column: str = "target_is_profitable_24h"
    regression_column: str = "target_future_net_return_bps_24h"
    timestamp_column: str = "timestamp"
    split_column: str = "split"
    ready_column: str = "supervised_ready"
    probability_threshold: float = 0.5
    trade_threshold_bps: float = 0.0


class SequenceSettings(SettingsBase):
    lookback_steps: int = 48
    allow_cross_split_context: bool = True


class DeepLearningModelSettings(SettingsBase):
    name: str = "lstm"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    tcn_hidden_channels: int = 64
    tcn_num_blocks: int = 4
    tcn_kernel_size: int = 3
    tcn_dilation_base: int = 2
    tcn_use_residual: bool = True
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 128
    transformer_pooling: str = "last"

    @model_validator(mode="after")
    def validate_model_name(self) -> "DeepLearningModelSettings":
        valid_names = {"lstm", "gru", "tcn", "transformer_encoder"}
        if self.name not in valid_names:
            raise ValueError(
                f"Deep-learning model.name must be one of {sorted(valid_names)}, got '{self.name}'."
            )
        if self.hidden_size <= 0:
            raise ValueError("Deep-learning hidden_size must be positive.")
        if self.num_layers <= 0:
            raise ValueError("Deep-learning num_layers must be positive.")
        if self.tcn_hidden_channels <= 0:
            raise ValueError("Deep-learning tcn_hidden_channels must be positive.")
        if self.tcn_num_blocks <= 0:
            raise ValueError("Deep-learning tcn_num_blocks must be positive.")
        if self.tcn_kernel_size <= 1:
            raise ValueError("Deep-learning tcn_kernel_size must be greater than 1.")
        if self.tcn_dilation_base <= 0:
            raise ValueError("Deep-learning tcn_dilation_base must be positive.")
        if self.transformer_d_model <= 0:
            raise ValueError("Deep-learning transformer_d_model must be positive.")
        if self.transformer_nhead <= 0:
            raise ValueError("Deep-learning transformer_nhead must be positive.")
        if self.transformer_num_layers <= 0:
            raise ValueError("Deep-learning transformer_num_layers must be positive.")
        if self.transformer_dim_feedforward <= 0:
            raise ValueError("Deep-learning transformer_dim_feedforward must be positive.")
        if self.transformer_d_model % self.transformer_nhead != 0:
            raise ValueError(
                "Deep-learning transformer_d_model must be divisible by transformer_nhead."
            )
        valid_pooling = {"last", "mean"}
        if self.transformer_pooling not in valid_pooling:
            raise ValueError(
                f"Deep-learning transformer_pooling must be one of {sorted(valid_pooling)}, got '{self.transformer_pooling}'."
            )
        return self


class DeepLearningPreprocessingSettings(SettingsBase):
    scaler: str = "standard"
    winsorize_lower_quantile: float | None = None
    winsorize_upper_quantile: float | None = None

    @model_validator(mode="after")
    def validate_preprocessing(self) -> "DeepLearningPreprocessingSettings":
        valid_scalers = {"standard", "robust"}
        if self.scaler not in valid_scalers:
            raise ValueError(
                f"Deep-learning preprocessing.scaler must be one of {sorted(valid_scalers)}, got '{self.scaler}'."
            )
        if self.winsorize_lower_quantile is not None and not 0.0 <= self.winsorize_lower_quantile < 0.5:
            raise ValueError(
                "Deep-learning winsorize_lower_quantile must be in [0.0, 0.5)."
            )
        if self.winsorize_upper_quantile is not None and not 0.5 < self.winsorize_upper_quantile <= 1.0:
            raise ValueError(
                "Deep-learning winsorize_upper_quantile must be in (0.5, 1.0]."
            )
        if (
            self.winsorize_lower_quantile is not None
            and self.winsorize_upper_quantile is not None
            and self.winsorize_lower_quantile >= self.winsorize_upper_quantile
        ):
            raise ValueError(
                "Deep-learning winsorize_lower_quantile must be smaller than winsorize_upper_quantile."
            )
        return self


class DeepLearningThresholdSearchSettings(SettingsBase):
    enabled: bool = True
    objective: str = "avg_signal_return_bps"
    probability_grid: list[float] = Field(default_factory=list)
    regression_threshold_grid_bps: list[float] = Field(default_factory=list)
    top_quantile: float = 0.1
    allow_degenerate_fallback: bool = False

    @model_validator(mode="after")
    def validate_threshold_search(self) -> "DeepLearningThresholdSearchSettings":
        if not 0.0 < self.top_quantile <= 0.5:
            raise ValueError(
                "Deep-learning threshold_search.top_quantile must be in (0, 0.5]."
            )
        return self


class DeepLearningTuningSettings(SettingsBase):
    enabled: bool = False
    mode: str = "expanding"
    n_splits: int = 2
    gap: int = 0
    min_train_size: int = 2000
    rolling_window_size: int | None = None
    max_candidates: int = 8
    trial_epochs: int = 3
    metric: str = "validation_avg_signal_return_bps"
    lookback_steps: list[int] = Field(default_factory=list)
    hidden_size: list[int] = Field(default_factory=list)
    num_layers: list[int] = Field(default_factory=list)
    dropout: list[float] = Field(default_factory=list)
    learning_rate: list[float] = Field(default_factory=list)
    weight_decay: list[float] = Field(default_factory=list)
    batch_size: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tuning(self) -> "DeepLearningTuningSettings":
        valid_modes = {"expanding", "rolling"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Deep-learning tuning mode must be one of {sorted(valid_modes)}, got '{self.mode}'."
            )
        if self.n_splits < 2:
            raise ValueError("Deep-learning tuning requires at least 2 splits when enabled.")
        if self.gap < 0:
            raise ValueError("Deep-learning tuning gap must be non-negative.")
        if self.min_train_size <= 0:
            raise ValueError("Deep-learning tuning min_train_size must be positive.")
        if self.max_candidates <= 0:
            raise ValueError("Deep-learning tuning max_candidates must be positive.")
        if self.trial_epochs <= 0:
            raise ValueError("Deep-learning tuning trial_epochs must be positive.")
        if self.mode == "rolling" and self.rolling_window_size is None:
            raise ValueError(
                "Deep-learning tuning rolling mode requires rolling_window_size."
            )
        return self


class DeepLearningPredictionSettings(SettingsBase):
    mode: str = "static"
    refit_every_n_periods: int = 168
    rolling_window_size: int | None = None
    expanding_window_start: int = 2000
    use_validation_history_for_test: bool = True

    @model_validator(mode="after")
    def validate_prediction(self) -> "DeepLearningPredictionSettings":
        valid_modes = {"static", "expanding", "rolling"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Deep-learning prediction.mode must be one of {sorted(valid_modes)}, got '{self.mode}'."
            )
        if self.refit_every_n_periods <= 0:
            raise ValueError(
                "Deep-learning prediction.refit_every_n_periods must be positive."
            )
        if self.expanding_window_start <= 0:
            raise ValueError(
                "Deep-learning prediction.expanding_window_start must be positive."
            )
        if self.mode == "rolling" and self.rolling_window_size is None:
            raise ValueError(
                "Deep-learning prediction rolling mode requires rolling_window_size."
            )
        return self


class DeepLearningInterpretabilitySettings(SettingsBase):
    enabled: bool = True
    ablation_splits: list[str] = Field(default_factory=lambda: ["validation"])
    calibration_bins: int = 10
    max_feature_groups: int | None = None

    @model_validator(mode="after")
    def validate_interpretability(self) -> "DeepLearningInterpretabilitySettings":
        if self.calibration_bins <= 1:
            raise ValueError(
                "Deep-learning interpretability.calibration_bins must be greater than 1."
            )
        return self


class DeepLearningTrainingSettings(SettingsBase):
    batch_size: int = 256
    epochs: int = 5
    learning_rate: float = 0.001
    weight_decay: float = 0.00001
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    clip_grad_norm: float | None = 1.0
    early_stopping_patience: int = 3
    deterministic: bool = True
    use_balanced_classification_loss: bool = True
    selection_metric: str = "validation_avg_signal_return_bps"
    allow_degenerate_fallback: bool = False
    regression_loss: str = "huber"
    huber_delta: float = 1.0
    internal_validation_fraction: float = 0.15

    @model_validator(mode="after")
    def validate_training(self) -> "DeepLearningTrainingSettings":
        valid_regression_losses = {"mse", "huber", "smooth_l1"}
        if self.regression_loss not in valid_regression_losses:
            raise ValueError(
                f"Deep-learning regression_loss must be one of {sorted(valid_regression_losses)}, got '{self.regression_loss}'."
            )
        if self.batch_size <= 0:
            raise ValueError("Deep-learning batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("Deep-learning epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("Deep-learning learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("Deep-learning weight_decay must be non-negative.")
        if self.early_stopping_patience <= 0:
            raise ValueError(
                "Deep-learning early_stopping_patience must be positive."
            )
        if not 0.0 < self.internal_validation_fraction < 0.5:
            raise ValueError(
                "Deep-learning internal_validation_fraction must be in (0, 0.5)."
            )
        if self.huber_delta <= 0.0:
            raise ValueError("Deep-learning huber_delta must be positive.")
        return self


class DeepLearningOutputSettings(SettingsBase):
    model_dir: str = "data/artifacts/models/dl"
    run_name: str = "default"
    write_csv: bool = True
    write_markdown_report: bool = True


class DeepLearningSettings(SettingsBase):
    input: BaselineInputSettings = Field(default_factory=BaselineInputSettings)
    target: DeepLearningTargetSettings = Field(
        default_factory=DeepLearningTargetSettings
    )
    feature_selection: BaselineFeatureSelectionSettings = Field(
        default_factory=BaselineFeatureSelectionSettings
    )
    sequence: SequenceSettings = Field(default_factory=SequenceSettings)
    model: DeepLearningModelSettings = Field(default_factory=DeepLearningModelSettings)
    preprocessing: DeepLearningPreprocessingSettings = Field(
        default_factory=DeepLearningPreprocessingSettings
    )
    threshold_search: DeepLearningThresholdSearchSettings = Field(
        default_factory=DeepLearningThresholdSearchSettings
    )
    tuning: DeepLearningTuningSettings = Field(
        default_factory=DeepLearningTuningSettings
    )
    prediction: DeepLearningPredictionSettings = Field(
        default_factory=DeepLearningPredictionSettings
    )
    interpretability: DeepLearningInterpretabilitySettings = Field(
        default_factory=DeepLearningInterpretabilitySettings
    )
    training: DeepLearningTrainingSettings = Field(
        default_factory=DeepLearningTrainingSettings
    )
    output: DeepLearningOutputSettings = Field(
        default_factory=DeepLearningOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class DeepLearningComparisonRunSettings(SettingsBase):
    """One model run inside a multi-model deep-learning comparison experiment."""

    name: str | None = None
    config_path: str
    overrides: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    force_retrain: bool = False


class DeepLearningComparisonRunnerSettings(SettingsBase):
    """Execution behavior for the deep-learning comparison runner."""

    train_if_missing: bool = True
    force_retrain_all: bool = False
    fail_fast: bool = True


class DeepLearningComparisonRankingSettings(SettingsBase):
    """Metrics used to rank models in comparison outputs."""

    validation_metric: str = "pearson_corr"
    test_metric: str = "pearson_corr"
    strategy_metric: str = "cumulative_signal_return_bps"
    strategy_split: str = "test"

    @model_validator(mode="after")
    def validate_ranking(self) -> "DeepLearningComparisonRankingSettings":
        valid_splits = {"validation", "test"}
        if self.strategy_split not in valid_splits:
            raise ValueError(
                f"Deep-learning comparison strategy_split must be one of {sorted(valid_splits)}, got '{self.strategy_split}'."
            )
        return self


class DeepLearningComparisonOutputSettings(SettingsBase):
    """Artifact settings for aggregated deep-learning comparison outputs."""

    output_dir: str = "data/artifacts/models/dl_comparisons"
    run_name: str = "sequence_regression_all"
    write_csv: bool = True
    write_markdown_report: bool = True
    write_plots: bool = True


class DeepLearningComparisonSettings(SettingsBase):
    """Bundle definition for multi-model deep-learning comparison experiments."""

    experiment_name: str
    description: str = ""
    runner: DeepLearningComparisonRunnerSettings = Field(
        default_factory=DeepLearningComparisonRunnerSettings
    )
    ranking: DeepLearningComparisonRankingSettings = Field(
        default_factory=DeepLearningComparisonRankingSettings
    )
    runs: list[DeepLearningComparisonRunSettings] = Field(default_factory=list)
    output: DeepLearningComparisonOutputSettings = Field(
        default_factory=DeepLearningComparisonOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_runs(self) -> "DeepLearningComparisonSettings":
        if not [run for run in self.runs if run.enabled]:
            raise ValueError(
                "Deep-learning comparison settings require at least one enabled run."
            )
        return self


class SignalInputSettings(SettingsBase):
    baseline_predictions_path: str = (
        "data/artifacts/models/baselines/binance/btcusdt/1h/btcusdt_24h_default/baseline_predictions.parquet"
    )
    baseline_manifest_path: str | None = (
        "data/artifacts/models/baselines/binance/btcusdt/1h/btcusdt_24h_default/baseline_manifest.json"
    )
    dl_predictions_path: str = (
        "data/artifacts/models/dl/binance/btcusdt/1h/transformer_regression_24h_default/dl_predictions.parquet"
    )
    dl_manifest_path: str | None = (
        "data/artifacts/models/dl/binance/btcusdt/1h/transformer_regression_24h_default/dl_manifest.json"
    )
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class SignalSourceSettings(SettingsBase):
    name: str = "baseline"
    baseline_mode: str = "all"
    model_names: list[str] = Field(default_factory=list)


class SignalOutputSettings(SettingsBase):
    output_dir: str = "data/artifacts/signals"
    artifact_name: str = "signals.parquet"
    manifest_name: str = "signals_manifest.json"
    write_csv: bool = True


class SignalSettings(SettingsBase):
    input: SignalInputSettings = Field(default_factory=SignalInputSettings)
    source: SignalSourceSettings = Field(default_factory=SignalSourceSettings)
    output: SignalOutputSettings = Field(default_factory=SignalOutputSettings)
    notes: dict[str, Any] = Field(default_factory=dict)


class ExploratoryDLSignalRunSettings(SettingsBase):
    name: str
    prediction_path: str
    manifest_path: str | None = None
    target_type: str
    task: str | None = None
    enabled: bool = True


class ExploratoryDLRankingRuleSettings(SettingsBase):
    enabled: bool = True
    name: str = "rolling_top_quantile_abs"
    percentile_threshold: float = 0.9
    window_size: int = 336
    min_history: int = 168

    @model_validator(mode="after")
    def validate_ranking_rule(self) -> "ExploratoryDLRankingRuleSettings":
        if not 0.5 <= self.percentile_threshold < 1.0:
            raise ValueError(
                "Exploratory ranking percentile_threshold must be in [0.5, 1.0)."
            )
        if self.window_size <= 1:
            raise ValueError("Exploratory ranking window_size must be greater than 1.")
        if self.min_history <= 0:
            raise ValueError("Exploratory ranking min_history must be positive.")
        return self


class ExploratoryDLThresholdRuleSettings(SettingsBase):
    enabled: bool = True
    name: str = "validation_tuned_abs_threshold"
    objective: str = "balanced_avg_return_support"
    candidate_quantiles: list[float] = Field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    )
    min_signal_count: int = 12
    min_signal_rate: float = 0.01
    support_weight: float = 0.5

    @model_validator(mode="after")
    def validate_threshold_rule(self) -> "ExploratoryDLThresholdRuleSettings":
        if any(not 0.0 < value < 1.0 for value in self.candidate_quantiles):
            raise ValueError(
                "Exploratory threshold_rule candidate_quantiles must all be in (0, 1)."
            )
        if self.min_signal_count <= 0:
            raise ValueError(
                "Exploratory threshold_rule min_signal_count must be positive."
            )
        if not 0.0 < self.min_signal_rate < 1.0:
            raise ValueError(
                "Exploratory threshold_rule min_signal_rate must be in (0, 1)."
            )
        if self.support_weight <= 0.0:
            raise ValueError(
                "Exploratory threshold_rule support_weight must be positive."
            )
        return self


class ExploratoryDLSignalInputSettings(SettingsBase):
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"
    runs: list[ExploratoryDLSignalRunSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_runs(self) -> "ExploratoryDLSignalInputSettings":
        if not [run for run in self.runs if run.enabled]:
            raise ValueError(
                "Exploratory signal generation requires at least one enabled run."
            )
        return self


class ExploratoryDLSignalOutputSettings(SettingsBase):
    output_dir: str = "data/artifacts/signals/exploratory_dl"
    artifact_name: str = "signals.parquet"
    manifest_name: str = "signals_manifest.json"
    strategy_catalog_name: str = "strategy_catalog.csv"
    diagnostics_dir_name: str = "diagnostics"
    write_csv: bool = True


class ExploratoryDLSignalSettings(SettingsBase):
    input: ExploratoryDLSignalInputSettings = Field(
        default_factory=ExploratoryDLSignalInputSettings
    )
    ranking_rule: ExploratoryDLRankingRuleSettings = Field(
        default_factory=ExploratoryDLRankingRuleSettings
    )
    threshold_rule: ExploratoryDLThresholdRuleSettings = Field(
        default_factory=ExploratoryDLThresholdRuleSettings
    )
    output: ExploratoryDLSignalOutputSettings = Field(
        default_factory=ExploratoryDLSignalOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class BacktestInputSettings(SettingsBase):
    signal_path: str = (
        "data/artifacts/signals/binance/btcusdt/1h/baseline/signals.parquet"
    )
    signal_manifest_path: str | None = (
        "data/artifacts/signals/binance/btcusdt/1h/baseline/signals_manifest.json"
    )
    market_dataset_path: str = (
        "data/processed/binance/btcusdt/1h/hourly_market_data.parquet"
    )
    market_manifest_path: str | None = "data/processed/binance/btcusdt/1h/manifest.json"
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class BacktestSelectionSettings(SettingsBase):
    strategy_names: list[str] = Field(default_factory=list)
    split_filter: list[str] = Field(
        default_factory=lambda: ["train", "validation", "test"]
    )
    direction: str = "short_perp_long_spot"
    require_should_trade: bool = True
    min_signal_score: float | None = None
    min_confidence: float | None = None
    min_expected_return_bps: float | None = None

    @model_validator(mode="after")
    def validate_selection(self) -> "BacktestSelectionSettings":
        valid_splits = {"train", "validation", "test"}
        valid_directions = {
            "short_perp_long_spot",
            "long_perp_short_spot",
            "any",
            "both",
        }
        unknown = sorted(set(self.split_filter) - valid_splits)
        if unknown:
            raise ValueError(
                "Backtest selection.split_filter can only contain "
                f"{sorted(valid_splits)}, got {unknown}."
            )
        if self.direction not in valid_directions:
            raise ValueError(
                "Backtest selection.direction must be one of "
                f"{sorted(valid_directions)}, got '{self.direction}'."
            )
        return self


class PortfolioSettings(SettingsBase):
    initial_capital: float
    position_notional: float
    max_open_positions: int = 1
    max_gross_leverage: float = 2.0
    leverage_check_mode: str = "warn"

    @model_validator(mode="after")
    def validate_portfolio(self) -> "PortfolioSettings":
        valid_modes = {"off", "warn", "fail"}
        if self.initial_capital <= 0.0:
            raise ValueError("Backtest portfolio.initial_capital must be positive.")
        if self.position_notional <= 0.0:
            raise ValueError("Backtest portfolio.position_notional must be positive.")
        if self.max_open_positions <= 0:
            raise ValueError("Backtest portfolio.max_open_positions must be positive.")
        if self.max_gross_leverage <= 0.0:
            raise ValueError("Backtest portfolio.max_gross_leverage must be positive.")
        if self.leverage_check_mode not in valid_modes:
            raise ValueError(
                f"Backtest portfolio.leverage_check_mode must be one of {sorted(valid_modes)}, got '{self.leverage_check_mode}'."
            )
        return self


class CostSettings(SettingsBase):
    taker_fee_bps: float
    maker_fee_bps: float
    slippage_bps: float
    gas_cost_usd: float
    other_friction_bps: float = 0.0


class ExecutionSettings(SettingsBase):
    entry_delay_bars: int = 1
    execution_price_field: str = "open"
    holding_window_hours: int = 24
    maximum_holding_hours: int = 48
    funding_interval_hours: int = 8
    funding_mode: str = "prototype_bar_sum"
    funding_notional_mode: str = "initial_notional"
    hedge_mode: str = "equal_notional_hedge"
    rebalance_frequency: str = "1h"
    exit_on_signal_off: bool = True
    stop_loss_bps: float | None = None
    take_profit_bps: float | None = None
    stop_observation_mode: str = "bar_close_observed"
    stop_execution_mode: str = "next_bar_executed"
    allow_partial_exit: bool = False

    @model_validator(mode="after")
    def validate_execution(self) -> "ExecutionSettings":
        valid_price_fields = {"open", "close"}
        valid_funding_modes = {"prototype_bar_sum", "event_aware"}
        valid_funding_notional_modes = {"initial_notional", "dynamic_position_value"}
        valid_hedge_modes = {
            "equal_notional_hedge",
            "equal_quantity_hedge",
            "contract_multiplier_adjusted_hedge",
        }
        if self.entry_delay_bars < 0:
            raise ValueError("Backtest execution.entry_delay_bars must be non-negative.")
        if self.execution_price_field not in valid_price_fields:
            raise ValueError(
                f"Backtest execution.execution_price_field must be one of {sorted(valid_price_fields)}, got '{self.execution_price_field}'."
            )
        if self.holding_window_hours <= 0:
            raise ValueError("Backtest execution.holding_window_hours must be positive.")
        if self.maximum_holding_hours <= 0:
            raise ValueError("Backtest execution.maximum_holding_hours must be positive.")
        if self.funding_interval_hours <= 0:
            raise ValueError("Backtest execution.funding_interval_hours must be positive.")
        if self.funding_mode not in valid_funding_modes:
            raise ValueError(
                f"Backtest execution.funding_mode must be one of {sorted(valid_funding_modes)}, got '{self.funding_mode}'."
            )
        if self.funding_notional_mode not in valid_funding_notional_modes:
            raise ValueError(
                "Backtest execution.funding_notional_mode must be one of "
                f"{sorted(valid_funding_notional_modes)}, got '{self.funding_notional_mode}'."
            )
        if self.hedge_mode not in valid_hedge_modes:
            raise ValueError(
                f"Backtest execution.hedge_mode must be one of {sorted(valid_hedge_modes)}, got '{self.hedge_mode}'."
            )
        if self.hedge_mode != "equal_notional_hedge":
            raise ValueError(
                "Backtest prototype currently implements only hedge_mode='equal_notional_hedge'. "
                "Other hedge modes are reserved for future work."
            )
        return self


class ReportingSettings(SettingsBase):
    output_dir: str = "data/artifacts/backtests"
    run_name: str = "baseline_signals_default"
    write_csv: bool = True
    write_markdown_report: bool = True
    figure_format: str = "png"
    dpi: int = 180
    top_n_strategies_for_plots: int = 5
    primary_split: str = "test"
    include_combined_summary: bool = True

    @model_validator(mode="after")
    def validate_reporting(self) -> "ReportingSettings":
        valid_primary_splits = {"train", "validation", "test", "combined"}
        if self.primary_split not in valid_primary_splits:
            raise ValueError(
                f"Backtest reporting.primary_split must be one of {sorted(valid_primary_splits)}, got '{self.primary_split}'."
            )
        if self.top_n_strategies_for_plots <= 0:
            raise ValueError("Backtest reporting.top_n_strategies_for_plots must be positive.")
        return self


class BacktestSettings(SettingsBase):
    input: BacktestInputSettings = Field(default_factory=BacktestInputSettings)
    selection: BacktestSelectionSettings = Field(
        default_factory=BacktestSelectionSettings
    )
    portfolio: PortfolioSettings
    costs: CostSettings
    execution: ExecutionSettings
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
    notes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backtest_splits(self) -> "BacktestSettings":
        if (
            self.reporting.primary_split != "combined"
            and self.selection.split_filter
            and self.reporting.primary_split not in set(self.selection.split_filter)
        ):
            raise ValueError(
                "Backtest reporting.primary_split must be included in "
                "selection.split_filter unless reporting.primary_split is 'combined'."
            )
        return self


class DataQualityReportInputSettings(SettingsBase):
    dataset_path: str = "data/processed/binance/btcusdt/1h/hourly_market_data.parquet"
    manifest_path: str | None = "data/processed/binance/btcusdt/1h/manifest.json"
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class DataQualityPlotSettings(SettingsBase):
    figure_format: str = "png"
    dpi: int = 180
    rolling_volatility_window_hours: int = 24
    funding_smoothing_window_hours: int = 24 * 7
    spread_smoothing_window_hours: int = 24
    annualization_factor_hours: int = 24 * 365


class DataQualityReportOutputSettings(SettingsBase):
    output_dir: str = "reports/data_quality"
    write_csv: bool = True
    write_markdown: bool = True
    write_json_summary: bool = True


class DataQualityReportSettings(SettingsBase):
    input: DataQualityReportInputSettings = Field(
        default_factory=DataQualityReportInputSettings
    )
    plots: DataQualityPlotSettings = Field(default_factory=DataQualityPlotSettings)
    output: DataQualityReportOutputSettings = Field(
        default_factory=DataQualityReportOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class FinalReportMetadataSettings(SettingsBase):
    title: str = "Deep Learning-Based Delta-Neutral Statistical Arbitrage on Perpetual Funding Rates"
    subtitle: str = "Final technical report for the project prototype."
    course: str = "Course Project"
    authors: list[str] = Field(default_factory=list)
    repository_url: str = ""
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    frequency: str = "1h"


class FinalReportInputSettings(SettingsBase):
    demo_snapshot_path: str = "data/artifacts/demo/demo_snapshot.json"
    robustness_summary_path: str | None = "reports/robustness/binance/btcusdt/1h/summary.json"
    exploratory_summary_path: str | None = None


class FinalReportSectionSettings(SettingsBase):
    executive_summary: list[str] = Field(default_factory=list)
    contributions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    future_work: list[str] = Field(default_factory=list)


class FinalReportOutputSettings(SettingsBase):
    artifact_dir: str = "reports/final"
    frontend_public_dir: str = "frontend/public/report"
    write_markdown: bool = True
    write_html: bool = True
    write_json_summary: bool = True
    copy_to_frontend_public: bool = True


class FinalReportSettings(SettingsBase):
    metadata: FinalReportMetadataSettings = Field(
        default_factory=FinalReportMetadataSettings
    )
    input: FinalReportInputSettings = Field(default_factory=FinalReportInputSettings)
    sections: FinalReportSectionSettings = Field(
        default_factory=FinalReportSectionSettings
    )
    output: FinalReportOutputSettings = Field(
        default_factory=FinalReportOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class ExploratoryDLReportInputSettings(SettingsBase):
    strict_demo_snapshot_path: str = "data/artifacts/demo/demo_snapshot.json"
    strict_final_report_summary_path: str | None = "reports/final/binance/btcusdt/1h/summary.json"
    strict_comparison_summary_path: str | None = (
        "data/artifacts/models/dl_comparisons/binance/btcusdt/1h/sequence_regression_all/comparison_summary.parquet"
    )
    exploratory_dataset_manifest_path: str = (
        "data/processed/exploratory_dl/binance/btcusdt/1h/btcusdt_exploratory_manifest.json"
    )
    exploratory_comparison_manifest_path: str = (
        "data/artifacts/models/exploratory_dl_comparisons/binance/btcusdt/1h/exploratory_showcase/comparison_manifest.json"
    )
    exploratory_comparison_summary_path: str = (
        "data/artifacts/models/exploratory_dl_comparisons/binance/btcusdt/1h/exploratory_showcase/comparison_summary.parquet"
    )
    exploratory_extra_comparison_summary_paths: list[str] = Field(
        default_factory=list
    )
    exploratory_signals_path: str = (
        "data/artifacts/signals/exploratory_dl/binance/btcusdt/1h/exploratory/signals.parquet"
    )
    exploratory_signals_manifest_path: str = (
        "data/artifacts/signals/exploratory_dl/binance/btcusdt/1h/exploratory/signals_manifest.json"
    )
    exploratory_backtest_manifest_path: str = (
        "data/artifacts/backtests/exploratory_dl/binance/btcusdt/1h/exploratory_dl_showcase/backtest_manifest.json"
    )
    exploratory_backtest_leaderboard_path: str = (
        "data/artifacts/backtests/exploratory_dl/binance/btcusdt/1h/exploratory_dl_showcase/leaderboard.parquet"
    )
    exploratory_trade_log_path: str | None = (
        "data/artifacts/backtests/exploratory_dl/binance/btcusdt/1h/exploratory_dl_showcase/trade_log.parquet"
    )


class ExploratoryDLReportOutputSettings(SettingsBase):
    output_dir: str = "reports/exploratory_dl"
    frontend_public_dir: str = "frontend/public/demo"
    write_csv: bool = True
    write_markdown: bool = True
    write_json_summary: bool = True
    write_plots: bool = True
    copy_to_frontend_public: bool = True


class ExploratoryDLReportSettings(SettingsBase):
    input: ExploratoryDLReportInputSettings = Field(
        default_factory=ExploratoryDLReportInputSettings
    )
    output: ExploratoryDLReportOutputSettings = Field(
        default_factory=ExploratoryDLReportOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class RobustnessInputSettings(SettingsBase):
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"
    signal_config_path: str = "configs/signals/default.yaml"
    baseline_config_path: str = "configs/models/baseline.yaml"
    dl_config_path: str = "configs/models/lstm.yaml"
    backtest_config_path: str = "configs/backtests/default.yaml"
    feature_manifest_path: str | None = (
        "data/processed/features/binance/btcusdt/1h/btcusdt_feature_manifest.json"
    )


class RobustnessFamilySettings(SettingsBase):
    name: str
    label: str | None = None
    source_name: str
    enabled: bool = True
    signal_path: str | None = None
    signal_manifest_path: str | None = None
    strategy_names: list[str] = Field(default_factory=list)
    regenerate_signal: bool = False


class RobustnessEvaluationSettings(SettingsBase):
    split_filter: list[str] = Field(default_factory=lambda: ["test"])
    primary_split: str = "test"
    ranking_metric: str = "cumulative_return"
    top_n_strategies: int = 1

    @model_validator(mode="after")
    def validate_evaluation(self) -> "RobustnessEvaluationSettings":
        valid_splits = {"train", "validation", "test", "combined"}
        if self.primary_split not in valid_splits:
            raise ValueError(
                f"Robustness evaluation.primary_split must be one of {sorted(valid_splits)}, got '{self.primary_split}'."
            )
        if self.primary_split != "combined" and self.primary_split not in set(self.split_filter):
            raise ValueError(
                "Robustness evaluation.primary_split must be included in split_filter unless it is 'combined'."
            )
        if self.top_n_strategies <= 0:
            raise ValueError("Robustness evaluation.top_n_strategies must be positive.")
        return self


class RobustnessCostScenario(SettingsBase):
    name: str
    taker_fee_bps: float | None = None
    slippage_bps: float | None = None
    gas_cost_usd: float | None = None
    other_friction_bps: float | None = None


class CostSensitivitySettings(SettingsBase):
    enabled: bool = True
    scenarios: list[RobustnessCostScenario] = Field(default_factory=list)


class RobustnessHoldingScenario(SettingsBase):
    name: str
    holding_window_hours: int
    maximum_holding_hours: int | None = None


class HoldingSensitivitySettings(SettingsBase):
    enabled: bool = True
    scenarios: list[RobustnessHoldingScenario] = Field(default_factory=list)


class RobustnessThresholdScenario(SettingsBase):
    name: str
    min_signal_score: float | None = None
    min_confidence: float | None = None
    min_expected_return_bps: float | None = None


class ThresholdSensitivitySettings(SettingsBase):
    enabled: bool = True
    family_name: str = "rule_based"
    scenarios: list[RobustnessThresholdScenario] = Field(default_factory=list)


class FeatureAblationSpec(SettingsBase):
    name: str
    feature_groups: list[str] = Field(default_factory=list)
    exclude_columns: list[str] = Field(default_factory=list)
    include_baseline_ml: bool = True
    include_deep_learning: bool = True


class FeatureAblationSettings(SettingsBase):
    enabled: bool = True
    signal_output_dir: str = "data/artifacts/robustness/signals"
    backtest_output_dir: str = "data/artifacts/robustness/backtests"
    baseline_run_name_prefix: str = "robustness_baseline"
    dl_run_name_prefix: str = "robustness_dl"
    allow_degenerate_fallback: bool = False
    groups: list[FeatureAblationSpec] = Field(default_factory=list)


class RobustnessReportingSettings(SettingsBase):
    output_dir: str = "reports/robustness"
    write_csv: bool = True
    write_markdown: bool = True
    write_json_summary: bool = True
    figure_format: str = "png"
    dpi: int = 180


class RobustnessReportSettings(SettingsBase):
    input: RobustnessInputSettings = Field(default_factory=RobustnessInputSettings)
    families: list[RobustnessFamilySettings] = Field(default_factory=list)
    evaluation: RobustnessEvaluationSettings = Field(
        default_factory=RobustnessEvaluationSettings
    )
    cost_sensitivity: CostSensitivitySettings = Field(
        default_factory=CostSensitivitySettings
    )
    holding_sensitivity: HoldingSensitivitySettings = Field(
        default_factory=HoldingSensitivitySettings
    )
    threshold_sensitivity: ThresholdSensitivitySettings = Field(
        default_factory=ThresholdSensitivitySettings
    )
    feature_ablation: FeatureAblationSettings = Field(
        default_factory=FeatureAblationSettings
    )
    reporting: RobustnessReportingSettings = Field(
        default_factory=RobustnessReportingSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class IntegrationInputSettings(SettingsBase):
    signals_path: str = (
        "data/artifacts/signals/binance/btcusdt/1h/baseline/signals.parquet"
    )
    signals_manifest_path: str | None = (
        "data/artifacts/signals/binance/btcusdt/1h/baseline/signals_manifest.json"
    )
    leaderboard_path: str = (
        "data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/leaderboard.parquet"
    )
    leaderboard_manifest_path: str | None = (
        "data/artifacts/backtests/binance/btcusdt/1h/baseline_signals_default/backtest_manifest.json"
    )
    provider: str = "binance"
    symbol: str = "BTCUSDT"
    venue: str = "binance"
    frequency: str = "1h"


class IntegrationSelectionSettings(SettingsBase):
    strategy_name: str | None = None
    ranking_metric: str = "total_net_pnl_usd"
    ranking_ascending: bool = False
    prefer_traded_strategy: bool = True
    split_preference: list[str] = Field(
        default_factory=lambda: ["test", "validation", "train"]
    )
    prefer_should_trade: bool = False
    require_should_trade: bool = False
    allow_flat_fallback: bool = True


class IntegrationContractSettings(SettingsBase):
    artifact_path: str = "contracts/out/DeltaNeutralVault.sol/DeltaNeutralVault.json"
    rpc_url: str = "http://127.0.0.1:8545"
    rpc_url_env: str | None = "VITE_RPC_URL"
    vault_address: str = "0x0000000000000000000000000000000000000000"
    vault_address_env: str | None = "VAULT_ADDRESS"
    operator_private_key_env: str = "PRIVATE_KEY"
    chain_id: int | None = 31337
    broadcast: bool = False
    update_strategy_state: bool = True
    update_nav: bool = True
    update_pnl: bool = False
    gas_limit: int = 500000
    gas_price_wei: int | None = None
    wait_for_receipt: bool = True

    @model_validator(mode="after")
    def validate_update_mode(self) -> "IntegrationContractSettings":
        if self.update_nav and self.update_pnl:
            raise ValueError(
                "Set at most one of update_nav or update_pnl to true for a single sync run."
            )
        return self


class IntegrationSemanticsSettings(SettingsBase):
    base_nav_assets: int = 100_000 * 10**6
    asset_decimals: int = 6
    asset_usd_price: float = 1.0
    nav_floor_assets: int = 0
    flat_strategy_state: str = "idle"
    active_strategy_state: str = "active"


class IntegrationOutputSettings(SettingsBase):
    output_dir: str = "data/artifacts/integration"
    run_name: str = "mock_operator_default"
    write_json: bool = True
    write_markdown_report: bool = True


class IntegrationSettings(SettingsBase):
    input: IntegrationInputSettings = Field(default_factory=IntegrationInputSettings)
    selection: IntegrationSelectionSettings = Field(
        default_factory=IntegrationSelectionSettings
    )
    contract: IntegrationContractSettings = Field(
        default_factory=IntegrationContractSettings
    )
    semantics: IntegrationSemanticsSettings = Field(
        default_factory=IntegrationSemanticsSettings
    )
    output: IntegrationOutputSettings = Field(default_factory=IntegrationOutputSettings)
    notes: dict[str, Any] = Field(default_factory=dict)


class DemoWorkflowStageSettings(SettingsBase):
    enabled: bool = True
    optional: bool = False


class DemoWorkflowCommandsSettings(SettingsBase):
    fetch_data_config_path: str = "configs/data/default.yaml"
    report_data_quality_config_path: str = "configs/reports/data_quality.yaml"
    features_config_path: str = "configs/features/default.yaml"
    labels_config_path: str = "configs/labels/default.yaml"
    exploratory_dataset_config_path: str = "configs/models/exploratory_dl/dataset.yaml"
    baseline_config_path: str = "configs/models/baseline.yaml"
    deep_learning_config_path: str = "configs/models/lstm.yaml"
    deep_learning_comparison_config_path: str = (
        "configs/experiments/dl/regression_all.yaml"
    )
    signals_config_path: str = "configs/signals/default.yaml"
    exploratory_signals_config_path: str = "configs/signals/exploratory_dl/default.yaml"
    backtest_config_path: str = "configs/backtests/default.yaml"
    exploratory_backtest_config_path: str = "configs/backtests/exploratory_dl/default.yaml"
    integration_config_path: str = "configs/integration/default.yaml"
    exploratory_report_config_path: str = "configs/reports/exploratory_dl/showcase.yaml"
    demo_snapshot_config_path: str = "configs/demo/default.yaml"
    exploratory_demo_snapshot_config_path: str = "configs/demo/exploratory_snapshot.yaml"
    exploratory_deep_learning_comparison_config_path: str = (
        "configs/experiments/dl/exploratory_gross_regression.yaml"
    )
    exploratory_direction_comparison_config_path: str = (
        "configs/experiments/dl/exploratory_direction_classification.yaml"
    )


class DemoWorkflowStages(SettingsBase):
    fetch_data: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    report_data_quality: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    build_features: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    build_labels: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    build_exploratory_dataset: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    train_baseline: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    train_deep_learning: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(optional=True)
    )
    compare_deep_learning: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(optional=True)
    )
    compare_exploratory_deep_learning: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    compare_exploratory_direction: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    generate_baseline_signals: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    generate_deep_learning_signals: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(optional=True)
    )
    generate_exploratory_signals: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    backtest: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    backtest_exploratory: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    sync_vault: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )
    report_exploratory: DemoWorkflowStageSettings = Field(
        default_factory=lambda: DemoWorkflowStageSettings(enabled=False, optional=True)
    )
    export_demo_snapshot: DemoWorkflowStageSettings = Field(
        default_factory=DemoWorkflowStageSettings
    )


class DemoWorkflowExecutionSettings(SettingsBase):
    continue_on_optional_failure: bool = True


class DemoWorkflowFrontendSettings(SettingsBase):
    frontend_dir: str = "frontend"
    public_snapshot_path: str = "frontend/public/demo/demo_snapshot.json"
    dashboard_url: str = "http://127.0.0.1:5173"
    dev_command: str = "npm run dev"
    build_command: str = "npm run build"


class DemoWorkflowOutputSettings(SettingsBase):
    output_dir: str = "data/artifacts/demo/workflow"
    run_name: str = "full_demo_default"
    log_level: str = "INFO"
    write_json: bool = True
    write_markdown_report: bool = True


class DemoWorkflowSettings(SettingsBase):
    commands: DemoWorkflowCommandsSettings = Field(
        default_factory=DemoWorkflowCommandsSettings
    )
    stages: DemoWorkflowStages = Field(default_factory=DemoWorkflowStages)
    execution: DemoWorkflowExecutionSettings = Field(
        default_factory=DemoWorkflowExecutionSettings
    )
    frontend: DemoWorkflowFrontendSettings = Field(
        default_factory=DemoWorkflowFrontendSettings
    )
    output: DemoWorkflowOutputSettings = Field(
        default_factory=DemoWorkflowOutputSettings
    )
    notes: dict[str, Any] = Field(default_factory=dict)
