from __future__ import annotations

import pandas as pd

from funding_arb.reporting.robustness import (
    _ablation_columns,
    _annotate_metrics,
    _best_rows,
)
from funding_arb.config.models import RobustnessFamilySettings


def test_ablation_columns_resolve_feature_groups_and_aliases() -> None:
    group_catalog = {
        "funding": ["funding_rate_bps", "funding_zscore_24h"],
        "basis": ["spread_bps"],
        "interaction_state": ["funding_x_spread_bps"],
    }

    columns = _ablation_columns(
        group_catalog,
        feature_groups=["funding", "interaction"],
        extra_columns=["manual_feature"],
    )

    assert columns == [
        "funding_rate_bps",
        "funding_x_spread_bps",
        "funding_zscore_24h",
        "manual_feature",
    ]


def test_best_rows_selects_highest_ranked_strategy_per_group() -> None:
    frame = pd.DataFrame(
        {
            "family_name": ["rule_based", "rule_based", "baseline_ml", "baseline_ml"],
            "scenario_name": ["base", "base", "base", "base"],
            "strategy_name": ["a_rule", "b_rule", "ridge", "logistic"],
            "trade_count": [10, 12, 0, 7],
            "cumulative_return": [0.01, 0.03, 0.02, -0.01],
        }
    )

    best = _best_rows(frame, "cumulative_return", ["family_name", "scenario_name"])

    assert set(best["family_name"]) == {"rule_based", "baseline_ml"}
    assert (
        best.loc[best["family_name"] == "rule_based", "strategy_name"].iloc[0]
        == "b_rule"
    )
    assert (
        best.loc[best["family_name"] == "baseline_ml", "strategy_name"].iloc[0]
        == "logistic"
    )


def test_annotate_metrics_preserves_baseline_strategy_metadata() -> None:
    metrics = pd.DataFrame(
        {
            "strategy_name": ["ridge_regression"],
            "source_subtype": ["baseline_linear"],
            "prediction_mode": ["expanding"],
            "calibration_method": ["none"],
            "signal_threshold": [4.5],
            "cumulative_return": [0.02],
            "trade_count": [12],
        }
    )
    family = RobustnessFamilySettings(
        name="baseline_ml",
        label="Simple ML Baseline",
        source_name="baseline-ml",
    )

    annotated = _annotate_metrics(
        metrics,
        experiment="family_comparison",
        family=family,
        scenario_name="base",
        scenario_order=1,
        run_name="family_comparison_baseline_ml_base",
        scenario_params={"scenario_label": "base"},
    )

    assert annotated["family_name"].iloc[0] == "baseline_ml"
    assert annotated["source_subtype"].iloc[0] == "baseline_linear"
    assert annotated["strategy_detail_label"].iloc[0] == "baseline_linear | expanding | thr=4.5"


def test_annotate_metrics_includes_deep_learning_strategy_metadata() -> None:
    metrics = pd.DataFrame(
        {
            "strategy_name": ["lstm"],
            "source_subtype": ["deep_learning"],
            "prediction_mode": ["rolling"],
            "calibration_method": ["none"],
            "signal_threshold": [0.0],
            "checkpoint_selection_effective_metric": ["validation_loss"],
            "checkpoint_selection_fallback_used": [True],
            "selected_loss": ["huber"],
            "preprocessing_scaler": ["robust"],
            "cumulative_return": [0.01],
            "trade_count": [4],
        }
    )
    family = RobustnessFamilySettings(
        name="deep_learning",
        label="Deep Learning Model",
        source_name="dl",
    )

    annotated = _annotate_metrics(
        metrics,
        experiment="family_comparison",
        family=family,
        scenario_name="base",
        scenario_order=1,
        run_name="family_comparison_deep_learning_base",
        scenario_params={"scenario_label": "base"},
    )

    assert annotated["strategy_detail_label"].iloc[0] == (
        "deep_learning | rolling | thr=0 | loss=huber | ckpt=validation_loss | scale=robust | fallback"
    )
