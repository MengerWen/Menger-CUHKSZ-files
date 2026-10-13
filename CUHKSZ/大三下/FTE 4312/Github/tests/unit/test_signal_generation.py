from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from funding_arb.config.models import SignalSettings
from funding_arb.signals.adapters import adapt_baseline_predictions, adapt_deep_learning_predictions
from funding_arb.signals.pipeline import run_signal_generation

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / 'tests' / '.tmp'
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / f'signal-tests-{uuid.uuid4().hex}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _baseline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC'),
            'split': ['train', 'validation', 'test'],
            'model_name': ['rule_a', 'logit_a', 'tree_reg_a'],
            'model_family': ['rule_based', 'linear', 'tree'],
            'task': ['classification', 'classification', 'regression'],
            'signal_direction': ['short_perp_long_spot', 'short_perp_long_spot', 'short_perp_long_spot'],
            'signal': [1, 0, 1],
            'decision_score': [1.5, 0.42, 6.8],
            'signal_threshold': [1.0, 0.5, 5.0],
            'signal_strength': [0.5, -0.08, 1.8],
            'predicted_probability': [None, 0.42, None],
            'predicted_return_bps': [None, None, 6.8],
            'predicted_label': [1, 0, 1],
            'actual_label': [1, 0, 1],
            'actual_return_bps': [5.0, -3.0, 4.5],
            'selected_threshold_objective': ['avg_signal_return_bps', 'precision', 'avg_signal_return_bps'],
            'calibration_method': ['none', 'sigmoid', 'none'],
            'feature_importance_method': ['not_applicable', 'permutation_validation', 'permutation_validation'],
            'prediction_mode': ['static', 'static', 'expanding'],
            'selected_hyperparameters_json': ['{}', '{"c": 0.1}', '{"max_depth": 4}'],
        }
    )


def _dl_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'timestamp': pd.date_range('2024-01-02', periods=2, freq='h', tz='UTC'),
            'split': ['validation', 'test'],
            'model_name': ['lstm', 'lstm'],
            'model_family': ['deep_learning', 'deep_learning'],
            'task': ['regression', 'regression'],
            'signal_direction': ['short_perp_long_spot', 'short_perp_long_spot'],
            'signal': [1, 0],
            'decision_score': [3.2, -1.4],
            'signal_threshold': [0.0, 0.0],
            'signal_strength': [3.2, -1.4],
            'predicted_probability': [None, None],
            'predicted_return_bps': [3.2, -1.4],
            'predicted_label': [1, 0],
            'actual_label': [1, 0],
            'actual_return_bps': [2.8, -2.1],
            'selected_threshold_objective': ['avg_signal_return_bps', 'avg_signal_return_bps'],
            'selected_threshold_objective_value': [1.25, 1.25],
            'prediction_mode': ['static', 'static'],
            'calibration_method': ['none', 'none'],
            'feature_importance_method': ['ablation_validation', 'ablation_validation'],
            'selected_hyperparameters_json': ['{"model_name":"lstm"}', '{"model_name":"lstm"}'],
            'checkpoint_selection_metric': ['validation_avg_signal_return_bps', 'validation_avg_signal_return_bps'],
            'best_checkpoint_metric_value': [None, None],
            'checkpoint_selection_effective_metric': ['validation_loss', 'validation_loss'],
            'best_checkpoint_effective_metric_value': [0.95, 0.95],
            'checkpoint_selection_fallback_used': [True, True],
            'selected_loss': ['huber', 'huber'],
            'regression_loss': ['huber', 'huber'],
            'use_balanced_classification_loss': [False, False],
            'preprocessing_scaler': ['robust', 'robust'],
            'winsorize_lower_quantile': [0.01, 0.01],
            'winsorize_upper_quantile': [0.99, 0.99],
        }
    )


def _settings(tmp_dir: Path, source_name: str) -> SignalSettings:
    baseline_path = tmp_dir / 'baseline.parquet'
    dl_path = tmp_dir / 'dl.parquet'
    _baseline_frame().to_parquet(baseline_path, index=False)
    _dl_frame().to_parquet(dl_path, index=False)
    return SignalSettings.model_validate(
        {
            'input': {
                'baseline_predictions_path': str(baseline_path),
                'baseline_manifest_path': None,
                'dl_predictions_path': str(dl_path),
                'dl_manifest_path': None,
                'provider': 'binance',
                'symbol': 'BTCUSDT',
                'venue': 'binance',
                'frequency': '1h',
            },
            'source': {
                'name': source_name,
                'baseline_mode': 'all',
                'model_names': [],
            },
            'output': {
                'output_dir': str(tmp_dir / 'signals_out'),
                'artifact_name': 'signals.parquet',
                'manifest_name': 'signals_manifest.json',
                'write_csv': True,
            },
        }
    )


def test_adapt_baseline_predictions_returns_rule_and_ml_subtypes() -> None:
    tmp_dir = _make_temp_dir()
    try:
        settings = _settings(tmp_dir, 'baseline')
        signals = adapt_baseline_predictions(settings)
        assert set(signals['source_subtype'].tolist()) == {'baseline_linear', 'baseline_tree', 'rule_based'}
        assert signals.loc[signals['strategy_name'] == 'rule_a', 'suggested_direction'].iloc[0] == 'short_perp_long_spot'
        assert signals.loc[signals['strategy_name'] == 'logit_a', 'suggested_direction'].iloc[0] == 'flat'
        assert signals.loc[signals['strategy_name'] == 'logit_a', 'calibration_method'].iloc[0] == 'sigmoid'
        assert signals.loc[signals['strategy_name'] == 'tree_reg_a', 'prediction_mode'].iloc[0] == 'expanding'
        assert float(signals.loc[signals['strategy_name'] == 'tree_reg_a', 'signal_threshold'].iloc[0]) == 5.0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_adapt_baseline_predictions_can_filter_rules_only() -> None:
    tmp_dir = _make_temp_dir()
    try:
        settings = _settings(tmp_dir, 'rules')
        signals = adapt_baseline_predictions(settings)
        assert signals['strategy_name'].tolist() == ['rule_a']
        assert signals['source_subtype'].tolist() == ['rule_based']
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_adapt_deep_learning_predictions_sets_expected_return_and_deep_learning_subtype() -> None:
    tmp_dir = _make_temp_dir()
    try:
        settings = _settings(tmp_dir, 'dl')
        signals = adapt_deep_learning_predictions(settings)
        assert signals['source_subtype'].tolist() == ['deep_learning', 'deep_learning']
        assert float(signals.loc[signals['should_trade'] == 1, 'expected_return_bps'].iloc[0]) == 3.2
        assert signals['checkpoint_selection_effective_metric'].iloc[0] == 'validation_loss'
        assert bool(signals['checkpoint_selection_fallback_used'].iloc[0]) is True
        assert signals['selected_loss'].iloc[0] == 'huber'
        assert signals['preprocessing_scaler'].iloc[0] == 'robust'
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_signal_generation_writes_artifacts_and_manifest() -> None:
    tmp_dir = _make_temp_dir()
    try:
        settings = _settings(tmp_dir, 'baseline')
        artifacts = run_signal_generation(settings)
        manifest_path = tmp_dir / 'signals_out' / 'binance' / 'btcusdt' / '1h' / 'baseline' / 'signals_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert artifacts.signals_path.endswith('signals.parquet')
        assert artifacts.signals_csv_path is not None
        assert manifest['summary']['active_signal_count'] == 2
        assert 'static' in manifest['summary']['prediction_modes']
        assert 'sigmoid' in manifest['summary']['calibration_methods']
        assert len(manifest['summary']['strategy_summary']) == 3
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_signal_generation_preserves_dl_metadata_in_manifest() -> None:
    tmp_dir = _make_temp_dir()
    try:
        settings = _settings(tmp_dir, 'dl')
        run_signal_generation(settings)
        manifest_path = tmp_dir / 'signals_out' / 'binance' / 'btcusdt' / '1h' / 'dl' / 'signals_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert 'validation_loss' in manifest['summary']['checkpoint_selection_effective_metrics']
        assert 'huber' in manifest['summary']['selected_losses']
        assert 'robust' in manifest['summary']['preprocessing_scalers']
        assert manifest['summary']['checkpoint_selection_fallback_count'] == 2
        assert manifest['degenerate_experiment'] is True
        assert manifest['status'] == 'warning'
        assert manifest['summary']['strategy_summary'][0]['status'] == 'no_tradable_signals'
        assert 'test' in manifest['summary']['strategy_summary'][0]['reason']
        assert 'threshold_search_summary' in manifest
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
