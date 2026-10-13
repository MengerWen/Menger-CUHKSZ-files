"""Export presentation-friendly demo artifacts for the frontend dashboard."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class DemoArtifacts:
    """Paths produced by the demo snapshot export."""

    artifact_snapshot_path: str
    public_snapshot_path: str
    public_assets_dir: str


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _load_json(path_text: str) -> dict[str, Any]:
    return json.loads(_resolve_path(path_text).read_text(encoding="utf-8"))


def _load_table(path_text: str) -> pd.DataFrame:
    path = _resolve_path(path_text)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for demo input: {path.suffix}")


def _load_optional_json(path_text: str | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_table(path_text: str | None) -> pd.DataFrame | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported table format for demo input: {path.suffix}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def _pick_best_baseline_row(frame: pd.DataFrame) -> dict[str, Any]:
    test_rows = frame.loc[frame["split"] == "test"].copy()
    if test_rows.empty:
        test_rows = frame.copy()
    regression_rows = test_rows.loc[test_rows["pearson_corr"].notna()].copy()
    if not regression_rows.empty:
        row = regression_rows.sort_values("pearson_corr", ascending=False, kind="stable").iloc[0]
    else:
        row = test_rows.sort_values(
            "cumulative_signal_return_bps",
            ascending=False,
            kind="stable",
            na_position="last",
        ).iloc[0]
    return {key: _json_ready(value) for key, value in row.to_dict().items()}


def _pick_best_dl_row(frame: pd.DataFrame) -> dict[str, Any]:
    test_rows = frame.loc[frame["split"] == "test"].copy()
    if test_rows.empty:
        test_rows = frame.copy()
    row = test_rows.sort_values("pearson_corr", ascending=False, kind="stable").iloc[0]
    return {key: _json_ready(value) for key, value in row.to_dict().items()}


def _pick_best_dl_comparison_row(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the top model-zoo row from a Phase 2 comparison leaderboard."""
    if frame.empty:
        return {}
    row = frame.sort_values("rank", ascending=True, kind="stable").iloc[0]
    return {key: _json_ready(value) for key, value in row.to_dict().items()}


def _sort_backtest_leaderboard(frame: pd.DataFrame) -> pd.DataFrame:
    """Align demo ordering with the upgraded backtest leaderboard contract."""
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    if "has_trades" in ranked.columns:
        ranked["_has_trades_for_demo"] = ranked["has_trades"].map(
            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        )
    elif "trade_count" in ranked.columns:
        ranked["_has_trades_for_demo"] = ranked["trade_count"].fillna(0).astype(float) > 0.0
    else:
        ranked["_has_trades_for_demo"] = True
    sort_columns = ["_has_trades_for_demo"]
    ascending = [False]
    for column in ("sharpe_ratio", "cumulative_return", "total_net_pnl_usd"):
        if column in ranked.columns:
            sort_columns.append(column)
            ascending.append(False)
    return (
        ranked.sort_values(sort_columns, ascending=ascending, kind="stable", na_position="last")
        .drop(columns="_has_trades_for_demo")
        .reset_index(drop=True)
    )


def _comparison_table_preview(frame: pd.DataFrame | None, limit: int = 4) -> list[dict[str, Any]]:
    """Build a small JSON-ready comparison table for the frontend."""
    if frame is None or frame.empty:
        return []
    columns = [
        "rank",
        "run_label",
        "model_name",
        "model_group",
        "task",
        "lookback_steps",
        "ranking_metric",
        "ranking_metric_value",
        "test_pearson_corr",
        "test_rmse",
        "test_top_quantile_avg_return_bps",
        "selected_loss",
    ]
    available_columns = [column for column in columns if column in frame.columns]
    preview = frame.loc[:, available_columns].head(limit)
    return [
        {key: _json_ready(value) for key, value in row.items()}
        for row in preview.to_dict("records")
    ]


def _copy_chart_assets(
    chart_specs: list[dict[str, Any]], public_assets_dir: Path
) -> list[dict[str, Any]]:
    ensure_directory(public_assets_dir)
    charts: list[dict[str, Any]] = []
    for item in chart_specs:
        source_path = _resolve_path(item["source_path"])
        if not source_path.exists():
            continue
        target_name = item.get("target_name") or source_path.name
        target_path = public_assets_dir / target_name
        shutil.copy2(source_path, target_path)
        charts.append(
            {
                "title": item["title"],
                "subtitle": item["subtitle"],
                "section": item["section"],
                "image": f"demo/assets/{target_name}",
                "source_path": str(source_path),
            }
        )
    return charts


def export_demo_snapshot(config: dict[str, Any]) -> DemoArtifacts:
    """Create a lightweight demo snapshot JSON plus copied chart assets."""
    demo_config = config.get("demo", {})
    input_config = config.get("inputs", {})
    frontend_config = config.get("frontend", {})
    contract_config = config.get("contract", {})

    artifact_dir = ensure_directory(_resolve_path(demo_config["artifact_dir"]))
    public_dir = ensure_directory(_resolve_path(demo_config["frontend_public_dir"]))
    public_assets_dir = ensure_directory(public_dir / "assets")

    data_manifest = _load_json(input_config["data_manifest_path"])
    quality_summary = _load_json(input_config["data_quality_summary_path"])
    backtest_manifest = _load_json(input_config["backtest_manifest_path"])
    baseline_leaderboard = _load_table(input_config["baseline_leaderboard_path"])
    dl_leaderboard = _load_optional_table(input_config.get("dl_leaderboard_path"))
    dl_comparison_manifest = _load_optional_json(
        input_config.get("dl_comparison_manifest_path")
    )
    dl_comparison_summary = _load_optional_table(
        input_config.get("dl_comparison_summary_path")
    )
    dl_comparison_test_leaderboard = _load_optional_table(
        input_config.get("dl_comparison_test_leaderboard_path")
    )
    backtest_leaderboard = _sort_backtest_leaderboard(
        _load_table(input_config["backtest_leaderboard_path"])
    )
    integration_selection = _load_optional_json(input_config.get("integration_selection_path"))
    integration_plan = _load_optional_json(input_config.get("integration_plan_path"))
    integration_calls = _load_optional_json(input_config.get("integration_call_summary_path"))
    exploratory_summary = _load_optional_json(input_config.get("exploratory_summary_path"))
    exploratory_leaderboard = _load_optional_table(
        input_config.get("exploratory_leaderboard_path")
    )
    exploratory_prediction_distribution = _load_optional_json(
        input_config.get("exploratory_prediction_distribution_path")
    )
    exploratory_quantile_analysis = _load_optional_json(
        input_config.get("exploratory_quantile_analysis_path")
    )

    charts = _copy_chart_assets(input_config["charts"], public_assets_dir)
    dl_comparison_available = (
        dl_comparison_test_leaderboard is not None
        and not dl_comparison_test_leaderboard.empty
    )
    dl_comparison_best = (
        _pick_best_dl_comparison_row(dl_comparison_test_leaderboard)
        if dl_comparison_available
        else None
    )
    single_dl_best = (
        _pick_best_dl_row(dl_leaderboard)
        if dl_leaderboard is not None and not dl_leaderboard.empty
        else {
            "model_name": "Deep learning not available",
            "task": "optional",
            "split": "n/a",
            "pearson_corr": None,
            "rmse": None,
            "note": "The demo snapshot was exported without a deep-learning leaderboard artifact.",
        }
    )

    best_backtest_row = {
        key: _json_ready(value)
        for key, value in backtest_leaderboard.iloc[0].to_dict().items()
    }
    top_backtests = [
        {key: _json_ready(value) for key, value in row.items()}
        for row in backtest_leaderboard.head(demo_config.get("top_strategies", 5)).to_dict("records")
    ]

    if integration_selection is None:
        # The dashboard should still tell a coherent story even when the
        # operator sync stage is skipped. In that case, fall back to the best
        # available backtest row and present the vault as idle.
        best_backtest = top_backtests[0]
        integration_selection = {
            "strategy_name": best_backtest.get("strategy_name", "not_available"),
            "timestamp": datetime.now(UTC).isoformat(),
            "leaderboard_summary": {
                "total_net_pnl_usd": best_backtest.get("total_net_pnl_usd", 0.0)
            },
        }
    if integration_plan is None:
        integration_plan = {
            "selected_strategy_name": integration_selection["strategy_name"],
            "strategy_state_name": "idle",
            "suggested_direction": "flat",
            "reported_nav_assets": 100_000 * 10**6,
            "summary_pnl_assets": 0,
            "summary_pnl_usd": 0.0,
            "should_trade": False,
        }
    if integration_calls is None:
        # Keep the snapshot export resilient when no live or dry-run contract
        # call summary exists yet. This makes the frontend usable for a pure
        # research-only demo path.
        integration_calls = {
            "calls": [],
            "execution_summary": {
                "mode": "not_run",
                "status": "The sync-vault stage was skipped or its artifacts were missing.",
            },
        }

    base_nav_assets = max(
        integration_plan["reported_nav_assets"] - integration_plan["summary_pnl_assets"],
        0,
    )
    base_total_shares = max(base_nav_assets, 1)

    snapshot = {
        "meta": {
            "title": demo_config.get("title", "Funding-Rate Arbitrage Prototype Dashboard"),
            "subtitle": demo_config.get(
                "subtitle",
                "A demo-friendly view of the research pipeline, model outputs, backtest results, and vault accounting layer.",
            ),
            "generated_at": datetime.now(UTC).isoformat(),
            "symbol": data_manifest["dataset"]["symbol"],
            "venue": data_manifest["dataset"]["venue"],
            "frequency": data_manifest["dataset"]["frequency"],
            "date_range": data_manifest["time_range"],
            "chain_name": contract_config.get("chain_name", "local_foundry"),
        },
        "overview": {
            "goal": "Show a complete prototype story from perpetual funding-rate data through modeling, backtesting, and on-chain vault state management.",
            "story_points": [
                "Historical Binance perpetual and spot data are normalized into a canonical hourly dataset.",
                "Funding, basis, volatility, liquidity, and interaction features drive both baseline and sequence models.",
                "Signals are evaluated in a delta-neutral backtester with fees, slippage, funding effects, and mark-to-market risk metrics.",
                "A prototype Solidity vault mirrors the off-chain strategy state through a trusted operator update path.",
            ],
            "layers": [
                {"label": "Data", "detail": "Binance BTCUSDT hourly perpetual, spot, and funding history over the main 2021-01-01 to 2026-04-07 UTC window."},
                {"label": "Models", "detail": "Interpretable rule-based and linear baselines plus a Phase 1/2 sequence-model zoo: LSTM, GRU, TCN, and TransformerEncoder."},
                {"label": "Backtest", "detail": "Explicit delta-neutral trade accounting with test-primary leaderboards, mark-to-market equity, realized audit columns, and robustness checks."},
                {"label": "Vault", "detail": "Mock stablecoin deposits, internal shares, strategy state, NAV/PnL updates, and event-rich accounting."},
            ],
        },
        "research": {
            "canonical_rows": data_manifest["canonical_row_count"],
            "perpetual_rows": data_manifest["row_counts"]["perpetual_bars"],
            "funding_events": quality_summary["funding_event_count"],
            "coverage_ratio": quality_summary["coverage"]["coverage_ratio"],
            "funding_mean_bps": quality_summary["funding_mean_bps"],
            "funding_std_bps": quality_summary["funding_std_bps"],
            "spread_mean_bps": quality_summary["spread_mean_bps"],
            "annualized_volatility": quality_summary["mean_perp_annualized_vol"],
        },
        "models": {
            "baseline_best": _pick_best_baseline_row(baseline_leaderboard),
            "deep_learning_best": dl_comparison_best or single_dl_best,
            "deep_learning_single_best": single_dl_best,
            "deep_learning_comparison": {
                "available": dl_comparison_available,
                "best_model_note": (
                    dl_comparison_manifest or {}
                ).get("best_model_note"),
                "run_count": (dl_comparison_manifest or {}).get("run_count"),
                "report_path": (dl_comparison_manifest or {}).get("report_path"),
                "summary_path": input_config.get("dl_comparison_summary_path")
                if dl_comparison_summary is not None
                else None,
                "test_leaderboard": _comparison_table_preview(
                    dl_comparison_test_leaderboard
                ),
            },
        },
        "exploratory_dl": (
            {
                "available": True,
                "summary": (
                    (exploratory_summary or {}).get("exploratory_summary", exploratory_summary)
                ),
                "summary_payload": exploratory_summary,
                "leaderboard_preview": (
                    [
                        {key: _json_ready(value) for key, value in row.items()}
                        for row in exploratory_leaderboard.head(8).to_dict("records")
                    ]
                    if exploratory_leaderboard is not None and not exploratory_leaderboard.empty
                    else []
                ),
                "prediction_distribution": exploratory_prediction_distribution,
                "quantile_analysis": exploratory_quantile_analysis,
                "disclaimer": (
                    exploratory_summary or {}
                ).get(
                    "disclaimer",
                    "Exploratory DL results are supplementary showcase results and do not replace the strict post-cost conclusion.",
                ),
            }
            if exploratory_summary is not None
            else None
        ),
        "backtest": {
            "summary": backtest_manifest["summary"],
            "diagnostics": backtest_manifest.get("diagnostics", {}),
            "risk_view": {
                "primary_split": backtest_manifest.get("summary", {}).get("primary_split"),
                "primary_trade_count": backtest_manifest.get("summary", {}).get("primary_trade_count"),
                "combined_trade_count": backtest_manifest.get("summary", {}).get("combined_trade_count"),
                "equity_basis": "mark_to_market",
                "drawdown_basis": "mark_to_market",
                "realized_audit_available": True,
            },
            "best_strategy": best_backtest_row,
            "top_strategies": top_backtests,
            "assumptions": backtest_manifest["assumptions"],
        },
        "charts": charts,
        "vault": {
            "chain_name": contract_config.get("chain_name", "local_foundry"),
            "vault_address": contract_config.get("vault_address", "0x0000000000000000000000000000000000000000"),
            "stablecoin_address": contract_config.get("stablecoin_address", "0x0000000000000000000000000000000000000000"),
            "selected_strategy": integration_selection["strategy_name"],
            "strategy_state": integration_plan["strategy_state_name"],
            "suggested_direction": integration_plan["suggested_direction"],
            "reported_nav_assets": integration_plan["reported_nav_assets"],
            "summary_pnl_assets": integration_plan["summary_pnl_assets"],
            "summary_pnl_usd": integration_plan["summary_pnl_usd"],
            "call_count": len(integration_calls["calls"]),
            "execution_summary": integration_calls["execution_summary"],
        },
        "activity_log": [
            {
                "timestamp": data_manifest["time_range"]["start"],
                "kind": "data",
                "title": "Hourly research dataset established",
                "detail": f"{data_manifest['canonical_row_count']:,} canonical rows covering multiple funding regimes.",
            },
            {
                "timestamp": integration_selection["timestamp"],
                "kind": "strategy",
                "title": "Top backtest strategy selected",
                "detail": f"{integration_selection['strategy_name']} is the current primary-split leaderboard row with ${integration_selection['leaderboard_summary']['total_net_pnl_usd']:.2f} net PnL.",
            },
            {
                "timestamp": integration_selection["timestamp"],
                "kind": "vault",
                "title": "Operator sync payload prepared",
                "detail": f"Dry-run call data prepared for {len(integration_calls['calls'])} vault update calls.",
            },
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "kind": "demo",
                "title": "Frontend snapshot exported",
                "detail": "Charts, metrics, and mock vault state were packaged for the local dashboard.",
            },
        ],
        "simulation": {
            "asset_symbol": contract_config.get("asset_symbol", "mUSDC"),
            "asset_decimals": contract_config.get("asset_decimals", 6),
            "wallet_cash_assets": contract_config.get("demo_wallet_cash_assets", 25_000 * 10**6),
            "base_vault_cash_assets": base_nav_assets,
            "base_reported_nav_assets": base_nav_assets,
            "base_total_shares": base_total_shares,
            "user_shares": 0,
            "strategy_state": "idle",
            "operator_plan": {
                "selected_strategy": integration_plan["selected_strategy_name"],
                "strategy_state": integration_plan["strategy_state_name"],
                "suggested_direction": integration_plan["suggested_direction"],
                "reported_nav_assets": integration_plan["reported_nav_assets"],
                "summary_pnl_assets": integration_plan["summary_pnl_assets"],
                "summary_pnl_usd": integration_plan["summary_pnl_usd"],
                "should_trade": integration_plan["should_trade"],
                "demo_activation_state": "active",
                "demo_activation_direction": "short_perp_long_spot",
            },
        },
    }

    artifact_snapshot_path = artifact_dir / "demo_snapshot.json"
    public_snapshot_path = public_dir / "demo_snapshot.json"
    artifact_snapshot_path.write_text(json.dumps(_json_ready(snapshot), indent=2), encoding="utf-8")
    public_snapshot_path.write_text(json.dumps(_json_ready(snapshot), indent=2), encoding="utf-8")

    return DemoArtifacts(
        artifact_snapshot_path=str(artifact_snapshot_path),
        public_snapshot_path=str(public_snapshot_path),
        public_assets_dir=str(public_assets_dir),
    )
