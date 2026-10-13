"""Stage 7 portfolio construction and backtesting on OOS model signals."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.evaluation.metrics import annualized_sharpe, max_drawdown
from src.training.non_graph_benchmark_pipeline import resolve_project_path

sns.set_theme(style="whitegrid")


@dataclass(frozen=True)
class PortfolioConfig:
    """Stage 7 portfolio construction configuration."""

    long_short_quantile: float = 0.1
    long_only_quantile: float = 0.1
    rebalance_frequency: str = "monthly"
    report_turnover: bool = True
    main_transaction_cost_bps: int = 10


def validate_portfolio_config(config: PortfolioConfig) -> None:
    """Validate portfolio construction choices."""

    if not 0 < config.long_short_quantile < 0.5:
        raise ValueError("long_short_quantile must be between 0 and 0.5")
    if not 0 < config.long_only_quantile <= 0.5:
        raise ValueError("long_only_quantile must be between 0 and 0.5")
    if config.rebalance_frequency != "monthly":
        raise ValueError("Stage 2 panel protocol assumes monthly rebalancing")
    if config.main_transaction_cost_bps < 0:
        raise ValueError("main_transaction_cost_bps must be non-negative")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load Stage 7 YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def project_relative_string(project_root: Path, path: Path | str) -> str:
    """Return project-relative path string when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def load_prediction_outputs(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load OOS predictions for all requested models."""

    frames: list[pd.DataFrame] = []
    for spec in config.get("models", []):
        name = spec["name"]
        path = resolve_project_path(project_root, spec["predictions_path"])
        frame = pd.read_pickle(path).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if "model" in frame.columns:
            frame = frame.loc[frame["model"] == name].copy()
        if frame.empty:
            raise ValueError(f"no prediction rows found for model {name} in {path}")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "stock_id", "model"]).reset_index(drop=True)


def align_common_signal_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align models on the common stock-date intersection for apples-to-apples backtests."""

    required_models = predictions["model"].nunique()
    common_keys = (
        predictions.groupby(["date", "stock_id"], sort=True)["model"]
        .nunique()
        .loc[lambda values: values == required_models]
        .reset_index()[["date", "stock_id"]]
    )
    aligned = predictions.merge(common_keys, on=["date", "stock_id"], how="inner")
    aligned = aligned.sort_values(["date", "stock_id", "model"]).reset_index(drop=True)
    coverage = (
        aligned.groupby("model", sort=True)
        .agg(
            n_obs=("stock_id", "size"),
            n_months=("date", "nunique"),
            n_unique_stocks=("stock_id", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )
    return aligned, coverage


def load_panel_fields(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load realized returns and month-t portfolio attributes from the cleaned panel."""

    panel_path = resolve_project_path(project_root, config.get("data", {}).get("panel_path", "outputs/panels/main_features500_panel.pkl"))
    panel = pd.read_pickle(panel_path)
    keep = ["date", "stock_id", "target_return", "target_excess_return", "rf_next_month", "mcap_t", "blacklisted_t", "untradable_t"]
    missing = [col for col in keep if col not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required fields for Stage 7: {missing}")
    panel = panel[keep].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    return panel


def monthly_flag_from_daily(path: Path, months: pd.DatetimeIndex, column_name: str) -> pd.DataFrame:
    """Aggregate daily blacklist/untradable records to signal-month flags."""

    raw = pd.read_pickle(path)
    index_frame = raw.index.to_frame(index=False).rename(columns={"asset": "stock_id"})
    index_frame["date"] = pd.to_datetime(index_frame["date"]) + pd.offsets.MonthEnd(0)
    index_frame = index_frame.loc[index_frame["date"].isin(months), ["date", "stock_id"]].drop_duplicates()
    index_frame[column_name] = True
    return index_frame


def merge_signal_inputs(project_root: Path, aligned: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge predictions with realized returns, mcap, and optional raw feasibility flags."""

    panel = load_panel_fields(project_root, config)
    signals = aligned.merge(panel, on=["date", "stock_id"], how="left", validate="many_to_one")
    if signals[["target_return", "target_excess_return", "rf_next_month"]].isna().any().any():
        raise ValueError("missing realized return fields after merging aligned signals with panel")

    months = pd.DatetimeIndex(signals["date"].drop_duplicates()).sort_values()
    data_config = config.get("data", {})
    merge_stats: dict[str, Any] = {}
    if config.get("portfolio", {}).get("filter_blacklist", True):
        blacklist_path = resolve_project_path(project_root, data_config.get("blacklist_path", "data/BLACKLIST.pkl"))
        blacklist = monthly_flag_from_daily(blacklist_path, months, "blacklisted_raw_t")
        signals = signals.merge(blacklist, on=["date", "stock_id"], how="left")
        signals["blacklisted_raw_t"] = signals["blacklisted_raw_t"].eq(True)
    else:
        signals["blacklisted_raw_t"] = False

    if config.get("portfolio", {}).get("filter_untradable", True):
        untradable_path = resolve_project_path(project_root, data_config.get("untradable_path", "data/UNTRADABLE.pkl"))
        untradable = monthly_flag_from_daily(untradable_path, months, "untradable_raw_t")
        signals = signals.merge(untradable, on=["date", "stock_id"], how="left")
        signals["untradable_raw_t"] = signals["untradable_raw_t"].eq(True)
    else:
        signals["untradable_raw_t"] = False

    y_diff = np.abs(signals["y_true"].to_numpy(dtype=np.float64) - signals["target_excess_return"].to_numpy(dtype=np.float64))
    merge_stats["max_abs_y_true_vs_panel_excess_diff"] = float(np.nanmax(y_diff)) if len(y_diff) else np.nan
    merge_stats["raw_blacklist_true_rows"] = int(signals["blacklisted_raw_t"].sum())
    merge_stats["raw_untradable_true_rows"] = int(signals["untradable_raw_t"].sum())
    merge_stats["panel_blacklisted_true_rows"] = int(signals["blacklisted_t"].sum())
    merge_stats["panel_untradable_true_rows"] = int(signals["untradable_t"].sum())
    return signals, merge_stats


def _n_select(n_assets: int, mode: str, quantile: float | None, count: int | None, min_names: int) -> int:
    if n_assets <= 0:
        return 0
    if mode == "count":
        n = int(count or 0)
    elif mode == "quantile":
        if quantile is None:
            raise ValueError("quantile selection mode requires quantile")
        n = int(np.floor(n_assets * float(quantile)))
    else:
        raise ValueError(f"unsupported selection mode: {mode}")
    return min(n_assets, max(min_names, n))


def _weighted_leg(block: pd.DataFrame, sign: float, weight_scheme: str) -> pd.Series:
    if block.empty:
        return pd.Series(dtype="float64")
    if weight_scheme == "equal":
        weights = pd.Series(sign / len(block), index=block.index, dtype="float64")
    elif weight_scheme == "value":
        mcap = pd.to_numeric(block["mcap_t"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = mcap.notna() & (mcap > 0)
        if not valid.any():
            return pd.Series(dtype="float64")
        scaled = mcap.loc[valid] / mcap.loc[valid].sum()
        weights = pd.Series(0.0, index=block.index, dtype="float64")
        weights.loc[scaled.index] = sign * scaled.astype("float64")
    else:
        raise ValueError(f"unsupported weight scheme: {weight_scheme}")
    return weights


def build_monthly_weights(signal_frame: pd.DataFrame, strategy: dict[str, Any], weight_scheme: str) -> pd.DataFrame:
    """Build long-only or long-short monthly weights for one model."""

    strategy_type = strategy["strategy_type"]
    selection_mode = strategy.get("selection_mode", "quantile")
    quantile = strategy.get("quantile")
    count = strategy.get("count")
    min_names = int(strategy.get("min_names", 1))

    rows: list[pd.DataFrame] = []
    for date, month in signal_frame.groupby("date", sort=True):
        clean = month.copy()
        clean = clean.loc[clean["eligible"] & clean["y_pred"].notna() & clean["target_return"].notna()].copy()
        clean = clean.sort_values(["y_pred", "stock_id"], ascending=[False, True]).reset_index(drop=True)
        if clean.empty:
            continue

        if strategy_type == "long_short":
            n_leg = _n_select(len(clean), selection_mode, quantile, count, min_names)
            if 2 * n_leg > len(clean):
                n_leg = len(clean) // 2
            if n_leg <= 0:
                continue
            long_block = clean.head(n_leg).copy()
            short_block = clean.tail(n_leg).copy()
            long_weights = _weighted_leg(long_block, sign=1.0, weight_scheme=weight_scheme)
            short_weights = _weighted_leg(short_block, sign=-1.0, weight_scheme=weight_scheme)
            if long_weights.empty or short_weights.empty:
                continue
            selected = pd.concat([
                long_block.assign(weight=long_weights, leg="long"),
                short_block.assign(weight=short_weights, leg="short"),
            ], ignore_index=True)
        elif strategy_type == "long_only":
            n_long = _n_select(len(clean), selection_mode, quantile, count, min_names)
            if n_long <= 0:
                continue
            long_block = clean.head(n_long).copy()
            long_weights = _weighted_leg(long_block, sign=1.0, weight_scheme=weight_scheme)
            if long_weights.empty:
                continue
            selected = long_block.assign(weight=long_weights, leg="long")
        else:
            raise ValueError(f"unsupported strategy_type: {strategy_type}")

        selected = selected.loc[selected["weight"].notna() & (selected["weight"] != 0)].copy()
        if selected.empty:
            continue
        selected["strategy_name"] = strategy["strategy_name"]
        selected["weight_scheme"] = weight_scheme
        selected["signal_date"] = pd.Timestamp(date)
        selected["holding_month"] = pd.Timestamp(date) + pd.offsets.MonthEnd(1)
        rows.append(
            selected[["model", "strategy_name", "weight_scheme", "signal_date", "holding_month", "stock_id", "leg", "y_pred", "target_return", "target_excess_return", "rf_next_month", "mcap_t", "weight"]].copy()
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def strategy_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand configured Stage 7 strategy definitions."""

    specs: list[dict[str, Any]] = []
    for strategy in config.get("strategies", []):
        strategy = strategy.copy()
        strategy_name = strategy.get("name")
        if not strategy_name:
            raise ValueError("each strategy needs a name")
        specs.append(
            {
                "strategy_name": strategy_name,
                "strategy_type": strategy["type"],
                "selection_mode": strategy.get("selection_mode", "quantile"),
                "quantile": strategy.get("quantile"),
                "count": strategy.get("count"),
                "min_names": strategy.get("min_names", 1),
            }
        )
    return specs


def build_all_weights(signals: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build portfolio weights for all models and configured strategies."""

    weight_schemes = config.get("portfolio", {}).get("weight_schemes", ["equal", "value"])
    all_weights: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for model, model_frame in signals.groupby("model", sort=True):
        for strategy in strategy_specs(config):
            for weight_scheme in weight_schemes:
                weights = build_monthly_weights(model_frame, strategy, weight_scheme)
                if weights.empty:
                    continue
                all_weights.append(weights)
                coverage_rows.append(
                    {
                        "model": model,
                        "strategy_name": strategy["strategy_name"],
                        "weight_scheme": weight_scheme,
                        "n_weight_rows": int(weights.shape[0]),
                        "n_signal_months": int(weights["signal_date"].nunique()),
                        "avg_names": float(weights.groupby("signal_date")["stock_id"].nunique().mean()),
                        "avg_abs_weight": float(weights["weight"].abs().mean()),
                    }
                )
    weight_frame = pd.concat(all_weights, ignore_index=True) if all_weights else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return weight_frame, coverage


def monthly_turnover(weights: pd.DataFrame, date_col: str = "signal_date", stock_col: str = "stock_id", weight_col: str = "weight") -> pd.Series:
    """Compute one-way monthly turnover from long-format weight data."""

    wide = weights.pivot_table(index=date_col, columns=stock_col, values=weight_col, fill_value=0.0).sort_index()
    turnover = wide.diff().abs().sum(axis=1) / 2.0
    if not turnover.empty:
        turnover.iloc[0] = np.nan
    return turnover.rename("turnover")


def compute_monthly_portfolio_returns(weights: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted stock returns to monthly portfolio returns."""

    if weights.empty:
        return pd.DataFrame()
    returns = (
        weights.assign(
            weighted_return=weights["weight"] * weights["target_return"],
            weighted_excess_return=weights["weight"] * weights["target_excess_return"],
        )
        .groupby(["model", "strategy_name", "weight_scheme", "signal_date", "holding_month"], sort=True)
        .agg(
            gross_return=("weighted_return", "sum"),
            excess_return=("weighted_excess_return", "sum"),
            n_holdings=("stock_id", "nunique"),
            gross_exposure=("weight", lambda values: float(np.abs(values).sum())),
            net_exposure=("weight", "sum"),
            long_count=("leg", lambda values: int((pd.Series(values) == "long").sum())),
            short_count=("leg", lambda values: int((pd.Series(values) == "short").sum())),
        )
        .reset_index()
    )
    turnover_list: list[pd.DataFrame] = []
    for keys, group in weights.groupby(["model", "strategy_name", "weight_scheme"], sort=True):
        turn = monthly_turnover(group)
        frame = turn.reset_index()
        frame[["model", "strategy_name", "weight_scheme"]] = list(keys)
        turnover_list.append(frame)
    turnover_frame = pd.concat(turnover_list, ignore_index=True) if turnover_list else pd.DataFrame(columns=["signal_date", "turnover", "model", "strategy_name", "weight_scheme"])
    returns = returns.merge(turnover_frame, on=["model", "strategy_name", "weight_scheme", "signal_date"], how="left")
    returns["turnover"] = returns["turnover"].astype("float64")
    return returns.sort_values(["model", "strategy_name", "weight_scheme", "signal_date"]).reset_index(drop=True)


def expand_transaction_cost_scenarios(monthly_returns: pd.DataFrame, cost_grid: list[int]) -> pd.DataFrame:
    """Create transaction-cost-adjusted return series."""

    rows: list[pd.DataFrame] = []
    for bps in cost_grid:
        frame = monthly_returns.copy()
        cost = frame["turnover"].fillna(0.0) * (float(bps) / 10000.0)
        frame["transaction_cost_bps"] = int(bps)
        frame["net_return"] = frame["gross_return"] - cost
        frame["net_excess_return"] = frame["excess_return"] - cost
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _annualized_return(monthly_returns: pd.Series) -> float:
    clean = pd.to_numeric(monthly_returns, errors="coerce").dropna()
    if clean.empty:
        return np.nan
    return float((1.0 + clean).prod() ** (12.0 / clean.size) - 1.0)


def _annualized_vol(monthly_returns: pd.Series) -> float:
    clean = pd.to_numeric(monthly_returns, errors="coerce").dropna()
    if clean.size < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(12.0))


def summarize_performance(net_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio performance metrics by model, strategy, and TC scenario."""

    rows: list[dict[str, Any]] = []
    for keys, group in net_returns.groupby(["model", "strategy_name", "weight_scheme", "transaction_cost_bps"], sort=True):
        raw = group.sort_values("holding_month")["net_return"]
        excess = group.sort_values("holding_month")["net_excess_return"]
        rows.append(
            {
                "model": keys[0],
                "strategy_name": keys[1],
                "weight_scheme": keys[2],
                "transaction_cost_bps": keys[3],
                "n_months": int(group["holding_month"].nunique()),
                "annualized_return": _annualized_return(raw),
                "annualized_excess_return": _annualized_return(excess),
                "annualized_volatility": _annualized_vol(raw),
                "sharpe_ratio": float(annualized_sharpe(excess)) if excess.dropna().size >= 2 and not np.isclose(excess.dropna().std(ddof=1), 0.0) else np.nan,
                "max_drawdown": float(max_drawdown(raw)) if raw.dropna().size > 0 else np.nan,
                "avg_monthly_turnover": float(group["turnover"].dropna().mean()) if group["turnover"].notna().any() else np.nan,
                "annualized_turnover": float(group["turnover"].dropna().mean() * 12.0) if group["turnover"].notna().any() else np.nan,
                "gross_exposure_mean": float(group["gross_exposure"].mean()),
                "avg_n_holdings": float(group["n_holdings"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategy_name", "weight_scheme", "transaction_cost_bps", "model"]).reset_index(drop=True)

def plot_cumulative_returns(net_returns: pd.DataFrame, output_dir: Path, main_cost_bps: int) -> dict[str, str]:
    """Plot cumulative returns by strategy and weight scheme for the main TC scenario."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    plot_frame = net_returns.loc[net_returns["transaction_cost_bps"] == main_cost_bps].copy()
    if plot_frame.empty:
        return plot_paths

    for (strategy_name, weight_scheme), group in plot_frame.groupby(["strategy_name", "weight_scheme"], sort=True):
        fig, ax = plt.subplots(figsize=(10, 5))
        for model, model_group in group.groupby("model", sort=True):
            series = model_group.sort_values("holding_month")
            wealth = (1.0 + series["net_return"].to_numpy(dtype=np.float64)).cumprod()
            ax.plot(series["holding_month"], wealth, marker="o", label=model)
        ax.set_title(f"Cumulative Returns: {strategy_name} / {weight_scheme} / {main_cost_bps} bps")
        ax.set_ylabel("Cumulative Wealth")
        ax.set_xlabel("")
        ax.legend(frameon=True)
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        path = output_dir / f"stage7_cumulative_{strategy_name}_{weight_scheme}_{main_cost_bps}bps.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths[f"{strategy_name}_{weight_scheme}"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    perf = summarize_performance(plot_frame)
    sns.barplot(data=perf, x="model", y="annualized_return", hue="strategy_name", ax=axes[0])
    axes[0].set_title(f"Annualized Return ({main_cost_bps} bps)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=perf, x="model", y="sharpe_ratio", hue="strategy_name", ax=axes[1])
    axes[1].set_title(f"Sharpe Ratio ({main_cost_bps} bps)")
    axes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    summary_path = output_dir / f"stage7_summary_bars_{main_cost_bps}bps.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths["summary_bars"] = str(summary_path)
    return plot_paths


def write_economic_value_note(project_root: Path, summary: pd.DataFrame, coverage: pd.DataFrame, plot_paths: dict[str, str], report_path: Path, main_cost_bps: int) -> None:
    """Write Stage 7 markdown interpretation note."""

    main = summary.loc[summary["transaction_cost_bps"] == main_cost_bps].copy()

    def best(strategy_name: str, column: str, higher: bool = True) -> str:
        block = main.loc[main["strategy_name"] == strategy_name, ["model", "weight_scheme", column]].dropna()
        if block.empty:
            return "N/A"
        row = block.sort_values(column, ascending=not higher).iloc[0]
        return f"{row['model']} ({row['weight_scheme']})"

    def metric(model: str, strategy_name: str, weight_scheme: str, column: str) -> str:
        block = main.loc[(main["model"] == model) & (main["strategy_name"] == strategy_name) & (main["weight_scheme"] == weight_scheme)]
        if block.empty or pd.isna(block.iloc[0][column]):
            return "N/A"
        return f"{float(block.iloc[0][column]):.4f}"

    text = f"""# Stage 7 Portfolio Results

## Scope

Stage 7 converts the saved OOS model predictions into investable monthly portfolio tests on the default `features500` main specification. The main comparison uses the common stock-date intersection across all models, so the portfolio results are as comparable as possible.

The default backtest uses month-`t` model scores to form portfolios at the end of month `t`, then applies realized month `t+1` returns from the cleaned Stage 2 panel. Value weights use `mcap_t`, which comes from the project market-cap data. Raw `BLACKLIST.pkl` and `UNTRADABLE.pkl` are aggregated to signal-month flags and can be used as additional formation filters.

## Core Result

At the current default settings, graph-enhanced exposure estimation does not yet produce a dominant investable winner across both long-short and long-only portfolios.

The strongest long-short performer at the main transaction-cost setting (`{main_cost_bps}` bps) is:

- `{best('long_short', 'annualized_return', higher=True)}` by annualized return
- `{best('long_short', 'sharpe_ratio', higher=True)}` by Sharpe ratio

The strongest long-only performer at the same setting is:

- `{best('long_only', 'annualized_return', higher=True)}` by annualized return
- `{best('long_only', 'sharpe_ratio', higher=True)}` by Sharpe ratio

## Graph Model Interpretation

The graph model should be read against the strongest characteristic-only alternatives, not against a naive predictor alone.

Examples from the saved summary table at `{main_cost_bps}` bps:

- Graph long-short equal-weight annualized return: `{metric('graph_conditional_latent_factor', 'long_short', 'equal', 'annualized_return')}`
- Graph long-short equal-weight Sharpe: `{metric('graph_conditional_latent_factor', 'long_short', 'equal', 'sharpe_ratio')}`
- Graph long-only value-weight annualized return: `{metric('graph_conditional_latent_factor', 'long_only', 'value', 'annualized_return')}`
- Graph long-only value-weight Sharpe: `{metric('graph_conditional_latent_factor', 'long_only', 'value', 'sharpe_ratio')}`

If the graph model outperforms, the current evidence should be interpreted mainly as improved **sorting quality**, because Stage 6 already suggested that graph structure helps rank-order stocks more than it improves raw OOS prediction fit. If the graph model does not dominate the portfolio table, that means the current graph beta function has not yet translated its ranking gains into a consistent investable edge after weighting, turnover, and transaction costs.

## Long-Short vs Long-Only

Long-short portfolios isolate cross-sectional sorting ability most directly. Long-only portfolios are more sensitive to whether the top-ranked names are robust enough to survive realistic weighting and costs.

In this stage, the right question is not only “which model has the highest return,” but also:

- whether the return comes with better Sharpe or just higher volatility,
- whether value weighting changes the ranking of models,
- whether turnover erodes the raw signal,
- and whether the graph model helps more in long-short ranking portfolios than in long-only implementations.

## Figures

"""
    for key, path in plot_paths.items():
        text += f"- `{project_relative_string(project_root, path)}`\n"
    text += """

## Strongest Remaining Limitations

1. The current portfolio test still relies on a short 24-month default OOS span.
2. The common stock-date intersection makes the comparison fairer, but it also narrows the investable universe.
3. The graph model is still a first-pass GCN/GAT latent-factor implementation rather than a stronger structural no-arbitrage system.
4. Turnover is computed with a simple one-way weight-change approximation and does not model post-return drift before rebalancing.
5. Transaction costs are only sensitivity adjustments, not a full microstructure model.
6. Because Stage 2 already filtered the main panel for feasibility, the raw blacklist/untradable overlays may have limited incremental effect in the default run.
7. There is still no broader-spec `features/` comparison in this stage.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def run_backtest(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run the Stage 7 portfolio backtest workflow."""

    started_at = time.time()
    config = load_config(config_path)
    portfolio_cfg = PortfolioConfig(
        long_short_quantile=float(config.get("portfolio", {}).get("long_short_quantile", 0.1)),
        long_only_quantile=float(config.get("portfolio", {}).get("long_only_quantile", 0.1)),
        main_transaction_cost_bps=int(config.get("portfolio", {}).get("main_transaction_cost_bps", 10)),
    )
    validate_portfolio_config(portfolio_cfg)

    predictions = load_prediction_outputs(project_root, config)
    aligned, signal_coverage = align_common_signal_panel(predictions)
    signals, merge_stats = merge_signal_inputs(project_root, aligned, config)
    signals["eligible"] = True
    if config.get("portfolio", {}).get("filter_blacklist", True):
        signals.loc[signals["blacklisted_raw_t"], "eligible"] = False
    if config.get("portfolio", {}).get("filter_untradable", True):
        signals.loc[signals["untradable_raw_t"], "eligible"] = False

    weights, strategy_coverage = build_all_weights(signals, config)
    monthly_returns = compute_monthly_portfolio_returns(weights)
    cost_grid = [int(value) for value in config.get("portfolio", {}).get("transaction_cost_bps_grid", [0, portfolio_cfg.main_transaction_cost_bps, 25])]
    net_returns = expand_transaction_cost_scenarios(monthly_returns, cost_grid)
    performance = summarize_performance(net_returns)

    outputs = config.get("outputs", {})
    table_dir = resolve_project_path(project_root, outputs.get("table_dir", "outputs/portfolio/stage7_tables"))
    plot_dir = resolve_project_path(project_root, outputs.get("plot_dir", "outputs/portfolio/stage7_plots"))
    weights_path = resolve_project_path(project_root, outputs.get("weights_path", "outputs/portfolio/stage7_weights.pkl"))
    returns_path = resolve_project_path(project_root, outputs.get("returns_path", "outputs/portfolio/stage7_monthly_returns.pkl"))
    performance_path = resolve_project_path(project_root, outputs.get("performance_path", "outputs/portfolio/stage7_performance_summary.csv"))
    signal_coverage_path = resolve_project_path(project_root, outputs.get("signal_coverage_path", "outputs/portfolio/stage7_signal_coverage.csv"))
    strategy_coverage_path = resolve_project_path(project_root, outputs.get("strategy_coverage_path", "outputs/portfolio/stage7_strategy_coverage.csv"))
    metadata_path = resolve_project_path(project_root, outputs.get("metadata_path", "outputs/metadata/stage7_portfolio_metadata.json"))
    report_path = resolve_project_path(project_root, outputs.get("report_path", "reports/stage7_portfolio_results.md"))
    for directory in [table_dir, plot_dir, weights_path.parent, returns_path.parent, performance_path.parent, metadata_path.parent, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    weights.to_pickle(weights_path)
    net_returns.to_pickle(returns_path)
    performance.to_csv(performance_path, index=False)
    signal_coverage.to_csv(signal_coverage_path, index=False)
    strategy_coverage.to_csv(strategy_coverage_path, index=False)
    plot_paths = plot_cumulative_returns(net_returns, plot_dir, portfolio_cfg.main_transaction_cost_bps)
    write_economic_value_note(project_root, performance, signal_coverage, plot_paths, report_path, portfolio_cfg.main_transaction_cost_bps)

    metadata = {
        "stage": "stage7_portfolio_backtest",
        "config_path": project_relative_string(project_root, config_path),
        "aligned_n_obs_per_model": int(signal_coverage["n_obs"].iloc[0]) if not signal_coverage.empty else 0,
        "aligned_n_months": int(signal_coverage["n_months"].iloc[0]) if not signal_coverage.empty else 0,
        "aligned_n_stocks": int(signal_coverage["n_unique_stocks"].iloc[0]) if not signal_coverage.empty else 0,
        "date_min": str(aligned["date"].min().date()) if not aligned.empty else None,
        "date_max": str(aligned["date"].max().date()) if not aligned.empty else None,
        "merge_stats": merge_stats,
        "transaction_cost_bps_grid": cost_grid,
        "main_transaction_cost_bps": portfolio_cfg.main_transaction_cost_bps,
        "outputs": {
            "weights": project_relative_string(project_root, weights_path),
            "monthly_returns": project_relative_string(project_root, returns_path),
            "performance": project_relative_string(project_root, performance_path),
            "signal_coverage": project_relative_string(project_root, signal_coverage_path),
            "strategy_coverage": project_relative_string(project_root, strategy_coverage_path),
            "report": project_relative_string(project_root, report_path),
            "plots": {key: project_relative_string(project_root, value) for key, value in plot_paths.items()},
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved Stage 7 portfolio outputs:")
    print(f"  weights:      {weights_path}")
    print(f"  returns:      {returns_path}")
    print(f"  performance:  {performance_path}")
    print(f"  report:       {report_path}")
    print(performance.to_string(index=False))
    return metadata


