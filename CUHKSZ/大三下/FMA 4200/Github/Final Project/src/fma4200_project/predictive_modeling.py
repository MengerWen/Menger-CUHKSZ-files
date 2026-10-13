from __future__ import annotations

import ast
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

from .config import (
    FAMA_FRENCH_FACTORS_PATH,
    PERCENT_COLUMNS,
    PORTFOLIO_GARCH_SUMMARY_PATH,
    PORTFOLIO_MODEL_COMPARISON_PATH,
    PORTFOLIO_TEST_SUMMARY_PATH,
    PREDICTIVE_FIGURES_DIR,
    PREDICTIVE_FORECAST_METRICS_PATH,
    PREDICTIVE_FORECASTS_PATH,
    PREDICTIVE_MODELING_LOG_PATH,
    PREDICTIVE_MODELS_DIR,
    PREDICTIVE_MODEL_SUMMARY_PATH,
    PREDICTIVE_TABLES_DIR,
    PREDICTOR_DATASET_PATH,
    PREDICTOR_SOURCE_SUMMARY_PATH,
)
from .data_pipeline import ensure_directories
from .univariate_modeling import (
    load_clean_returns,
    portfolio_label,
    portfolio_slug,
    write_appendix as write_individual_returns_appendix,
    write_section_03 as write_individual_returns_section,
)


OUT_OF_SAMPLE_MONTHS = 120
REFIT_FREQUENCY = 1
MIN_TRAIN_OBS = 240
FORECAST_LAGS = [12, 24]


def configure_predictive_logger() -> logging.Logger:
    logger = logging.getLogger("fma4200_predictive")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(PREDICTIVE_MODELING_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def predictive_paths(portfolio: str) -> dict[str, Path]:
    slug = portfolio_slug(portfolio)
    figure_dir = PREDICTIVE_FIGURES_DIR / slug
    table_dir = PREDICTIVE_TABLES_DIR / slug
    model_dir = PREDICTIVE_MODELS_DIR / slug
    for path in (figure_dir, table_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"figure_dir": figure_dir, "table_dir": table_dir, "model_dir": model_dir}


def normalize_month_end_dates(date_series: pd.Series) -> pd.Series:
    return pd.to_datetime(date_series).dt.to_period("M").dt.to_timestamp("M")


def predictor_source_label(source_flag: str) -> str:
    labels = {
        "authoritative_fama_french_cached": "cached authoritative Fama-French monthly factors",
        "authoritative_fama_french_downloaded": "downloaded authoritative Fama-French monthly factors",
        "internal_fallback_only": "internally constructed fallback predictors",
    }
    return labels.get(source_flag, source_flag.replace("_", " "))


def load_cached_fama_french(logger: logging.Logger) -> tuple[pd.DataFrame | None, str]:
    if FAMA_FRENCH_FACTORS_PATH.exists():
        factors = pd.read_csv(FAMA_FRENCH_FACTORS_PATH, parse_dates=["date"])
        factors["date"] = normalize_month_end_dates(factors["date"])
        logger.info("Loaded cached Fama-French factors from %s", FAMA_FRENCH_FACTORS_PATH)
        return factors, "authoritative_fama_french_cached"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pandas_datareader.data as web

            ff = web.DataReader("F-F_Research_Data_Factors", "famafrench", start="1926-07-01")[0].copy()
        ff.index = ff.index.to_timestamp(how="end")
        ff.index = pd.to_datetime(ff.index).to_period("M").to_timestamp("M")
        ff = ff.rename(
            columns={
                "Mkt-RF": "ff_mkt_rf_pct",
                "SMB": "ff_smb_pct",
                "HML": "ff_hml_pct",
                "RF": "ff_rf_pct",
            }
        ).reset_index(names="date")
        ff["date"] = normalize_month_end_dates(ff["date"])
        ff.to_csv(FAMA_FRENCH_FACTORS_PATH, index=False)
        logger.info("Downloaded and cached Fama-French factors to %s", FAMA_FRENCH_FACTORS_PATH)
        return ff, "authoritative_fama_french_downloaded"
    except Exception as exc:
        logger.warning("Unable to fetch Fama-French factors; using internal fallback predictors only. Error: %s", exc)
        return None, "internal_fallback_only"


def build_predictor_dataset(returns_df: pd.DataFrame, factors_df: pd.DataFrame | None, source_flag: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = returns_df.copy()
    df["market_proxy_pct"] = df[PERCENT_COLUMNS].mean(axis=1)
    df["small_avg_pct"] = df[["small_lobm_vwret_pct", "me1_bm2_vwret_pct", "small_hibm_vwret_pct"]].mean(axis=1)
    df["big_avg_pct"] = df[["big_lobm_vwret_pct", "me2_bm2_vwret_pct", "big_hibm_vwret_pct"]].mean(axis=1)
    df["lobm_avg_pct"] = df[["small_lobm_vwret_pct", "big_lobm_vwret_pct"]].mean(axis=1)
    df["hibm_avg_pct"] = df[["small_hibm_vwret_pct", "big_hibm_vwret_pct"]].mean(axis=1)
    df["size_spread_pct"] = df["small_avg_pct"] - df["big_avg_pct"]
    df["value_spread_pct"] = df["hibm_avg_pct"] - df["lobm_avg_pct"]
    df["market_vol_12m_pct"] = df["market_proxy_pct"].rolling(12).std()
    df["market_momentum_12m_pct"] = df["market_proxy_pct"].rolling(12).mean()
    market_growth = (1.0 + df["market_proxy_pct"] / 100.0).cumprod()
    df["market_drawdown"] = market_growth / market_growth.cummax() - 1.0

    if factors_df is not None:
        df = df.merge(factors_df, on="date", how="left")
        df["ff_mkt_total_pct"] = df["ff_mkt_rf_pct"] + df["ff_rf_pct"]

    lag_base_cols = [
        "market_proxy_pct",
        "size_spread_pct",
        "value_spread_pct",
        "market_vol_12m_pct",
        "market_momentum_12m_pct",
        "market_drawdown",
    ]
    if factors_df is not None:
        lag_base_cols.extend(["ff_mkt_rf_pct", "ff_smb_pct", "ff_hml_pct", "ff_rf_pct", "ff_mkt_total_pct"])

    for column in lag_base_cols:
        df[f"{column.replace('_pct', '')}_lag1_pct" if column.endswith("_pct") else f"{column}_lag1"] = df[column].shift(1)

    factor_predictors = []
    if factors_df is not None:
        factor_predictors = [
            "ff_mkt_rf_lag1_pct",
            "ff_smb_lag1_pct",
            "ff_hml_lag1_pct",
            "ff_rf_lag1_pct",
        ]
    internal_predictors = [
        "value_spread_lag1_pct",
        "size_spread_lag1_pct",
        "market_vol_12m_lag1_pct",
        "market_momentum_12m_lag1_pct",
        "market_drawdown_lag1",
    ]

    df["predictor_source"] = source_flag
    df.to_csv(PREDICTOR_DATASET_PATH, index=False)

    source_summary = pd.DataFrame(
        [
            {
                "predictor_source": source_flag,
                "fama_french_available": bool(factors_df is not None),
                "sample_start": df["date"].min().date().isoformat(),
                "sample_end": df["date"].max().date().isoformat(),
                "n_rows": len(df),
                "factor_columns": ", ".join(factor_predictors) if factor_predictors else "",
                "internal_columns": ", ".join(internal_predictors),
            }
        ]
    )
    source_summary.to_csv(PREDICTOR_SOURCE_SUMMARY_PATH, index=False)
    return df, factor_predictors, internal_predictors


def load_univariate_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_summary = pd.read_csv(PORTFOLIO_TEST_SUMMARY_PATH)
    model_summary = pd.read_csv(PORTFOLIO_MODEL_COMPARISON_PATH)
    garch_summary = pd.read_csv(PORTFOLIO_GARCH_SUMMARY_PATH)
    return test_summary, model_summary, garch_summary


def parse_arima_order(order_text: str) -> tuple[int, int, int]:
    return ast.literal_eval(order_text)


def fit_arima_model(
    y: pd.Series,
    order: tuple[int, int, int],
    exog: pd.DataFrame | None = None,
):
    if exog is not None and exog.shape[1] == 0:
        exog = None
    trend = "n" if order[1] > 0 else "c"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(
            y,
            exog=exog,
            order=order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(method_kwargs={"maxiter": 300})
    return result


def build_portfolio_modeling_frame(
    predictor_df: pd.DataFrame,
    portfolio: str,
    factor_predictors: list[str],
    internal_predictors: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = predictor_df.copy()
    df["target_pct"] = df[portfolio]
    df["target_lag1_pct"] = df["target_pct"].shift(1)

    arimax_predictors = factor_predictors if factor_predictors else internal_predictors[:4]
    regression_predictors = ["target_lag1_pct", *factor_predictors, *internal_predictors[:4]]
    if not factor_predictors:
        regression_predictors = ["target_lag1_pct", *internal_predictors[:4]]
    regression_predictors = list(dict.fromkeys(regression_predictors))

    needed_cols = ["date", "target_pct", *arimax_predictors, *regression_predictors]
    modeling_df = df.loc[:, list(dict.fromkeys(needed_cols))].dropna().reset_index(drop=True)
    return modeling_df, arimax_predictors, regression_predictors


def fit_regression_model(y: pd.Series, x: pd.DataFrame):
    x_with_const = sm.add_constant(x, has_constant="add")
    return sm.OLS(y, x_with_const).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def arima_parameter_table(result) -> pd.DataFrame:
    params = np.asarray(result.params)
    bse = np.asarray(result.bse)
    pvalues = np.asarray(result.pvalues)
    z_stats = np.divide(params, bse, out=np.full_like(params, np.nan, dtype=float), where=bse != 0.0)
    return pd.DataFrame(
        {
            "parameter": list(result.param_names),
            "estimate": params,
            "std_error": bse,
            "z_stat": z_stats,
            "pvalue": pvalues,
        }
    )


def regression_parameter_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": result.params.index,
            "estimate": result.params.values,
            "std_error": result.bse.values,
            "t_stat": result.tvalues.values,
            "pvalue": result.pvalues.values,
        }
    )


def compute_ljung_box(series: pd.Series) -> dict[str, float]:
    lb = acorr_ljungbox(series, lags=FORECAST_LAGS, return_df=True)
    return {
        "lb_stat_lag_12": float(lb.loc[12, "lb_stat"]),
        "lb_pvalue_lag_12": float(lb.loc[12, "lb_pvalue"]),
        "lb_stat_lag_24": float(lb.loc[24, "lb_stat"]),
        "lb_pvalue_lag_24": float(lb.loc[24, "lb_pvalue"]),
    }


def parameter_significance_share(parameter_table: pd.DataFrame) -> float:
    relevant = parameter_table.loc[
        ~parameter_table["parameter"].isin(["const", "sigma2", "intercept"])
    ].copy()
    if relevant.empty:
        return 1.0
    return float((relevant["pvalue"] < 0.05).mean())


def summarize_significant_terms(parameter_table: pd.DataFrame) -> str:
    relevant = parameter_table.loc[
        (~parameter_table["parameter"].isin(["const", "sigma2", "intercept"]))
        & (parameter_table["pvalue"] < 0.10)
    ].copy()
    if relevant.empty:
        return "No non-constant terms are significant at the 10% level."

    stat_col = "t_stat" if "t_stat" in relevant.columns else "z_stat"
    relevant["abs_stat"] = np.abs(relevant[stat_col])
    relevant = relevant.sort_values(["pvalue", "abs_stat"], ascending=[True, False]).head(3)

    parts: list[str] = []
    for row in relevant.itertuples(index=False):
        sign = "positive" if row.estimate >= 0.0 else "negative"
        parts.append(f"{row.parameter} ({sign}, p={row.pvalue:.3f})")
    return "; ".join(parts)


def compute_forecast_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    paired = pd.DataFrame({"actual": actual, "forecast": forecast}).dropna()
    errors = paired["actual"] - paired["forecast"]
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "directional_accuracy": float(np.mean(np.sign(paired["actual"]) == np.sign(paired["forecast"]))),
        "n_forecasts": int(len(paired)),
    }


def in_sample_metrics(actual: pd.Series, fitted: pd.Series) -> dict[str, float]:
    paired = pd.DataFrame({"actual": actual, "fitted": fitted}).dropna()
    errors = paired["actual"] - paired["fitted"]
    return {
        "in_sample_rmse": float(np.sqrt(np.mean(errors**2))),
        "in_sample_mae": float(np.mean(np.abs(errors))),
    }


def candidate_model_specs(
    arimax_predictors: list[str],
    regression_predictors: list[str],
    factor_predictors: list[str],
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = [
        {
            "model_key": "benchmark_arima",
            "model_label": "Benchmark ARIMA",
            "model_type": "arima",
            "predictors": [],
            "predictor_set": "none",
        }
    ]

    if factor_predictors:
        arimax_label = "ARIMAX with lagged Fama-French factors"
        regression_label = "Predictive regression with lagged factors and internal signals"
        predictor_set = "fama_french_plus_internal"
    else:
        arimax_label = "ARIMAX with internal fallback predictors"
        regression_label = "Predictive regression with internal fallback predictors"
        predictor_set = "internal_fallback"

    specs.append(
        {
            "model_key": "arimax_exog",
            "model_label": arimax_label,
            "model_type": "arima",
            "predictors": arimax_predictors,
            "predictor_set": predictor_set,
        }
    )
    specs.append(
        {
            "model_key": "predictive_regression",
            "model_label": regression_label,
            "model_type": "regression",
            "predictors": regression_predictors,
            "predictor_set": predictor_set,
        }
    )
    return specs


def fit_full_sample_candidate(
    modeling_df: pd.DataFrame,
    portfolio: str,
    spec: dict[str, object],
    order: tuple[int, int, int],
) -> dict[str, object]:
    indexed = modeling_df.set_index("date")
    y = indexed["target_pct"]
    predictors = list(spec["predictors"])
    x = indexed[predictors] if predictors else None

    if spec["model_type"] == "arima":
        result = fit_arima_model(y, order, x)
        fitted = pd.Series(result.fittedvalues, index=y.index)
        residuals = pd.Series(result.resid, index=y.index)
        parameter_table = arima_parameter_table(result)
        summary_text = result.summary().as_text()
        llf = float(result.llf)
        aic = float(result.aic)
        bic = float(result.bic)
    else:
        result = fit_regression_model(y, x)
        fitted = pd.Series(result.fittedvalues, index=y.index)
        residuals = pd.Series(result.resid, index=y.index)
        parameter_table = regression_parameter_table(result)
        summary_text = result.summary().as_text()
        llf = float(result.llf)
        aic = float(result.aic)
        bic = float(result.bic)

    diagnostics = compute_ljung_box(residuals)
    metrics = in_sample_metrics(y, fitted)
    return {
        "portfolio": portfolio,
        "model_key": spec["model_key"],
        "model_label": spec["model_label"],
        "model_type": spec["model_type"],
        "predictor_set": spec["predictor_set"],
        "predictors": predictors,
        "order": order,
        "n_predictors": len(predictors),
        "n_obs": int(len(indexed)),
        "result": result,
        "fitted": fitted,
        "residuals": residuals,
        "parameter_table": parameter_table,
        "summary_text": summary_text,
        "aic": aic,
        "bic": bic,
        "llf": llf,
        "significant_share": parameter_significance_share(parameter_table),
        "top_significant_terms": summarize_significant_terms(parameter_table),
        **metrics,
        **diagnostics,
    }


def generate_expanding_forecasts(
    modeling_df: pd.DataFrame,
    portfolio: str,
    spec: dict[str, object],
    order: tuple[int, int, int],
    logger: logging.Logger,
) -> pd.DataFrame:
    indexed = modeling_df.set_index("date")
    y = indexed["target_pct"]
    predictors = list(spec["predictors"])
    x = indexed[predictors] if predictors else None

    oos_months = min(OUT_OF_SAMPLE_MONTHS, len(indexed) - MIN_TRAIN_OBS)
    if oos_months <= 0:
        raise ValueError(f"Not enough observations for out-of-sample evaluation of {portfolio}.")

    train_end = len(indexed) - oos_months
    rows: list[dict[str, object]] = []

    for step, idx in enumerate(range(train_end, len(indexed))):
        date = indexed.index[idx]
        y_train = y.iloc[:idx]
        x_train = x.iloc[:idx] if x is not None else None

        if spec["model_type"] == "arima":
            fitted_result = fit_arima_model(y_train, order, x_train)
            x_next = x.iloc[[idx]] if x is not None else None
            forecast_result = fitted_result.forecast(steps=1, exog=x_next)
            forecast_value = float(pd.Series(forecast_result).iloc[0])
            actual_value = float(y.iloc[idx])
        else:
            fitted_result = fit_regression_model(y_train, x_train)
            x_next = sm.add_constant(x.iloc[[idx]], has_constant="add")
            forecast_value = float(np.asarray(fitted_result.predict(x_next))[0])
            actual_value = float(y.iloc[idx])

        rows.append(
            {
                "portfolio": portfolio,
                "model_key": spec["model_key"],
                "model_label": spec["model_label"],
                "date": date,
                "actual_pct": actual_value,
                "forecast_pct": forecast_value,
                "forecast_error_pct": actual_value - forecast_value,
            }
        )

    return pd.DataFrame(rows)


def select_preferred_predictive_model(merged_metrics: pd.DataFrame) -> str:
    predictive = merged_metrics.loc[merged_metrics["model_key"] != "benchmark_arima"].copy()
    if predictive.empty:
        return "benchmark_arima"

    best_rmse = float(predictive["rmse"].min())
    finalists = predictive.loc[predictive["rmse"] <= best_rmse * 1.01].copy()
    acceptable_residuals = finalists.loc[finalists["lb_pvalue_lag_12"] >= 0.05].copy()
    if not acceptable_residuals.empty:
        finalists = acceptable_residuals

    finalists = finalists.sort_values(
        ["rmse", "mae", "bic", "directional_accuracy", "n_predictors"],
        ascending=[True, True, True, False, True],
    )
    return str(finalists.iloc[0]["model_key"])


def save_forecast_plot(forecast_df: pd.DataFrame, figure_path: Path, title: str) -> None:
    plot_df = forecast_df.copy().sort_values("date")
    actual = plot_df[["date", "actual_pct"]].drop_duplicates().sort_values("date")
    pivot = plot_df.pivot(index="date", columns="model_label", values="forecast_pct")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(actual["date"], actual["actual_pct"], color="black", linewidth=1.3, label="Actual return")
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column], linewidth=1.0, alpha=0.9, label=column)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title}: Expanding-Window Forecast Comparison")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_fit_plot(
    actual: pd.Series,
    benchmark_fitted: pd.Series,
    preferred_fitted: pd.Series,
    preferred_label: str,
    figure_path: Path,
    title: str,
) -> None:
    recent_actual = actual.tail(240)
    recent_benchmark = benchmark_fitted.reindex(recent_actual.index)
    recent_preferred = preferred_fitted.reindex(recent_actual.index)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(recent_actual.index, recent_actual.values, color="black", linewidth=1.1, label="Actual return")
    ax.plot(recent_benchmark.index, recent_benchmark.values, color="#1f5aa6", linewidth=1.0, label="Benchmark ARIMA fitted")
    ax.plot(recent_preferred.index, recent_preferred.values, color="#ba4a00", linewidth=1.0, label=preferred_label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{title}: Recent In-Sample Fitted Comparison")
    ax.set_ylabel("Return (%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(
    residuals: pd.Series,
    figure_path: Path,
    title: str,
    model_label: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    axes[0].plot(residuals.index, residuals.values, color="#7a1f5c", linewidth=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title(f"{title}: {model_label} Residuals")
    plot_acf(residuals, ax=axes[1], lags=24, zero=False)
    axes[1].set_title(f"{title}: {model_label} Residual ACF")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_interpretation_text(
    portfolio: str,
    predictor_source: str,
    merged_metrics: pd.DataFrame,
    preferred_model_key: str,
) -> str:
    source_label = predictor_source_label(predictor_source)
    benchmark_row = merged_metrics.loc[merged_metrics["model_key"] == "benchmark_arima"].iloc[0]
    preferred_row = merged_metrics.loc[merged_metrics["model_key"] == preferred_model_key].iloc[0]
    rmse_improvement = 100.0 * (benchmark_row["rmse"] - preferred_row["rmse"]) / benchmark_row["rmse"]
    mae_improvement = 100.0 * (benchmark_row["mae"] - preferred_row["mae"]) / benchmark_row["mae"]

    if preferred_model_key == "benchmark_arima":
        comparison_text = "The predictive extensions do not outperform the univariate benchmark on the current out-of-sample RMSE criterion."
    elif rmse_improvement > 0.0:
        comparison_text = (
            f"The preferred predictive model improves out-of-sample RMSE by {rmse_improvement:.2f}% "
            f"and MAE by {mae_improvement:.2f}% relative to the univariate benchmark."
        )
    else:
        comparison_text = (
            f"The preferred predictive model remains economically interpretable, but the benchmark still wins on RMSE "
            f"by {abs(rmse_improvement):.2f}%."
        )

    return "\n".join(
        [
            f"# {portfolio_label(portfolio)} Predictive Modeling",
            "",
            f"- Predictor source: {source_label}.",
            f"- Benchmark model: {benchmark_row['model_label']} with out-of-sample RMSE = {benchmark_row['rmse']:.4f}, MAE = {benchmark_row['mae']:.4f}, and directional accuracy = {benchmark_row['directional_accuracy']:.3f}.",
            f"- Preferred predictive model: {preferred_row['model_label']} with out-of-sample RMSE = {preferred_row['rmse']:.4f}, MAE = {preferred_row['mae']:.4f}, and directional accuracy = {preferred_row['directional_accuracy']:.3f}.",
            f"- Full-sample fit: AIC = {preferred_row['aic']:.2f}, BIC = {preferred_row['bic']:.2f}, Ljung-Box p-value at lag 12 = {preferred_row['lb_pvalue_lag_12']:.3f}, significant-share = {preferred_row['significant_share']:.3f}.",
            f"- Most informative terms: {preferred_row['top_significant_terms']}",
            f"- Interpretation: {comparison_text}",
        ]
    )


def estimate_portfolio_predictive_models(
    predictor_df: pd.DataFrame,
    portfolio: str,
    selected_order: tuple[int, int, int],
    factor_predictors: list[str],
    internal_predictors: list[str],
    predictor_source: str,
    logger: logging.Logger,
) -> dict[str, object]:
    title = portfolio_label(portfolio)
    paths = predictive_paths(portfolio)
    modeling_df, arimax_predictors, regression_predictors = build_portfolio_modeling_frame(
        predictor_df,
        portfolio,
        factor_predictors,
        internal_predictors,
    )
    specs = candidate_model_specs(arimax_predictors, regression_predictors, factor_predictors)

    summary_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    forecast_frames: list[pd.DataFrame] = []
    result_lookup: dict[str, dict[str, object]] = {}

    logger.info("Estimating predictive models for %s", portfolio)
    for spec in specs:
        candidate = fit_full_sample_candidate(modeling_df, portfolio, spec, selected_order)
        forecast_df = generate_expanding_forecasts(modeling_df, portfolio, spec, selected_order, logger)
        forecast_metrics = compute_forecast_metrics(forecast_df["actual_pct"], forecast_df["forecast_pct"])

        summary_rows.append(
            {
                "portfolio": portfolio,
                "model_key": candidate["model_key"],
                "model_label": candidate["model_label"],
                "model_type": candidate["model_type"],
                "selected_arima_order": str(selected_order),
                "predictor_source": predictor_source,
                "predictor_set": candidate["predictor_set"],
                "predictor_columns": ", ".join(candidate["predictors"]),
                "n_predictors": candidate["n_predictors"],
                "n_obs": candidate["n_obs"],
                "aic": candidate["aic"],
                "bic": candidate["bic"],
                "llf": candidate["llf"],
                "in_sample_rmse": candidate["in_sample_rmse"],
                "in_sample_mae": candidate["in_sample_mae"],
                "lb_pvalue_lag_12": candidate["lb_pvalue_lag_12"],
                "lb_pvalue_lag_24": candidate["lb_pvalue_lag_24"],
                "significant_share": candidate["significant_share"],
                "top_significant_terms": candidate["top_significant_terms"],
            }
        )
        metric_rows.append(
            {
                "portfolio": portfolio,
                "model_key": candidate["model_key"],
                "model_label": candidate["model_label"],
                **forecast_metrics,
            }
        )
        forecast_frames.append(forecast_df)
        result_lookup[candidate["model_key"]] = candidate

        save_dataframe(candidate["parameter_table"], paths["table_dir"] / f"{candidate['model_key']}_parameters.csv")
        write_text_file(paths["model_dir"] / f"{candidate['model_key']}_summary.txt", candidate["summary_text"] + "\n")

    summary_df = pd.DataFrame(summary_rows)
    metrics_df = pd.DataFrame(metric_rows)
    merged_metrics = summary_df.merge(
        metrics_df,
        on=["portfolio", "model_key", "model_label"],
        how="left",
    )

    preferred_model_key = select_preferred_predictive_model(merged_metrics)
    merged_metrics["is_preferred_predictive"] = merged_metrics["model_key"] == preferred_model_key
    merged_metrics["is_benchmark"] = merged_metrics["model_key"] == "benchmark_arima"

    benchmark_rmse = float(merged_metrics.loc[merged_metrics["model_key"] == "benchmark_arima", "rmse"].iloc[0])
    merged_metrics["rmse_vs_benchmark_pct"] = 100.0 * (merged_metrics["rmse"] - benchmark_rmse) / benchmark_rmse

    forecast_output = pd.concat(forecast_frames, ignore_index=True)
    save_dataframe(merged_metrics, paths["table_dir"] / "model_comparison_full_sample_and_oos.csv")
    save_dataframe(metrics_df, paths["table_dir"] / "forecast_metrics.csv")
    save_dataframe(forecast_output, paths["table_dir"] / "forecast_paths.csv")

    preferred_result = result_lookup[preferred_model_key]
    benchmark_result = result_lookup["benchmark_arima"]
    actual_series = modeling_df.set_index("date")["target_pct"]

    write_text_file(
        paths["model_dir"] / "interpretation.md",
        build_interpretation_text(portfolio, predictor_source, merged_metrics, preferred_model_key) + "\n",
    )

    save_forecast_plot(forecast_output, paths["figure_dir"] / "oos_forecast_comparison.png", title)
    save_fit_plot(
        actual_series,
        benchmark_result["fitted"],
        preferred_result["fitted"],
        preferred_result["model_label"],
        paths["figure_dir"] / "recent_fitted_comparison.png",
        title,
    )
    save_residual_plot(
        preferred_result["residuals"],
        paths["figure_dir"] / "preferred_model_residuals.png",
        title,
        preferred_result["model_label"],
    )

    return {
        "summary_df": merged_metrics,
        "metrics_df": metrics_df,
        "forecast_df": forecast_output,
    }


def run_predictive_modeling_pipeline() -> dict[str, object]:
    ensure_directories()
    logger = configure_predictive_logger()
    logger.info("Starting predictive modeling pipeline.")

    returns_df = load_clean_returns()
    test_summary, model_summary, garch_summary = load_univariate_summaries()
    factors_df, predictor_source = load_cached_fama_french(logger)
    predictor_df, factor_predictors, internal_predictors = build_predictor_dataset(
        returns_df,
        factors_df,
        predictor_source,
    )

    portfolio_summaries: list[pd.DataFrame] = []
    portfolio_metrics: list[pd.DataFrame] = []
    portfolio_forecasts: list[pd.DataFrame] = []

    for portfolio in PERCENT_COLUMNS:
        order_text = model_summary.loc[model_summary["portfolio"] == portfolio, "selected_arima_order"].iloc[0]
        selected_order = parse_arima_order(order_text)
        result = estimate_portfolio_predictive_models(
            predictor_df,
            portfolio,
            selected_order,
            factor_predictors,
            internal_predictors,
            predictor_source,
            logger,
        )
        portfolio_summaries.append(result["summary_df"])
        portfolio_metrics.append(result["metrics_df"])
        portfolio_forecasts.append(result["forecast_df"])

    predictive_summary = pd.concat(portfolio_summaries, ignore_index=True)
    predictive_metrics = pd.concat(portfolio_metrics, ignore_index=True)
    predictive_forecasts = pd.concat(portfolio_forecasts, ignore_index=True)

    save_dataframe(predictive_summary, PREDICTIVE_MODEL_SUMMARY_PATH)
    save_dataframe(predictive_metrics, PREDICTIVE_FORECAST_METRICS_PATH)
    save_dataframe(predictive_forecasts, PREDICTIVE_FORECASTS_PATH)

    write_individual_returns_section(test_summary, model_summary, garch_summary, predictive_summary)
    write_individual_returns_appendix(test_summary, model_summary, garch_summary, predictive_summary)

    logger.info("Predictive modeling pipeline completed successfully.")
    return {
        "n_portfolios": len(PERCENT_COLUMNS),
        "predictor_source": predictor_source,
        "predictor_dataset_path": str(PREDICTOR_DATASET_PATH),
        "predictive_summary_path": str(PREDICTIVE_MODEL_SUMMARY_PATH),
        "predictive_metrics_path": str(PREDICTIVE_FORECAST_METRICS_PATH),
        "predictive_forecasts_path": str(PREDICTIVE_FORECASTS_PATH),
    }
