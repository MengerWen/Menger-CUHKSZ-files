from __future__ import annotations

import ast
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GeneralizedError, Normal, SkewStudent, StudentsT
from scipy.stats import anderson, gaussian_kde, kstest, norm, norminvgauss, shapiro, t as student_t
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from .config import (
    APPENDIX_INDIVIDUAL_PATH,
    CLEAN_PERCENT_DATA_PATH,
    INDIVIDUAL_FIGURES_DIR,
    INDIVIDUAL_MODELING_LOG_PATH,
    INDIVIDUAL_MODELS_DIR,
    INDIVIDUAL_TABLES_DIR,
    PERCENT_COLUMNS,
    PORTFOLIO_GARCH_SUMMARY_PATH,
    PORTFOLIO_MODEL_COMPARISON_PATH,
    PORTFOLIO_TEST_SUMMARY_PATH,
    SECTION_03_PATH,
)
from .data_pipeline import ensure_directories, extract_monthly_value_weighted_data


ANNUALIZATION_FACTOR = np.sqrt(12.0)
LJUNG_BOX_LAGS = [12, 24]
ARCH_TEST_LAGS = 12
MAX_AR_ORDER = 2
MAX_MA_ORDER = 2
VOLATILITY_MODELS = (
    ("Gaussian GARCH(1,1)", "normal"),
    ("Student-t GARCH(1,1)", "t"),
    ("Skewed Student-t GARCH(1,1)", "skewt"),
    ("GED GARCH(1,1)", "ged"),
)

SHORT_PREDICTIVE_LABELS = {
    "Benchmark ARIMA": "Benchmark ARIMA",
    "ARIMAX with lagged Fama-French factors": "ARIMAX",
    "ARIMAX with internal fallback predictors": "ARIMAX",
    "Predictive regression with lagged factors and internal signals": "Predictive regression",
    "Predictive regression with internal fallback predictors": "Predictive regression",
}


@dataclass
class DistributionFitResult:
    distribution: str
    distribution_code: str
    loglik: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    params: dict[str, float]
    parameter_text: str
    complexity_rank: int


@dataclass
class VolatilityFitResult:
    model_label: str
    distribution: str
    distribution_code: str
    success: bool
    convergence_flag: int
    loglik: float
    aic: float
    bic: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    nu: float | None
    distribution_params: dict[str, float]
    distribution_param_text: str
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    residuals: np.ndarray
    param_table: pd.DataFrame
    summary_text: str
    diagnostics: dict[str, float]


def configure_modeling_logger() -> logging.Logger:
    logger = logging.getLogger("fma4200_univariate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(INDIVIDUAL_MODELING_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def portfolio_slug(portfolio: str) -> str:
    return portfolio.replace("_vwret_pct", "")


def portfolio_label(portfolio: str) -> str:
    labels = {
        "small_lobm_vwret_pct": "Small LoBM",
        "me1_bm2_vwret_pct": "ME1 BM2",
        "small_hibm_vwret_pct": "Small HiBM",
        "big_lobm_vwret_pct": "Big LoBM",
        "me2_bm2_vwret_pct": "ME2 BM2",
        "big_hibm_vwret_pct": "Big HiBM",
    }
    return labels.get(portfolio, portfolio.replace("_vwret_pct", "").replace("_", " ").title())


def predictor_source_label(source_flag: str) -> str:
    labels = {
        "authoritative_fama_french_cached": "cached authoritative Fama-French monthly factors",
        "authoritative_fama_french_downloaded": "downloaded authoritative Fama-French monthly factors",
        "internal_fallback_only": "internally constructed fallback predictors",
    }
    return labels.get(source_flag, source_flag.replace("_", " "))


def format_pvalue(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(dataframe: pd.DataFrame) -> str:
    headers = list(dataframe.columns)
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in dataframe.astype(str).values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def portfolio_paths(portfolio: str) -> dict[str, Path]:
    slug = portfolio_slug(portfolio)
    figure_dir = INDIVIDUAL_FIGURES_DIR / slug
    table_dir = INDIVIDUAL_TABLES_DIR / slug
    model_dir = INDIVIDUAL_MODELS_DIR / slug
    for path in (figure_dir, table_dir, model_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"figure_dir": figure_dir, "table_dir": table_dir, "model_dir": model_dir}


def load_clean_returns() -> pd.DataFrame:
    if CLEAN_PERCENT_DATA_PATH.exists():
        df = pd.read_csv(CLEAN_PERCENT_DATA_PATH, parse_dates=["date"])
    else:
        df = extract_monthly_value_weighted_data()
        df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def safe_kpss(series: pd.Series) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, _, _ = kpss(series, regression="c", nlags="auto")
    return float(stat), float(pvalue)


def safe_adf(series: pd.Series) -> tuple[float, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, pvalue, _, _, _, _ = adfuller(series, autolag="AIC")
    return float(stat), float(pvalue)


def compute_ljung_box(series: pd.Series) -> dict[str, float]:
    clean = pd.Series(series, copy=False).dropna()
    lb = acorr_ljungbox(clean, lags=LJUNG_BOX_LAGS, return_df=True)
    output: dict[str, float] = {}
    for lag in LJUNG_BOX_LAGS:
        output[f"lb_stat_lag_{lag}"] = float(lb.loc[lag, "lb_stat"])
        output[f"lb_pvalue_lag_{lag}"] = float(lb.loc[lag, "lb_pvalue"])
    return output


def compute_arch_test(series: pd.Series) -> dict[str, float]:
    clean = pd.Series(series, copy=False).dropna()
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(clean, nlags=ARCH_TEST_LAGS)
    return {
        "arch_lm_stat": float(lm_stat),
        "arch_lm_pvalue": float(lm_pvalue),
        "arch_f_stat": float(f_stat),
        "arch_f_pvalue": float(f_pvalue),
    }


def format_parameter_text(params: dict[str, float]) -> str:
    if not params:
        return ""
    return ", ".join(f"{name}={value:.4f}" for name, value in params.items())


def marginal_distribution_label(distribution_code: str) -> str:
    mapping = {
        "normal": "Normal",
        "t": "Student-t",
        "nig": "NIG",
    }
    return mapping[distribution_code]


def marginal_distribution_complexity(distribution_code: str) -> int:
    mapping = {
        "normal": 0,
        "t": 1,
        "nig": 2,
    }
    return mapping[distribution_code]


def arch_distribution_label(distribution_code: str) -> str:
    mapping = {
        "normal": "Gaussian",
        "t": "Student-t",
        "skewt": "Skewed Student-t",
        "ged": "GED",
    }
    return mapping[distribution_code]


def arch_distribution_complexity(distribution_code: str) -> int:
    mapping = {
        "normal": 0,
        "t": 1,
        "ged": 1,
        "skewt": 2,
    }
    return mapping[distribution_code]


def arch_distribution_parameter_names(distribution_code: str) -> list[str]:
    mapping = {
        "normal": [],
        "t": ["nu"],
        "skewt": ["eta", "lambda"],
        "ged": ["nu"],
    }
    return mapping[distribution_code]


def arch_distribution_object(distribution_code: str):
    mapping = {
        "normal": Normal(),
        "t": StudentsT(),
        "skewt": SkewStudent(),
        "ged": GeneralizedError(),
    }
    return mapping[distribution_code]


def arch_distribution_parameter_vector(distribution_code: str, params: dict[str, float]) -> np.ndarray | None:
    names = arch_distribution_parameter_names(distribution_code)
    if not names:
        return None
    return np.array([params[name] for name in names], dtype=float)


def arch_distribution_cdf(
    values: np.ndarray | pd.Series,
    distribution_code: str,
    params: dict[str, float],
) -> np.ndarray:
    distribution = arch_distribution_object(distribution_code)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, params)
    return np.asarray(distribution.cdf(values, parameters=parameter_vector), dtype=float)


def arch_distribution_ppf(
    probabilities: np.ndarray,
    distribution_code: str,
    params: dict[str, float],
) -> np.ndarray:
    distribution = arch_distribution_object(distribution_code)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, params)
    return np.asarray(distribution.ppf(probabilities, parameters=parameter_vector), dtype=float)


def fit_distribution(values: np.ndarray, distribution_code: str) -> DistributionFitResult:
    n_obs = len(values)
    if distribution_code == "normal":
        loc, scale = norm.fit(values)
        loglik = float(np.sum(norm.logpdf(values, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(values, lambda x: norm.cdf(x, loc=loc, scale=scale))
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="normal",
            loglik=loglik,
            aic=float(2 * 2 - 2 * loglik),
            bic=float(np.log(n_obs) * 2 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params={
                "loc": float(loc),
                "scale": float(scale),
            },
            parameter_text=format_parameter_text({"loc": float(loc), "scale": float(scale)}),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    if distribution_code == "t":
        df_param, loc, scale = student_t.fit(values)
        loglik = float(np.sum(student_t.logpdf(values, df=df_param, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(values, lambda x: student_t.cdf(x, df=df_param, loc=loc, scale=scale))
        params = {
            "df": float(df_param),
            "loc": float(loc),
            "scale": float(scale),
        }
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="t",
            loglik=loglik,
            aic=float(2 * 3 - 2 * loglik),
            bic=float(np.log(n_obs) * 3 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params=params,
            parameter_text=format_parameter_text(params),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    if distribution_code == "nig":
        alpha, beta, loc, scale = norminvgauss.fit(values)
        loglik = float(np.sum(norminvgauss.logpdf(values, alpha, beta, loc=loc, scale=scale)))
        ks_stat, ks_pvalue = kstest(
            values,
            lambda x: norminvgauss.cdf(x, alpha, beta, loc=loc, scale=scale),
        )
        params = {
            "alpha": float(alpha),
            "beta": float(beta),
            "loc": float(loc),
            "scale": float(scale),
        }
        return DistributionFitResult(
            distribution=marginal_distribution_label(distribution_code),
            distribution_code="nig",
            loglik=loglik,
            aic=float(2 * 4 - 2 * loglik),
            bic=float(np.log(n_obs) * 4 - 2 * loglik),
            ks_stat=float(ks_stat),
            ks_pvalue=float(ks_pvalue),
            params=params,
            parameter_text=format_parameter_text(params),
            complexity_rank=marginal_distribution_complexity(distribution_code),
        )

    raise ValueError(f"Unsupported distribution code: {distribution_code}")


def distribution_results_table(
    results: list[DistributionFitResult],
) -> tuple[pd.DataFrame, DistributionFitResult, str]:
    rows = []
    for result in results:
        rows.append(
            {
                "distribution": result.distribution,
                "distribution_code": result.distribution_code,
                "parameter_estimates": result.parameter_text,
                "loglik": result.loglik,
                "aic": result.aic,
                "bic": result.bic,
                "ks_stat": result.ks_stat,
                "ks_pvalue": result.ks_pvalue,
                "complexity_rank": result.complexity_rank,
            }
        )
    table = pd.DataFrame(rows)
    best_aic = float(table["aic"].min())
    shortlist = table.loc[table["aic"] <= best_aic + 4.0].copy()
    good_fit = shortlist.loc[shortlist["ks_pvalue"] >= 0.05].copy()

    if not good_fit.empty:
        chosen_row = good_fit.sort_values(
            ["ks_pvalue", "bic", "complexity_rank", "aic"],
            ascending=[False, True, True, True],
        ).iloc[0]
        selection_reason = (
            "highest KS p-value among distributions within 4 AIC points of the best fit, "
            "with BIC and parsimony used as tie-breakers"
        )
    else:
        chosen_row = shortlist.sort_values(
            ["ks_stat", "bic", "complexity_rank", "aic"],
            ascending=[True, True, True, True],
        ).iloc[0]
        selection_reason = (
            "lowest KS distance among distributions within 4 AIC points of the best fit "
            "because no candidate passed the KS threshold"
        )

    selected_code = str(chosen_row["distribution_code"])
    selected_result = next(result for result in results if result.distribution_code == selected_code)
    table["recommended"] = table["distribution_code"] == selected_code
    table["selection_reason"] = ""
    table.loc[table["recommended"], "selection_reason"] = selection_reason
    table = table.sort_values(["recommended", "aic", "bic"], ascending=[False, True, True]).reset_index(drop=True)
    return table, selected_result, selection_reason


def descriptive_row(
    series: pd.Series,
) -> tuple[dict[str, float | str], pd.DataFrame, dict[str, DistributionFitResult], DistributionFitResult]:
    values = series.to_numpy(dtype=float)
    jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(values)
    shapiro_stat, shapiro_pvalue = shapiro(values)
    anderson_result = anderson(values, dist="norm")
    sig_levels = np.asarray(anderson_result.significance_level, dtype=float)
    crit_values = np.asarray(anderson_result.critical_values, dtype=float)
    crit_5 = float(crit_values[np.argmin(np.abs(sig_levels - 5.0))])
    adf_stat, adf_pvalue = safe_adf(series)
    kpss_stat, kpss_pvalue = safe_kpss(series)
    lb = compute_ljung_box(series)
    arch = compute_arch_test(series)

    fitted_distributions = {
        "normal": fit_distribution(values, "normal"),
        "t": fit_distribution(values, "t"),
        "nig": fit_distribution(values, "nig"),
    }
    distribution_table, recommended_distribution, selection_reason = distribution_results_table(
        list(fitted_distributions.values())
    )
    best_distribution = recommended_distribution.distribution
    normal_fit = fitted_distributions["normal"]
    student_fit = fitted_distributions["t"]
    nig_fit = fitted_distributions["nig"]

    row: dict[str, float | str] = {
        "count": int(series.count()),
        "mean_pct": float(series.mean()),
        "annualized_mean_pct": float(series.mean() * 12.0),
        "std_pct": float(series.std()),
        "annualized_vol_pct": float(series.std() * ANNUALIZATION_FACTOR),
        "min_pct": float(series.min()),
        "p25_pct": float(series.quantile(0.25)),
        "median_pct": float(series.median()),
        "p75_pct": float(series.quantile(0.75)),
        "max_pct": float(series.max()),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "excess_kurtosis": float(kurtosis - 3.0),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "shapiro_wilk_stat": float(shapiro_stat),
        "shapiro_wilk_pvalue": float(shapiro_pvalue),
        "anderson_normal_stat": float(anderson_result.statistic),
        "anderson_5pct_critical": crit_5,
        "normal_ks_stat": normal_fit.ks_stat,
        "normal_ks_pvalue": normal_fit.ks_pvalue,
        "normal_aic": normal_fit.aic,
        "normal_bic": normal_fit.bic,
        "student_t_ks_stat": student_fit.ks_stat,
        "student_t_ks_pvalue": student_fit.ks_pvalue,
        "student_t_aic": student_fit.aic,
        "student_t_bic": student_fit.bic,
        "student_t_df": float(student_fit.params["df"]),
        "student_t_loc": float(student_fit.params["loc"]),
        "student_t_scale": float(student_fit.params["scale"]),
        "student_t_aic_gain_vs_normal": float(normal_fit.aic - student_fit.aic),
        "nig_ks_stat": nig_fit.ks_stat,
        "nig_ks_pvalue": nig_fit.ks_pvalue,
        "nig_aic": nig_fit.aic,
        "nig_bic": nig_fit.bic,
        "nig_alpha": float(nig_fit.params["alpha"]),
        "nig_beta": float(nig_fit.params["beta"]),
        "nig_loc": float(nig_fit.params["loc"]),
        "nig_scale": float(nig_fit.params["scale"]),
        "nig_aic_gain_vs_normal": float(normal_fit.aic - nig_fit.aic),
        "best_marginal_fit": best_distribution,
        "best_marginal_fit_code": recommended_distribution.distribution_code,
        "best_marginal_fit_ks_stat": recommended_distribution.ks_stat,
        "best_marginal_fit_ks_pvalue": recommended_distribution.ks_pvalue,
        "best_marginal_fit_aic_gain_vs_normal": float(normal_fit.aic - recommended_distribution.aic),
        "distribution_selection_reason": selection_reason,
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "kpss_stat": float(kpss_stat),
        "kpss_pvalue": float(kpss_pvalue),
        **lb,
        **arch,
    }
    return row, distribution_table, fitted_distributions, recommended_distribution


def build_distribution_summary_table(test_summary: pd.DataFrame) -> str:
    table = test_summary.copy()
    table["Portfolio"] = table["portfolio"].map(portfolio_label)
    table["Skewness"] = table["skewness"].map(lambda value: format_float(float(value), 2))
    table["Excess kurtosis"] = table["excess_kurtosis"].map(lambda value: format_float(float(value), 2))
    table["Jarque-Bera p"] = table["jarque_bera_pvalue"].map(format_pvalue)
    table["Shapiro-Wilk p"] = table["shapiro_wilk_pvalue"].map(format_pvalue)
    table["Recommended fit"] = table["best_marginal_fit"]
    table["Best-fit KS p"] = table["best_marginal_fit_ks_pvalue"].map(format_pvalue)
    table["AIC gain vs Gaussian"] = table["best_marginal_fit_aic_gain_vs_normal"].map(
        lambda value: format_float(float(value), 1)
    )
    table["ADF p"] = table["adf_pvalue"].map(format_pvalue)
    table["ARCH-LM p"] = table["arch_lm_pvalue"].map(format_pvalue)
    return markdown_table(
        table[
            [
                "Portfolio",
                "Skewness",
                "Excess kurtosis",
                "Jarque-Bera p",
                "Shapiro-Wilk p",
                "Recommended fit",
                "Best-fit KS p",
                "AIC gain vs Gaussian",
                "ADF p",
                "ARCH-LM p",
            ]
        ]
    )


def build_mean_volatility_table(model_summary: pd.DataFrame, garch_summary: pd.DataFrame) -> str:
    merged = model_summary.merge(
        garch_summary[
            [
                "portfolio",
                "model_label",
                "distribution",
                "persistence",
                "std_resid_sq_lb_pvalue_lag_12",
                "std_resid_arch_lm_pvalue",
                "innovation_ks_pvalue",
            ]
        ],
        on="portfolio",
        how="inner",
    )
    merged["Portfolio"] = merged["portfolio"].map(portfolio_label)
    merged["Selected ARIMA"] = merged["selected_arima_order"]
    merged["Selected volatility"] = merged["model_label"]
    merged["Innovation dist."] = merged["distribution"]
    merged["Residual Ljung-Box p (12)"] = merged["residual_lb_pvalue_lag_12"].map(format_pvalue)
    merged["GARCH persistence"] = merged["persistence"].map(lambda value: format_float(float(value), 3))
    merged["Std. sq. resid. LB p (12)"] = merged["std_resid_sq_lb_pvalue_lag_12"].map(format_pvalue)
    merged["Std. resid. ARCH-LM p"] = merged["std_resid_arch_lm_pvalue"].map(format_pvalue)
    merged["Innovation KS p"] = merged["innovation_ks_pvalue"].map(format_pvalue)
    return markdown_table(
        merged[
            [
                "Portfolio",
                "Selected ARIMA",
                "Selected volatility",
                "Innovation dist.",
                "Residual Ljung-Box p (12)",
                "GARCH persistence",
                "Std. sq. resid. LB p (12)",
                "Std. resid. ARCH-LM p",
                "Innovation KS p",
            ]
        ]
    )


def build_predictive_summary_table(predictive_summary: pd.DataFrame) -> str:
    benchmark = predictive_summary.loc[predictive_summary["is_benchmark"]].copy()
    preferred = predictive_summary.loc[predictive_summary["is_preferred_predictive"]].copy()
    merged = preferred.merge(
        benchmark[["portfolio", "rmse"]].rename(columns={"rmse": "benchmark_rmse"}),
        on="portfolio",
        how="left",
    )
    merged["Portfolio"] = merged["portfolio"].map(portfolio_label)
    merged["Preferred model"] = merged["model_label"].map(
        lambda value: SHORT_PREDICTIVE_LABELS.get(str(value), str(value))
    )
    merged["Benchmark RMSE"] = merged["benchmark_rmse"].map(lambda value: format_float(float(value), 3))
    merged["Predictive RMSE"] = merged["rmse"].map(lambda value: format_float(float(value), 3))
    merged["RMSE gain vs benchmark (%)"] = (
        ((merged["benchmark_rmse"] - merged["rmse"]) / merged["benchmark_rmse"]) * 100.0
    ).map(lambda value: format_float(float(value), 2))
    merged["Directional accuracy"] = merged["directional_accuracy"].map(
        lambda value: format_float(float(value), 3)
    )
    return markdown_table(
        merged[
            [
                "Portfolio",
                "Preferred model",
                "Benchmark RMSE",
                "Predictive RMSE",
                "RMSE gain vs benchmark (%)",
                "Directional accuracy",
            ]
        ]
    )


def choose_d_candidates(test_row: dict[str, float | str]) -> list[int]:
    if float(test_row["adf_pvalue"]) < 0.05 and float(test_row["kpss_pvalue"]) >= 0.05:
        return [0]
    return [0, 1]


def candidate_orders(d_candidates: list[int]) -> list[tuple[int, int, int]]:
    orders: list[tuple[int, int, int]] = []
    for d in d_candidates:
        for p in range(MAX_AR_ORDER + 1):
            for q in range(MAX_MA_ORDER + 1):
                if p == 0 and q == 0 and d == 0:
                    orders.append((p, d, q))
                elif p + q > 0:
                    orders.append((p, d, q))
    seen: set[tuple[int, int, int]] = set()
    unique_orders: list[tuple[int, int, int]] = []
    for order in orders:
        if order not in seen:
            unique_orders.append(order)
            seen.add(order)
    return unique_orders


def fit_single_arima(series: pd.Series, order: tuple[int, int, int]) -> dict[str, object]:
    trend = "n" if order[1] > 0 else "c"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                series,
                order=order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 300})

        param_names = list(result.param_names)
        dynamic_mask = [name.startswith("ar.") or name.startswith("ma.") for name in param_names]
        dynamic_pvalues = [float(result.pvalues.iloc[i]) for i, is_dynamic in enumerate(dynamic_mask) if is_dynamic]
        significant_share = (
            float(np.mean(np.array(dynamic_pvalues) < 0.05)) if dynamic_pvalues else 1.0
        )
        dynamic_count = int(sum(dynamic_mask))
        lb = acorr_ljungbox(result.resid, lags=LJUNG_BOX_LAGS, return_df=True)
        return {
            "order": order,
            "success": True,
            "result": result,
            "aic": float(result.aic),
            "bic": float(result.bic),
            "hqic": float(result.hqic),
            "llf": float(result.llf),
            "lb_pvalue_lag_12": float(lb.loc[12, "lb_pvalue"]),
            "lb_pvalue_lag_24": float(lb.loc[24, "lb_pvalue"]),
            "significant_dynamic_share": float(significant_share),
            "dynamic_param_count": dynamic_count,
            "order_complexity": int(sum(order)),
        }
    except Exception as exc:
        return {
            "order": order,
            "success": False,
            "error": str(exc),
        }


def select_best_arima(candidates: list[dict[str, object]]) -> tuple[dict[str, object], pd.DataFrame]:
    successful = [candidate for candidate in candidates if candidate["success"]]
    if not successful:
        raise RuntimeError("No ARIMA candidate converged successfully.")

    candidate_rows = []
    for candidate in successful:
        candidate_rows.append(
            {
                "order": str(candidate["order"]),
                "aic": candidate["aic"],
                "bic": candidate["bic"],
                "hqic": candidate["hqic"],
                "llf": candidate["llf"],
                "lb_pvalue_lag_12": candidate["lb_pvalue_lag_12"],
                "lb_pvalue_lag_24": candidate["lb_pvalue_lag_24"],
                "significant_dynamic_share": candidate["significant_dynamic_share"],
                "dynamic_param_count": candidate["dynamic_param_count"],
                "order_complexity": candidate["order_complexity"],
            }
        )
    comparison_df = pd.DataFrame(candidate_rows).sort_values(["aic", "bic", "order_complexity"]).reset_index(drop=True)

    best_aic = float(comparison_df["aic"].min())
    shortlisted = comparison_df[comparison_df["aic"] <= best_aic + 2.0].copy()
    preferred = shortlisted[
        (shortlisted["lb_pvalue_lag_12"] >= 0.05)
        & (
            (shortlisted["significant_dynamic_share"] >= 0.5)
            | (shortlisted["dynamic_param_count"] == 0)
        )
    ].copy()

    if not preferred.empty:
        chosen_row = preferred.sort_values(["bic", "order_complexity", "aic"]).iloc[0]
        selection_reason = "lowest BIC within 2 AIC points among models with acceptable residual autocorrelation and mostly significant dynamic terms"
    else:
        chosen_row = shortlisted.sort_values(["bic", "order_complexity", "aic"]).iloc[0]
        selection_reason = "lowest BIC within 2 AIC points because no candidate met the preferred residual-and-significance filter"

    selected_order = ast.literal_eval(chosen_row["order"])
    selected = next(candidate for candidate in successful if candidate["order"] == selected_order)
    comparison_df["selected"] = comparison_df["order"] == str(selected_order)
    comparison_df["selection_reason"] = ""
    comparison_df.loc[comparison_df["selected"], "selection_reason"] = selection_reason
    selected["selection_reason"] = selection_reason
    return selected, comparison_df


def fit_garch_candidate(residuals: pd.Series, model_label: str, distribution_code: str) -> VolatilityFitResult:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            residuals,
            mean="Zero",
            vol="GARCH",
            p=1,
            o=0,
            q=1,
            dist=distribution_code,
            rescale=False,
        )
        fitted = model.fit(update_freq=0, disp="off", cov_type="robust", show_warning=False)

    params = fitted.params
    std_errors = fitted.std_err
    tvalues = fitted.tvalues
    pvalues = fitted.pvalues

    param_table = pd.DataFrame(
        {
            "parameter": params.index,
            "estimate": params.values,
            "std_error": std_errors.reindex(params.index).values,
            "t_stat": tvalues.reindex(params.index).values,
            "pvalue": pvalues.reindex(params.index).values,
        }
    )

    residual_values = pd.Series(np.asarray(fitted.resid), index=residuals.index, dtype=float)
    conditional_sigma = pd.Series(np.asarray(fitted.conditional_volatility), index=residuals.index, dtype=float)
    std_resid = pd.Series(np.asarray(fitted.std_resid), index=residuals.index, dtype=float)
    std_resid = std_resid.replace([np.inf, -np.inf], np.nan)
    clean_std_resid = std_resid.dropna()

    lb_resid = compute_ljung_box(clean_std_resid)
    lb_sq = compute_ljung_box(clean_std_resid**2)
    arch_diag = compute_arch_test(clean_std_resid)

    volatility_pvalues = [
        float(pvalues[name])
        for name in ("omega", "alpha[1]", "beta[1]")
        if name in pvalues.index and np.isfinite(pvalues[name])
    ]
    volatility_significant_share = (
        float(np.mean(np.array(volatility_pvalues) < 0.05)) if volatility_pvalues else np.nan
    )

    omega = float(params.get("omega", np.nan))
    alpha = float(params.get("alpha[1]", np.nan))
    beta = float(params.get("beta[1]", np.nan))
    persistence = float(alpha + beta) if np.isfinite(alpha + beta) else np.nan
    nu = float(params.get("nu", np.nan)) if "nu" in params.index else None
    distribution_param_names = arch_distribution_parameter_names(distribution_code)
    distribution_params = {
        name: float(params[name])
        for name in distribution_param_names
        if name in params.index and np.isfinite(float(params[name]))
    }
    distribution_param_text = format_parameter_text(distribution_params)
    parameter_vector = arch_distribution_parameter_vector(distribution_code, distribution_params)
    innovation_ks_stat, innovation_ks_pvalue = kstest(
        clean_std_resid.to_numpy(dtype=float),
        lambda x: arch_distribution_cdf(x, distribution_code, distribution_params),
    )
    std_jb_stat, std_jb_pvalue, std_skewness, std_kurtosis = jarque_bera(clean_std_resid)
    std_shapiro_stat, std_shapiro_pvalue = shapiro(clean_std_resid)

    diagnostics = {
        "std_resid_lb_stat_lag_12": lb_resid["lb_stat_lag_12"],
        "std_resid_lb_pvalue_lag_12": lb_resid["lb_pvalue_lag_12"],
        "std_resid_lb_stat_lag_24": lb_resid["lb_stat_lag_24"],
        "std_resid_lb_pvalue_lag_24": lb_resid["lb_pvalue_lag_24"],
        "std_resid_sq_lb_stat_lag_12": lb_sq["lb_stat_lag_12"],
        "std_resid_sq_lb_pvalue_lag_12": lb_sq["lb_pvalue_lag_12"],
        "std_resid_sq_lb_stat_lag_24": lb_sq["lb_stat_lag_24"],
        "std_resid_sq_lb_pvalue_lag_24": lb_sq["lb_pvalue_lag_24"],
        "std_resid_arch_lm_stat": arch_diag["arch_lm_stat"],
        "std_resid_arch_lm_pvalue": arch_diag["arch_lm_pvalue"],
        "innovation_ks_stat": float(innovation_ks_stat),
        "innovation_ks_pvalue": float(innovation_ks_pvalue),
        "std_resid_jarque_bera_stat": float(std_jb_stat),
        "std_resid_jarque_bera_pvalue": float(std_jb_pvalue),
        "std_resid_shapiro_wilk_stat": float(std_shapiro_stat),
        "std_resid_shapiro_wilk_pvalue": float(std_shapiro_pvalue),
        "std_resid_skewness": float(std_skewness),
        "std_resid_kurtosis": float(std_kurtosis),
        "volatility_param_significant_share": volatility_significant_share,
    }

    distribution = arch_distribution_label(distribution_code)
    return VolatilityFitResult(
        model_label=model_label,
        distribution=distribution,
        distribution_code=distribution_code,
        success=True,
        convergence_flag=int(fitted.convergence_flag),
        loglik=float(fitted.loglikelihood),
        aic=float(fitted.aic),
        bic=float(fitted.bic),
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        nu=nu,
        distribution_params=distribution_params,
        distribution_param_text=distribution_param_text,
        conditional_volatility=conditional_sigma.to_numpy(dtype=float),
        standardized_residuals=std_resid.to_numpy(dtype=float),
        residuals=residual_values.to_numpy(dtype=float),
        param_table=param_table,
        summary_text=str(fitted.summary()),
        diagnostics=diagnostics,
    )


def select_best_volatility_model(
    residuals: pd.Series,
) -> tuple[VolatilityFitResult, pd.DataFrame, str]:
    fitted_candidates: list[VolatilityFitResult] = []
    failed_rows: list[dict[str, object]] = []

    for model_label, distribution_code in VOLATILITY_MODELS:
        try:
            fitted_candidates.append(fit_garch_candidate(residuals, model_label, distribution_code))
        except Exception as exc:
            failed_rows.append(
                {
                    "model_label": model_label,
                    "distribution": arch_distribution_label(distribution_code),
                    "distribution_code": distribution_code,
                    "success": False,
                    "error": str(exc),
                }
            )

    if not fitted_candidates:
        raise RuntimeError("No arch-based volatility model converged successfully.")

    candidate_rows = []
    for result in fitted_candidates:
        candidate_rows.append(
            {
                "model_label": result.model_label,
                "distribution": result.distribution,
                "aic": result.aic,
                "bic": result.bic,
                "loglik": result.loglik,
                "persistence": result.persistence,
                "nu": result.nu,
                "distribution_param_text": result.distribution_param_text,
                "std_resid_lb_pvalue_lag_12": result.diagnostics["std_resid_lb_pvalue_lag_12"],
                "std_resid_lb_pvalue_lag_24": result.diagnostics["std_resid_lb_pvalue_lag_24"],
                "std_resid_sq_lb_pvalue_lag_12": result.diagnostics["std_resid_sq_lb_pvalue_lag_12"],
                "std_resid_sq_lb_pvalue_lag_24": result.diagnostics["std_resid_sq_lb_pvalue_lag_24"],
                "std_resid_arch_lm_pvalue": result.diagnostics["std_resid_arch_lm_pvalue"],
                "innovation_ks_stat": result.diagnostics["innovation_ks_stat"],
                "innovation_ks_pvalue": result.diagnostics["innovation_ks_pvalue"],
                "std_resid_jarque_bera_pvalue": result.diagnostics["std_resid_jarque_bera_pvalue"],
                "std_resid_shapiro_wilk_pvalue": result.diagnostics["std_resid_shapiro_wilk_pvalue"],
                "volatility_param_significant_share": result.diagnostics["volatility_param_significant_share"],
                "distribution_complexity": arch_distribution_complexity(result.distribution_code),
                "success": True,
            }
        )

    comparison_df = pd.DataFrame(candidate_rows + failed_rows)
    successful_df = comparison_df.loc[comparison_df["success"]].copy()
    best_aic = float(successful_df["aic"].min())
    shortlisted = successful_df.loc[successful_df["aic"] <= best_aic + 4.0].copy()
    shortlisted["diagnostic_score"] = (
        (shortlisted["std_resid_lb_pvalue_lag_12"] >= 0.05).astype(int)
        + (shortlisted["std_resid_sq_lb_pvalue_lag_12"] >= 0.05).astype(int)
        + (shortlisted["std_resid_arch_lm_pvalue"] >= 0.05).astype(int)
        + (shortlisted["innovation_ks_pvalue"] >= 0.05).astype(int)
        + (
            (shortlisted["volatility_param_significant_share"] >= 0.5)
            | shortlisted["volatility_param_significant_share"].isna()
        ).astype(int)
    )

    chosen_row = shortlisted.sort_values(
        [
            "diagnostic_score",
            "bic",
            "distribution_complexity",
            "innovation_ks_stat",
            "aic",
        ],
        ascending=[False, True, True, True, True],
    ).iloc[0]
    selection_reason = (
        "highest joint diagnostic score among models within 4 AIC points of the best fit, where the score "
        "combines residual Ljung-Box, squared-residual Ljung-Box, residual ARCH-LM, innovation KS, and "
        "core-parameter significance; BIC and parsimony break ties"
    )

    selected_label = str(chosen_row["model_label"])
    selected = next(result for result in fitted_candidates if result.model_label == selected_label)
    comparison_df["selected"] = comparison_df["model_label"] == selected_label
    comparison_df["selection_reason"] = ""
    comparison_df.loc[comparison_df["selected"], "selection_reason"] = selection_reason
    return (
        selected,
        comparison_df.sort_values(["success", "aic", "bic"], ascending=[False, True, True]).reset_index(drop=True),
        selection_reason,
    )


def save_time_series_plot(dates: pd.Series, series: pd.Series, figure_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, series, linewidth=0.9, color="#1f5aa6")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{title}: Monthly Returns")
    ax.set_ylabel("Return (%)")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_distribution_plot(
    series: pd.Series,
    fitted_distributions: dict[str, DistributionFitResult],
    figure_path: Path,
    title: str,
) -> None:
    values = series.to_numpy(dtype=float)
    x_grid = np.linspace(values.min(), values.max(), 400)
    kde = gaussian_kde(values)
    normal_fit = fitted_distributions["normal"]
    student_fit = fitted_distributions["t"]
    nig_fit = fitted_distributions["nig"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(values, bins=36, density=True, alpha=0.55, color="#5b8bd1", edgecolor="white")
    ax.plot(x_grid, kde(x_grid), color="#ba4a00", linewidth=2, label="Kernel density")
    ax.plot(
        x_grid,
        norm.pdf(x_grid, loc=normal_fit.params["loc"], scale=normal_fit.params["scale"]),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Fitted normal",
    )
    ax.plot(
        x_grid,
        student_t.pdf(
            x_grid,
            df=float(student_fit.params["df"]),
            loc=float(student_fit.params["loc"]),
            scale=float(student_fit.params["scale"]),
        ),
        color="#117864",
        linestyle="-.",
        linewidth=1.8,
        label="Fitted Student-t",
    )
    ax.plot(
        x_grid,
        norminvgauss.pdf(
            x_grid,
            a=float(nig_fit.params["alpha"]),
            b=float(nig_fit.params["beta"]),
            loc=float(nig_fit.params["loc"]),
            scale=float(nig_fit.params["scale"]),
        ),
        color="#7d3c98",
        linestyle=":",
        linewidth=1.8,
        label="Fitted NIG",
    )
    ax.set_title(f"{title}: Histogram, KDE, and Fitted Densities")
    ax.set_xlabel("Return (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_qq_plot(
    series: pd.Series,
    fitted_distributions: dict[str, DistributionFitResult],
    recommended_distribution: DistributionFitResult,
    figure_path: Path,
    title: str,
) -> None:
    normal_fit = fitted_distributions["normal"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    qqplot(
        series,
        dist=norm,
        distargs=(),
        loc=float(normal_fit.params["loc"]),
        scale=float(normal_fit.params["scale"]),
        line="45",
        ax=axes[0],
    )
    axes[0].set_title(f"{title}: Normal QQ")

    if recommended_distribution.distribution_code == "t":
        qqplot(
            series,
            dist=student_t,
            distargs=(float(recommended_distribution.params["df"]),),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    elif recommended_distribution.distribution_code == "nig":
        qqplot(
            series,
            dist=norminvgauss,
            distargs=(
                float(recommended_distribution.params["alpha"]),
                float(recommended_distribution.params["beta"]),
            ),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    else:
        qqplot(
            series,
            dist=norm,
            distargs=(),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
            line="45",
            ax=axes[1],
        )
    axes[1].set_title(f"{title}: Recommended {recommended_distribution.distribution} QQ")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_recommended_distribution_diagnostic_plot(
    series: pd.Series,
    recommended_distribution: DistributionFitResult,
    figure_path: Path,
    title: str,
) -> None:
    values = np.sort(series.to_numpy(dtype=float))
    empirical_cdf = np.arange(1, len(values) + 1, dtype=float) / len(values)

    if recommended_distribution.distribution_code == "normal":
        fitted_cdf = norm.cdf(
            values,
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    elif recommended_distribution.distribution_code == "t":
        fitted_cdf = student_t.cdf(
            values,
            df=float(recommended_distribution.params["df"]),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    elif recommended_distribution.distribution_code == "nig":
        fitted_cdf = norminvgauss.cdf(
            values,
            a=float(recommended_distribution.params["alpha"]),
            b=float(recommended_distribution.params["beta"]),
            loc=float(recommended_distribution.params["loc"]),
            scale=float(recommended_distribution.params["scale"]),
        )
    else:
        raise ValueError(f"Unsupported recommended distribution: {recommended_distribution.distribution_code}")

    absolute_error = np.abs(empirical_cdf - fitted_cdf)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(values, empirical_cdf, color="#1f5aa6", linewidth=1.6, label="Empirical CDF")
    axes[0].plot(values, fitted_cdf, color="#ba4a00", linewidth=1.6, linestyle="--", label="Fitted CDF")
    axes[0].set_title(f"{title}: Recommended Fit CDF")
    axes[0].set_xlabel("Return (%)")
    axes[0].legend()

    axes[1].plot(values, absolute_error, color="#7d3c98", linewidth=1.5)
    axes[1].set_title(f"{title}: Absolute CDF Error")
    axes[1].set_xlabel("Return (%)")
    axes[1].set_ylabel("|F_n(x) - F(x)|")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_acf_pacf_plot(series: pd.Series, figure_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(series, ax=axes[0], lags=24, zero=False)
    axes[0].set_title(f"{title}: ACF")
    plot_pacf(series, ax=axes[1], lags=24, zero=False, method="ywm")
    axes[1].set_title(f"{title}: PACF")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_volatility_clustering_plot(dates: pd.Series, series: pd.Series, figure_path: Path, title: str) -> None:
    squared = series**2
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(dates, squared, linewidth=0.9, color="#7a1f5c")
    axes[0].set_title(f"{title}: Squared Returns")
    axes[0].set_ylabel("Squared return")
    plot_acf(squared, ax=axes[1], lags=24, zero=False)
    axes[1].set_title(f"{title}: ACF of Squared Returns")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_residual_diagnostics_plot(
    dates: pd.Series,
    residuals: pd.Series,
    fitted: pd.Series,
    figure_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(dates, residuals, linewidth=0.9, color="#1f5aa6")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[0, 0].set_title(f"{title}: Residuals")

    axes[0, 1].plot(dates, fitted, linewidth=0.9, color="#ba4a00", label="Fitted")
    axes[0, 1].plot(dates, fitted + residuals, linewidth=0.7, color="#1f5aa6", alpha=0.5, label="Actual")
    axes[0, 1].set_title(f"{title}: Actual vs Fitted")
    axes[0, 1].legend()

    plot_acf(residuals.dropna(), ax=axes[1, 0], lags=24, zero=False)
    axes[1, 0].set_title(f"{title}: Residual ACF")
    plot_acf((residuals.dropna()) ** 2, ax=axes[1, 1], lags=24, zero=False)
    axes[1, 1].set_title(f"{title}: Squared Residual ACF")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_garch_plot(
    dates: pd.Series,
    arima_residuals: pd.Series,
    volatility_result: VolatilityFitResult,
    figure_path: Path,
    title: str,
) -> None:
    std_resid_series = pd.Series(volatility_result.standardized_residuals, index=dates, dtype=float).dropna()
    sigma_series = pd.Series(volatility_result.conditional_volatility, index=dates, dtype=float)
    residual_series = pd.Series(arima_residuals.to_numpy(dtype=float), index=dates, dtype=float)
    sorted_resid = np.sort(std_resid_series.to_numpy(dtype=float))
    probabilities = (np.arange(1, len(sorted_resid) + 1, dtype=float) - 0.5) / len(sorted_resid)
    theoretical_quantiles = arch_distribution_ppf(
        probabilities,
        volatility_result.distribution_code,
        volatility_result.distribution_params,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(dates, residual_series.abs(), linewidth=0.9, color="#1f5aa6", label="|ARIMA residual|")
    axes[0, 0].plot(dates, sigma_series, linewidth=1.2, color="#ba4a00", label="Conditional sigma")
    axes[0, 0].set_title(f"{title}: Residual Magnitude and Conditional Volatility")
    axes[0, 0].legend()

    axes[0, 1].scatter(theoretical_quantiles, sorted_resid, s=10, alpha=0.7, color="#117864")
    qq_min = float(min(theoretical_quantiles.min(), sorted_resid.min()))
    qq_max = float(max(theoretical_quantiles.max(), sorted_resid.max()))
    axes[0, 1].plot([qq_min, qq_max], [qq_min, qq_max], color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title(f"{title}: Std. Residual QQ vs {volatility_result.distribution}")
    axes[0, 1].set_xlabel("Theoretical quantiles")
    axes[0, 1].set_ylabel("Empirical quantiles")

    plot_acf(std_resid_series, ax=axes[1, 0], lags=24, zero=False)
    axes[1, 0].set_title(f"{title}: ACF of Std. Residuals")

    plot_acf(std_resid_series**2, ax=axes[1, 1], lags=24, zero=False)
    axes[1, 1].set_title(f"{title}: ACF of Squared Std. Residuals")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_text_file(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def arima_parameter_table(result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": result.param_names,
            "estimate": result.params,
            "std_error": result.bse,
            "z_stat": result.params / result.bse,
            "pvalue": result.pvalues,
        }
    )


def garch_summary_table(
    selected_volatility: VolatilityFitResult,
    selection_reason: str,
    arima_residual_arch: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_label": selected_volatility.model_label,
                "distribution": selected_volatility.distribution,
                "distribution_param_text": selected_volatility.distribution_param_text,
                "success": selected_volatility.success,
                "convergence_flag": selected_volatility.convergence_flag,
                "omega": selected_volatility.omega,
                "alpha": selected_volatility.alpha,
                "beta": selected_volatility.beta,
                "persistence": selected_volatility.persistence,
                "nu": selected_volatility.nu,
                "loglik": selected_volatility.loglik,
                "aic": selected_volatility.aic,
                "bic": selected_volatility.bic,
                "volatility_param_significant_share": selected_volatility.diagnostics[
                    "volatility_param_significant_share"
                ],
                "arima_resid_arch_lm_pvalue": arima_residual_arch["arch_lm_pvalue"],
                "std_resid_lb_pvalue_lag_12": selected_volatility.diagnostics["std_resid_lb_pvalue_lag_12"],
                "std_resid_lb_pvalue_lag_24": selected_volatility.diagnostics["std_resid_lb_pvalue_lag_24"],
                "std_resid_sq_lb_pvalue_lag_12": selected_volatility.diagnostics["std_resid_sq_lb_pvalue_lag_12"],
                "std_resid_sq_lb_pvalue_lag_24": selected_volatility.diagnostics["std_resid_sq_lb_pvalue_lag_24"],
                "std_resid_arch_lm_pvalue": selected_volatility.diagnostics["std_resid_arch_lm_pvalue"],
                "innovation_ks_pvalue": selected_volatility.diagnostics["innovation_ks_pvalue"],
                "std_resid_jarque_bera_pvalue": selected_volatility.diagnostics["std_resid_jarque_bera_pvalue"],
                "std_resid_shapiro_wilk_pvalue": selected_volatility.diagnostics["std_resid_shapiro_wilk_pvalue"],
                "selection_reason": selection_reason,
            }
        ]
    )


def summarize_stationarity(test_row: dict[str, float | str]) -> str:
    adf_stationary = float(test_row["adf_pvalue"]) < 0.05
    kpss_stationary = float(test_row["kpss_pvalue"]) >= 0.05
    if adf_stationary and kpss_stationary:
        return "Both ADF and KPSS support treating the return series as stationary in levels."
    if adf_stationary and not kpss_stationary:
        return "ADF rejects a unit root, but KPSS still flags some persistence; level models remain plausible but should be checked carefully."
    return "Stationarity evidence is mixed enough that ARIMA candidates allow both level and differenced specifications."


def build_interpretation_text(
    portfolio: str,
    test_row: dict[str, float | str],
    selected_arima: dict[str, object],
    selected_volatility: VolatilityFitResult,
    garch_summary: pd.DataFrame,
) -> str:
    order = selected_arima["order"]
    residual_lb = float(selected_arima["lb_pvalue_lag_12"])
    raw_arch_pvalue = float(garch_summary.loc[0, "arima_resid_arch_lm_pvalue"])
    std_arch_pvalue = float(garch_summary.loc[0, "std_resid_arch_lm_pvalue"])
    recommended_fit_gain = float(test_row["best_marginal_fit_aic_gain_vs_normal"])
    recommended_fit = str(test_row["best_marginal_fit"])
    normality_text = (
        "Jarque-Bera and Shapiro-Wilk both reject normality."
        if float(test_row["jarque_bera_pvalue"]) < 0.05 and float(test_row["shapiro_wilk_pvalue"]) < 0.05
        else "Normality evidence is weaker once multiple tests are considered."
    )
    marginal_text = (
        f"The recommended marginal model is {recommended_fit}, which improves AIC by {recommended_fit_gain:.1f} points relative to the Gaussian fit."
        if recommended_fit_gain > 0.0
        else f"The recommended marginal model is {recommended_fit}, but it does not improve on the Gaussian fit by AIC."
    )
    arch_text = (
        "ARCH effects are materially weaker after the arch-based volatility filter."
        if std_arch_pvalue > raw_arch_pvalue
        else "Standardized residual diagnostics still show nontrivial leftover ARCH effects, so volatility fit should be read cautiously."
    )
    innovation_ks_pvalue = float(selected_volatility.diagnostics["innovation_ks_pvalue"])
    return "\n".join(
        [
            f"# {portfolio_label(portfolio)}",
            "",
            (
                f"- Distribution: mean = {float(test_row['mean_pct']):.3f}% per month, volatility = "
                f"{float(test_row['std_pct']):.3f}% per month, skewness = {float(test_row['skewness']):.3f}, "
                f"kurtosis = {float(test_row['kurtosis']):.3f}."
            ),
            f"- Normality: {normality_text} {marginal_text}",
            f"- Stationarity: {summarize_stationarity(test_row)}",
            f"- Selected mean model: ARIMA{order} chosen because {selected_arima['selection_reason']}.",
            f"- Residual autocorrelation: Ljung-Box p-value at lag 12 = {residual_lb:.3f}.",
            (
                f"- Volatility model: {selected_volatility.model_label} estimated on ARIMA residuals with "
                f"alpha = {selected_volatility.alpha:.3f}, beta = {selected_volatility.beta:.3f}, "
                f"and persistence = {selected_volatility.persistence:.3f}. "
                f"Assumed innovation distribution parameters: {selected_volatility.distribution_param_text or 'none'}."
            ),
            (
                f"- Volatility clustering: ARIMA-residual ARCH-LM p-value = {raw_arch_pvalue:.4f}; "
                f"standardized-residual ARCH-LM p-value = {std_arch_pvalue:.4f}. {arch_text}"
            ),
            (
                f"- Innovation fit: standardized-residual KS p-value under the selected {selected_volatility.distribution} "
                f"assumption = {innovation_ks_pvalue:.4f}."
            ),
        ]
    )


def write_section_03(
    test_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    garch_summary: pd.DataFrame,
    predictive_summary: pd.DataFrame | None = None,
) -> None:
    best_mean_portfolio = test_summary.sort_values("mean_pct", ascending=False).iloc[0]
    highest_persistence = garch_summary.sort_values("persistence", ascending=False).iloc[0]
    strongest_non_normal = test_summary.sort_values("best_marginal_fit_aic_gain_vs_normal", ascending=False).iloc[0]
    strongest_arch = test_summary.sort_values("arch_lm_stat", ascending=False).iloc[0]
    weak_residual_fit = model_summary.loc[model_summary["residual_lb_pvalue_lag_12"] < 0.05, "portfolio"].tolist()
    weak_residual_text = ", ".join(portfolio_label(portfolio) for portfolio in weak_residual_fit) or "none of the portfolios"
    marginal_counts = test_summary["best_marginal_fit"].value_counts().to_dict()
    volatility_counts = garch_summary["distribution"].value_counts().to_dict()
    marginal_count_text = ", ".join(f"{name}: {count}" for name, count in sorted(marginal_counts.items()))
    volatility_count_text = ", ".join(f"{name}: {count}" for name, count in sorted(volatility_counts.items()))

    section_lines = [
        "# Modeling the Individual Portfolio Returns",
        "",
        "## Distributional Properties and Stationarity",
        "",
        (
            "Each portfolio was analyzed with the lecture-style univariate toolkit from Lecture Slides 1 and 2: "
            "time-series plots, histogram-density plots with fitted Gaussian, Student-t, and NIG densities, "
            "normal-versus-recommended QQ diagnostics, recommended-fit CDF diagnostics, full descriptive statistics "
            "(mean, median, standard deviation, quantiles, skewness, and kurtosis), Jarque-Bera and Shapiro-Wilk "
            "tests, MLE-based distribution fitting, KS goodness-of-fit tests, ADF and KPSS stationarity checks, "
            "ACF/PACF, Ljung-Box tests, and ARCH-LM diagnostics."
        ),
        "",
        "**Table 3. Distribution, normality, stationarity, and ARCH diagnostics.**",
        "",
        build_distribution_summary_table(test_summary),
        "",
        (
            f"Table 3 shows that the six monthly return series are stationary in levels but decisively non-Gaussian. "
            f"Jarque-Bera and Shapiro-Wilk p-values are effectively zero across the panel, while the preferred "
            f"marginal-fit counts are **{marginal_count_text}**. The most extreme departure from the Gaussian benchmark "
            f"appears in **{portfolio_label(strongest_non_normal['portfolio'])}**, where the recommended "
            f"**{strongest_non_normal['best_marginal_fit']}** fit improves AIC by "
            f"**{float(strongest_non_normal['best_marginal_fit_aic_gain_vs_normal']):.1f}** points relative to the "
            "normal fit."
        ),
        "",
        (
            "The stationarity evidence supports modeling monthly returns directly rather than differencing them again: "
            "ADF rejects a unit root throughout the panel, while KPSS does not overturn the practical use of level "
            "ARIMA models. At the same time, the raw series exhibit clear volatility clustering. ARCH-LM tests reject "
            f"homoskedasticity for all six portfolios, with the strongest raw ARCH signal in "
            f"**{portfolio_label(strongest_arch['portfolio'])}**."
        ),
        "",
        "Detailed artifacts for this subsection are saved under:",
        "",
        "- `output/figures/individual_returns/<portfolio>/`",
        "- `output/tables/individual_returns/<portfolio>/`",
        "- `output/models/individual_returns/<portfolio>/`",
        "",
        "## ARIMA Benchmarks and `arch`-Based Volatility Models",
        "",
        (
            "AR, MA, and ARMA/ARIMA candidates were estimated with `statsmodels` using the lecture-slide logic of "
            "ACF/PACF identification, low-order interpretable grids, and explicit residual checks. Candidate mean "
            "models were compared with AIC, BIC, residual Ljung-Box tests, and the significance share of dynamic "
            "parameters. The volatility stage then used the canonical `arch` package on the selected ARIMA residuals, "
            "comparing Gaussian, Student-t, Skewed Student-t, and GED GARCH(1,1) specifications."
        ),
        "",
        (
            "Preferred volatility selection is intentionally multi-criterion rather than single-metric. It combines "
            "AIC/BIC, Ljung-Box tests on standardized residuals and squared standardized residuals, standardized-"
            "residual ARCH-LM tests, innovation KS checks under the assumed distribution, and the significance share "
            "of the core volatility parameters."
        ),
        "",
        "**Table 4. Selected mean and volatility models.**",
        "",
        build_mean_volatility_table(model_summary, garch_summary),
        "",
        (
            f"Mean dynamics remain modest. The selected ARIMA models are low-order benchmarks, and the clearest "
            f"remaining residual autocorrelation at lag 12 appears in **{weak_residual_text}**. By contrast, the "
            f"conditional-variance evidence is stronger and more systematic. The selected volatility-model counts are "
            f"**{volatility_count_text}**, and the most persistent selected process belongs to "
            f"**{portfolio_label(highest_persistence['portfolio'])}** with alpha + beta = "
            f"**{float(highest_persistence['persistence']):.3f}**."
        ),
        "",
        (
            f"The portfolio with the highest average monthly return remains **{portfolio_label(best_mean_portfolio['portfolio'])}**, "
            f"at **{float(best_mean_portfolio['mean_pct']):.3f}%** per month. Across the panel, the evidence supports "
            "a standard finance interpretation: conditional mean dynamics are weak, but conditional risk is persistent "
            "and is better captured by heavy-tailed innovation assumptions than by the Gaussian benchmark alone."
        ),
        "",
        "## Predictive Models with Exogenous Variables",
        "",
    ]

    if predictive_summary is not None and not predictive_summary.empty:
        preferred_predictive = predictive_summary.loc[predictive_summary["is_preferred_predictive"]].copy()
        benchmark_rows = predictive_summary.loc[
            predictive_summary["is_benchmark"],
            ["portfolio", "rmse", "mae"],
        ].rename(columns={"rmse": "benchmark_rmse", "mae": "benchmark_mae"})
        preferred_predictive = preferred_predictive.merge(benchmark_rows, on="portfolio", how="left")
        preferred_predictive["rmse_improvement_pct"] = (
            100.0
            * (preferred_predictive["benchmark_rmse"] - preferred_predictive["rmse"])
            / preferred_predictive["benchmark_rmse"]
        )
        beating_benchmark = preferred_predictive.loc[preferred_predictive["rmse_improvement_pct"] > 0.0].copy()
        source_value = predictor_source_label(str(preferred_predictive["predictor_source"].mode().iloc[0]))
        common_family = str(preferred_predictive["model_label"].mode().iloc[0])
        best_gain_row = preferred_predictive.sort_values("rmse_improvement_pct", ascending=False).iloc[0]

        section_lines.extend(
            [
                (
                    f"The predictive extension adds lagged exogenous information to the univariate benchmarks. In the "
                    f"current run, the project uses **{source_value}**, together with internally constructed signals "
                    "such as lagged size and value spreads, a 12-month rolling market-volatility proxy, a 12-month "
                    "momentum proxy, and a drawdown proxy. Three classes are compared portfolio by portfolio: the "
                    "selected ARIMA benchmark, an ARIMAX specification with lagged exogenous predictors, and a "
                    "predictive regression with lagged returns plus the broader predictor set."
                ),
                "",
                (
                    "Predictive evaluation follows the course requirement to compare both in-sample fit and "
                    "out-of-sample performance. Full-sample model quality is summarized with AIC, BIC, residual "
                    "Ljung-Box diagnostics, and parameter significance. Out-of-sample performance is evaluated with a "
                    "120-month expanding one-step-ahead forecast exercise using RMSE, MAE, and directional accuracy."
                ),
                "",
                "**Table 5. Preferred predictive model versus the univariate benchmark.**",
                "",
                build_predictive_summary_table(predictive_summary),
                "",
                (
                    f"The most common preferred predictive family is **{common_family}**. The out-of-sample gains are "
                    f"modest rather than dramatic, which is consistent with the literature on monthly return "
                    f"predictability. Even so, **{len(beating_benchmark)} of 6** portfolios improve on the benchmark "
                    f"RMSE once the best predictive extension is used. The strongest improvement occurs for "
                    f"**{portfolio_label(best_gain_row['portfolio'])}**, where the preferred "
                    f"**{SHORT_PREDICTIVE_LABELS.get(str(best_gain_row['model_label']), str(best_gain_row['model_label']))}** "
                    f"changes RMSE by **{float(best_gain_row['rmse_improvement_pct']):.2f}%** relative to the "
                    "benchmark."
                ),
                "",
                (
                    "These results still point to the same economic conclusion: exogenous signals can help selectively, "
                    "but they do not overturn the broader Section 3 lesson that conditional variance and tail behavior "
                    "are more reliable features of the monthly portfolio returns than large and stable conditional-mean "
                    "predictability."
                ),
                "",
                "Predictive-modeling artifacts are saved under:",
                "",
                "- `data/processed/predictor_dataset_monthly.csv`",
                "- `data/processed/predictor_source_summary.csv`",
                "- `output/figures/predictive_individual_returns/<portfolio>/`",
                "- `output/tables/predictive_individual_returns/<portfolio>/`",
                "- `output/models/predictive_individual_returns/<portfolio>/`",
                "- `output/tables/predictive_individual_returns/predictive_model_summary.csv`",
                "- `output/tables/predictive_individual_returns/predictive_forecast_metrics.csv`",
                "- `output/tables/predictive_individual_returns/predictive_forecasts.csv`",
                "",
            ]
        )
    else:
        section_lines.extend(
            [
                (
                    "The predictive-modeling subsection is filled by the exogenous-predictor pipeline. If this file is "
                    "being regenerated from the univariate stage alone, rerun `scripts/run_predictive_modeling.py` to "
                    "append the ARIMAX and predictive-regression comparison with out-of-sample forecast metrics."
                ),
                "",
            ]
        )

    section_lines.extend(
        [
            (
                "Overall, Section 3 delivers three main conclusions. First, the monthly portfolio returns are heavy "
                "tailed enough that Gaussian-only diagnostics are too narrow, so the analysis now uses multiple "
                "normality checks, MLE-based Normal/Student-t/NIG comparisons, and KS goodness-of-fit tests. Second, "
                "low-order ARIMA models are adequate mean benchmarks, but they reveal only modest conditional-mean "
                "structure. Third, volatility clustering is strong enough to justify `arch`-based GARCH modeling, and "
                "non-Gaussian innovation assumptions often dominate the Gaussian benchmark even at the monthly "
                "frequency."
            ),
            "",
            "Portfolio-by-portfolio notes and the denser output inventory are moved to `report/sections/appendix_individual_returns_modeling.md`.",
        ]
    )
    SECTION_03_PATH.write_text("\n".join(section_lines) + "\n", encoding="utf-8")


def write_appendix(
    test_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    garch_summary: pd.DataFrame,
    predictive_summary: pd.DataFrame | None = None,
) -> None:
    appendix_lines = [
        "# Appendix: Individual Portfolio Modeling Details",
        "",
        "This appendix records concise portfolio-by-portfolio notes from the univariate benchmark stage and, when available, the exogenous predictive-modeling stage.",
        "",
    ]

    benchmark_lookup = None
    preferred_lookup = None
    if predictive_summary is not None and not predictive_summary.empty:
        benchmark_lookup = predictive_summary.loc[predictive_summary["is_benchmark"]].set_index("portfolio")
        preferred_lookup = predictive_summary.loc[predictive_summary["is_preferred_predictive"]].set_index("portfolio")

    for portfolio in PERCENT_COLUMNS:
        test_row = test_summary.loc[test_summary["portfolio"] == portfolio].iloc[0]
        model_row = model_summary.loc[model_summary["portfolio"] == portfolio].iloc[0]
        garch_row = garch_summary.loc[garch_summary["portfolio"] == portfolio].iloc[0]
        appendix_lines.extend(
            [
                f"## {portfolio_label(portfolio)}",
                "",
                (
                    f"- Distribution: mean = {float(test_row['mean_pct']):.3f}% per month, median = "
                    f"{float(test_row['median_pct']):.3f}%, volatility = {float(test_row['std_pct']):.3f}% per month, "
                    f"skewness = {float(test_row['skewness']):.3f}, kurtosis = {float(test_row['kurtosis']):.3f}, "
                    f"Jarque-Bera p-value = {float(test_row['jarque_bera_pvalue']):.4f}, Shapiro-Wilk p-value = "
                    f"{float(test_row['shapiro_wilk_pvalue']):.4f}, and recommended marginal fit = "
                    f"{test_row['best_marginal_fit']} with KS p-value = {float(test_row['best_marginal_fit_ks_pvalue']):.4f}."
                ),
                (
                    f"- Stationarity and dependence: ADF p-value = {float(test_row['adf_pvalue']):.4f}, KPSS p-value = "
                    f"{float(test_row['kpss_pvalue']):.4f}, raw Ljung-Box p-value at lag 12 = "
                    f"{float(test_row['lb_pvalue_lag_12']):.4f}, and raw ARCH-LM p-value = "
                    f"{float(test_row['arch_lm_pvalue']):.4f}."
                ),
                (
                    f"- Selected ARIMA model: {model_row['selected_arima_order']} with AIC = "
                    f"{float(model_row['selected_arima_aic']):.2f}, BIC = {float(model_row['selected_arima_bic']):.2f}, "
                    f"and residual Ljung-Box p-value at lag 12 = {float(model_row['residual_lb_pvalue_lag_12']):.4f}."
                ),
                (
                    f"- Selected volatility model: {garch_row['model_label']} with innovation distribution "
                    f"{garch_row['distribution']}, alpha = {float(garch_row['alpha']):.4f}, beta = "
                    f"{float(garch_row['beta']):.4f}, persistence = {float(garch_row['persistence']):.4f}, "
                    f"innovation KS p-value = {float(garch_row['innovation_ks_pvalue']):.4f}, standardized-squared-"
                    f"residual Ljung-Box p-value at lag 12 = {float(garch_row['std_resid_sq_lb_pvalue_lag_12']):.4f}, "
                    f"and standardized-residual ARCH-LM p-value = {float(garch_row['std_resid_arch_lm_pvalue']):.4f}."
                ),
            ]
        )

        if benchmark_lookup is not None and preferred_lookup is not None:
            benchmark_row = benchmark_lookup.loc[portfolio]
            preferred_row = preferred_lookup.loc[portfolio]
            rmse_improvement = 100.0 * (benchmark_row["rmse"] - preferred_row["rmse"]) / benchmark_row["rmse"]
            appendix_lines.extend(
                [
                    (
                        f"- Predictive benchmark: RMSE = {float(benchmark_row['rmse']):.4f}, MAE = "
                        f"{float(benchmark_row['mae']):.4f}, directional accuracy = "
                        f"{float(benchmark_row['directional_accuracy']):.3f}."
                    ),
                    (
                        f"- Preferred predictive model: {preferred_row['model_label']} with RMSE = "
                        f"{float(preferred_row['rmse']):.4f}, MAE = {float(preferred_row['mae']):.4f}, "
                        f"directional accuracy = {float(preferred_row['directional_accuracy']):.3f}, and RMSE change "
                        f"vs benchmark = {rmse_improvement:.2f}%."
                    ),
                    f"- Most informative predictive terms: {preferred_row['top_significant_terms']}",
                ]
            )

        appendix_lines.extend(
            [
                f"- Saved outputs: `output/figures/individual_returns/{portfolio_slug(portfolio)}/`, `output/tables/individual_returns/{portfolio_slug(portfolio)}/`, and `output/models/individual_returns/{portfolio_slug(portfolio)}/`.",
                (
                    f"- Distribution-fit comparison: `output/tables/individual_returns/{portfolio_slug(portfolio)}/distribution_fit_comparison.csv`; "
                    f"volatility-model comparison: `output/tables/individual_returns/{portfolio_slug(portfolio)}/garch_candidate_models.csv`."
                ),
                "",
            ]
        )
    APPENDIX_INDIVIDUAL_PATH.write_text("\n".join(appendix_lines) + "\n", encoding="utf-8")


def analyze_single_portfolio(df: pd.DataFrame, portfolio: str, logger: logging.Logger) -> dict[str, object]:
    title = portfolio_label(portfolio)
    paths = portfolio_paths(portfolio)
    series = df[portfolio].astype(float)
    dates = df["date"]

    logger.info("Analyzing %s", portfolio)
    test_row, distribution_table, fitted_distributions, recommended_distribution = descriptive_row(series)
    test_row["portfolio"] = portfolio

    save_time_series_plot(dates, series, paths["figure_dir"] / "time_series.png", title)
    save_distribution_plot(series, fitted_distributions, paths["figure_dir"] / "histogram_density.png", title)
    save_qq_plot(series, fitted_distributions, recommended_distribution, paths["figure_dir"] / "qq_plot.png", title)
    save_recommended_distribution_diagnostic_plot(
        series,
        recommended_distribution,
        paths["figure_dir"] / "recommended_distribution_diagnostic.png",
        title,
    )
    save_acf_pacf_plot(series, paths["figure_dir"] / "acf_pacf.png", title)
    save_volatility_clustering_plot(dates, series, paths["figure_dir"] / "volatility_clustering.png", title)

    d_candidates = choose_d_candidates(test_row)
    fitted_candidates = [fit_single_arima(series, order) for order in candidate_orders(d_candidates)]
    selected_arima, arima_comparison = select_best_arima(fitted_candidates)
    arima_result = selected_arima["result"]
    fitted_values = pd.Series(arima_result.fittedvalues, index=series.index)
    residuals = pd.Series(arima_result.resid, index=series.index)

    save_residual_diagnostics_plot(
        dates,
        residuals,
        fitted_values,
        paths["figure_dir"] / "residual_diagnostics.png",
        title,
    )

    residual_lb = compute_ljung_box(residuals)
    residual_arch = compute_arch_test(residuals)
    residual_summary = pd.DataFrame(
        [
            {
                "lb_stat_lag_12": residual_lb["lb_stat_lag_12"],
                "lb_pvalue_lag_12": residual_lb["lb_pvalue_lag_12"],
                "lb_stat_lag_24": residual_lb["lb_stat_lag_24"],
                "lb_pvalue_lag_24": residual_lb["lb_pvalue_lag_24"],
                "arch_lm_stat": residual_arch["arch_lm_stat"],
                "arch_lm_pvalue": residual_arch["arch_lm_pvalue"],
            }
        ]
    )

    selected_volatility, volatility_comparison, selection_reason = select_best_volatility_model(residuals)
    garch_summary = garch_summary_table(selected_volatility, selection_reason, residual_arch)
    save_garch_plot(dates, residuals, selected_volatility, paths["figure_dir"] / "garch_diagnostics.png", title)

    conditional_volatility_df = pd.DataFrame(
        {
            "date": dates,
            "return_pct": series,
            "arima_fitted_pct": fitted_values,
            "arima_residual_pct": residuals,
            "conditional_sigma_pct": selected_volatility.conditional_volatility,
            "standardized_residual": selected_volatility.standardized_residuals,
        }
    )

    save_dataframe(pd.DataFrame([test_row]), paths["table_dir"] / "descriptive_and_test_statistics.csv")
    save_dataframe(distribution_table, paths["table_dir"] / "distribution_fit_comparison.csv")
    save_dataframe(arima_comparison, paths["table_dir"] / "arima_candidate_models.csv")
    save_dataframe(arima_parameter_table(arima_result), paths["table_dir"] / "selected_arima_parameters.csv")
    save_dataframe(residual_summary, paths["table_dir"] / "selected_arima_residual_diagnostics.csv")
    save_dataframe(volatility_comparison, paths["table_dir"] / "garch_candidate_models.csv")
    save_dataframe(garch_summary, paths["table_dir"] / "garch_summary.csv")
    save_dataframe(selected_volatility.param_table, paths["table_dir"] / "selected_garch_parameters.csv")
    save_dataframe(conditional_volatility_df, paths["table_dir"] / "garch_conditional_volatility.csv")

    write_text_file(paths["model_dir"] / "selected_arima_summary.txt", arima_result.summary().as_text())
    write_text_file(paths["model_dir"] / "distribution_fit_summary.txt", distribution_table.to_string(index=False) + "\n")
    write_text_file(paths["model_dir"] / "garch_summary.txt", selected_volatility.summary_text + "\n")

    interpretation_text = build_interpretation_text(
        portfolio,
        test_row,
        selected_arima,
        selected_volatility,
        garch_summary,
    )
    write_text_file(paths["model_dir"] / "interpretation.md", interpretation_text + "\n")

    model_row = {
        "portfolio": portfolio,
        "selected_arima_order": str(selected_arima["order"]),
        "selected_arima_aic": float(selected_arima["aic"]),
        "selected_arima_bic": float(selected_arima["bic"]),
        "selected_arima_hqic": float(selected_arima["hqic"]),
        "residual_lb_pvalue_lag_12": float(selected_arima["lb_pvalue_lag_12"]),
        "residual_lb_pvalue_lag_24": float(selected_arima["lb_pvalue_lag_24"]),
        "significant_dynamic_share": float(selected_arima["significant_dynamic_share"]),
        "selection_reason": str(selected_arima["selection_reason"]),
    }

    garch_row = garch_summary.iloc[0].to_dict()
    garch_row["portfolio"] = portfolio

    logger.info(
        "Completed %s with %s and %s",
        portfolio,
        model_row["selected_arima_order"],
        garch_row["model_label"],
    )

    return {
        "test_row": test_row,
        "model_row": model_row,
        "garch_row": garch_row,
    }


def run_individual_modeling_pipeline() -> dict[str, object]:
    ensure_directories()
    logger = configure_modeling_logger()
    logger.info("Starting individual portfolio modeling pipeline.")

    df = load_clean_returns()

    test_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    garch_rows: list[dict[str, object]] = []

    for portfolio in PERCENT_COLUMNS:
        result = analyze_single_portfolio(df, portfolio, logger)
        test_rows.append(result["test_row"])
        model_rows.append(result["model_row"])
        garch_rows.append(result["garch_row"])

    test_summary = pd.DataFrame(test_rows)
    model_summary = pd.DataFrame(model_rows)
    garch_summary = pd.DataFrame(garch_rows)

    save_dataframe(test_summary, PORTFOLIO_TEST_SUMMARY_PATH)
    save_dataframe(model_summary, PORTFOLIO_MODEL_COMPARISON_PATH)
    save_dataframe(garch_summary, PORTFOLIO_GARCH_SUMMARY_PATH)

    write_section_03(test_summary, model_summary, garch_summary)
    write_appendix(test_summary, model_summary, garch_summary)

    logger.info("Individual portfolio modeling pipeline completed successfully.")
    return {
        "n_portfolios": len(PERCENT_COLUMNS),
        "test_summary_path": str(PORTFOLIO_TEST_SUMMARY_PATH),
        "model_summary_path": str(PORTFOLIO_MODEL_COMPARISON_PATH),
        "garch_summary_path": str(PORTFOLIO_GARCH_SUMMARY_PATH),
    }
