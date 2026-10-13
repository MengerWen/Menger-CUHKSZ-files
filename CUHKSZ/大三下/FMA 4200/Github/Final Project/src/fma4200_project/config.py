from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent
RAW_DATA_PATH = ROOT / "Data.csv"
GUIDANCE_PATH = ROOT / "Guidance.md"
LECTURE_SLIDES_DIR = WORKSPACE_ROOT / "Lecture Slides"
LECTURE_01_PATH = LECTURE_SLIDES_DIR / "1_Univariate Data Analysis.md"
LECTURE_02_PATH = LECTURE_SLIDES_DIR / "2_Time Series Data Analysis.md"
LECTURE_03_PATH = LECTURE_SLIDES_DIR / "3_Multivariate Time Series Data Analysis.md"
LECTURE_04_PATH = LECTURE_SLIDES_DIR / "4_Portfolio Optimization.md"

SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
REPORT_DIR = ROOT / "report"
REPORT_SECTIONS_DIR = REPORT_DIR / "sections"
LOGS_DIR = ROOT / "logs"
SUBMISSION_READY_DIR = ROOT / "submission_ready"

README_PATH = ROOT / "README.md"
PROJECT_STATUS_PATH = ROOT / "PROJECT_STATUS.md"
ENVIRONMENT_USED_PATH = ROOT / "environment_used.md"
RUNBOOK_PATH = ROOT / "SUBMISSION_RUNBOOK.md"

AUTOGEN_DATA_SNAPSHOT_PATH = REPORT_SECTIONS_DIR / "_02_data_snapshot_autogen.md"
SECTION_01_PATH = REPORT_SECTIONS_DIR / "01_introduction.md"
SECTION_02_PATH = REPORT_SECTIONS_DIR / "02_data_source_processing.md"
ENVIRONMENT_LOG_PATH = LOGS_DIR / "environment_check.txt"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline_run.log"

CLEAN_PERCENT_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_clean.csv"
CLEAN_DECIMAL_DATA_PATH = PROCESSED_DATA_DIR / "monthly_portfolio_returns_decimal.csv"
DATA_DICTIONARY_PATH = PROCESSED_DATA_DIR / "data_dictionary.csv"
FAMA_FRENCH_FACTORS_PATH = PROCESSED_DATA_DIR / "fama_french_3f_monthly.csv"
PREDICTOR_DATASET_PATH = PROCESSED_DATA_DIR / "predictor_dataset_monthly.csv"
PREDICTOR_SOURCE_SUMMARY_PATH = PROCESSED_DATA_DIR / "predictor_source_summary.csv"

DATE_COVERAGE_PATH = TABLES_DIR / "date_coverage_check.csv"
DUPLICATE_CHECK_PATH = TABLES_DIR / "duplicate_check.csv"
MISSINGNESS_PATH = TABLES_DIR / "missingness_check.csv"
DESCRIPTIVE_STATS_PCT_PATH = TABLES_DIR / "descriptive_statistics_pct.csv"
DESCRIPTIVE_STATS_DEC_PATH = TABLES_DIR / "descriptive_statistics_decimal.csv"
SUMMARY_SNAPSHOT_PATH = TABLES_DIR / "portfolio_summary_snapshot.csv"
CORRELATION_MATRIX_PATH = TABLES_DIR / "portfolio_correlation_matrix.csv"
SANITY_SUMMARY_PATH = TABLES_DIR / "sanity_checks_summary.csv"

RETURNS_FIGURE_PATH = FIGURES_DIR / "monthly_returns_overview.png"
GROWTH_FIGURE_PATH = FIGURES_DIR / "cumulative_growth_of_1.png"
CORRELATION_HEATMAP_PATH = FIGURES_DIR / "portfolio_correlation_heatmap.png"
MODELS_README_PATH = MODELS_DIR / "README.md"

INDIVIDUAL_FIGURES_DIR = FIGURES_DIR / "individual_returns"
INDIVIDUAL_TABLES_DIR = TABLES_DIR / "individual_returns"
INDIVIDUAL_MODELS_DIR = MODELS_DIR / "individual_returns"
PREDICTIVE_FIGURES_DIR = FIGURES_DIR / "predictive_individual_returns"
PREDICTIVE_TABLES_DIR = TABLES_DIR / "predictive_individual_returns"
PREDICTIVE_MODELS_DIR = MODELS_DIR / "predictive_individual_returns"
TRADING_FIGURES_DIR = FIGURES_DIR / "trading_strategies"
TRADING_TABLES_DIR = TABLES_DIR / "trading_strategies"
TRADING_MODELS_DIR = MODELS_DIR / "trading_strategies"

SECTION_03_PATH = REPORT_SECTIONS_DIR / "03_individual_returns_modeling.md"
SECTION_04_PATH = REPORT_SECTIONS_DIR / "04_trading_strategies.md"
SECTION_05_PATH = REPORT_SECTIONS_DIR / "05_conclusions.md"
APPENDIX_INDIVIDUAL_PATH = REPORT_SECTIONS_DIR / "appendix_individual_returns_modeling.md"
FINAL_REPORT_MD_PATH = REPORT_DIR / "final_report.md"
FINAL_REPORT_NOTEBOOK_PATH = REPORT_DIR / "final_report.ipynb"
FINAL_REPORT_HTML_PATH = REPORT_DIR / "final_report.html"
FINAL_REPORT_PDF_PATH = REPORT_DIR / "final_report.pdf"
FINAL_APPENDICES_PATH = REPORT_DIR / "final_appendices.md"
EXPORT_NOTES_PATH = REPORT_DIR / "export_notes.md"
REFERENCES_PATH = REPORT_DIR / "references.md"
REFERENCE_VERIFICATION_PATH = REPORT_DIR / "references_verification.md"
GUIDANCE_LECTURE_MAPPING_PATH = REPORT_DIR / "guidance_lecture_mapping.md"
RUBRIC_CHECKLIST_PATH = REPORT_DIR / "rubric_checklist.md"
AUDIT_REPORT_PATH = REPORT_DIR / "audit_report.md"
SUBMISSION_MANIFEST_PATH = ROOT / "submission_ready_manifest.txt"
SUBMISSION_ZIP_PATH = ROOT / "submission_ready.zip"
INDIVIDUAL_MODELING_LOG_PATH = LOGS_DIR / "individual_modeling.log"
PREDICTIVE_MODELING_LOG_PATH = LOGS_DIR / "predictive_modeling.log"
TRADING_STRATEGIES_LOG_PATH = LOGS_DIR / "trading_strategies.log"
FINAL_REPORT_LOG_PATH = LOGS_DIR / "final_report.log"

PORTFOLIO_MODEL_COMPARISON_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_model_comparison_summary.csv"
PORTFOLIO_TEST_SUMMARY_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_statistical_test_summary.csv"
PORTFOLIO_GARCH_SUMMARY_PATH = INDIVIDUAL_TABLES_DIR / "portfolio_garch_summary.csv"
PREDICTIVE_MODEL_SUMMARY_PATH = PREDICTIVE_TABLES_DIR / "predictive_model_summary.csv"
PREDICTIVE_FORECAST_METRICS_PATH = PREDICTIVE_TABLES_DIR / "predictive_forecast_metrics.csv"
PREDICTIVE_FORECASTS_PATH = PREDICTIVE_TABLES_DIR / "predictive_forecasts.csv"
VAR_LAG_SELECTION_PATH = TRADING_TABLES_DIR / "var_lag_selection.csv"
VAR_DIAGNOSTICS_PATH = TRADING_TABLES_DIR / "var_diagnostics.csv"
VAR_STABILITY_PATH = TRADING_TABLES_DIR / "var_stability_roots.csv"
COINTEGRATION_ORDER_TESTS_PATH = TRADING_TABLES_DIR / "cointegration_integration_order_tests.csv"
COINTEGRATION_SUMMARY_PATH = TRADING_TABLES_DIR / "cointegration_summary.csv"
COINTEGRATION_VECTORS_PATH = TRADING_TABLES_DIR / "cointegration_vectors.csv"
STAT_ARB_BACKTEST_PATH = TRADING_TABLES_DIR / "stat_arb_backtest.csv"
STAT_ARB_SIGNAL_PATH = TRADING_TABLES_DIR / "stat_arb_signals.csv"
STRATEGY_RETURNS_PATH = TRADING_TABLES_DIR / "strategy_returns.csv"
STRATEGY_METRICS_PATH = TRADING_TABLES_DIR / "strategy_metrics.csv"
STRATEGY_WEIGHTS_PATH = TRADING_TABLES_DIR / "strategy_weights.csv"
EFFICIENT_FRONTIER_POINTS_PATH = TRADING_TABLES_DIR / "efficient_frontier_points.csv"
MULTIVARIATE_WEALTH_PANEL_PATH = PROCESSED_DATA_DIR / "portfolio_wealth_indices.csv"

REQUIRED_IMPORTS = (
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "statsmodels",
    "arch",
    "pandas_datareader",
    "cvxpy",
    "sklearn",
    "nbformat",
    "nbconvert",
)

RAW_TO_PERCENT_COLUMNS = {
    "SMALL LoBM": "small_lobm_vwret_pct",
    "ME1 BM2": "me1_bm2_vwret_pct",
    "SMALL HiBM": "small_hibm_vwret_pct",
    "BIG LoBM": "big_lobm_vwret_pct",
    "ME2 BM2": "me2_bm2_vwret_pct",
    "BIG HiBM": "big_hibm_vwret_pct",
}

PERCENT_COLUMNS = list(RAW_TO_PERCENT_COLUMNS.values())
DECIMAL_COLUMNS = [column.replace("_pct", "_dec") for column in PERCENT_COLUMNS]

PROJECT_DIRECTORIES = (
    SRC_DIR,
    SCRIPTS_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    INDIVIDUAL_FIGURES_DIR,
    PREDICTIVE_FIGURES_DIR,
    TRADING_FIGURES_DIR,
    TABLES_DIR,
    INDIVIDUAL_TABLES_DIR,
    PREDICTIVE_TABLES_DIR,
    TRADING_TABLES_DIR,
    MODELS_DIR,
    INDIVIDUAL_MODELS_DIR,
    PREDICTIVE_MODELS_DIR,
    TRADING_MODELS_DIR,
    REPORT_SECTIONS_DIR,
    LOGS_DIR,
)
