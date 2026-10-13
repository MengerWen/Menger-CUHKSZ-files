# Final Project Full File Map Explained

这份说明把整个 [`Final Project/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project) 按阶段重新整理了一遍，并把文中提到的文件和目录都换成了 GitHub 链接。

说明：

- 目录链接指向 GitHub `tree`。
- 文件链接指向 GitHub `blob`。
- [`outputs/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs) 下部分自动生成文件如果没有被提交，GitHub 上可能打不开；这里保留链接主要是为了让你看清它们在项目结构中的位置。

---

## 1. 顶层总览

顶层核心文件：

- [`Guideline.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Guideline.md)
- [`Proposal.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Proposal.md)
- [`README.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/README.md)
- [`requirements.txt`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/requirements.txt)
- [`.gitignore`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/.gitignore)

顶层目录：

- [`configs/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/configs)
- [`data/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/data)
- [`scripts/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/scripts)
- [`src/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/src)
- [`outputs/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs)
- [`reports/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/reports)

这一层决定了整个项目的阅读路线：先看研究目标，再看数据，再看模型，再看结果。

---

## 2. 主流水线

最重要的入口脚本顺序：

1. [`scripts/check_environment.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/check_environment.py)
2. [`scripts/load_data.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/load_data.py)
3. [`scripts/inspect_data.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/inspect_data.py)
4. [`scripts/build_panel.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/build_panel.py)
5. [`scripts/train_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/train_benchmarks.py)
6. [`scripts/build_graph.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/build_graph.py)
7. [`scripts/train_graph_model.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/train_graph_model.py)
8. [`scripts/evaluate_model_comparison.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/evaluate_model_comparison.py)
9. [`scripts/backtest_portfolio.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/backtest_portfolio.py)
10. [`scripts/run_stage8_analysis.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/run_stage8_analysis.py)
11. [`scripts/generate_report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/generate_report_figures.py)

可以把它理解成：研究设计 -> 数据检查 -> 面板构建 -> benchmark -> graph -> graph model -> 比较 -> 回测 -> 解释性与稳健性 -> 最终出图。

---

## 3. 环节 0：课程要求与研究协议

关联文件：

- [`Guideline.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Guideline.md)
- [`Proposal.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Proposal.md)
- [`README.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/README.md)
- [`configs/main_spec.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/main_spec.yaml)
- [`reports/project_protocol.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/project_protocol.md)
- [`reports/data_role_mapping.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/data_role_mapping.md)

这一层负责回答“为什么做这个项目、课程真正要求什么、主规格实验怎么定义”。如果你不先看这一层，后面看到模型代码时会只感觉是在跑机器学习，而不知道项目真正关心的是 `month t -> month t+1` 的 OOS 资产定价与组合构建。

---

## 4. 环节 1：环境与基础设施

关联文件：

- [`scripts/check_environment.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/check_environment.py)
- [`scripts/load_data.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/load_data.py)
- [`scripts/evaluate.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/evaluate.py)
- [`scripts/list_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/list_benchmarks.py)
- [`scripts/train.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/train.py)
- [`src/data/loaders.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/data/loaders.py)
- [`src/evaluation/metrics.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/metrics.py)
- [`src/models/benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/benchmarks.py)
- [`src/training/train.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/training/train.py)

这一层不直接产出主结果，但负责统一路径、训练工具、指标函数和 benchmark 注册，是后面所有阶段的公共基础。

---

## 5. 环节 2：原始数据检查与 Stage 2 数据审计

原始数据：

- [`data/monthly_returns.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/monthly_returns.pkl)
- [`data/risk_free.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/risk_free.csv)
- [`data/mcap.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/mcap.pkl)
- [`data/BLACKLIST.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/BLACKLIST.pkl)
- [`data/UNTRADABLE.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/UNTRADABLE.pkl)
- [`data/csi500_mask_monthly.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/csi500_mask_monthly.pkl)
- [`data/FF5.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/FF5.csv)
- [`data/HXZ.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/HXZ.csv)
- [`data/price.h5`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/price.h5)
- [`data/instruction.txt`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/instruction.txt)
- [`data/features500/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/data/features500)
- [`data/features/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/data/features)

检查与报告：

- [`scripts/inspect_data.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/inspect_data.py)
- [`src/data/inspection.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/data/inspection.py)
- [`reports/data_audit_stage2.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/data_audit_stage2.md)
- [`outputs/metadata/data_inspection_stage2.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/data_inspection_stage2.json)

这一层负责回答：数据到底是什么格式、日期和股票代码怎么对齐、[`data/features500/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/data/features500) 是否真的对应 CSI 500、[`data/BLACKLIST.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/BLACKLIST.pkl) 和 [`data/UNTRADABLE.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/data/UNTRADABLE.pkl) 的时间语义是什么。

---

## 6. 环节 3：构建清洗后的月度面板

关联文件：

- [`configs/cleaning_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/cleaning_features500.yaml)
- [`scripts/build_panel.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/build_panel.py)
- [`scripts/run_preprocessing.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/run_preprocessing.py)
- [`src/data/preprocessing.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/data/preprocessing.py)
- [`src/features/build_features.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/features/build_features.py)
- [`outputs/panels/main_features500_panel.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/panels/main_features500_panel.pkl)
- [`outputs/metadata/main_features500_panel_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/main_features500_panel_metadata.json)

这里真正把原始数据变成后续所有模型共用的“主面板”。核心逻辑是：month `t` 的特征对齐到 `t+1` 的收益，日频 risk-free 合成月频后得到 `target_excess_return`，blacklist / untradable 聚合并过滤，再做横截面 winsorize、impute、normalize，最终输出 [`outputs/panels/main_features500_panel.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/panels/main_features500_panel.pkl)。

---

## 7. 环节 4：Stage 3 无图 benchmark

关联文件：

- [`configs/benchmarks_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/benchmarks_features500.yaml)
- [`scripts/train_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/train_benchmarks.py)
- [`scripts/list_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/list_benchmarks.py)
- [`src/training/non_graph_benchmark_pipeline.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/training/non_graph_benchmark_pipeline.py)
- [`src/models/non_graph_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/non_graph_benchmarks.py)
- [`src/models/torch_models.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/torch_models.py)
- [`src/models/benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/benchmarks.py)
- [`reports/benchmark_definitions_stage3.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/benchmark_definitions_stage3.md)

主要输出：

- [`outputs/predictions/stage3_non_graph_predictions.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/predictions/stage3_non_graph_predictions.pkl)
- [`outputs/latent/stage3_non_graph_exposures.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/latent/stage3_non_graph_exposures.pkl)
- [`outputs/latent/stage3_non_graph_factors.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/latent/stage3_non_graph_factors.pkl)
- [`outputs/metrics/stage3_non_graph_metrics.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metrics/stage3_non_graph_metrics.csv)
- [`outputs/metadata/stage3_non_graph_run_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/stage3_non_graph_run_metadata.json)

这一阶段的角色是给 graph 模型树立公平 benchmark：MLP、IPCA-style、CAE-style 都在这里定义并统一按 OOS 协议训练。

---

## 8. 环节 5：Stage 4 股票图构建

关联文件：

- [`configs/graphs_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/graphs_features500.yaml)
- [`scripts/build_graph.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/build_graph.py)
- [`src/graphs/monthly_graphs.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/graphs/monthly_graphs.py)
- [`src/graphs/build_graph.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/graphs/build_graph.py)
- [`reports/graph_design_stage4.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/graph_design_stage4.md)

主要输出：

- [`outputs/graphs/features500_similarity_hybrid_manifest.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/graphs/features500_similarity_hybrid_manifest.csv)
- [`outputs/graphs/features500_similarity_hybrid_stats.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/graphs/features500_similarity_hybrid_stats.csv)
- [`outputs/graphs/features500_similarity_hybrid/edges/*.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/graphs/features500_similarity_hybrid/edges)
- [`outputs/graphs/features500_similarity_hybrid/pyg/*.pt`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/graphs/features500_similarity_hybrid/pyg)
- [`outputs/metadata/stage4_graph_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/stage4_graph_metadata.json)

这一层不是预测，而是在构造“股票之间的关系上下文”。当前主规格图是相似性图，不是行业图。

---

## 9. 环节 6：Stage 5 图增强条件潜在因子模型

关联文件：

- [`configs/graph_model_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/graph_model_features500.yaml)
- [`scripts/train_graph_model.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/train_graph_model.py)
- [`src/models/graph_latent_factor.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/graph_latent_factor.py)
- [`src/training/graph_model_pipeline.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/training/graph_model_pipeline.py)
- [`src/models/torch_models.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/torch_models.py)
- [`reports/graph_model_architecture_stage5.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/graph_model_architecture_stage5.md)

主要输出：

- [`outputs/predictions/stage5_graph_predictions.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/predictions/stage5_graph_predictions.pkl)
- [`outputs/latent/stage5_graph_exposures.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/latent/stage5_graph_exposures.pkl)
- [`outputs/latent/stage5_graph_factors.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/latent/stage5_graph_factors.pkl)
- [`outputs/attention/stage5_graph_attention.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/attention/stage5_graph_attention.pkl)
- [`outputs/metrics/stage5_graph_metrics.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metrics/stage5_graph_metrics.csv)
- [`outputs/metadata/stage5_graph_model_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/stage5_graph_model_metadata.json)

这一步的关键不是“直接用 GNN 预测收益”，而是“先用图和特征学 beta，再用 beta × latent factor 生成收益预测”。

---

## 10. 环节 7：Stage 6 模型横向比较

关联文件：

- [`configs/evaluation_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/evaluation_features500.yaml)
- [`scripts/evaluate_model_comparison.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/evaluate_model_comparison.py)
- [`src/evaluation/model_comparison.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/model_comparison.py)
- [`src/evaluation/metrics.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/metrics.py)
- [`reports/stage6_model_comparison.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage6_model_comparison.md)

主要输出：

- [`outputs/comparison/stage6_tables/stage6_summary_metrics.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_tables/stage6_summary_metrics.csv)
- [`outputs/comparison/stage6_tables/stage6_monthly_metrics.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_tables/stage6_monthly_metrics.csv)
- [`outputs/comparison/stage6_tables/stage6_latent_diagnostics.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_tables/stage6_latent_diagnostics.csv)
- [`outputs/comparison/stage6_tables/stage6_prediction_correlation.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_tables/stage6_prediction_correlation.csv)
- [`outputs/comparison/stage6_plots/stage6_overall_metric_bars.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_plots/stage6_overall_metric_bars.png)
- [`outputs/comparison/stage6_plots/stage6_monthly_rank_ic.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_plots/stage6_monthly_rank_ic.png)
- [`outputs/comparison/stage6_plots/stage6_monthly_cross_sectional_corr.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_plots/stage6_monthly_cross_sectional_corr.png)
- [`outputs/comparison/stage6_plots/stage6_prediction_correlation_heatmap.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/comparison/stage6_plots/stage6_prediction_correlation_heatmap.png)
- [`outputs/metadata/stage6_comparison_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/stage6_comparison_metadata.json)

这一阶段是“预测 / 定价层面的横向比较”，还没有进入真实投资组合。

---

## 11. 环节 8：Stage 7 投资组合回测

关联文件：

- [`configs/portfolio_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/portfolio_features500.yaml)
- [`scripts/backtest_portfolio.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/backtest_portfolio.py)
- [`src/portfolio/backtest.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/portfolio/backtest.py)
- [`reports/stage7_portfolio_results.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage7_portfolio_results.md)

主要输出：

- [`outputs/portfolio/stage7_weights.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_weights.pkl)
- [`outputs/portfolio/stage7_monthly_returns.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_monthly_returns.pkl)
- [`outputs/portfolio/stage7_performance_summary.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_performance_summary.csv)
- [`outputs/portfolio/stage7_signal_coverage.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_signal_coverage.csv)
- [`outputs/portfolio/stage7_strategy_coverage.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_strategy_coverage.csv)
- [`outputs/portfolio/stage7_plots/stage7_cumulative_long_only_equal_10bps.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_plots/stage7_cumulative_long_only_equal_10bps.png)
- [`outputs/portfolio/stage7_plots/stage7_cumulative_long_only_value_10bps.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_plots/stage7_cumulative_long_only_value_10bps.png)
- [`outputs/portfolio/stage7_plots/stage7_cumulative_long_short_equal_10bps.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_plots/stage7_cumulative_long_short_equal_10bps.png)
- [`outputs/portfolio/stage7_plots/stage7_cumulative_long_short_value_10bps.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_plots/stage7_cumulative_long_short_value_10bps.png)
- [`outputs/portfolio/stage7_plots/stage7_summary_bars_10bps.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/portfolio/stage7_plots/stage7_summary_bars_10bps.png)
- [`outputs/metadata/stage7_portfolio_metadata.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/metadata/stage7_portfolio_metadata.json)

这一步把预测信号真正转成了 long-short / long-only 组合，是整个项目“经济价值”最重要的一层。

---

## 12. 环节 9：Stage 8 可解释性与稳健性分析

关联文件：

- [`configs/stage8_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/stage8_features500.yaml)
- [`scripts/run_stage8_analysis.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/run_stage8_analysis.py)
- [`src/evaluation/stage8_analysis.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/stage8_analysis.py)
- [`reports/stage8_final_summary.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage8_final_summary.md)

主要表、图与派生目录：

- [`outputs/stage8/tables/stage8_main_results_table.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_main_results_table.csv)
- [`outputs/stage8/tables/stage8_feature_exposure_association.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_feature_exposure_association.csv)
- [`outputs/stage8/tables/stage8_feature_exposure_top_links.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_feature_exposure_top_links.csv)
- [`outputs/stage8/tables/stage8_permutation_importance.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_permutation_importance.csv)
- [`outputs/stage8/tables/stage8_neighbor_monthly_summary.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_neighbor_monthly_summary.csv)
- [`outputs/stage8/tables/stage8_neighbor_edge_mix.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_neighbor_edge_mix.csv)
- [`outputs/stage8/tables/stage8_graph_robustness_summary.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_graph_robustness_summary.csv)
- [`outputs/stage8/tables/stage8_graph_robustness_portfolio.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_graph_robustness_portfolio.csv)
- [`outputs/stage8/tables/stage8_gat_attention_summary.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/tables/stage8_gat_attention_summary.csv)
- [`outputs/stage8/plots/stage8_feature_exposure_heatmap.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_feature_exposure_heatmap.png)
- [`outputs/stage8/plots/stage8_permutation_importance.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_permutation_importance.png)
- [`outputs/stage8/plots/stage8_neighbor_summary.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_neighbor_summary.png)
- [`outputs/stage8/plots/stage8_neighbor_edge_mix.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_neighbor_edge_mix.png)
- [`outputs/stage8/plots/stage8_robustness_prediction_bars.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_robustness_prediction_bars.png)
- [`outputs/stage8/plots/stage8_robustness_portfolio_bars.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_robustness_portfolio_bars.png)
- [`outputs/stage8/plots/stage8_gat_attention_edge_types.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/outputs/stage8/plots/stage8_gat_attention_edge_types.png)
- [`outputs/stage8/predictions/*.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/predictions)
- [`outputs/stage8/latent/*.pkl`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/latent)
- [`outputs/stage8/metrics/*.csv`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/metrics)
- [`outputs/stage8/metadata/*.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/metadata)
- [`outputs/stage8/graphs/graph_return_only/...`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/graphs/graph_return_only)
- [`outputs/stage8/graphs/graph_lookback6/...`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/outputs/stage8/graphs/graph_lookback6)

这一层负责“解释”和“稳健性”：模型依赖哪些特征、graph 邻域有什么结构、如果换图或换模型设定，结论还是否成立。

---

## 13. 环节 10：最终报告图生成

关联文件：

- [`configs/report_figures.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/report_figures.yaml)
- [`scripts/generate_report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/generate_report_figures.py)
- [`src/evaluation/report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/report_figures.py)
- [`reports/figure_guide.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figure_guide.md)
- [`reports/figures/figure_manifest.json`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_manifest.json)
- [`reports/figures/*`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/reports/figures)

主要 final report 图：

- [`reports/figures/figure_1_sample_coverage.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_1_sample_coverage.png)
- [`reports/figures/figure_2_graph_overview.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_2_graph_overview.png)
- [`reports/figures/figure_3_model_comparison.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_3_model_comparison.png)
- [`reports/figures/figure_4_portfolio_cumulative.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_4_portfolio_cumulative.png)
- [`reports/figures/figure_5_portfolio_summary.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_5_portfolio_summary.png)
- [`reports/figures/figure_6_interpretability.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_6_interpretability.png)
- [`reports/figures/figure_7_graph_robustness.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_7_graph_robustness.png)
- [`reports/figures/figure_8_gat_attention_exploratory.png`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figures/figure_8_gat_attention_exploratory.png)

你当前打开的 [`scripts/generate_report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/scripts/generate_report_figures.py) 只是一个薄入口，真正负责画图的是 [`src/evaluation/report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/report_figures.py)。

---

## 14. reports/ 的推荐阅读顺序

1. [`reports/project_protocol.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/project_protocol.md)
2. [`reports/data_role_mapping.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/data_role_mapping.md)
3. [`reports/data_audit_stage2.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/data_audit_stage2.md)
4. [`reports/benchmark_definitions_stage3.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/benchmark_definitions_stage3.md)
5. [`reports/graph_design_stage4.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/graph_design_stage4.md)
6. [`reports/graph_model_architecture_stage5.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/graph_model_architecture_stage5.md)
7. [`reports/stage6_model_comparison.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage6_model_comparison.md)
8. [`reports/stage7_portfolio_results.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage7_portfolio_results.md)
9. [`reports/stage8_final_summary.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage8_final_summary.md)
10. [`reports/figure_guide.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figure_guide.md)

---

## 15. 作为新手，最推荐的阅读路线

第一轮：先看“这个项目在研究什么”

- [`Guideline.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Guideline.md)
- [`Proposal.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Proposal.md)
- [`README.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/README.md)
- [`reports/project_protocol.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/project_protocol.md)

第二轮：再看“数据如何变成 panel”

- [`reports/data_audit_stage2.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/data_audit_stage2.md)
- [`configs/cleaning_features500.yaml`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/configs/cleaning_features500.yaml)
- [`src/data/preprocessing.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/data/preprocessing.py)

第三轮：再看“模型怎么建”

- [`reports/benchmark_definitions_stage3.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/benchmark_definitions_stage3.md)
- [`src/models/non_graph_benchmarks.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/non_graph_benchmarks.py)
- [`reports/graph_model_architecture_stage5.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/graph_model_architecture_stage5.md)
- [`src/models/graph_latent_factor.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/models/graph_latent_factor.py)

第四轮：最后看“结果怎么样”

- [`reports/stage6_model_comparison.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage6_model_comparison.md)
- [`reports/stage7_portfolio_results.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage7_portfolio_results.md)
- [`reports/stage8_final_summary.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/stage8_final_summary.md)
- [`reports/figure_guide.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/reports/figure_guide.md)

---

## 16. 一句话总结

> [`Guideline.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Guideline.md) 和 [`Proposal.md`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/Proposal.md) 定义“研究什么”，[`data/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project/data) 与 [`src/data/preprocessing.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/data/preprocessing.py) 定义“拿什么数据研究”，Stage 3 / 4 / 5 的脚本、配置和源码定义“模型怎么建”，Stage 6 / 7 / 8 的比较、回测和总结定义“模型到底好不好”，而 [`src/evaluation/report_figures.py`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/blob/main/Final%20Project/src/evaluation/report_figures.py) 负责把这些结果整理成最终报告图。

这就是整个 [`Final Project/`](https://github.com/MengerWen/MFE5340-AI-in-Financial-Engineering/tree/main/Final%20Project) 的完整文件地图。
