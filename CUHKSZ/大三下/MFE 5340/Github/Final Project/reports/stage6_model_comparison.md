# Stage 6 Model Comparison

## Scope

Stage 6 compares all implemented default main-spec models on the saved OOS outputs from Stages 3 and 5:

- `mlp_predictor`
- `ipca_style`
- `conditional_autoencoder_style`
- `graph_conditional_latent_factor`

The comparison uses the common stock-date intersection across all models, so the main tables are as close to apples-to-apples as possible. Coverage is summarized in the saved CSV tables.

## Common OOS Sample

All four models overlap on the same 24 OOS months from 2016-01-31 through 2017-12-31. The aligned sample uses the stock-date intersection across all models before calculating summary metrics.

## Main Result

The graph model does show incremental value relative to the characteristic-only CAE-style benchmark on ranking-oriented metrics, but not a clean across-the-board win over all characteristic-only models.

- Graph model OOS R^2: -0.0111
- Graph model rank IC mean: 0.0374
- Graph model cross-sectional corr mean: 0.0431
- Graph model pricing error monthly RMSE: 0.0488

Against the nonlinear characteristic-only exposure model (`conditional_autoencoder_style`):

- Graph rank IC is higher: 0.0374 vs -0.0008
- Graph cross-sectional corr is higher: 0.0431 vs 0.0022
- Graph OOS R^2 is slightly worse: -0.0111 vs -0.0073

Against the direct nonlinear predictor (`mlp_predictor`):

- Graph rank IC is lower: 0.0374 vs 0.0582
- Graph OOS R^2 is better than MLP but still negative: -0.0111 vs -0.0769
- Graph pricing error RMSE is slightly lower: 0.0488 vs 0.0495

Against the linear dynamic-beta benchmark (`ipca_style`):

- Graph rank IC is slightly higher: 0.0374 vs 0.0354
- Graph OOS R^2 is slightly worse: -0.0111 vs -0.0071

## Interpretation

At this point, the gains from graph structure look more like **ranking gains than raw prediction gains**.

- Best OOS R^2 on the aligned sample: `ipca_style`
- Best rank IC mean: `mlp_predictor`
- Best cross-sectional correlation mean: `graph_conditional_latent_factor`
- Lowest pricing error monthly RMSE: `ipca_style`

This suggests the current graph model is helping the conditional beta function capture some useful cross-sectional ordering information relative to CAE-style and IPCA-style latent factor models, but it is not yet producing a decisive OOS prediction improvement over the strongest characteristic-only benchmark set.

## Latent-Factor Diagnostics

Latent-factor-related diagnostics are available for IPCA-style, CAE-style, and the graph model, but not for the direct MLP predictor. These diagnostics are reported in the latent diagnostics CSV. They should be interpreted carefully because latent factors are only identified up to standard rotation/sign conventions and the current factor forecast is just the train-window mean.

## Figures

- Overall metric bars: `outputs/comparison/stage6_plots/stage6_overall_metric_bars.png`
- Monthly rank IC: `outputs/comparison/stage6_plots/stage6_monthly_rank_ic.png`
- Monthly cross-sectional correlation: `outputs/comparison/stage6_plots/stage6_monthly_cross_sectional_corr.png`
- Prediction correlation heatmap: `outputs/comparison/stage6_plots/stage6_prediction_correlation_heatmap.png`

## Strongest Remaining Limitations

1. The OOS comparison window is short: only 24 months in the current default run.
2. The graph model is still a simple first-pass GCN/GAT beta encoder, not a stronger structural asset-pricing system.
3. The graph is a homogeneous combination of multiple similarity layers, not a true multi-relation graph model.
4. The factor forecast uses the train-window mean latent factor premium rather than a dedicated factor forecasting model.
5. There is still no Stage 6 portfolio comparison yet, so the economic value is only partially observed through prediction/ranking/pricing diagnostics.
6. Industry edges are still unavailable because the stored data does not contain point-in-time industry classifications.
7. Hyperparameter tuning is still shallow, so the current graph result should be interpreted as a baseline rather than an optimized ceiling.
