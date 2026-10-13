# Stage 3 Benchmark Definitions

## Scope

Stage 3 implements characteristic-only, non-graph benchmarks on the cleaned monthly `features500` panel. The purpose is not to exactly replicate every detail of IPCA or the conditional autoencoder papers. The purpose is to build an academically defensible baseline suite that asks whether firm characteristics alone can learn dynamic exposures and pricing information before graph structure is added.

The default target is `target_excess_return`, defined in Stage 2 as next-month stock return minus the next-month compounded risk-free return. A row at month `t` uses month-`t` characteristics to predict the realized target for month `t+1`.

## Shared Protocol

All Stage 3 models use the same cleaned panel, the same feature columns, and the same expanding out-of-sample schedule:

- Training window: historical months before the validation block.
- Validation window: the trailing months immediately before the test block, used for neural early stopping.
- Test window: the next OOS refit block.
- Refit frequency: configurable in `configs/benchmarks_features500.yaml`.
- Graph inputs: not used in Stage 3.

The default config runs a computationally manageable OOS subset. Set `oos.max_oos_months: null` to run the full available OOS span.

## MLP Predictor

The MLP benchmark predicts next-month excess return directly from month-`t` firm characteristics:

```text
y_hat_{i,t+1} = h_theta(x_{i,t})
```

It does not estimate latent factors or conditional betas. Its role is to measure how much OOS performance can come from generic nonlinear characteristic prediction alone. This benchmark is important because a later graph model must beat a strong no-graph nonlinear predictor, not only a linear model.

Simplifications:

- Direct supervised MSE objective.
- No explicit asset pricing restriction.
- No latent factor decomposition.

## IPCA-Style Benchmark

The IPCA-style benchmark uses a linear characteristic-driven loading function and latent factor returns:

```text
beta_{i,t} = x_{i,t}' Gamma
r_{i,t+1} ~= beta_{i,t}' f_t
```

The implementation estimates `Gamma` and monthly latent factors by ridge-regularized alternating least squares on the training window. OOS prediction uses the exposure estimated from month-`t` characteristics and the historical mean factor premium from the training window.

This model maps most directly to conditional beta learning: characteristics are instruments for dynamic loadings, while factors are latent pricing components.

Simplifications relative to canonical IPCA:

- Uses a transparent ridge ALS implementation rather than a full paper replication.
- Does not implement all IPCA normalization and identification conventions.
- Uses historical mean latent factor premium for OOS forecasts.
- Factor estimates are identified only up to the usual rotation/sign ambiguity.

## Conditional Autoencoder-Style Benchmark

The CAE-style benchmark keeps the latent factor structure but replaces the linear loading map with a neural network:

```text
beta_{i,t} = g_theta(x_{i,t})
r_{i,t+1} ~= beta_{i,t}' f_t
```

During training, the model learns a nonlinear beta network and train-month factor embeddings. For OOS prediction, it uses the month-`t` nonlinear exposure and the historical mean factor embedding from the training window.

This benchmark answers whether nonlinear characteristic-to-beta mapping improves over the IPCA-style linear map even before graph context is introduced.

Simplifications relative to canonical conditional autoencoder asset pricing models:

- Uses train-month factor embeddings rather than a full encoder/decoder architecture with every paper-specific restriction.
- Uses MSE reconstruction/prediction loss as the main objective.
- Does not yet add explicit no-arbitrage or SDF pricing penalties.
- Does not use macro state variables or graph relations.

## Comparability

Comparable across Stage 3 models:

- Same Stage 2 cleaned `features500` panel.
- Same target column.
- Same feature matrix.
- Same expanding OOS blocks.
- Same OOS prediction dates.
- Same evaluation metrics.

Not perfectly comparable:

- MLP is a direct predictor and does not produce economically identified latent factors.
- IPCA-style and CAE-style models produce latent factors/loadings, but factor signs and rotations are not uniquely identified.
- Neural models use validation early stopping; the IPCA-style model does not need a validation loss in the same way.
- These are benchmark approximations, not exact reproductions of the original papers.

## What Stage 5 Graph Models Must Beat

The graph model must show incremental value relative to:

- MLP: beat generic nonlinear no-graph prediction.
- IPCA-style: beat linear characteristic-driven dynamic beta learning.
- CAE-style: beat nonlinear characteristic-driven dynamic beta learning without graph context.

The strongest claim in Stage 5 would be that graph structure improves OOS `R^2`, rank IC, pricing error, and later portfolio performance after controlling for both nonlinearity and characteristic-only conditional exposure learning.
