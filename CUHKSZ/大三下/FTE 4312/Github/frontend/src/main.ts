import "./style.css";

interface DemoMeta {
  title: string;
  subtitle: string;
  generated_at: string;
  symbol: string;
  venue: string;
  frequency: string;
  date_range: {
    start: string;
    end_exclusive: string;
  };
  chain_name: string;
  artifact_label?: string;
  artifact_note?: string;
  bundle_name?: string;
}

interface DemoChart {
  title: string;
  subtitle: string;
  section: string;
  image: string;
  source_path: string;
}

interface StrategyRow {
  strategy_name: string;
  source: string;
  source_subtype: string;
  task: string;
  evaluation_split?: string;
  has_trades?: boolean;
  trade_count: number;
  cumulative_return: number;
  realized_cumulative_return?: number;
  annualized_return: number;
  realized_annualized_return?: number;
  sharpe_ratio: number;
  raw_period_sharpe?: number;
  autocorr_adjusted_sharpe?: number;
  realized_sharpe_ratio?: number;
  max_drawdown: number;
  realized_max_drawdown?: number;
  mark_to_market_max_drawdown?: number;
  win_rate: number;
  profit_factor?: number;
  average_trade_return_bps: number;
  median_trade_return_bps?: number;
  exposure_time_fraction?: number;
  funding_contribution_share?: number;
  total_net_pnl_usd: number;
  total_funding_pnl_usd?: number;
  final_equity_usd: number;
  realized_final_equity_usd?: number;
}

interface ActivityItem {
  timestamp: string;
  kind: string;
  title: string;
  detail: string;
}

interface DemoSnapshot {
  meta: DemoMeta;
  overview: {
    goal: string;
    story_points: string[];
    layers: Array<{ label: string; detail: string }>;
  };
  research: {
    canonical_rows: number;
    perpetual_rows: number;
    funding_events: number;
    coverage_ratio: number;
    funding_mean_bps: number;
    funding_std_bps: number;
    spread_mean_bps: number;
    annualized_volatility: number;
  };
  models: {
    baseline_best: Record<string, unknown>;
    deep_learning_best: Record<string, unknown>;
    deep_learning_single_best: Record<string, unknown>;
    deep_learning_comparison: {
      available: boolean;
      best_model_note: string | null;
      run_count: number | null;
      report_path: string | null;
      summary_path: string | null;
      test_leaderboard: Array<Record<string, unknown>>;
    };
  };
  backtest: {
    summary: Record<string, unknown>;
    diagnostics?: Record<string, unknown>;
    risk_view?: {
      primary_split?: string | null;
      primary_trade_count?: number | null;
      combined_trade_count?: number | null;
      equity_basis?: string;
      drawdown_basis?: string;
      realized_audit_available?: boolean;
    };
    best_strategy: StrategyRow;
    top_strategies: StrategyRow[];
    assumptions: string[];
  };
  charts: DemoChart[];
  vault: {
    chain_name: string;
    vault_address: string;
    stablecoin_address: string;
    selected_strategy: string;
    strategy_state: string;
    suggested_direction: string;
    reported_nav_assets: number;
    summary_pnl_assets: number;
    summary_pnl_usd: number;
    call_count: number;
    execution_summary: Record<string, unknown>;
  };
  activity_log: ActivityItem[];
  simulation: {
    asset_symbol: string;
    asset_decimals: number;
    wallet_cash_assets: number;
    base_vault_cash_assets: number;
    base_reported_nav_assets: number;
    base_total_shares: number;
    user_shares: number;
    strategy_state: string;
    operator_plan: {
      selected_strategy: string;
      strategy_state: string;
      suggested_direction: string;
      reported_nav_assets: number;
      summary_pnl_assets: number;
      summary_pnl_usd: number;
      should_trade: boolean;
      demo_activation_state: string;
      demo_activation_direction: string;
    };
  };
}

interface ExploratoryFigureAsset {
  label: string;
  image: string;
  file_name: string;
}

interface ExploratorySummaryPayload {
  strict_summary?: Record<string, unknown>;
  strict_final_summary?: Record<string, unknown>;
  exploratory_summary?: {
    strategy_count?: number;
    nonzero_trade_strategy_count?: number;
    best_showcase_row?: Record<string, unknown>;
    full_leaderboard_path?: string;
    showcase_leaderboard_path?: string;
    prediction_distribution_path?: string;
    quantile_analysis_path?: string;
    figure_assets?: ExploratoryFigureAsset[];
  };
  disclaimer?: string;
}

interface ExploratoryArtifacts {
  summary: ExploratorySummaryPayload | null;
  leaderboard: Array<Record<string, unknown>>;
  predictionDistribution: Array<Record<string, unknown>>;
  quantileAnalysis: Array<Record<string, unknown>>;
}

interface SimulationState {
  walletCashAssets: number;
  vaultCashAssets: number;
  reportedNavAssets: number;
  totalShares: number;
  userShares: number;
  strategyState: string;
  suggestedDirection: string;
  selectedStrategy: string;
  amountInputAssets: number;
  activityLog: ActivityItem[];
}

const appNode = document.querySelector<HTMLDivElement>("#app");

if (!appNode) {
  throw new Error("App root not found.");
}

const app = appNode;
const BASE_URL = import.meta.env.BASE_URL;
const urlParams = new URLSearchParams(window.location.search);
const DEMO_BUNDLE = urlParams.get("mode") || "demo";
const REPORT_BUNDLE =
  urlParams.get("reportMode") ||
  (DEMO_BUNDLE === "demo" ? "report" : `${DEMO_BUNDLE}/report`);
const DEMO_SNAPSHOT_URL = `${BASE_URL}${DEMO_BUNDLE}/demo_snapshot.json`;
const EXPLORATORY_SUMMARY_URL = `${BASE_URL}${DEMO_BUNDLE}/exploratory_dl_summary.json`;
const EXPLORATORY_LEADERBOARD_URL = `${BASE_URL}${DEMO_BUNDLE}/exploratory_dl_leaderboard.json`;
const EXPLORATORY_PREDICTION_DISTRIBUTION_URL = `${BASE_URL}${DEMO_BUNDLE}/exploratory_prediction_distribution.json`;
const EXPLORATORY_QUANTILE_ANALYSIS_URL = `${BASE_URL}${DEMO_BUNDLE}/exploratory_quantile_analysis.json`;
const FINAL_REPORT_URL = `${BASE_URL}${REPORT_BUNDLE}/`;
const FINAL_REPORT_MARKDOWN_URL = `${BASE_URL}${REPORT_BUNDLE}/final_report.md`;
const REPOSITORY_URL =
  "https://github.com/MengerWen/Deep-Learning-Based-Delta-Neutral-Statistical-Arbitrage-on-Perpetual-Funding-Rates";

const DEFAULT_DEPOSIT_ASSETS = 2_500 * 10 ** 6;
const DEFAULT_WITHDRAW_ASSETS = 1_000 * 10 ** 6;

let snapshotState: DemoSnapshot | null = null;
let simulationState: SimulationState | null = null;
let exploratoryArtifactsState: ExploratoryArtifacts | null = null;

function formatDate(dateText: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(new Date(dateText));
}

function formatDateTime(dateText: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(dateText));
}

function formatUsd(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(2)}%`;
}

function formatBps(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${value.toFixed(2)} bps`;
}

function formatAssets(
  assets: number,
  assetSymbol: string,
  decimals: number,
  digits = 2,
): string {
  return `${formatNumber(assets / 10 ** decimals, digits)} ${assetSymbol}`;
}

function formatAddress(address: string): string {
  if (!address || address.length < 12) {
    return address || "n/a";
  }
  return `${address.slice(0, 6)}...${address.slice(-4)}`;
}

function formatYesNo(value: boolean | null | undefined): string {
  if (value == null) {
    return "n/a";
  }
  return value ? "yes" : "no";
}

function createInitialSimulationState(snapshot: DemoSnapshot): SimulationState {
  return {
    walletCashAssets: snapshot.simulation.wallet_cash_assets,
    vaultCashAssets: snapshot.simulation.base_vault_cash_assets,
    reportedNavAssets: snapshot.simulation.base_reported_nav_assets,
    totalShares: snapshot.simulation.base_total_shares,
    userShares: snapshot.simulation.user_shares,
    strategyState: snapshot.simulation.strategy_state,
    suggestedDirection: "flat",
    selectedStrategy: snapshot.vault.selected_strategy,
    amountInputAssets: DEFAULT_DEPOSIT_ASSETS,
    activityLog: [...snapshot.activity_log].reverse(),
  };
}

function convertToShares(state: SimulationState, assets: number): number {
  if (state.totalShares === 0 || state.reportedNavAssets === 0) {
    return assets;
  }
  return Math.floor((assets * state.totalShares) / state.reportedNavAssets);
}

function previewWithdrawShares(state: SimulationState, assets: number): number {
  if (state.totalShares === 0 || state.reportedNavAssets === 0) {
    return assets;
  }
  return Math.ceil((assets * state.totalShares) / state.reportedNavAssets);
}

function pushActivity(state: SimulationState, item: ActivityItem): SimulationState {
  return {
    ...state,
    activityLog: [item, ...state.activityLog].slice(0, 12),
  };
}

function renderLoading(): void {
  app.innerHTML = `
    <main class="loading-shell">
      <div class="loading-panel">
        <p class="eyebrow">Preparing Demo</p>
        <h1>Loading the funding-rate arbitrage dashboard...</h1>
        <p>Reading the local demo snapshot and chart assets.</p>
      </div>
    </main>
  `;
}

function renderError(errorMessage: string): void {
  const refreshCommand =
    DEMO_BUNDLE === "demo_showcase"
      ? "& 'd:\\MG\\anaconda3\\python.exe' -m src.main build-demo-showcase --config configs/demo/showcase.yaml"
      : "& 'd:\\MG\\anaconda3\\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml";
  const snapshotCommand =
    DEMO_BUNDLE === "demo_showcase"
      ? "& 'd:\\MG\\anaconda3\\python.exe' scripts/demo/build_demo_showcase.py --config configs/demo/showcase.yaml"
      : "& 'd:\\MG\\anaconda3\\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml";
  app.innerHTML = `
    <main class="loading-shell">
      <div class="loading-panel error-panel">
        <p class="eyebrow">Snapshot Missing</p>
        <h1>The frontend could not load <code>${DEMO_SNAPSHOT_URL}</code>.</h1>
        <p>${errorMessage}</p>
        <pre class="command-block">${refreshCommand}</pre>
        <pre class="command-block"># or only refresh the frontend snapshot
${snapshotCommand}</pre>
        <pre class="command-block">cd frontend
npm install
npm run dev</pre>
      </div>
    </main>
  `;
}

function metricCard(label: string, value: string, note: string): string {
  return `
    <article class="metric-card">
      <p class="metric-label">${label}</p>
      <h3>${value}</h3>
      <p class="metric-note">${note}</p>
    </article>
  `;
}

function deliverableCard(
  label: string,
  title: string,
  body: string,
  href: string,
  cta: string,
): string {
  return `
    <article class="deliverable-card">
      <p class="deliverable-label">${label}</p>
      <h3>${title}</h3>
      <p>${body}</p>
      <a class="action-link" href="${href}" target="_blank" rel="noreferrer">${cta}</a>
    </article>
  `;
}

function renderExploratoryShowcase(): string {
  if (!exploratoryArtifactsState?.summary?.exploratory_summary) {
    return "";
  }

  const summary = exploratoryArtifactsState.summary.exploratory_summary;
  const disclaimer =
    exploratoryArtifactsState.summary.disclaimer ??
    "Exploratory results are supplementary showcase artifacts.";
  const bestRow = summary.best_showcase_row ?? {};
  const figures = summary.figure_assets ?? [];
  const distributionPreview = exploratoryArtifactsState.predictionDistribution.slice(0, 4);
  const quantilePreview = exploratoryArtifactsState.quantileAnalysis.slice(0, 8);
  const leaderboardPreview = exploratoryArtifactsState.leaderboard.slice(0, 6);

  return `
    <section class="section-block exploratory-panel">
      <div class="section-heading">
        <p class="section-kicker">Exploratory DL Showcase</p>
        <h2>Supplementary DL results that stay separate from the strict conclusion</h2>
        <p class="panel-text">${disclaimer}</p>
      </div>
      <div class="exploratory-metrics">
        ${metricCard(
          "Showcase Strategies",
          formatNumber(summary.strategy_count ?? null, 0),
          "Independent exploratory DL strategy variants generated from ranking-based and support-aware signal rules.",
        )}
        ${metricCard(
          "Nonzero Trade Runs",
          formatNumber(summary.nonzero_trade_strategy_count ?? null, 0),
          "Only the exploratory track is counted here; strict primary outputs remain unchanged.",
        )}
        ${metricCard(
          "Best Showcase PnL",
          formatUsd((bestRow["total_net_pnl_usd"] as number | null | undefined) ?? null),
          `${String(bestRow["strategy_name"] ?? "n/a")} on ${String(bestRow["evaluation_split"] ?? "n/a")} split.`,
        )}
        ${metricCard(
          "Best Showcase Trades",
          formatNumber((bestRow["trade_count"] as number | null | undefined) ?? null, 0),
          `${String(bestRow["model_name"] ?? "n/a")} with ${String(bestRow["target_type"] ?? "n/a")} and ${String(bestRow["signal_rule"] ?? "n/a")}.`,
        )}
      </div>
      <div class="table-shell">
        <table class="showcase-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Model</th>
              <th>Target</th>
              <th>Signal Rule</th>
              <th>Split</th>
              <th>Trades</th>
              <th>Cum Return</th>
              <th>Net PnL</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${leaderboardPreview
              .map(
                (row) => `
                  <tr>
                    <td>${String(row["strategy_name"] ?? "n/a")}</td>
                    <td>${String(row["model_name"] ?? "n/a")}</td>
                    <td>${String(row["target_type"] ?? "n/a")}</td>
                    <td>${String(row["signal_rule"] ?? "n/a")}</td>
                    <td>${String(row["evaluation_split"] ?? "n/a")}</td>
                    <td>${formatNumber(row["trade_count"] as number | null | undefined, 0)}</td>
                    <td>${formatPercent(row["cumulative_return"] as number | null | undefined)}</td>
                    <td>${formatUsd(row["total_net_pnl_usd"] as number | null | undefined)}</td>
                    <td>${String(row["status"] ?? "n/a")}</td>
                  </tr>
                `,
              )
              .join("")}
          </tbody>
        </table>
      </div>
      ${
        figures.length > 0
          ? `
            <div class="chart-gallery exploratory-gallery">
              ${figures
                .map(
                  (figure) => `
                    <figure class="chart-card">
                      <div class="chart-header">
                        <p class="chart-section">exploratory</p>
                        <h3>${figure.label.split("_").join(" ")}</h3>
                        <p>Generated from the best exploratory showcase run and copied into the frontend-ready demo bundle.</p>
                      </div>
                      <img src="${BASE_URL}${figure.image}" alt="${figure.label}" loading="lazy" />
                    </figure>
                  `,
                )
                .join("")}
            </div>
          `
          : ""
      }
      <div class="exploratory-grid">
        <article class="story-panel compact-panel">
          <div class="section-heading">
            <p class="section-kicker">Prediction Distribution</p>
            <h2>How the exploratory scores spread across splits</h2>
          </div>
          <div class="table-shell compact-table-shell">
            <table class="mini-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Split</th>
                  <th>Mean Score</th>
                  <th>Std</th>
                  <th>Pred Short Rate</th>
                </tr>
              </thead>
              <tbody>
                ${distributionPreview
                  .map(
                    (row) => `
                      <tr>
                        <td>${String(row["run_name"] ?? "n/a")}</td>
                        <td>${String(row["split"] ?? "n/a")}</td>
                        <td>${formatNumber(row["signed_score_mean"] as number | null | undefined, 3)}</td>
                        <td>${formatNumber(row["signed_score_std"] as number | null | undefined, 3)}</td>
                        <td>${formatPercent(row["predicted_short_rate"] as number | null | undefined)}</td>
                      </tr>
                    `,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </article>
        <article class="story-panel compact-panel">
          <div class="section-heading">
            <p class="section-kicker">Quantile Lens</p>
            <h2>Do higher-score buckets carry better directional returns?</h2>
          </div>
          <div class="table-shell compact-table-shell">
            <table class="mini-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Bucket</th>
                  <th>Rows</th>
                  <th>Avg Return</th>
                  <th>Cum Return</th>
                </tr>
              </thead>
              <tbody>
                ${quantilePreview
                  .map(
                    (row) => `
                      <tr>
                        <td>${String(row["run_name"] ?? "n/a")}</td>
                        <td>${String(row["absolute_score_quantile"] ?? "n/a")}</td>
                        <td>${formatNumber(row["row_count"] as number | null | undefined, 0)}</td>
                        <td>${formatBps(row["avg_directional_return_bps"] as number | null | undefined)}</td>
                        <td>${formatBps(row["cumulative_directional_return_bps"] as number | null | undefined)}</td>
                      </tr>
                    `,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  `;
}

function renderDashboard(snapshot: DemoSnapshot, state: SimulationState): void {
  const baselineBest = snapshot.models.baseline_best;
  const dlBest = snapshot.models.deep_learning_best;
  const dlComparison = snapshot.models.deep_learning_comparison;
  const assetSymbol = snapshot.simulation.asset_symbol;
  const assetDecimals = snapshot.simulation.asset_decimals;
  const visibleEvents = state.activityLog.slice(0, 8);
  const amountInputValue = (state.amountInputAssets / 10 ** assetDecimals).toFixed(2);
  const baselineCorr = (baselineBest["pearson_corr"] as number | null | undefined) ?? null;
  const dlMetric =
    ((dlBest["ranking_metric_value"] as number | null | undefined) ??
      (dlBest["pearson_corr"] as number | null | undefined)) ??
    null;
  const bestPnl = snapshot.backtest.best_strategy.total_net_pnl_usd;
  const verdictTitle =
    bestPnl > 0
      ? "Current artifact set preserves positive post-cost alpha."
      : "Predictive structure exists, but the current post-cost strategy set is still unprofitable.";
  const verdictBody =
    bestPnl > 0
      ? "That makes the showcase a clean demonstration of both research quality and strategy viability under the configured assumptions."
      : "That is still a strong course-project outcome because the repository makes the negative result explicit instead of hiding it behind pre-cost metrics.";
  const signalBody =
    bestPnl > 0
      ? "The synthetic strict bundle keeps the baseline modestly positive, then improves the payoff profile across the DL variants without removing path volatility."
      : "Both simple and sequence models learn some structure in the 24-hour post-cost target, which is why the modeling layer remains meaningful even when trading outcomes stay weak.";
  const executionBody =
    bestPnl > 0
      ? "The best strict strategy remains profitable after fees, slippage, and funding, but it still experiences visible drawdowns and non-monotonic recovery."
      : "The current best test-period strategy is still negative after fees, slippage, gas, and next-bar execution, so the repository is explicit about the true cost of monetizing funding dislocations.";
  const architectureBody =
    snapshot.meta.artifact_label
      ? "The same frontend and report contract can now render either real artifacts or the isolated DEMO ONLY showcase bundle without changing the default pipeline."
      : "The project already demonstrates an end-to-end hybrid design: off-chain modeling, transparent backtests, dry-run operator sync, and on-chain share-accounting logic in one coherent story.";
  const bannerHtml = snapshot.meta.artifact_label
    ? `
      <div class="demo-flag">
        <strong>${snapshot.meta.artifact_label}</strong>
        <span>${snapshot.meta.artifact_note ?? "Illustrative showcase bundle"}</span>
      </div>
    `
    : "";

  app.innerHTML = `
    <div class="page-shell">
      <div class="page-noise"></div>
      <header class="masthead" id="top">
        <div class="masthead-copy">
          ${bannerHtml}
          <p class="eyebrow">Signal To Settlement</p>
          <h1>${snapshot.meta.title}</h1>
          <p class="lede">${snapshot.meta.subtitle}</p>
          <div class="hero-meta">
            <span>${snapshot.meta.symbol} on ${snapshot.meta.venue}</span>
            <span>${formatDate(snapshot.meta.date_range.start)} to ${formatDate(snapshot.meta.date_range.end_exclusive)}</span>
            <span>${snapshot.meta.frequency} research frequency</span>
            <span>${snapshot.meta.chain_name}</span>
          </div>
          <div class="cta-row">
            <a class="action-link" href="${FINAL_REPORT_URL}" target="_blank" rel="noreferrer">Open final report</a>
            <a class="action-link ghost-link" href="${FINAL_REPORT_MARKDOWN_URL}" target="_blank" rel="noreferrer">Markdown version</a>
            <a class="action-link ghost-link" href="${DEMO_SNAPSHOT_URL}" target="_blank" rel="noreferrer">Snapshot JSON</a>
            <a class="action-link ghost-link" href="${REPOSITORY_URL}" target="_blank" rel="noreferrer">GitHub repo</a>
          </div>
        </div>
        <aside class="masthead-panel">
          <p class="panel-label">Showcase Verdict</p>
          <p class="panel-text"><strong>${verdictTitle}</strong></p>
          <p class="panel-text">${verdictBody}</p>
          <div class="status-chip-row">
            <span class="status-chip">Research pipeline ready</span>
            <span class="status-chip">Backtest artifacts loaded</span>
            <span class="status-chip">Vault prototype synced locally</span>
            <span class="status-chip">Static-pages ready</span>
          </div>
        </aside>
      </header>

      <section class="metrics-ribbon">
        ${metricCard(
          "Data Window",
          `${formatNumber(snapshot.research.canonical_rows, 0)} hours`,
          `${formatNumber(snapshot.research.funding_events, 0)} funding events across the primary BTCUSDT sample.`,
        )}
        ${metricCard(
          "Best Backtest",
          formatUsd(snapshot.backtest.best_strategy.total_net_pnl_usd),
          `${snapshot.backtest.best_strategy.strategy_name} on ${snapshot.backtest.best_strategy.evaluation_split ?? snapshot.backtest.risk_view?.primary_split ?? "primary"} split with ${formatNumber(snapshot.backtest.best_strategy.trade_count, 0)} trades.`,
        )}
        ${metricCard(
          "Baseline Benchmark",
          formatNumber(baselineCorr, 3),
          `${String(baselineBest["model_name"] ?? "baseline")} test-set correlation.`,
        )}
        ${metricCard(
          "DL Zoo Leader",
          formatNumber(dlMetric, 3),
          `${String(dlBest["model_name"] ?? dlBest["run_label"] ?? "deep learning")} under the configured comparison metric.`,
        )}
      </section>

      <section class="section-block delivery-panel">
        <div class="section-heading">
          <p class="section-kicker">Submission Kit</p>
          <h2>Two deliverables now sit on top of the same artifact pipeline</h2>
        </div>
        <div class="deliverable-grid">
          ${deliverableCard(
            "Final Report",
            "Technical write-up with figures and conclusions",
            "A static HTML report plus markdown version summarize the dataset, models, backtest assumptions, robustness interpretation, and vault architecture.",
            FINAL_REPORT_URL,
            "Read the report",
          )}
          ${deliverableCard(
            "Showcase Website",
            "This dashboard can be built as a static site",
            "The frontend now uses relative asset paths and build-time base URLs, so the same page works locally and under GitHub Pages-style subpaths.",
            REPOSITORY_URL,
            "View repository",
          )}
          ${deliverableCard(
            "Artifact Snapshot",
            "Inspectable JSON for reproducible demos",
            "The exported snapshot keeps the presentation layer honest by sourcing metrics, charts, and vault state directly from generated research artifacts.",
            DEMO_SNAPSHOT_URL,
            "Inspect snapshot",
          )}
        </div>
      </section>

      <section class="section-grid verdict-zone">
        <section class="story-panel">
          <div class="section-heading">
            <p class="section-kicker">Project Takeaways</p>
            <h2>Three things the showcase makes clear</h2>
          </div>
          <div class="verdict-grid">
            <article class="verdict-card">
              <p class="story-index">Signals</p>
              <h3>${formatNumber(baselineCorr, 3)} vs ${formatNumber(dlMetric, 3)}</h3>
              <p>${signalBody}</p>
            </article>
            <article class="verdict-card">
              <p class="story-index">Execution</p>
              <h3>${formatUsd(bestPnl)}</h3>
              <p>${executionBody}</p>
            </article>
            <article class="verdict-card">
              <p class="story-index">Architecture</p>
              <h3>Research -> Vault</h3>
              <p>${architectureBody}</p>
            </article>
          </div>
        </section>

        <aside class="research-panel">
          <div class="section-heading">
            <p class="section-kicker">Public Demo</p>
            <h2>Why this is a showcase page, not only a local frontend</h2>
          </div>
          <div class="story-points">
            <div class="story-point">The dashboard is now buildable into a static bundle with no backend dependency.</div>
            <div class="story-point">A generated HTML final report is copied into the public site so reviewers can open a formal write-up from the same deployment.</div>
            <div class="story-point">Relative asset paths keep the site compatible with GitHub Pages and similar subpath hosting.</div>
          </div>
        </aside>
      </section>

      <section class="section-grid">
        <section class="story-panel">
          <div class="section-heading">
            <p class="section-kicker">Project Overview</p>
            <h2>One prototype, four connected layers</h2>
          </div>
          <div class="story-list">
            ${snapshot.overview.layers
              .map(
                (layer) => `
                  <article class="story-card">
                    <p class="story-index">${layer.label}</p>
                    <p>${layer.detail}</p>
                  </article>
                `,
              )
              .join("")}
          </div>
          <div class="story-points">
            ${snapshot.overview.story_points
              .map((point) => `<div class="story-point">${point}</div>`)
              .join("")}
          </div>
        </section>

        <aside class="research-panel">
          <div class="section-heading">
            <p class="section-kicker">Research Snapshot</p>
            <h2>Why this market is interesting</h2>
          </div>
          <div class="research-grid">
            ${metricCard("Coverage", formatPercent(snapshot.research.coverage_ratio), "No missing hours inside the observed research range.")}
            ${metricCard("Funding Mean", formatBps(snapshot.research.funding_mean_bps), "Persistent positive funding dominates the sample.") }
            ${metricCard("Spread Mean", formatBps(snapshot.research.spread_mean_bps), "Basis is small on average but wide enough for short-lived dislocations.") }
            ${metricCard("Perp Vol", formatPercent(snapshot.research.annualized_volatility), "Average realized annualized volatility from hourly returns.") }
          </div>
        </aside>
      </section>

      <section class="section-block">
        <div class="section-heading">
          <p class="section-kicker">Key Charts</p>
          <h2>From funding regimes to backtest outcomes</h2>
        </div>
        <div class="chart-gallery">
          ${snapshot.charts
            .map(
              (chart) => `
                <figure class="chart-card">
                  <div class="chart-header">
                    <p class="chart-section">${chart.section}</p>
                    <h3>${chart.title}</h3>
                    <p>${chart.subtitle}</p>
                  </div>
                  <img src="${chart.image}" alt="${chart.title}" loading="lazy" />
                </figure>
              `,
            )
            .join("")}
        </div>
      </section>

      <section class="section-grid">
        <section class="results-panel">
          <div class="section-heading">
            <p class="section-kicker">Strategy Metrics</p>
            <h2>Benchmark the rule-based, ML, and deep-learning outputs</h2>
          </div>
          <div class="model-strip">
            <article class="model-card">
              <p class="model-type">Baseline Leader</p>
              <h3>${String(baselineBest["model_name"] ?? "n/a")}</h3>
              <p>Task: ${String(baselineBest["task"] ?? "n/a")}</p>
              <dl>
                <div><dt>Split</dt><dd>${String(baselineBest["split"] ?? "n/a")}</dd></div>
                <div><dt>Pearson</dt><dd>${formatNumber((baselineBest["pearson_corr"] as number | null | undefined) ?? null, 3)}</dd></div>
                <div><dt>RMSE</dt><dd>${formatNumber((baselineBest["rmse"] as number | null | undefined) ?? null, 3)}</dd></div>
              </dl>
            </article>
            <article class="model-card">
              <p class="model-type">${dlComparison.available ? "DL Zoo Winner" : "Deep Learning"}</p>
              <h3>${String(dlBest["run_label"] ?? dlBest["model_name"] ?? "n/a")}</h3>
              <p>${dlComparison.available ? `${formatNumber(dlComparison.run_count, 0)} model families compared` : "Single-model artifact"}</p>
              <dl>
                <div><dt>Metric</dt><dd>${String(dlBest["ranking_metric"] ?? "pearson_corr")}</dd></div>
                <div><dt>Score</dt><dd>${formatNumber(((dlBest["ranking_metric_value"] as number | null | undefined) ?? (dlBest["pearson_corr"] as number | null | undefined)) ?? null, 3)}</dd></div>
                <div><dt>Group</dt><dd>${String(dlBest["model_group"] ?? "sequence")}</dd></div>
              </dl>
            </article>
            <article class="model-card">
              <p class="model-type">Backtest Winner</p>
              <h3>${snapshot.backtest.best_strategy.strategy_name}</h3>
              <p>${snapshot.backtest.best_strategy.source_subtype}</p>
              <dl>
                <div><dt>PnL</dt><dd>${formatUsd(snapshot.backtest.best_strategy.total_net_pnl_usd)}</dd></div>
                <div><dt>MTM Sharpe</dt><dd>${formatNumber(snapshot.backtest.best_strategy.sharpe_ratio, 3)}</dd></div>
                <div><dt>MTM Drawdown</dt><dd>${formatPercent(snapshot.backtest.best_strategy.mark_to_market_max_drawdown ?? snapshot.backtest.best_strategy.max_drawdown)}</dd></div>
                <div><dt>Has Trades</dt><dd>${formatYesNo(snapshot.backtest.best_strategy.has_trades ?? snapshot.backtest.best_strategy.trade_count > 0)}</dd></div>
              </dl>
            </article>
          </div>
          <div class="table-shell">
            ${
              dlComparison.available && dlComparison.test_leaderboard.length > 0
                ? `
                  <div class="comparison-note">
                    ${dlComparison.best_model_note ?? "Deep-learning model zoo comparison is available."}
                  </div>
                  <table class="comparison-table">
                    <thead>
                      <tr>
                        <th>DL Rank</th>
                        <th>Model</th>
                        <th>Group</th>
                        <th>Metric</th>
                        <th>Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      ${dlComparison.test_leaderboard
                        .map(
                          (row) => `
                            <tr>
                              <td>${formatNumber(row["rank"] as number | null | undefined, 0)}</td>
                              <td>${String(row["run_label"] ?? row["model_name"] ?? "n/a")}</td>
                              <td>${String(row["model_group"] ?? "n/a")}</td>
                              <td>${String(row["ranking_metric"] ?? "n/a")}</td>
                              <td>${formatNumber(row["ranking_metric_value"] as number | null | undefined, 3)}</td>
                            </tr>
                          `,
                        )
                        .join("")}
                    </tbody>
                  </table>
                `
                : ""
            }
            <table>
              <thead>
                <tr>
                  <th>Strategy</th>
                  <th>Source</th>
                  <th>Split</th>
                  <th>Trades</th>
                  <th>Cum Return</th>
                  <th>MTM DD</th>
                  <th>MTM Sharpe</th>
                  <th>Net PnL</th>
                </tr>
              </thead>
              <tbody>
                ${snapshot.backtest.top_strategies
                  .map(
                    (row) => `
                      <tr>
                        <td>${row.strategy_name}</td>
                        <td>${row.source_subtype}</td>
                        <td>${row.evaluation_split ?? snapshot.backtest.risk_view?.primary_split ?? "n/a"}</td>
                        <td>${formatNumber(row.trade_count, 0)}</td>
                        <td>${formatPercent(row.cumulative_return)}</td>
                        <td>${formatPercent(row.mark_to_market_max_drawdown ?? row.max_drawdown)}</td>
                        <td>${formatNumber(row.sharpe_ratio, 3)}</td>
                        <td>${formatUsd(row.total_net_pnl_usd)}</td>
                      </tr>
                    `,
                  )
                  .join("")}
              </tbody>
            </table>
          </div>
        </section>

        <aside class="assumptions-panel">
          <div class="section-heading">
            <p class="section-kicker">Backtest Assumptions</p>
            <h2>Explicit and presentation-friendly</h2>
          </div>
          <div class="risk-note">
            <p><strong>Primary split:</strong> ${String(snapshot.backtest.risk_view?.primary_split ?? snapshot.backtest.summary["primary_split"] ?? "test")}</p>
            <p><strong>Risk view:</strong> mark-to-market equity for drawdown and Sharpe; realized-only equity is retained for audit.</p>
            <p><strong>Trades:</strong> ${formatNumber((snapshot.backtest.risk_view?.primary_trade_count as number | null | undefined) ?? (snapshot.backtest.summary["primary_trade_count"] as number | null | undefined), 0)} primary-split trades, ${formatNumber((snapshot.backtest.risk_view?.combined_trade_count as number | null | undefined) ?? (snapshot.backtest.summary["combined_trade_count"] as number | null | undefined), 0)} combined trades.</p>
          </div>
          <ul class="assumption-list">
            ${snapshot.backtest.assumptions
              .slice(0, 6)
              .map((item) => `<li>${item}</li>`)
              .join("")}
          </ul>
        </aside>
      </section>

      ${renderExploratoryShowcase()}

      <section class="section-grid vault-zone">
        <section class="vault-panel">
          <div class="section-heading">
            <p class="section-kicker">Vault Status</p>
            <h2>Mock operator state mirrored into the on-chain vault</h2>
          </div>
          <div class="vault-status-grid">
            ${metricCard("Strategy State", snapshot.vault.strategy_state.toUpperCase(), `Selected strategy: ${snapshot.vault.selected_strategy}`)}
            ${metricCard("Reported NAV", formatAssets(snapshot.vault.reported_nav_assets, assetSymbol, assetDecimals), `Summary PnL: ${formatUsd(snapshot.vault.summary_pnl_usd)}`)}
            ${metricCard("Suggested Direction", snapshot.vault.suggested_direction, `${snapshot.vault.call_count} prepared contract calls in the latest operator payload.`)}
            ${metricCard("Execution Mode", String(snapshot.vault.execution_summary["mode"] ?? "dry-run"), `Vault address ${formatAddress(snapshot.vault.vault_address)}`)}
          </div>
          <div class="vault-meta-grid">
            <div>
              <p class="meta-label">Chain</p>
              <p>${snapshot.vault.chain_name}</p>
            </div>
            <div>
              <p class="meta-label">Vault</p>
              <p>${formatAddress(snapshot.vault.vault_address)}</p>
            </div>
            <div>
              <p class="meta-label">Stablecoin</p>
              <p>${formatAddress(snapshot.vault.stablecoin_address)}</p>
            </div>
            <div>
              <p class="meta-label">Operator Mode</p>
              <p>${String(snapshot.vault.execution_summary["mode"] ?? "dry-run")}</p>
            </div>
          </div>
        </section>

        <section class="console-panel">
          <div class="section-heading">
            <p class="section-kicker">Local Demo Console</p>
            <h2>Deposit, update strategy state, apply NAV, and inspect accounting</h2>
          </div>
          <p class="console-note">
            This console is intentionally local and educational. It simulates the vault accounting flow using the exported snapshot rather than a live wallet connection.
          </p>
          <div class="console-grid">
            <article class="console-card">
              <p class="console-label">Wallet Cash</p>
              <h3>${formatAssets(state.walletCashAssets, assetSymbol, assetDecimals)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">Vault Cash</p>
              <h3>${formatAssets(state.vaultCashAssets, assetSymbol, assetDecimals)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">Reported NAV</p>
              <h3>${formatAssets(state.reportedNavAssets, assetSymbol, assetDecimals)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">User Shares</p>
              <h3>${formatAssets(state.userShares, "shares", assetDecimals)}</h3>
            </article>
          </div>
          <div class="console-status-row">
            <span class="status-chip accent">${state.strategyState.toUpperCase()}</span>
            <span class="status-chip">${state.selectedStrategy}</span>
            <span class="status-chip">${state.suggestedDirection}</span>
          </div>
          <div class="control-panel">
            <label class="control-label" for="amount-input">Demo amount (${assetSymbol})</label>
            <div class="control-row">
              <input id="amount-input" class="amount-input" type="number" min="0" step="100" value="${amountInputValue}" />
              <button class="action-button" data-action="deposit">Deposit</button>
              <button class="action-button ghost" data-action="withdraw">Withdraw</button>
            </div>
            <div class="control-row secondary">
              <button class="action-button" data-action="toggle-strategy">Strategy Update</button>
              <button class="action-button" data-action="apply-operator">NAV/PnL Update</button>
              <button class="action-button ghost" data-action="reset">Reset</button>
            </div>
          </div>
          <div class="console-footnote">
            <p>Demo operator plan: ${snapshot.simulation.operator_plan.selected_strategy}, target NAV ${formatAssets(snapshot.simulation.operator_plan.reported_nav_assets, assetSymbol, assetDecimals)}, summary PnL ${formatUsd(snapshot.simulation.operator_plan.summary_pnl_usd)}.</p>
          </div>
        </section>
      </section>

      <section class="section-block log-panel">
        <div class="section-heading">
          <p class="section-kicker">Example Activity Log</p>
          <h2>Project milestones plus live demo interactions</h2>
        </div>
        <div class="log-grid">
          ${visibleEvents
            .map(
              (item) => `
                <article class="log-entry log-${item.kind}">
                  <p class="log-time">${formatDateTime(item.timestamp)}</p>
                  <h3>${item.title}</h3>
                  <p>${item.detail}</p>
                </article>
              `,
            )
            .join("")}
        </div>
      </section>
    </div>
  `;

  const amountInput = document.querySelector<HTMLInputElement>("#amount-input");
  if (amountInput) {
    amountInput.addEventListener("input", () => {
      if (!simulationState) {
        return;
      }
      const parsed = Number(amountInput.value);
      simulationState = {
        ...simulationState,
        amountInputAssets: Number.isFinite(parsed) ? Math.max(parsed, 0) * 10 ** assetDecimals : 0,
      };
    });
  }

  document.querySelectorAll<HTMLButtonElement>("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!snapshotState || !simulationState) {
        return;
      }
      const action = button.dataset.action;
      const amountAssets = Math.round(simulationState.amountInputAssets);
      const now = new Date().toISOString();

      if (action === "deposit") {
        if (amountAssets <= 0 || amountAssets > simulationState.walletCashAssets) {
          simulationState = pushActivity(simulationState, {
            timestamp: now,
            kind: "demo",
            title: "Deposit rejected",
            detail: "Choose a positive amount that is smaller than the demo wallet cash balance.",
          });
          renderDashboard(snapshotState, simulationState);
          return;
        }
        const mintedShares = convertToShares(simulationState, amountAssets);
        simulationState = pushActivity(
          {
            ...simulationState,
            walletCashAssets: simulationState.walletCashAssets - amountAssets,
            vaultCashAssets: simulationState.vaultCashAssets + amountAssets,
            reportedNavAssets: simulationState.reportedNavAssets + amountAssets,
            totalShares: simulationState.totalShares + mintedShares,
            userShares: simulationState.userShares + mintedShares,
          },
          {
            timestamp: now,
            kind: "vault",
            title: "Deposit simulated",
            detail: `Deposited ${formatAssets(amountAssets, assetSymbol, assetDecimals)} and minted ${formatAssets(mintedShares, "shares", assetDecimals)}.`,
          },
        );
      }

      if (action === "withdraw") {
        const burnedShares = previewWithdrawShares(simulationState, amountAssets);
        if (
          amountAssets <= 0 ||
          burnedShares > simulationState.userShares ||
          amountAssets > simulationState.vaultCashAssets
        ) {
          simulationState = pushActivity(simulationState, {
            timestamp: now,
            kind: "demo",
            title: "Withdraw rejected",
            detail: "The demo user needs enough shares and the vault needs enough cash for the requested withdrawal.",
          });
          renderDashboard(snapshotState, simulationState);
          return;
        }
        simulationState = pushActivity(
          {
            ...simulationState,
            walletCashAssets: simulationState.walletCashAssets + amountAssets,
            vaultCashAssets: simulationState.vaultCashAssets - amountAssets,
            reportedNavAssets: Math.max(simulationState.reportedNavAssets - amountAssets, 0),
            totalShares: Math.max(simulationState.totalShares - burnedShares, 0),
            userShares: Math.max(simulationState.userShares - burnedShares, 0),
          },
          {
            timestamp: now,
            kind: "vault",
            title: "Withdrawal simulated",
            detail: `Withdrew ${formatAssets(amountAssets, assetSymbol, assetDecimals)} and burned ${formatAssets(burnedShares, "shares", assetDecimals)}.`,
          },
        );
      }

      if (action === "toggle-strategy") {
        const active = simulationState.strategyState !== "active";
        simulationState = pushActivity(
          {
            ...simulationState,
            strategyState: active
              ? snapshotState.simulation.operator_plan.demo_activation_state
              : "idle",
            suggestedDirection: active
              ? snapshotState.simulation.operator_plan.demo_activation_direction
              : "flat",
            selectedStrategy: snapshotState.simulation.operator_plan.selected_strategy,
          },
          {
            timestamp: now,
            kind: "strategy",
            title: active ? "Strategy activated" : "Strategy idled",
            detail: active
              ? `Demo operator switched the strategy to ${snapshotState.simulation.operator_plan.demo_activation_direction}.`
              : "The local demo returned the strategy to an idle flat state.",
          },
        );
      }

      if (action === "apply-operator") {
        const targetNav = snapshotState.simulation.operator_plan.reported_nav_assets;
        const pnlDelta = snapshotState.simulation.operator_plan.summary_pnl_assets;
        const cashDelta = targetNav - simulationState.reportedNavAssets;
        simulationState = pushActivity(
          {
            ...simulationState,
            reportedNavAssets: targetNav,
            vaultCashAssets: Math.max(simulationState.vaultCashAssets + cashDelta, 0),
            selectedStrategy: snapshotState.simulation.operator_plan.selected_strategy,
          },
          {
            timestamp: now,
            kind: "vault",
            title: "Operator NAV update applied",
            detail: `Reported NAV moved to ${formatAssets(targetNav, assetSymbol, assetDecimals)} with a demo PnL delta of ${formatUsd(snapshotState.simulation.operator_plan.summary_pnl_usd)}.`,
          },
        );
        if (pnlDelta !== 0) {
          simulationState = pushActivity(simulationState, {
            timestamp: now,
            kind: "demo",
            title: "Mock liquidity synchronized",
            detail: "For demo readability, vault cash was mirrored with the operator NAV update so withdrawals remain intuitive.",
          });
        }
      }

      if (action === "reset") {
        simulationState = createInitialSimulationState(snapshotState);
      }

      renderDashboard(snapshotState, simulationState);
    });
  });
}

async function loadOptionalJson<T>(url: string): Promise<T | null> {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as T;
  } catch {
    return null;
  }
}

async function loadDashboard(): Promise<void> {
  renderLoading();
  try {
    const response = await fetch(DEMO_SNAPSHOT_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    snapshotState = (await response.json()) as DemoSnapshot;
    const [exploratorySummary, exploratoryLeaderboard, exploratoryPredictionDistribution, exploratoryQuantileAnalysis] =
      await Promise.all([
        loadOptionalJson<ExploratorySummaryPayload>(EXPLORATORY_SUMMARY_URL),
        loadOptionalJson<Array<Record<string, unknown>>>(EXPLORATORY_LEADERBOARD_URL),
        loadOptionalJson<Array<Record<string, unknown>>>(EXPLORATORY_PREDICTION_DISTRIBUTION_URL),
        loadOptionalJson<Array<Record<string, unknown>>>(EXPLORATORY_QUANTILE_ANALYSIS_URL),
      ]);
    exploratoryArtifactsState = {
      summary: exploratorySummary,
      leaderboard: exploratoryLeaderboard ?? [],
      predictionDistribution: exploratoryPredictionDistribution ?? [],
      quantileAnalysis: exploratoryQuantileAnalysis ?? [],
    };
    simulationState = createInitialSimulationState(snapshotState);
    renderDashboard(snapshotState, simulationState);
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unknown error while loading the demo snapshot.";
    renderError(message);
  }
}

void loadDashboard();
