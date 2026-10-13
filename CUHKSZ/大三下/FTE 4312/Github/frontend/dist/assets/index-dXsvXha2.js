(function(){const a=document.createElement("link").relList;if(a&&a.supports&&a.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))c(r);new MutationObserver(r=>{for(const l of r)if(l.type==="childList")for(const d of l.addedNodes)d.tagName==="LINK"&&d.rel==="modulepreload"&&c(d)}).observe(document,{childList:!0,subtree:!0});function i(r){const l={};return r.integrity&&(l.integrity=r.integrity),r.referrerPolicy&&(l.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?l.credentials="include":r.crossOrigin==="anonymous"?l.credentials="omit":l.credentials="same-origin",l}function c(r){if(r.ep)return;r.ep=!0;const l=i(r);fetch(r.href,l)}})();const V=document.querySelector("#app");if(!V)throw new Error("App root not found.");const P=V,$="/",z=new URLSearchParams(window.location.search),_=z.get("mode")||"demo",W=z.get("reportMode")||(_==="demo"?"report":`${_}/report`),D=`${$}${_}/demo_snapshot.json`,et=`${$}${_}/exploratory_dl_summary.json`,at=`${$}${_}/exploratory_dl_leaderboard.json`,st=`${$}${_}/exploratory_prediction_distribution.json`,rt=`${$}${_}/exploratory_quantile_analysis.json`,j=`${$}${W}/`,it=`${$}${W}/final_report.md`,F="https://github.com/MengerWen/Deep-Learning-Based-Delta-Neutral-Statistical-Arbitrage-on-Perpetual-Funding-Rates",nt=2500*10**6;let u=null,e=null,v=null;function q(t){return new Intl.DateTimeFormat("en-US",{year:"numeric",month:"short",day:"2-digit"}).format(new Date(t))}function ot(t){return new Intl.DateTimeFormat("en-US",{year:"numeric",month:"short",day:"2-digit",hour:"2-digit",minute:"2-digit"}).format(new Date(t))}function y(t){return t==null||Number.isNaN(t)?"n/a":new Intl.NumberFormat("en-US",{style:"currency",currency:"USD",maximumFractionDigits:2}).format(t)}function o(t,a=2){return t==null||Number.isNaN(t)?"n/a":new Intl.NumberFormat("en-US",{maximumFractionDigits:a}).format(t)}function k(t){return t==null||Number.isNaN(t)?"n/a":`${(t*100).toFixed(2)}%`}function T(t){return t==null||Number.isNaN(t)?"n/a":`${t.toFixed(2)} bps`}function g(t,a,i,c=2){return`${o(t/10**i,c)} ${a}`}function R(t){return!t||t.length<12?t||"n/a":`${t.slice(0,6)}...${t.slice(-4)}`}function lt(t){return t==null?"n/a":t?"yes":"no"}function H(t){return{walletCashAssets:t.simulation.wallet_cash_assets,vaultCashAssets:t.simulation.base_vault_cash_assets,reportedNavAssets:t.simulation.base_reported_nav_assets,totalShares:t.simulation.base_total_shares,userShares:t.simulation.user_shares,strategyState:t.simulation.strategy_state,suggestedDirection:"flat",selectedStrategy:t.vault.selected_strategy,amountInputAssets:nt,activityLog:[...t.activity_log].reverse()}}function ct(t,a){return t.totalShares===0||t.reportedNavAssets===0?a:Math.floor(a*t.totalShares/t.reportedNavAssets)}function dt(t,a){return t.totalShares===0||t.reportedNavAssets===0?a:Math.ceil(a*t.totalShares/t.reportedNavAssets)}function S(t,a){return{...t,activityLog:[a,...t.activityLog].slice(0,12)}}function pt(){P.innerHTML=`
    <main class="loading-shell">
      <div class="loading-panel">
        <p class="eyebrow">Preparing Demo</p>
        <h1>Loading the funding-rate arbitrage dashboard...</h1>
        <p>Reading the local demo snapshot and chart assets.</p>
      </div>
    </main>
  `}function ut(t){const a=_==="demo_showcase"?"& 'd:\\MG\\anaconda3\\python.exe' -m src.main build-demo-showcase --config configs/demo/showcase.yaml":"& 'd:\\MG\\anaconda3\\python.exe' -m src.main run-demo --config configs/demo/workflow.yaml",i=_==="demo_showcase"?"& 'd:\\MG\\anaconda3\\python.exe' scripts/demo/build_demo_showcase.py --config configs/demo/showcase.yaml":"& 'd:\\MG\\anaconda3\\python.exe' scripts/demo/export_demo_snapshot.py --config configs/demo/default.yaml";P.innerHTML=`
    <main class="loading-shell">
      <div class="loading-panel error-panel">
        <p class="eyebrow">Snapshot Missing</p>
        <h1>The frontend could not load <code>${D}</code>.</h1>
        <p>${t}</p>
        <pre class="command-block">${a}</pre>
        <pre class="command-block"># or only refresh the frontend snapshot
${i}</pre>
        <pre class="command-block">cd frontend
npm install
npm run dev</pre>
      </div>
    </main>
  `}function m(t,a,i){return`
    <article class="metric-card">
      <p class="metric-label">${t}</p>
      <h3>${a}</h3>
      <p class="metric-note">${i}</p>
    </article>
  `}function M(t,a,i,c,r){return`
    <article class="deliverable-card">
      <p class="deliverable-label">${t}</p>
      <h3>${a}</h3>
      <p>${i}</p>
      <a class="action-link" href="${c}" target="_blank" rel="noreferrer">${r}</a>
    </article>
  `}function mt(){var A;if(!((A=v==null?void 0:v.summary)!=null&&A.exploratory_summary))return"";const t=v.summary.exploratory_summary,a=v.summary.disclaimer??"Exploratory results are supplementary showcase artifacts.",i=t.best_showcase_row??{},c=t.figure_assets??[],r=v.predictionDistribution.slice(0,4),l=v.quantileAnalysis.slice(0,8),d=v.leaderboard.slice(0,6);return`
    <section class="section-block exploratory-panel">
      <div class="section-heading">
        <p class="section-kicker">Exploratory DL Showcase</p>
        <h2>Supplementary DL results that stay separate from the strict conclusion</h2>
        <p class="panel-text">${a}</p>
      </div>
      <div class="exploratory-metrics">
        ${m("Showcase Strategies",o(t.strategy_count??null,0),"Independent exploratory DL strategy variants generated from ranking-based and support-aware signal rules.")}
        ${m("Nonzero Trade Runs",o(t.nonzero_trade_strategy_count??null,0),"Only the exploratory track is counted here; strict primary outputs remain unchanged.")}
        ${m("Best Showcase PnL",y(i.total_net_pnl_usd??null),`${String(i.strategy_name??"n/a")} on ${String(i.evaluation_split??"n/a")} split.`)}
        ${m("Best Showcase Trades",o(i.trade_count??null,0),`${String(i.model_name??"n/a")} with ${String(i.target_type??"n/a")} and ${String(i.signal_rule??"n/a")}.`)}
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
            ${d.map(n=>`
                  <tr>
                    <td>${String(n.strategy_name??"n/a")}</td>
                    <td>${String(n.model_name??"n/a")}</td>
                    <td>${String(n.target_type??"n/a")}</td>
                    <td>${String(n.signal_rule??"n/a")}</td>
                    <td>${String(n.evaluation_split??"n/a")}</td>
                    <td>${o(n.trade_count,0)}</td>
                    <td>${k(n.cumulative_return)}</td>
                    <td>${y(n.total_net_pnl_usd)}</td>
                    <td>${String(n.status??"n/a")}</td>
                  </tr>
                `).join("")}
          </tbody>
        </table>
      </div>
      ${c.length>0?`
            <div class="chart-gallery exploratory-gallery">
              ${c.map(n=>`
                    <figure class="chart-card">
                      <div class="chart-header">
                        <p class="chart-section">exploratory</p>
                        <h3>${n.label.split("_").join(" ")}</h3>
                        <p>Generated from the best exploratory showcase run and copied into the frontend-ready demo bundle.</p>
                      </div>
                      <img src="${$}${n.image}" alt="${n.label}" loading="lazy" />
                    </figure>
                  `).join("")}
            </div>
          `:""}
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
                ${r.map(n=>`
                      <tr>
                        <td>${String(n.run_name??"n/a")}</td>
                        <td>${String(n.split??"n/a")}</td>
                        <td>${o(n.signed_score_mean,3)}</td>
                        <td>${o(n.signed_score_std,3)}</td>
                        <td>${k(n.predicted_short_rate)}</td>
                      </tr>
                    `).join("")}
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
                ${l.map(n=>`
                      <tr>
                        <td>${String(n.run_name??"n/a")}</td>
                        <td>${String(n.absolute_score_quantile??"n/a")}</td>
                        <td>${o(n.row_count,0)}</td>
                        <td>${T(n.avg_directional_return_bps)}</td>
                        <td>${T(n.cumulative_directional_return_bps)}</td>
                      </tr>
                    `).join("")}
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  `}function x(t,a){var C,U,I,B;const i=t.models.baseline_best,c=t.models.deep_learning_best,r=t.models.deep_learning_comparison,l=t.simulation.asset_symbol,d=t.simulation.asset_decimals,A=a.activityLog.slice(0,8),n=(a.amountInputAssets/10**d).toFixed(2),O=i.pearson_corr??null,E=c.ranking_metric_value??c.pearson_corr??null,w=t.backtest.best_strategy.total_net_pnl_usd,G=w>0?"Current artifact set preserves positive post-cost alpha.":"Predictive structure exists, but the current post-cost strategy set is still unprofitable.",Y=w>0?"That makes the showcase a clean demonstration of both research quality and strategy viability under the configured assumptions.":"That is still a strong course-project outcome because the repository makes the negative result explicit instead of hiding it behind pre-cost metrics.",K=w>0?"The synthetic strict bundle keeps the baseline modestly positive, then improves the payoff profile across the DL variants without removing path volatility.":"Both simple and sequence models learn some structure in the 24-hour post-cost target, which is why the modeling layer remains meaningful even when trading outcomes stay weak.",X=w>0?"The best strict strategy remains profitable after fees, slippage, and funding, but it still experiences visible drawdowns and non-monotonic recovery.":"The current best test-period strategy is still negative after fees, slippage, gas, and next-bar execution, so the repository is explicit about the true cost of monetizing funding dislocations.",J=t.meta.artifact_label?"The same frontend and report contract can now render either real artifacts or the isolated DEMO ONLY showcase bundle without changing the default pipeline.":"The project already demonstrates an end-to-end hybrid design: off-chain modeling, transparent backtests, dry-run operator sync, and on-chain share-accounting logic in one coherent story.",Q=t.meta.artifact_label?`
      <div class="demo-flag">
        <strong>${t.meta.artifact_label}</strong>
        <span>${t.meta.artifact_note??"Illustrative showcase bundle"}</span>
      </div>
    `:"";P.innerHTML=`
    <div class="page-shell">
      <div class="page-noise"></div>
      <header class="masthead" id="top">
        <div class="masthead-copy">
          ${Q}
          <p class="eyebrow">Signal To Settlement</p>
          <h1>${t.meta.title}</h1>
          <p class="lede">${t.meta.subtitle}</p>
          <div class="hero-meta">
            <span>${t.meta.symbol} on ${t.meta.venue}</span>
            <span>${q(t.meta.date_range.start)} to ${q(t.meta.date_range.end_exclusive)}</span>
            <span>${t.meta.frequency} research frequency</span>
            <span>${t.meta.chain_name}</span>
          </div>
          <div class="cta-row">
            <a class="action-link" href="${j}" target="_blank" rel="noreferrer">Open final report</a>
            <a class="action-link ghost-link" href="${it}" target="_blank" rel="noreferrer">Markdown version</a>
            <a class="action-link ghost-link" href="${D}" target="_blank" rel="noreferrer">Snapshot JSON</a>
            <a class="action-link ghost-link" href="${F}" target="_blank" rel="noreferrer">GitHub repo</a>
          </div>
        </div>
        <aside class="masthead-panel">
          <p class="panel-label">Showcase Verdict</p>
          <p class="panel-text"><strong>${G}</strong></p>
          <p class="panel-text">${Y}</p>
          <div class="status-chip-row">
            <span class="status-chip">Research pipeline ready</span>
            <span class="status-chip">Backtest artifacts loaded</span>
            <span class="status-chip">Vault prototype synced locally</span>
            <span class="status-chip">Static-pages ready</span>
          </div>
        </aside>
      </header>

      <section class="metrics-ribbon">
        ${m("Data Window",`${o(t.research.canonical_rows,0)} hours`,`${o(t.research.funding_events,0)} funding events across the primary BTCUSDT sample.`)}
        ${m("Best Backtest",y(t.backtest.best_strategy.total_net_pnl_usd),`${t.backtest.best_strategy.strategy_name} on ${t.backtest.best_strategy.evaluation_split??((C=t.backtest.risk_view)==null?void 0:C.primary_split)??"primary"} split with ${o(t.backtest.best_strategy.trade_count,0)} trades.`)}
        ${m("Baseline Benchmark",o(O,3),`${String(i.model_name??"baseline")} test-set correlation.`)}
        ${m("DL Zoo Leader",o(E,3),`${String(c.model_name??c.run_label??"deep learning")} under the configured comparison metric.`)}
      </section>

      <section class="section-block delivery-panel">
        <div class="section-heading">
          <p class="section-kicker">Submission Kit</p>
          <h2>Two deliverables now sit on top of the same artifact pipeline</h2>
        </div>
        <div class="deliverable-grid">
          ${M("Final Report","Technical write-up with figures and conclusions","A static HTML report plus markdown version summarize the dataset, models, backtest assumptions, robustness interpretation, and vault architecture.",j,"Read the report")}
          ${M("Showcase Website","This dashboard can be built as a static site","The frontend now uses relative asset paths and build-time base URLs, so the same page works locally and under GitHub Pages-style subpaths.",F,"View repository")}
          ${M("Artifact Snapshot","Inspectable JSON for reproducible demos","The exported snapshot keeps the presentation layer honest by sourcing metrics, charts, and vault state directly from generated research artifacts.",D,"Inspect snapshot")}
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
              <h3>${o(O,3)} vs ${o(E,3)}</h3>
              <p>${K}</p>
            </article>
            <article class="verdict-card">
              <p class="story-index">Execution</p>
              <h3>${y(w)}</h3>
              <p>${X}</p>
            </article>
            <article class="verdict-card">
              <p class="story-index">Architecture</p>
              <h3>Research -> Vault</h3>
              <p>${J}</p>
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
            ${t.overview.layers.map(s=>`
                  <article class="story-card">
                    <p class="story-index">${s.label}</p>
                    <p>${s.detail}</p>
                  </article>
                `).join("")}
          </div>
          <div class="story-points">
            ${t.overview.story_points.map(s=>`<div class="story-point">${s}</div>`).join("")}
          </div>
        </section>

        <aside class="research-panel">
          <div class="section-heading">
            <p class="section-kicker">Research Snapshot</p>
            <h2>Why this market is interesting</h2>
          </div>
          <div class="research-grid">
            ${m("Coverage",k(t.research.coverage_ratio),"No missing hours inside the observed research range.")}
            ${m("Funding Mean",T(t.research.funding_mean_bps),"Persistent positive funding dominates the sample.")}
            ${m("Spread Mean",T(t.research.spread_mean_bps),"Basis is small on average but wide enough for short-lived dislocations.")}
            ${m("Perp Vol",k(t.research.annualized_volatility),"Average realized annualized volatility from hourly returns.")}
          </div>
        </aside>
      </section>

      <section class="section-block">
        <div class="section-heading">
          <p class="section-kicker">Key Charts</p>
          <h2>From funding regimes to backtest outcomes</h2>
        </div>
        <div class="chart-gallery">
          ${t.charts.map(s=>`
                <figure class="chart-card">
                  <div class="chart-header">
                    <p class="chart-section">${s.section}</p>
                    <h3>${s.title}</h3>
                    <p>${s.subtitle}</p>
                  </div>
                  <img src="${s.image}" alt="${s.title}" loading="lazy" />
                </figure>
              `).join("")}
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
              <h3>${String(i.model_name??"n/a")}</h3>
              <p>Task: ${String(i.task??"n/a")}</p>
              <dl>
                <div><dt>Split</dt><dd>${String(i.split??"n/a")}</dd></div>
                <div><dt>Pearson</dt><dd>${o(i.pearson_corr??null,3)}</dd></div>
                <div><dt>RMSE</dt><dd>${o(i.rmse??null,3)}</dd></div>
              </dl>
            </article>
            <article class="model-card">
              <p class="model-type">${r.available?"DL Zoo Winner":"Deep Learning"}</p>
              <h3>${String(c.run_label??c.model_name??"n/a")}</h3>
              <p>${r.available?`${o(r.run_count,0)} model families compared`:"Single-model artifact"}</p>
              <dl>
                <div><dt>Metric</dt><dd>${String(c.ranking_metric??"pearson_corr")}</dd></div>
                <div><dt>Score</dt><dd>${o(c.ranking_metric_value??c.pearson_corr??null,3)}</dd></div>
                <div><dt>Group</dt><dd>${String(c.model_group??"sequence")}</dd></div>
              </dl>
            </article>
            <article class="model-card">
              <p class="model-type">Backtest Winner</p>
              <h3>${t.backtest.best_strategy.strategy_name}</h3>
              <p>${t.backtest.best_strategy.source_subtype}</p>
              <dl>
                <div><dt>PnL</dt><dd>${y(t.backtest.best_strategy.total_net_pnl_usd)}</dd></div>
                <div><dt>MTM Sharpe</dt><dd>${o(t.backtest.best_strategy.sharpe_ratio,3)}</dd></div>
                <div><dt>MTM Drawdown</dt><dd>${k(t.backtest.best_strategy.mark_to_market_max_drawdown??t.backtest.best_strategy.max_drawdown)}</dd></div>
                <div><dt>Has Trades</dt><dd>${lt(t.backtest.best_strategy.has_trades??t.backtest.best_strategy.trade_count>0)}</dd></div>
              </dl>
            </article>
          </div>
          <div class="table-shell">
            ${r.available&&r.test_leaderboard.length>0?`
                  <div class="comparison-note">
                    ${r.best_model_note??"Deep-learning model zoo comparison is available."}
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
                      ${r.test_leaderboard.map(s=>`
                            <tr>
                              <td>${o(s.rank,0)}</td>
                              <td>${String(s.run_label??s.model_name??"n/a")}</td>
                              <td>${String(s.model_group??"n/a")}</td>
                              <td>${String(s.ranking_metric??"n/a")}</td>
                              <td>${o(s.ranking_metric_value,3)}</td>
                            </tr>
                          `).join("")}
                    </tbody>
                  </table>
                `:""}
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
                ${t.backtest.top_strategies.map(s=>{var b;return`
                      <tr>
                        <td>${s.strategy_name}</td>
                        <td>${s.source_subtype}</td>
                        <td>${s.evaluation_split??((b=t.backtest.risk_view)==null?void 0:b.primary_split)??"n/a"}</td>
                        <td>${o(s.trade_count,0)}</td>
                        <td>${k(s.cumulative_return)}</td>
                        <td>${k(s.mark_to_market_max_drawdown??s.max_drawdown)}</td>
                        <td>${o(s.sharpe_ratio,3)}</td>
                        <td>${y(s.total_net_pnl_usd)}</td>
                      </tr>
                    `}).join("")}
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
            <p><strong>Primary split:</strong> ${String(((U=t.backtest.risk_view)==null?void 0:U.primary_split)??t.backtest.summary.primary_split??"test")}</p>
            <p><strong>Risk view:</strong> mark-to-market equity for drawdown and Sharpe; realized-only equity is retained for audit.</p>
            <p><strong>Trades:</strong> ${o(((I=t.backtest.risk_view)==null?void 0:I.primary_trade_count)??t.backtest.summary.primary_trade_count,0)} primary-split trades, ${o(((B=t.backtest.risk_view)==null?void 0:B.combined_trade_count)??t.backtest.summary.combined_trade_count,0)} combined trades.</p>
          </div>
          <ul class="assumption-list">
            ${t.backtest.assumptions.slice(0,6).map(s=>`<li>${s}</li>`).join("")}
          </ul>
        </aside>
      </section>

      ${mt()}

      <section class="section-grid vault-zone">
        <section class="vault-panel">
          <div class="section-heading">
            <p class="section-kicker">Vault Status</p>
            <h2>Mock operator state mirrored into the on-chain vault</h2>
          </div>
          <div class="vault-status-grid">
            ${m("Strategy State",t.vault.strategy_state.toUpperCase(),`Selected strategy: ${t.vault.selected_strategy}`)}
            ${m("Reported NAV",g(t.vault.reported_nav_assets,l,d),`Summary PnL: ${y(t.vault.summary_pnl_usd)}`)}
            ${m("Suggested Direction",t.vault.suggested_direction,`${t.vault.call_count} prepared contract calls in the latest operator payload.`)}
            ${m("Execution Mode",String(t.vault.execution_summary.mode??"dry-run"),`Vault address ${R(t.vault.vault_address)}`)}
          </div>
          <div class="vault-meta-grid">
            <div>
              <p class="meta-label">Chain</p>
              <p>${t.vault.chain_name}</p>
            </div>
            <div>
              <p class="meta-label">Vault</p>
              <p>${R(t.vault.vault_address)}</p>
            </div>
            <div>
              <p class="meta-label">Stablecoin</p>
              <p>${R(t.vault.stablecoin_address)}</p>
            </div>
            <div>
              <p class="meta-label">Operator Mode</p>
              <p>${String(t.vault.execution_summary.mode??"dry-run")}</p>
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
              <h3>${g(a.walletCashAssets,l,d)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">Vault Cash</p>
              <h3>${g(a.vaultCashAssets,l,d)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">Reported NAV</p>
              <h3>${g(a.reportedNavAssets,l,d)}</h3>
            </article>
            <article class="console-card">
              <p class="console-label">User Shares</p>
              <h3>${g(a.userShares,"shares",d)}</h3>
            </article>
          </div>
          <div class="console-status-row">
            <span class="status-chip accent">${a.strategyState.toUpperCase()}</span>
            <span class="status-chip">${a.selectedStrategy}</span>
            <span class="status-chip">${a.suggestedDirection}</span>
          </div>
          <div class="control-panel">
            <label class="control-label" for="amount-input">Demo amount (${l})</label>
            <div class="control-row">
              <input id="amount-input" class="amount-input" type="number" min="0" step="100" value="${n}" />
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
            <p>Demo operator plan: ${t.simulation.operator_plan.selected_strategy}, target NAV ${g(t.simulation.operator_plan.reported_nav_assets,l,d)}, summary PnL ${y(t.simulation.operator_plan.summary_pnl_usd)}.</p>
          </div>
        </section>
      </section>

      <section class="section-block log-panel">
        <div class="section-heading">
          <p class="section-kicker">Example Activity Log</p>
          <h2>Project milestones plus live demo interactions</h2>
        </div>
        <div class="log-grid">
          ${A.map(s=>`
                <article class="log-entry log-${s.kind}">
                  <p class="log-time">${ot(s.timestamp)}</p>
                  <h3>${s.title}</h3>
                  <p>${s.detail}</p>
                </article>
              `).join("")}
        </div>
      </section>
    </div>
  `;const N=document.querySelector("#amount-input");N&&N.addEventListener("input",()=>{if(!e)return;const s=Number(N.value);e={...e,amountInputAssets:Number.isFinite(s)?Math.max(s,0)*10**d:0}}),document.querySelectorAll("[data-action]").forEach(s=>{s.addEventListener("click",()=>{if(!u||!e)return;const b=s.dataset.action,h=Math.round(e.amountInputAssets),f=new Date().toISOString();if(b==="deposit"){if(h<=0||h>e.walletCashAssets){e=S(e,{timestamp:f,kind:"demo",title:"Deposit rejected",detail:"Choose a positive amount that is smaller than the demo wallet cash balance."}),x(u,e);return}const p=ct(e,h);e=S({...e,walletCashAssets:e.walletCashAssets-h,vaultCashAssets:e.vaultCashAssets+h,reportedNavAssets:e.reportedNavAssets+h,totalShares:e.totalShares+p,userShares:e.userShares+p},{timestamp:f,kind:"vault",title:"Deposit simulated",detail:`Deposited ${g(h,l,d)} and minted ${g(p,"shares",d)}.`})}if(b==="withdraw"){const p=dt(e,h);if(h<=0||p>e.userShares||h>e.vaultCashAssets){e=S(e,{timestamp:f,kind:"demo",title:"Withdraw rejected",detail:"The demo user needs enough shares and the vault needs enough cash for the requested withdrawal."}),x(u,e);return}e=S({...e,walletCashAssets:e.walletCashAssets+h,vaultCashAssets:e.vaultCashAssets-h,reportedNavAssets:Math.max(e.reportedNavAssets-h,0),totalShares:Math.max(e.totalShares-p,0),userShares:Math.max(e.userShares-p,0)},{timestamp:f,kind:"vault",title:"Withdrawal simulated",detail:`Withdrew ${g(h,l,d)} and burned ${g(p,"shares",d)}.`})}if(b==="toggle-strategy"){const p=e.strategyState!=="active";e=S({...e,strategyState:p?u.simulation.operator_plan.demo_activation_state:"idle",suggestedDirection:p?u.simulation.operator_plan.demo_activation_direction:"flat",selectedStrategy:u.simulation.operator_plan.selected_strategy},{timestamp:f,kind:"strategy",title:p?"Strategy activated":"Strategy idled",detail:p?`Demo operator switched the strategy to ${u.simulation.operator_plan.demo_activation_direction}.`:"The local demo returned the strategy to an idle flat state."})}if(b==="apply-operator"){const p=u.simulation.operator_plan.reported_nav_assets,Z=u.simulation.operator_plan.summary_pnl_assets,tt=p-e.reportedNavAssets;e=S({...e,reportedNavAssets:p,vaultCashAssets:Math.max(e.vaultCashAssets+tt,0),selectedStrategy:u.simulation.operator_plan.selected_strategy},{timestamp:f,kind:"vault",title:"Operator NAV update applied",detail:`Reported NAV moved to ${g(p,l,d)} with a demo PnL delta of ${y(u.simulation.operator_plan.summary_pnl_usd)}.`}),Z!==0&&(e=S(e,{timestamp:f,kind:"demo",title:"Mock liquidity synchronized",detail:"For demo readability, vault cash was mirrored with the operator NAV update so withdrawals remain intuitive."}))}b==="reset"&&(e=H(u)),x(u,e)})})}async function L(t){try{const a=await fetch(t,{cache:"no-store"});return a.ok?await a.json():null}catch{return null}}async function ht(){pt();try{const t=await fetch(D,{cache:"no-store"});if(!t.ok)throw new Error(`HTTP ${t.status}`);u=await t.json();const[a,i,c,r]=await Promise.all([L(et),L(at),L(st),L(rt)]);v={summary:a,leaderboard:i??[],predictionDistribution:c??[],quantileAnalysis:r??[]},e=H(u),x(u,e)}catch(t){const a=t instanceof Error?t.message:"Unknown error while loading the demo snapshot.";ut(a)}}ht();
