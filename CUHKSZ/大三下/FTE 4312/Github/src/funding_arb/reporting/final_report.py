"""Generate a project-level final report from the latest demo artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from funding_arb.config.models import FinalReportSettings
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class FinalReportArtifacts:
    """Files produced by the final-report command."""

    artifact_output_dir: str
    markdown_report_path: str | None
    html_report_path: str | None
    summary_json_path: str | None
    public_report_dir: str | None


def describe_final_report_job(config: FinalReportSettings | dict[str, Any]) -> str:
    """Return a short human-readable description of the job."""
    settings = (
        config
        if isinstance(config, FinalReportSettings)
        else FinalReportSettings.model_validate(config)
    )
    return (
        f"Final report generation ready for {settings.metadata.symbol} on "
        f"{settings.metadata.provider} at {settings.metadata.frequency}, "
        f"reading {settings.input.demo_snapshot_path}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _load_json(path_text: str | Path) -> dict[str, Any]:
    return json.loads(_resolve_path(path_text).read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_int(value: Any) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{int(round(number)):,}"


def _fmt_num(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:,.{digits}f}"


def _fmt_pct(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number * 100:.{digits}f}%"


def _fmt_bps(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"{number:.{digits}f} bps"


def _fmt_usd(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    return "n/a" if number is None else f"${number:,.{digits}f}"


def _fmt_date(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return value


def _copy_charts(
    charts: list[dict[str, Any]],
    artifact_assets_dir: Path,
    public_assets_dir: Path | None,
) -> list[dict[str, Any]]:
    ensure_directory(artifact_assets_dir)
    if public_assets_dir is not None:
        ensure_directory(public_assets_dir)

    copied: list[dict[str, Any]] = []
    for chart in charts:
        source_path_text = chart.get("source_path")
        if not source_path_text:
            continue
        source_path = _resolve_path(source_path_text)
        if not source_path.exists():
            continue
        target_name = source_path.name
        artifact_target = artifact_assets_dir / target_name
        shutil.copy2(source_path, artifact_target)
        if public_assets_dir is not None:
            shutil.copy2(source_path, public_assets_dir / target_name)
        copied.append(
            {
                "title": chart.get("title", target_name),
                "subtitle": chart.get("subtitle", ""),
                "section": chart.get("section", "general"),
                "artifact_path": f"assets/{target_name}",
            }
        )
    return copied


def _best_family(robustness_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if robustness_summary is None:
        return None
    family_rows = robustness_summary.get("family_comparison", [])
    if not family_rows:
        return None
    return max(
        family_rows,
        key=lambda row: float(row.get("cumulative_return", float("-inf")) or float("-inf")),
    )


def _table_md(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "| Note |\n| --- |\n| No rows available |"
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append(
            "| " + " | ".join(str(row.get(key, "")) for _, key in columns) + " |"
        )
    return "\n".join([header, divider, *body])


def _table_html(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "<p>No rows available.</p>"
    header = "".join(f"<th>{escape(title)}</th>" for title, _ in columns)
    body = []
    for row in rows:
        cells = "".join(
            f"<td>{escape(str(row.get(key, '')))}</td>" for _, key in columns
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _sorted_strategies(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(snapshot["backtest"].get("top_strategies", []))
    return sorted(
        rows,
        key=lambda row: (
            1 if row.get("has_trades") else 0,
            float(row.get("sharpe_ratio", float("-inf")) or float("-inf")),
            float(row.get("cumulative_return", float("-inf")) or float("-inf")),
        ),
        reverse=True,
    )


def _build_summary(
    settings: FinalReportSettings,
    snapshot: dict[str, Any],
    robustness_summary: dict[str, Any] | None,
    exploratory_summary: dict[str, Any] | None,
    charts: list[dict[str, Any]],
) -> dict[str, Any]:
    best_strategy = snapshot["backtest"].get("best_strategy", {})
    best_family = _best_family(robustness_summary)
    best_family_note = None
    if best_family is not None and (_safe_float(best_family.get("trade_count")) or 0.0) == 0.0:
        best_family_note = (
            "The robustness family winner is a no-trade configuration, so it should be interpreted as abstention rather than monetized alpha."
        )

    verdict = (
        "No positive post-cost out-of-sample strategy survives the current friction model, but the repository now demonstrates a coherent research-to-vault prototype."
        if (_safe_float(best_strategy.get("total_net_pnl_usd")) or 0.0) <= 0.0
        else "The current artifact set contains a positive post-cost out-of-sample strategy under the configured assumptions."
    )
    return {
        "meta": {
            "title": settings.metadata.title,
            "subtitle": settings.metadata.subtitle,
            "course": settings.metadata.course,
            "authors": settings.metadata.authors,
            "repository_url": settings.metadata.repository_url,
            "provider": settings.metadata.provider,
            "symbol": settings.metadata.symbol,
            "frequency": settings.metadata.frequency,
            "date_range": snapshot["meta"].get("date_range", {}),
            "generated_at": datetime.now(UTC).isoformat(),
            "artifact_label": snapshot["meta"].get("artifact_label"),
            "artifact_note": snapshot["meta"].get("artifact_note"),
            "bundle_name": snapshot["meta"].get("bundle_name"),
        },
        "executive_summary": settings.sections.executive_summary,
        "contributions": settings.sections.contributions,
        "limitations": settings.sections.limitations,
        "future_work": settings.sections.future_work,
        "verdict": verdict,
        "best_family_note": best_family_note,
        "research": snapshot.get("research", {}),
        "models": snapshot.get("models", {}),
        "backtest": {
            "summary": snapshot["backtest"].get("summary", {}),
            "best_strategy": best_strategy,
            "top_strategies": _sorted_strategies(snapshot)[:5],
            "assumptions": snapshot["backtest"].get("assumptions", []),
        },
        "exploratory": exploratory_summary or {},
        "robustness": robustness_summary or {},
        "vault": snapshot.get("vault", {}),
        "charts": charts,
        "layers": [
            "Data ingestion and canonicalization",
            "Feature engineering and supervised learning targets",
            "Predictive modeling plus standardized signals",
            "Cost-aware backtesting and vault-state mirroring",
        ],
    }


def _model_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_best = summary["models"].get("baseline_best", {})
    dl_best = summary["models"].get("deep_learning_best", {})
    return [
        {
            "family": "Best baseline",
            "model": baseline_best.get("model_name", "n/a"),
            "metric": "pearson_corr",
            "score": _fmt_num(baseline_best.get("pearson_corr"), 3),
            "rmse": _fmt_num(baseline_best.get("rmse"), 3),
            "signals": _fmt_int(baseline_best.get("signal_count")),
        },
        {
            "family": "Best deep learning",
            "model": dl_best.get("model_name") or dl_best.get("run_label", "n/a"),
            "metric": str(dl_best.get("ranking_metric", "pearson_corr")),
            "score": _fmt_num(
                dl_best.get("ranking_metric_value", dl_best.get("test_pearson_corr")), 3
            ),
            "rmse": _fmt_num(dl_best.get("test_rmse"), 3),
            "signals": _fmt_int(dl_best.get("test_signal_count")),
        },
    ]


def _strategy_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["backtest"].get("top_strategies", []):
        rows.append(
            {
                "strategy": row.get("strategy_name", "n/a"),
                "source": row.get("source_subtype", "n/a"),
                "split": row.get("evaluation_split", "n/a"),
                "status": row.get("status", "ok"),
                "trades": _fmt_int(row.get("trade_count")),
                "cum_return": _fmt_pct(row.get("cumulative_return")),
                "mtm_dd": _fmt_pct(
                    row.get("mark_to_market_max_drawdown", row.get("max_drawdown"))
                ),
                "mtm_sharpe": _fmt_num(row.get("sharpe_ratio"), 3),
                "net_pnl": _fmt_usd(row.get("total_net_pnl_usd")),
                "reason": row.get("diagnostic_reason") or row.get("skip_reason") or "",
            }
        )
    return rows


def _robustness_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["robustness"].get("family_comparison", []):
        rows.append(
            {
                "family": row.get("family_label", "n/a"),
                "strategy": row.get("strategy_name", "n/a"),
                "trades": _fmt_int(row.get("trade_count")),
                "cum_return": _fmt_pct(row.get("cumulative_return")),
                "sharpe": _fmt_num(row.get("sharpe_ratio"), 3),
                "net_pnl": _fmt_usd(row.get("total_net_pnl_usd")),
            }
        )
    return rows


def _exploratory_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    exploratory = summary.get("exploratory", {}) or {}
    showcase = exploratory.get("exploratory_summary", {}) or {}
    best_row = showcase.get("best_showcase_row", {}) or {}
    if not best_row:
        return []
    return [
        {
            "strategy": best_row.get("strategy_name", "n/a"),
            "model": best_row.get("model_name", "n/a"),
            "target": best_row.get("target_type", "n/a"),
            "signal_rule": best_row.get("signal_rule", "n/a"),
            "split": best_row.get("evaluation_split", "n/a"),
            "trades": _fmt_int(best_row.get("trade_count")),
            "cum_return": _fmt_pct(best_row.get("cumulative_return")),
            "mtm_sharpe": _fmt_num(best_row.get("sharpe_ratio"), 3),
            "net_pnl": _fmt_usd(best_row.get("total_net_pnl_usd")),
            "status": best_row.get("status", "n/a"),
            "reason": best_row.get("reason", ""),
        }
    ]


def _build_markdown(summary: dict[str, Any]) -> str:
    research = summary["research"]
    best_strategy = summary["backtest"]["best_strategy"]
    demo_banner = ""
    if summary["meta"].get("artifact_label"):
        banner_note = summary["meta"].get("artifact_note") or "Illustrative results"
        demo_banner = f"> **{summary['meta']['artifact_label']}** - {banner_note}\n\n"
    chart_md = "\n\n".join(
        f"### {chart['title']}\n\n{chart['subtitle']}\n\n![{chart['title']}]({chart['artifact_path']})"
        for chart in summary["charts"]
    )
    note = (
        f"\n\n**Important note:** {summary['best_family_note']}"
        if summary.get("best_family_note")
        else ""
    )
    exploratory_section = ""
    if summary.get("exploratory"):
        exploratory_section = f"""

## Exploratory DL Showcase

Exploratory DL results are supplementary showcase results designed to demonstrate model learning behavior, ranking ability, and alternative opportunity definitions. They do not replace the strict post-cost primary conclusion.

{_table_md(
    _exploratory_rows(summary),
    [
        ("Strategy", "strategy"),
        ("Model", "model"),
        ("Target", "target"),
        ("Signal Rule", "signal_rule"),
        ("Split", "split"),
        ("Trades", "trades"),
        ("Cum Return", "cum_return"),
        ("MTM Sharpe", "mtm_sharpe"),
        ("Net PnL", "net_pnl"),
        ("Status", "status"),
        ("Reason", "reason"),
    ],
)}
"""
    return f"""# {summary['meta']['title']}

{demo_banner}{summary['meta']['subtitle']}

## Metadata

- Course: `{summary['meta']['course']}`
- Authors: `{', '.join(summary['meta']['authors'])}`
- Repository: `{summary['meta']['repository_url']}`
- Market: `{summary['meta']['symbol']}` on `{summary['meta']['provider']}` at `{summary['meta']['frequency']}`
- Sample window: `{_fmt_date(summary['meta']['date_range'].get('start'))}` to `{_fmt_date(summary['meta']['date_range'].get('end_exclusive'))}`
- Generated at: `{summary['meta']['generated_at']}`

## Executive Summary

{"\n".join(f"- {item}" for item in summary['executive_summary'])}

**Verdict:** {summary['verdict']}

## System Scope

{"\n".join(f"- {item}" for item in summary['layers'])}

## Dataset And Data Quality

- Canonical hourly rows: `{_fmt_int(research.get('canonical_rows'))}`
- Funding events: `{_fmt_int(research.get('funding_events'))}`
- Coverage ratio: `{_fmt_pct(research.get('coverage_ratio'))}`
- Average funding rate: `{_fmt_bps(research.get('funding_mean_bps'))}`
- Funding standard deviation: `{_fmt_bps(research.get('funding_std_bps'))}`
- Average perp-vs-spot spread: `{_fmt_bps(research.get('spread_mean_bps'))}`
- Mean annualized volatility: `{_fmt_pct(research.get('annualized_volatility'))}`

## Modeling Summary

{_table_md(
    _model_rows(summary),
    [
        ("Family", "family"),
        ("Model", "model"),
        ("Metric", "metric"),
        ("Score", "score"),
        ("RMSE", "rmse"),
        ("Signals", "signals"),
    ],
)}

## Backtest Summary

- Primary split: `{summary['backtest']['summary'].get('primary_split', 'test')}`
- Best strategy: `{best_strategy.get('strategy_name', 'n/a')}`
- Trade count: `{_fmt_int(best_strategy.get('trade_count'))}`
- Cumulative return: `{_fmt_pct(best_strategy.get('cumulative_return'))}`
- Mark-to-market Sharpe: `{_fmt_num(best_strategy.get('sharpe_ratio'), 3)}`
- Net PnL: `{_fmt_usd(best_strategy.get('total_net_pnl_usd'))}`

{_table_md(
    _strategy_rows(summary),
    [
        ("Strategy", "strategy"),
        ("Source", "source"),
        ("Split", "split"),
        ("Status", "status"),
        ("Trades", "trades"),
        ("Cum Return", "cum_return"),
        ("MTM Drawdown", "mtm_dd"),
        ("MTM Sharpe", "mtm_sharpe"),
        ("Net PnL", "net_pnl"),
        ("Reason", "reason"),
    ],
)}

### Core Assumptions

{"\n".join(f"- {item}" for item in summary['backtest'].get('assumptions', [])[:8])}

## Robustness Interpretation

{_table_md(
    _robustness_rows(summary),
    [
        ("Family", "family"),
        ("Representative Strategy", "strategy"),
        ("Trades", "trades"),
        ("Cum Return", "cum_return"),
        ("Sharpe", "sharpe"),
        ("Net PnL", "net_pnl"),
    ],
)}{note}
{exploratory_section}

## Vault Prototype

- Selected strategy: `{summary['vault'].get('selected_strategy', 'n/a')}`
- Strategy state: `{summary['vault'].get('strategy_state', 'n/a')}`
- Suggested direction: `{summary['vault'].get('suggested_direction', 'n/a')}`
- Reported NAV assets: `{_fmt_int(summary['vault'].get('reported_nav_assets'))}`
- Summary PnL: `{_fmt_usd(summary['vault'].get('summary_pnl_usd'))}`
- Prepared contract calls: `{_fmt_int(summary['vault'].get('call_count'))}`

## Contributions

{"\n".join(f"- {item}" for item in summary['contributions'])}

## Limitations

{"\n".join(f"- {item}" for item in summary['limitations'])}

## Future Work

{"\n".join(f"- {item}" for item in summary['future_work'])}

## Figures

{chart_md}
"""


def _build_html(summary: dict[str, Any]) -> str:
    research = summary["research"]
    best_strategy = summary["backtest"]["best_strategy"]
    demo_banner_html = ""
    if summary["meta"].get("artifact_label"):
        artifact_note = escape(str(summary["meta"].get("artifact_note") or "Illustrative results"))
        demo_banner_html = f"""
        <div class="demo-banner">
          <strong>{escape(str(summary["meta"]["artifact_label"]))}</strong>
          <span>{artifact_note}</span>
        </div>
        """
    chart_cards = "".join(
        f"""
        <figure class="chart-card">
          <div class="chart-copy">
            <p class="kicker">{escape(chart['section'])}</p>
            <h3>{escape(chart['title'])}</h3>
            <p>{escape(chart['subtitle'])}</p>
          </div>
          <img src="{escape(chart['artifact_path'])}" alt="{escape(chart['title'])}" />
        </figure>
        """
        for chart in summary["charts"]
    )
    note_html = (
        f"<p class='note'><strong>Interpretation:</strong> {escape(summary['best_family_note'])}</p>"
        if summary.get("best_family_note")
        else ""
    )
    exploratory_html = ""
    if summary.get("exploratory"):
        exploratory_html = f"""
      <section class="section">
        <p class="eyebrow">Exploratory</p>
        <h2>Supplementary DL showcase results stay separate from the strict conclusion</h2>
        <p>Exploratory DL results are supplementary showcase results designed to demonstrate model learning behavior, ranking ability, and alternative opportunity definitions. They do not replace the strict post-cost primary conclusion.</p>
        <div class="table-shell">
          {_table_html(
              _exploratory_rows(summary),
              [
                  ("Strategy", "strategy"),
                  ("Model", "model"),
                  ("Target", "target"),
                  ("Signal Rule", "signal_rule"),
                  ("Split", "split"),
                  ("Trades", "trades"),
                  ("Cum Return", "cum_return"),
                  ("MTM Sharpe", "mtm_sharpe"),
                  ("Net PnL", "net_pnl"),
                  ("Status", "status"),
                  ("Reason", "reason"),
              ],
          )}
        </div>
      </section>
        """
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{escape(summary['meta']['title'])}</title>
    <style>
      :root {{
        --bg: #f7f1e7;
        --panel: rgba(255, 251, 245, 0.92);
        --line: rgba(23, 21, 26, 0.12);
        --ink: #16161b;
        --muted: #5e6068;
        --accent: #0f766e;
        --shadow: 0 20px 60px rgba(38, 25, 10, 0.08);
        --display: "Iowan Old Style", Georgia, serif;
        --body: "Avenir Next", "Segoe UI", sans-serif;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        font-family: var(--body);
        color: var(--ink);
        background: linear-gradient(180deg, #faf5ed 0%, #f1e5d7 100%);
      }}
      .page {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 72px; }}
      .hero, .section {{
        border: 1px solid var(--line);
        border-radius: 26px;
        background: var(--panel);
        box-shadow: var(--shadow);
        padding: 28px;
      }}
      .section {{ margin-top: 20px; }}
      .grid, .metrics, .charts {{ display: grid; gap: 16px; }}
      .grid {{ grid-template-columns: 1.6fr 1fr; }}
      .metrics {{ grid-template-columns: repeat(4, minmax(0, 1fr)); margin-top: 18px; }}
      .charts {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .metric, .side, .chart-card {{
        border: 1px solid var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,0.72);
      }}
      .metric, .side, .chart-copy {{ padding: 16px; }}
      h1, h2, h3 {{ font-family: var(--display); margin: 0; letter-spacing: -0.03em; }}
      h1 {{ font-size: clamp(2.8rem, 5vw, 4.8rem); line-height: 0.94; margin-top: 8px; }}
      h2 {{ font-size: clamp(1.8rem, 3vw, 2.6rem); margin-bottom: 10px; }}
      h3 {{ font-size: 1.25rem; }}
      p, li {{ color: var(--muted); line-height: 1.7; }}
      .eyebrow, .kicker {{
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.73rem;
        color: var(--accent);
        font-weight: 700;
        margin: 0;
      }}
      .chips {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
      .chip {{
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.8);
      }}
      table {{ width: 100%; border-collapse: collapse; }}
      th, td {{
        text-align: left;
        padding: 12px 14px;
        border-bottom: 1px solid rgba(23,21,26,0.08);
        white-space: nowrap;
      }}
      thead th {{
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        background: rgba(247, 240, 230, 0.92);
      }}
      .table-shell {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 20px; background: rgba(255,255,255,0.75); }}
      img {{ width: 100%; display: block; border-top: 1px solid var(--line); }}
      ul {{ margin: 0; padding-left: 20px; }}
      .note {{ margin-top: 14px; }}
      .demo-banner {{
        display: inline-flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 18px;
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid rgba(185, 28, 28, 0.18);
        background: rgba(254, 242, 242, 0.88);
        color: #991b1b;
        font-size: 0.92rem;
      }}
      @media (max-width: 900px) {{
        .grid, .metrics, .charts {{ grid-template-columns: 1fr; }}
      }}
    </style>
  </head>
  <body>
    <main class="page">
      <section class="hero">
        {demo_banner_html}
        <div class="grid">
          <div>
            <p class="eyebrow">Final Report</p>
            <h1>{escape(summary['meta']['title'])}</h1>
            <p>{escape(summary['meta']['subtitle'])}</p>
            <div class="chips">
              <span class="chip">{escape(summary['meta']['course'])}</span>
              <span class="chip">{escape(', '.join(summary['meta']['authors']))}</span>
              <span class="chip">{escape(summary['meta']['symbol'])} / {escape(summary['meta']['provider'])} / {escape(summary['meta']['frequency'])}</span>
            </div>
          </div>
          <div class="side">
            <p class="kicker">Verdict</p>
            <p><strong>{escape(summary['verdict'])}</strong></p>
            <ul>{"".join(f"<li>{escape(item)}</li>" for item in summary['executive_summary'])}</ul>
          </div>
        </div>
        <div class="metrics">
          <article class="metric"><p class="kicker">Canonical Hours</p><h3>{escape(_fmt_int(research.get('canonical_rows')))}</h3><p>Aligned hourly research rows.</p></article>
          <article class="metric"><p class="kicker">Funding Events</p><h3>{escape(_fmt_int(research.get('funding_events')))}</h3><p>Observed settlements in the sample.</p></article>
          <article class="metric"><p class="kicker">Best Baseline</p><h3>{escape(_fmt_num(summary['models'].get('baseline_best', {}).get('pearson_corr'), 3))}</h3><p>Elastic-net test Pearson correlation.</p></article>
          <article class="metric"><p class="kicker">Best Backtest</p><h3>{escape(_fmt_pct(best_strategy.get('cumulative_return')))}</h3><p>{escape(best_strategy.get('strategy_name', 'n/a'))} with {escape(_fmt_usd(best_strategy.get('total_net_pnl_usd')))} net PnL.</p></article>
        </div>
      </section>

      <section class="section">
        <p class="eyebrow">Dataset</p>
        <h2>Clean data supports a credible prototype, not an automatic edge</h2>
        <ul>
          <li>Coverage ratio: {escape(_fmt_pct(research.get('coverage_ratio')))}</li>
          <li>Average funding rate: {escape(_fmt_bps(research.get('funding_mean_bps')))}</li>
          <li>Funding standard deviation: {escape(_fmt_bps(research.get('funding_std_bps')))}</li>
          <li>Average perp-vs-spot spread: {escape(_fmt_bps(research.get('spread_mean_bps')))}</li>
          <li>Mean annualized volatility: {escape(_fmt_pct(research.get('annualized_volatility')))}</li>
        </ul>
      </section>

      <section class="section">
        <p class="eyebrow">Modeling</p>
        <h2>Predictive structure exists, but the current decision rules mostly abstain</h2>
        <div class="table-shell">
          {_table_html(
              _model_rows(summary),
              [
                  ("Family", "family"),
                  ("Model", "model"),
                  ("Metric", "metric"),
                  ("Score", "score"),
                  ("RMSE", "rmse"),
                  ("Signals", "signals"),
              ],
          )}
        </div>
      </section>

      <section class="section">
        <p class="eyebrow">Backtest</p>
        <h2>Explicit costs are the main reason the current strategy set fails</h2>
        <div class="table-shell">
          {_table_html(
              _strategy_rows(summary),
              [
                  ("Strategy", "strategy"),
                  ("Source", "source"),
                  ("Split", "split"),
                  ("Status", "status"),
                  ("Trades", "trades"),
                  ("Cum Return", "cum_return"),
                  ("MTM Drawdown", "mtm_dd"),
                  ("MTM Sharpe", "mtm_sharpe"),
                  ("Net PnL", "net_pnl"),
                  ("Reason", "reason"),
              ],
          )}
        </div>
        <p class="note">Primary split: {escape(str(summary['backtest']['summary'].get('primary_split', 'test')))}. Assumptions remain explicit and prototype-scoped.</p>
      </section>

      <section class="section">
        <p class="eyebrow">Robustness</p>
        <h2>Family comparisons should be read together with trade counts</h2>
        <div class="table-shell">
          {_table_html(
              _robustness_rows(summary),
              [
                  ("Family", "family"),
                  ("Representative Strategy", "strategy"),
                  ("Trades", "trades"),
                  ("Cum Return", "cum_return"),
                  ("Sharpe", "sharpe"),
                  ("Net PnL", "net_pnl"),
              ],
          )}
        </div>
        {note_html}
      </section>

      {exploratory_html}

      <section class="section">
        <p class="eyebrow">Vault</p>
        <h2>The on-chain layer mirrors accounting and state, not live exchange execution</h2>
        <ul>
          <li>Selected strategy: {escape(str(summary['vault'].get('selected_strategy', 'n/a')))}</li>
          <li>Strategy state: {escape(str(summary['vault'].get('strategy_state', 'n/a')))}</li>
          <li>Suggested direction: {escape(str(summary['vault'].get('suggested_direction', 'n/a')))}</li>
          <li>Reported NAV assets: {escape(_fmt_int(summary['vault'].get('reported_nav_assets')))}</li>
          <li>Summary PnL: {escape(_fmt_usd(summary['vault'].get('summary_pnl_usd')))}</li>
          <li>Prepared contract calls: {escape(_fmt_int(summary['vault'].get('call_count')))}</li>
        </ul>
      </section>

      <section class="section">
        <p class="eyebrow">Figures</p>
        <h2>Charts copied directly from the artifact pipeline</h2>
        <div class="charts">{chart_cards}</div>
      </section>

      <section class="section">
        <p class="eyebrow">Conclusion</p>
        <h2>What this project contributes</h2>
        <div class="grid">
          <div class="side">
            <p class="kicker">Contributions</p>
            <ul>{"".join(f"<li>{escape(item)}</li>" for item in summary['contributions'])}</ul>
          </div>
          <div class="side">
            <p class="kicker">Limitations</p>
            <ul>{"".join(f"<li>{escape(item)}</li>" for item in summary['limitations'])}</ul>
            <p class="kicker" style="margin-top:16px;">Future Work</p>
            <ul>{"".join(f"<li>{escape(item)}</li>" for item in summary['future_work'])}</ul>
          </div>
        </div>
      </section>
    </main>
  </body>
</html>
"""


def run_final_report(settings: FinalReportSettings) -> FinalReportArtifacts:
    """Generate markdown, HTML, and summary artifacts for the final report."""
    snapshot = _load_json(settings.input.demo_snapshot_path)
    robustness_summary = None
    if settings.input.robustness_summary_path is not None:
        robustness_path = _resolve_path(settings.input.robustness_summary_path)
        if robustness_path.exists():
            robustness_summary = _load_json(robustness_path)
    exploratory_summary = None
    if settings.input.exploratory_summary_path is not None:
        exploratory_path = _resolve_path(settings.input.exploratory_summary_path)
        if exploratory_path.exists():
            exploratory_summary = _load_json(exploratory_path)

    artifact_output_dir = ensure_directory(
        _resolve_path(settings.output.artifact_dir)
        / settings.metadata.provider
        / settings.metadata.symbol.lower()
        / settings.metadata.frequency
    )
    public_report_dir = (
        ensure_directory(_resolve_path(settings.output.frontend_public_dir))
        if settings.output.copy_to_frontend_public
        else None
    )
    charts = _copy_charts(
        snapshot.get("charts", []),
        artifact_output_dir / "assets",
        public_report_dir / "assets" if public_report_dir is not None else None,
    )
    summary = _build_summary(
        settings,
        snapshot,
        robustness_summary,
        exploratory_summary,
        charts,
    )

    markdown_report_path: str | None = None
    if settings.output.write_markdown:
        markdown_text = _build_markdown(summary)
        markdown_path = artifact_output_dir / "final_report.md"
        markdown_path.write_text(markdown_text, encoding="utf-8")
        markdown_report_path = str(markdown_path)
        if public_report_dir is not None:
            (public_report_dir / "final_report.md").write_text(markdown_text, encoding="utf-8")

    html_report_path: str | None = None
    if settings.output.write_html:
        html_text = _build_html(summary)
        html_path = artifact_output_dir / "final_report.html"
        html_path.write_text(html_text, encoding="utf-8")
        html_report_path = str(html_path)
        if public_report_dir is not None:
            (public_report_dir / "index.html").write_text(html_text, encoding="utf-8")
            (public_report_dir / "final_report.html").write_text(html_text, encoding="utf-8")

    summary_json_path: str | None = None
    if settings.output.write_json_summary:
        summary_path = artifact_output_dir / "summary.json"
        summary_text = json.dumps(summary, indent=2)
        summary_path.write_text(summary_text, encoding="utf-8")
        summary_json_path = str(summary_path)
        if public_report_dir is not None:
            (public_report_dir / "summary.json").write_text(summary_text, encoding="utf-8")

    return FinalReportArtifacts(
        artifact_output_dir=str(artifact_output_dir),
        markdown_report_path=markdown_report_path,
        html_report_path=html_report_path,
        summary_json_path=summary_json_path,
        public_report_dir=str(public_report_dir) if public_report_dir is not None else None,
    )
