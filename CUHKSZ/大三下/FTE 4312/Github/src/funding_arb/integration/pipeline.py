"""Prototype operator/oracle-style flow from strategy artifacts to vault updates."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from web3 import Web3

from funding_arb.config.models import IntegrationSettings
from funding_arb.utils.paths import ensure_directory, repo_path

STATE_NAME_TO_CODE = {"idle": 0, "active": 1, "emergency": 2, "settled": 3}


@dataclass(frozen=True)
class VaultSyncArtifacts:
    """Files produced by the mock off-chain to on-chain integration flow."""

    selection_summary_path: str
    plan_path: str
    call_summary_path: str
    markdown_report_path: str | None
    output_dir: str


@dataclass(frozen=True)
class SelectedStrategySnapshot:
    """Compact summary of the chosen strategy and latest signal row."""

    strategy_name: str
    split: str
    ranking_metric: str
    ranking_metric_value: float | int | None
    timestamp: str
    should_trade: bool
    suggested_direction: str
    signal_score: float | None
    expected_return_bps: float | None
    confidence: float | None
    source: str
    source_subtype: str
    model_family: str
    task: str
    signal_metadata: dict[str, Any]
    leaderboard_summary: dict[str, Any]


@dataclass(frozen=True)
class VaultUpdatePlan:
    """Mock vault update payload derived from strategy artifacts."""

    strategy_state_name: str
    strategy_state_code: int
    signal_hash: str
    metadata_hash: str
    report_hash: str
    reported_nav_assets: int
    summary_pnl_assets: int
    summary_pnl_usd: float
    selected_strategy_name: str
    selected_split: str
    source_timestamp: str
    should_trade: bool
    suggested_direction: str


def describe_integration_job(config: IntegrationSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the integration job."""
    settings = (
        config
        if isinstance(config, IntegrationSettings)
        else IntegrationSettings.model_validate(config)
    )
    mode = "broadcast" if settings.contract.broadcast else "dry-run"
    return (
        f"Vault sync ready for {settings.input.symbol} ({settings.input.provider}, "
        f"{settings.input.frequency}) using {settings.input.signals_path} and "
        f"{settings.input.leaderboard_path}; mode={mode}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _output_dir(settings: IntegrationSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / settings.output.run_name
    )


def _load_table(path_text: str) -> pd.DataFrame:
    path = _resolve_path(path_text)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for integration input: {path.suffix}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return "0x" + sha256(encoded.encode("utf-8")).hexdigest()


def _parse_signal_metadata(raw_value: Any) -> dict[str, Any]:
    if raw_value is None or (isinstance(raw_value, float) and math.isnan(raw_value)):
        return {}
    if isinstance(raw_value, dict):
        return _json_ready(raw_value)
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        return _json_ready(json.loads(text))
    except json.JSONDecodeError:
        return {"raw_metadata": text}


def _choose_leaderboard_row(
    signals: pd.DataFrame,
    leaderboard: pd.DataFrame,
    settings: IntegrationSettings,
) -> tuple[str, dict[str, Any]]:
    if leaderboard.empty:
        strategies = sorted(signals["strategy_name"].dropna().astype(str).unique().tolist())
        if not strategies:
            raise ValueError("No strategies found in signals artifact.")
        return settings.selection.strategy_name or strategies[0], {}

    if settings.selection.strategy_name is not None:
        filtered = leaderboard.loc[leaderboard["strategy_name"] == settings.selection.strategy_name]
        if filtered.empty:
            raise ValueError(f"Requested strategy '{settings.selection.strategy_name}' was not found in leaderboard.")
        selected = filtered.iloc[0]
    else:
        metric = settings.selection.ranking_metric
        if metric not in leaderboard.columns:
            raise ValueError(f"Ranking metric '{metric}' is missing from leaderboard artifact.")
        ranked = leaderboard.copy()
        if settings.selection.prefer_traded_strategy:
            if "has_trades" in ranked.columns:
                ranked["_has_trades_for_selection"] = ranked["has_trades"].map(
                    lambda value: str(value).strip().lower() in {"true", "1", "yes"}
                )
            elif "trade_count" in ranked.columns:
                ranked["_has_trades_for_selection"] = ranked["trade_count"].fillna(0).astype(float) > 0.0
            else:
                ranked["_has_trades_for_selection"] = True
            ranked = ranked.sort_values(
                ["_has_trades_for_selection", metric],
                ascending=[False, settings.selection.ranking_ascending],
                kind="stable",
                na_position="last",
            )
        else:
            ranked = ranked.sort_values(
                metric,
                ascending=settings.selection.ranking_ascending,
                kind="stable",
                na_position="last",
            )
        selected = ranked.iloc[0]

    selected_payload = selected.to_dict()
    selected_payload.pop("_has_trades_for_selection", None)
    return str(selected["strategy_name"]), {key: _json_ready(value) for key, value in selected_payload.items()}


def _choose_signal_row(
    signals: pd.DataFrame,
    strategy_name: str,
    settings: IntegrationSettings,
) -> tuple[pd.Series, str]:
    subset = signals.loc[signals["strategy_name"] == strategy_name].copy()
    if subset.empty:
        raise ValueError(f"No signals found for strategy '{strategy_name}'.")

    subset["timestamp"] = pd.to_datetime(subset["timestamp"], utc=True)
    for split_name in settings.selection.split_preference:
        split_subset = subset.loc[subset["split"] == split_name].copy()
        if split_subset.empty:
            continue

        candidate = split_subset
        if settings.selection.prefer_should_trade or settings.selection.require_should_trade:
            tradeable = split_subset.loc[split_subset["should_trade"].astype(bool)]
            if not tradeable.empty:
                candidate = tradeable
            elif settings.selection.require_should_trade:
                continue

        return candidate.sort_values("timestamp").iloc[-1], split_name

    if not settings.selection.allow_flat_fallback:
        raise ValueError("No signal row matched the requested split/tradeability filters.")
    fallback = subset.sort_values("timestamp").iloc[-1]
    return fallback, str(fallback["split"])


def _build_selected_snapshot(
    signal_row: pd.Series,
    split_name: str,
    strategy_name: str,
    leaderboard_summary: dict[str, Any],
    settings: IntegrationSettings,
) -> SelectedStrategySnapshot:
    return SelectedStrategySnapshot(
        strategy_name=strategy_name,
        split=split_name,
        ranking_metric=settings.selection.ranking_metric,
        ranking_metric_value=leaderboard_summary.get(settings.selection.ranking_metric),
        timestamp=_json_ready(signal_row["timestamp"]),
        should_trade=bool(signal_row["should_trade"]),
        suggested_direction=str(signal_row["suggested_direction"]),
        signal_score=_json_ready(signal_row.get("signal_score")),
        expected_return_bps=_json_ready(signal_row.get("expected_return_bps")),
        confidence=_json_ready(signal_row.get("confidence")),
        source=str(signal_row["source"]),
        source_subtype=str(signal_row["source_subtype"]),
        model_family=str(signal_row["model_family"]),
        task=str(signal_row["task"]),
        signal_metadata=_parse_signal_metadata(signal_row.get("metadata_json")),
        leaderboard_summary=leaderboard_summary,
    )


def _state_name_from_snapshot(
    snapshot: SelectedStrategySnapshot, settings: IntegrationSettings
) -> str:
    if snapshot.should_trade and snapshot.suggested_direction != "flat":
        return settings.semantics.active_strategy_state.lower()
    return settings.semantics.flat_strategy_state.lower()


def _usd_to_asset_units(usd_value: float, decimals: int, asset_usd_price: float) -> int:
    if asset_usd_price <= 0:
        raise ValueError("asset_usd_price must be positive.")
    return int(round((usd_value / asset_usd_price) * (10**decimals)))


def _build_update_plan(
    snapshot: SelectedStrategySnapshot,
    settings: IntegrationSettings,
) -> VaultUpdatePlan:
    pnl_usd = float(snapshot.leaderboard_summary.get("total_net_pnl_usd", 0.0) or 0.0)
    pnl_assets = _usd_to_asset_units(
        pnl_usd,
        settings.semantics.asset_decimals,
        settings.semantics.asset_usd_price,
    )
    reported_nav_assets = max(
        settings.semantics.nav_floor_assets,
        settings.semantics.base_nav_assets + pnl_assets,
    )
    strategy_state_name = _state_name_from_snapshot(snapshot, settings)
    return VaultUpdatePlan(
        strategy_state_name=strategy_state_name,
        strategy_state_code=STATE_NAME_TO_CODE[strategy_state_name],
        signal_hash=_hash_payload(
            {
                "strategy_name": snapshot.strategy_name,
                "timestamp": snapshot.timestamp,
                "split": snapshot.split,
                "should_trade": snapshot.should_trade,
                "suggested_direction": snapshot.suggested_direction,
                "signal_score": snapshot.signal_score,
                "confidence": snapshot.confidence,
            }
        ),
        metadata_hash=_hash_payload(
            {
                "selection": asdict(snapshot),
                "assumptions": {
                    "base_nav_assets": settings.semantics.base_nav_assets,
                    "asset_decimals": settings.semantics.asset_decimals,
                    "asset_usd_price": settings.semantics.asset_usd_price,
                },
            }
        ),
        report_hash=_hash_payload(
            {
                "strategy_name": snapshot.strategy_name,
                "leaderboard_summary": snapshot.leaderboard_summary,
                "reported_nav_assets": reported_nav_assets,
                "summary_pnl_assets": pnl_assets,
                "summary_pnl_usd": pnl_usd,
            }
        ),
        reported_nav_assets=reported_nav_assets,
        summary_pnl_assets=pnl_assets,
        summary_pnl_usd=pnl_usd,
        selected_strategy_name=snapshot.strategy_name,
        selected_split=snapshot.split,
        source_timestamp=snapshot.timestamp,
        should_trade=snapshot.should_trade,
        suggested_direction=snapshot.suggested_direction,
    )


def _load_contract_abi(artifact_path: str) -> list[dict[str, Any]]:
    artifact = json.loads(_resolve_path(artifact_path).read_text(encoding="utf-8"))
    abi = artifact.get("abi")
    if not abi:
        raise ValueError(f"Artifact '{artifact_path}' does not contain an ABI.")
    return abi


def _resolve_with_env(value: str, env_name: str | None) -> str:
    env_value = os.getenv(env_name) if env_name else None
    return env_value or value


def _build_contract(web3: Web3, settings: IntegrationSettings):
    abi = _load_contract_abi(settings.contract.artifact_path)
    address_text = _resolve_with_env(settings.contract.vault_address, settings.contract.vault_address_env)
    checksum = Web3.to_checksum_address(address_text)
    return web3.eth.contract(address=checksum, abi=abi), checksum


def _planned_calls(contract: Any, plan: VaultUpdatePlan, settings: IntegrationSettings) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    signal_hash = bytes.fromhex(plan.signal_hash.removeprefix("0x"))
    metadata_hash = bytes.fromhex(plan.metadata_hash.removeprefix("0x"))
    report_hash = bytes.fromhex(plan.report_hash.removeprefix("0x"))

    if settings.contract.update_strategy_state:
        args = [plan.strategy_state_code, signal_hash, metadata_hash]
        fn = contract.functions.updateStrategyState(*args)
        calls.append(
            {
                "name": "updateStrategyState",
                "args": {"new_state": plan.strategy_state_code, "signal_hash": plan.signal_hash, "metadata_hash": plan.metadata_hash},
                "function_args": args,
                "calldata": fn._encode_transaction_data(),
            }
        )
    if settings.contract.update_nav:
        args = [plan.reported_nav_assets, report_hash]
        fn = contract.functions.updateNav(*args)
        calls.append(
            {
                "name": "updateNav",
                "args": {"new_reported_nav_assets": plan.reported_nav_assets, "report_hash": plan.report_hash},
                "function_args": args,
                "calldata": fn._encode_transaction_data(),
            }
        )
    if settings.contract.update_pnl:
        args = [plan.summary_pnl_assets, report_hash]
        fn = contract.functions.updatePnl(*args)
        calls.append(
            {
                "name": "updatePnl",
                "args": {"pnl_delta_assets": plan.summary_pnl_assets, "report_hash": plan.report_hash},
                "function_args": args,
                "calldata": fn._encode_transaction_data(),
            }
        )
    return calls


def _broadcast_calls(settings: IntegrationSettings, calls: list[dict[str, Any]]) -> dict[str, Any]:
    rpc_url = _resolve_with_env(settings.contract.rpc_url, settings.contract.rpc_url_env)
    web3 = Web3(Web3.HTTPProvider(rpc_url))
    if not web3.is_connected():
        raise RuntimeError(f"Could not connect to RPC endpoint '{rpc_url}'.")

    private_key = os.getenv(settings.contract.operator_private_key_env)
    if not private_key:
        raise RuntimeError(
            f"Environment variable '{settings.contract.operator_private_key_env}' is required for broadcast mode."
        )

    account = web3.eth.account.from_key(private_key)
    contract, checksum = _build_contract(web3, settings)
    nonce = web3.eth.get_transaction_count(account.address)
    receipts: list[dict[str, Any]] = []

    for item in calls:
        fn = getattr(contract.functions, item["name"])(*item["function_args"])
        tx = fn.build_transaction(
            {
                "from": account.address,
                "nonce": nonce,
                "chainId": settings.contract.chain_id or web3.eth.chain_id,
                "gas": settings.contract.gas_limit,
                "gasPrice": settings.contract.gas_price_wei or web3.eth.gas_price,
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        if settings.contract.wait_for_receipt:
            receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            receipts.append(
                {
                    "function": item["name"],
                    "tx_hash": tx_hash.hex(),
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed,
                    "status": receipt.status,
                }
            )
        else:
            receipts.append({"function": item["name"], "tx_hash": tx_hash.hex(), "status": "submitted"})
        nonce += 1

    result: dict[str, Any] = {
        "mode": "broadcast",
        "rpc_url": rpc_url,
        "vault_address": checksum,
        "receipts": receipts,
    }
    if settings.contract.wait_for_receipt:
        result["state_snapshot"] = {
            "strategy_state": int(contract.functions.strategyState().call()),
            "reported_nav_assets": int(contract.functions.reportedNavAssets().call()),
            "cumulative_pnl_assets": int(contract.functions.cumulativePnlAssets().call()),
            "last_signal_hash": contract.functions.lastSignalHash().call().hex(),
            "last_report_hash": contract.functions.lastReportHash().call().hex(),
        }
    return result


def _markdown_report(
    snapshot: SelectedStrategySnapshot,
    plan: VaultUpdatePlan,
    calls: list[dict[str, Any]],
    execution_summary: dict[str, Any],
) -> str:
    call_lines = "\n".join(f"- `{item['name']}` with calldata `{item['calldata']}`" for item in calls)
    exec_lines = "\n".join(f"- {key}: {value}" for key, value in execution_summary.items())
    return (
        "# Mock Off-Chain to On-Chain Integration Report\n\n"
        "## Selected Strategy Context\n\n"
        f"- strategy: `{snapshot.strategy_name}`\n"
        f"- split: `{snapshot.split}`\n"
        f"- timestamp: `{snapshot.timestamp}`\n"
        f"- should_trade: `{snapshot.should_trade}`\n"
        f"- suggested_direction: `{snapshot.suggested_direction}`\n"
        f"- ranking metric `{snapshot.ranking_metric}`: `{snapshot.ranking_metric_value}`\n\n"
        "## Planned Vault Update\n\n"
        f"- strategy_state: `{plan.strategy_state_name}` (`{plan.strategy_state_code}`)\n"
        f"- reported_nav_assets: `{plan.reported_nav_assets}`\n"
        f"- summary_pnl_assets: `{plan.summary_pnl_assets}`\n"
        f"- summary_pnl_usd: `{plan.summary_pnl_usd}`\n\n"
        "## Contract Calls\n\n"
        f"{call_lines}\n\n"
        "## Execution Summary\n\n"
        f"{exec_lines}\n\n"
        "## Prototype Assumptions\n\n"
        "- This flow is a trusted operator/oracle-style prototype.\n"
        "- It reuses local strategy artifacts and converts them into simplified vault updates.\n"
        "- It is educational and demo-oriented, not a production oracle network.\n"
    )


def run_vault_sync_pipeline(settings: IntegrationSettings) -> VaultSyncArtifacts:
    """Run the lightweight mock operator flow from strategy artifacts to vault calls."""
    signals = _load_table(settings.input.signals_path)
    leaderboard = _load_table(settings.input.leaderboard_path)
    strategy_name, leaderboard_summary = _choose_leaderboard_row(signals, leaderboard, settings)
    signal_row, split_name = _choose_signal_row(signals, strategy_name, settings)
    snapshot = _build_selected_snapshot(signal_row, split_name, strategy_name, leaderboard_summary, settings)
    plan = _build_update_plan(snapshot, settings)

    dry_run_web3 = Web3()
    contract, checksum = _build_contract(dry_run_web3, settings)
    calls = _planned_calls(contract, plan, settings)

    execution_summary: dict[str, Any] = {
        "mode": "dry-run",
        "rpc_url": _resolve_with_env(settings.contract.rpc_url, settings.contract.rpc_url_env),
        "vault_address": checksum,
        "operator_private_key_env": settings.contract.operator_private_key_env,
    }
    if settings.contract.broadcast:
        execution_summary = _broadcast_calls(settings, calls)

    output_dir = _output_dir(settings)
    selection_path = output_dir / "selected_strategy_summary.json"
    plan_path = output_dir / "vault_update_plan.json"
    calls_path = output_dir / "contract_call_summary.json"
    selection_path.write_text(json.dumps(_json_ready(asdict(snapshot)), indent=2), encoding="utf-8")
    plan_path.write_text(json.dumps(_json_ready(asdict(plan)), indent=2), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "calls": [{key: _json_ready(value) for key, value in item.items() if key != "function_args"} for item in calls],
                "execution_summary": _json_ready(execution_summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    markdown_report_path: str | None = None
    if settings.output.write_markdown_report:
        report_path = output_dir / "integration_report.md"
        report_path.write_text(_markdown_report(snapshot, plan, calls, execution_summary), encoding="utf-8")
        markdown_report_path = str(report_path)

    return VaultSyncArtifacts(
        selection_summary_path=str(selection_path),
        plan_path=str(plan_path),
        call_summary_path=str(calls_path),
        markdown_report_path=markdown_report_path,
        output_dir=str(output_dir),
    )
