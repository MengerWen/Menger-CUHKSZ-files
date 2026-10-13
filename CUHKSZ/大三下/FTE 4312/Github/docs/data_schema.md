# Data Schema

## Purpose

This project uses a normalized hourly dataset for perpetual-funding arbitrage research. The first working implementation targets `BTCUSDT` on Binance and writes three layers of artifacts:

- raw exchange extracts for reproducibility
- cleaned per-source tables for inspection
- one canonical hourly market table for downstream features, labels, and backtests

The primary research frequency is `1h`.

## Current Assumptions

- The first source path uses Binance public REST endpoints.
- The default market is `BTCUSDT`, but the code is structured so `ETHUSDT` and other symbols can be added by config.
- The default primary extraction window is `2021-01-01` to `2026-04-07` UTC for the main Binance `BTCUSDT` research dataset.
- The canonical table aligns perpetual bars, spot bars, funding events, and optional open interest onto a shared hourly UTC grid.
- Date-only `end` values in config are treated as inclusive UTC dates and expanded to the next day internally.
- Funding events are sparse; non-event hours are stored with `funding_rate = 0.0` and `funding_event = 0`.

## Source Tables

### `perpetual_bars`

One row per Binance futures kline.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | Candle open timestamp |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | float | Base asset volume |
| `quote_volume` | float | Quote asset volume |
| `trade_count` | float | Number of trades |
| `taker_buy_base_volume` | float | Taker buy base volume |
| `taker_buy_quote_volume` | float | Taker buy quote volume |
| `close_time` | float | Exchange-reported close time in ms |

### `spot_bars`

Same schema as `perpetual_bars`, but from Binance spot klines.

### `funding_rates`

One row per funding settlement event.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | Funding settlement timestamp |
| `funding_rate` | float | Realized funding rate |
| `mark_price` | float | Exchange-reported mark price when available |

### `open_interest`

Optional venue-specific hourly history.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | Observation timestamp |
| `open_interest` | float | Aggregate open interest |
| `open_interest_value` | float | Quote-value open interest |

## Canonical Hourly Dataset

Output file: `hourly_market_data.parquet` and optionally `hourly_market_data.csv`.

| Column | Type | Description |
| --- | --- | --- |
| `timestamp` | datetime64[ns, UTC] | Hourly grid timestamp |
| `symbol` | string | Research symbol, e.g. `BTCUSDT` |
| `venue` | string | Venue identifier, currently `binance` |
| `frequency` | string | Primary research frequency, currently `1h` |
| `perp_open` | float | Perpetual open |
| `perp_high` | float | Perpetual high |
| `perp_low` | float | Perpetual low |
| `perp_close` | float | Perpetual close |
| `perp_volume` | float | Perpetual volume |
| `spot_open` | float | Spot open |
| `spot_high` | float | Spot high |
| `spot_low` | float | Spot low |
| `spot_close` | float | Spot close |
| `spot_volume` | float | Spot volume |
| `funding_rate` | float | Funding rate, zero-filled on non-event hours |
| `funding_event` | int | `1` if a funding event occurred on that hour, else `0` |
| `open_interest` | float nullable | Optional open interest |
| `perp_close_was_missing` | bool | Whether perpetual close was missing before fill |
| `spot_close_was_missing` | bool | Whether spot close was missing before fill |
| `open_interest_was_missing` | bool | Whether open interest was missing before fill |

## Output Folder Layout

```text
data/
+-- raw/
|   `-- binance/
|       `-- btcusdt/
|           `-- 1h/
|               +-- perpetual_bars.parquet
|               +-- spot_bars.parquet
|               +-- funding_rates.parquet
|               `-- open_interest.parquet
+-- interim/
|   `-- binance/
|       `-- btcusdt/
|           `-- 1h/
|               +-- perpetual_bars_clean.parquet
|               +-- spot_bars_clean.parquet
|               +-- funding_rates_clean.parquet
|               `-- open_interest_clean.parquet
`-- processed/
    `-- binance/
        `-- btcusdt/
            `-- 1h/
                +-- hourly_market_data.parquet
                +-- hourly_market_data.csv
                `-- manifest.json
```

Files are omitted when a source is disabled. For example, the default config disables `open_interest`.

## Cleaning Rules

- Normalize all timestamps to UTC.
- Drop duplicate timestamps and keep the last observation from the source extract.
- Sort ascending by timestamp.
- Forward-fill price columns up to the configured maximum gap. The current default is `6` hours for the main Binance sample because the long spot history contains a few rare multi-hour outages.
- Fill missing volume with the configured numeric default, currently `0.0`.
- Fill missing funding rates on non-event hours with the configured value, currently `0.0`.
- Raise an error if aligned `perp_close` or `spot_close` still contains missing values after allowed filling.

## Downstream Reporting

The canonical hourly dataset is the input to the `report-data-quality` command. That command reads `hourly_market_data.parquet` and writes presentation-friendly tables, figures, and a markdown summary under `reports/data_quality/<provider>/<symbol>/<frequency>/`.

## Current Data Gaps

- Index-price history is not yet wired into the first working pipeline.
- Open-interest history is interface-ready but disabled by default.
- Multi-exchange normalization is not implemented yet.
- Long lookback runs may need retry logic and rate-limit handling beyond the current simple REST pagination.