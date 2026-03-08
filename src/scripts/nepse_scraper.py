from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd

URL_TEMPLATE_ENV = "NEPSE_PRICE_VOLUME_URL_TEMPLATE"
DEFAULT_URL_TEMPLATE = "http://0.0.0.0:8000/PriceVolumeHistory?symbol={symbol}"
LOCAL_HOSTS = ("0.0.0.0", "127.0.0.1", "localhost", "host.docker.internal")
LOCAL_BANK_DATA_DIR_ENV = "NEPSE_LOCAL_BANK_DATA_DIR"
DEFAULT_LOCAL_BANK_DATA_DIR = "data/raw"
NEPSE_INDEX_JSON_PATH_ENV = "NEPSE_INDEX_JSON_PATH"
DEFAULT_NEPSE_INDEX_JSON_PATH = "src/scripts/nepse_index.json"
FETCH_ALL_ACTIVE_BANKS_ENV = "NEPSE_FETCH_ALL_ACTIVE_BANKS"
FORCE_ACTIVE_BANK_CONTEXT_ENV = "NEPSE_FORCE_ACTIVE_BANK_CONTEXT"
PREFER_LOCAL_DATA_ENV = "NEPSE_PREFER_LOCAL_DATA"


def scrape_market_data(symbol: str, timeframe: str = "1d", lookback_days: int = 320):
    target_symbol = symbol.strip().upper()
    if not target_symbol:
        raise ValueError("symbol is required.")
    if timeframe != "1d":
        raise ValueError("Only timeframe='1d' is currently supported.")

    symbols_to_fetch = _resolve_symbols_to_fetch(target_symbol)
    histories: dict[str, list[dict[str, Any]]] = {}
    optional_errors: list[str] = []

    for bank_symbol in symbols_to_fetch:
        try:
            raw_history = _fetch_price_volume_history(bank_symbol)
            normalized = _normalize_ohlcv_rows(
                symbol=bank_symbol,
                raw_rows=raw_history,
                lookback_days=lookback_days,
            )
            if normalized:
                histories[bank_symbol] = normalized
        except Exception as exc:  # noqa: BLE001
            if bank_symbol == target_symbol:
                raise RuntimeError(
                    f"Failed to fetch data for requested symbol '{target_symbol}': {exc}"
                ) from exc
            optional_errors.append(f"{bank_symbol}: {exc}")

    if target_symbol not in histories:
        extra = f" Optional fetch errors: {optional_errors}" if optional_errors else ""
        raise RuntimeError(
            f"No OHLCV rows available for requested symbol '{target_symbol}'.{extra}"
        )

    ohlcv_rows: list[dict[str, Any]] = []
    for rows in histories.values():
        ohlcv_rows.extend(rows)
    ohlcv_rows.sort(key=lambda row: (row["bank"], row["date"]))

    nepse_rows = _build_nepse_rows(histories=histories, lookback_days=lookback_days)
    fundamentals = _build_fundamentals(list(histories.keys()))

    return {
        "ohlcv": ohlcv_rows,
        "nepse": nepse_rows,
        "fundamentals": fundamentals,
        "policy_rate": _get_env_float("NEPSE_POLICY_RATE", 4.5),
    }


def list_supported_symbols() -> list[str]:
    configured = Path(os.getenv(LOCAL_BANK_DATA_DIR_ENV, DEFAULT_LOCAL_BANK_DATA_DIR))
    candidates = [configured, Path(DEFAULT_LOCAL_BANK_DATA_DIR)]
    symbols: set[str] = set(_load_active_banks_from_meta())

    seen_paths: set[Path] = set()
    for directory in candidates:
        resolved = directory.resolve()
        if resolved in seen_paths or not resolved.exists() or not resolved.is_dir():
            continue
        seen_paths.add(resolved)
        for csv_path in resolved.glob("*.csv"):
            if csv_path.stem.strip():
                symbols.add(csv_path.stem.strip().upper())

    return sorted(symbols)


def _resolve_symbols_to_fetch(target_symbol: str) -> list[str]:
    symbols = [target_symbol]
    fetch_all_active = _is_truthy(os.getenv(FETCH_ALL_ACTIVE_BANKS_ENV, "false"))
    force_active_context = _is_truthy(os.getenv(FORCE_ACTIVE_BANK_CONTEXT_ENV, "true"))
    if fetch_all_active or force_active_context:
        symbols.extend(_load_active_banks_from_meta())

    extras = os.getenv("NEPSE_EXTRA_SYMBOLS", "")
    if extras:
        symbols.extend([item.strip().upper() for item in extras.split(",") if item.strip()])

    deduped: list[str] = []
    seen: set[str] = set()
    for bank_symbol in symbols:
        normalized = bank_symbol.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _load_active_banks_from_meta() -> list[str]:
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    meta_path = model_dir / "model_meta.json"
    if not meta_path.exists():
        return []
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []

    active_banks = meta.get("active_banks")
    if not isinstance(active_banks, list):
        return []
    return [str(bank).strip().upper() for bank in active_banks if str(bank).strip()]


def _fetch_price_volume_history(symbol: str) -> list[dict[str, Any]]:
    errors: list[str] = []
    prefer_local = _is_truthy(os.getenv(PREFER_LOCAL_DATA_ENV, "true"))

    if prefer_local:
        local_rows = _load_local_price_volume_history(symbol)
        if local_rows:
            return local_rows

    timeout = _get_env_float("NEPSE_API_TIMEOUT_SECONDS", 15.0)

    for url in _candidate_urls(symbol):
        try:
            request = Request(
                url=url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ml-ete-nepse-scraper/1.0",
                },
            )
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                body = response.read().decode("utf-8")
            payload = json.loads(body)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
            continue

        rows = _extract_list_payload(payload)
        if rows is None:
            errors.append(f"{url} -> Unexpected JSON payload shape.")
            continue
        return rows

    if not prefer_local:
        local_rows = _load_local_price_volume_history(symbol)
        if local_rows:
            return local_rows

    raise RuntimeError(" | ".join(errors) if errors else "No endpoint URL resolved.")


def _load_local_price_volume_history(symbol: str) -> list[dict[str, Any]]:
    csv_path = _resolve_local_bank_csv_path(symbol)
    if not csv_path:
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001
        return []
    if df.empty:
        return []

    df.columns = [str(col).strip() for col in df.columns]
    date_column = next(
        (
            col
            for col in ("published_date", "businessDate", "tradeDate", "tradingDate", "date")
            if col in df.columns
        ),
        None,
    )
    if date_column is None:
        return []

    rows: list[dict[str, Any]] = []
    for _, raw_row in df.iterrows():
        business_date = _normalize_date_text(raw_row.get(date_column))
        close = _coerce_optional_float(raw_row.get("close"))
        if not business_date or close is None:
            continue

        row: dict[str, Any] = {
            "businessDate": business_date,
            "openPrice": _coerce_optional_float(raw_row.get("open")),
            "highPrice": _coerce_optional_float(raw_row.get("high")),
            "lowPrice": _coerce_optional_float(raw_row.get("low")),
            "closePrice": close,
            "totalTradedQuantity": _coerce_optional_float(raw_row.get("traded_quantity")),
            "totalTradedValue": _coerce_optional_float(raw_row.get("traded_amount")),
        }
        if row["totalTradedQuantity"] is None:
            row["totalTradedQuantity"] = _coerce_optional_float(raw_row.get("volume"))
        if row["totalTradedValue"] is None:
            row["totalTradedValue"] = _coerce_optional_float(raw_row.get("amount"))
        rows.append(row)

    return rows


def _resolve_local_bank_csv_path(symbol: str) -> Path | None:
    configured = Path(os.getenv(LOCAL_BANK_DATA_DIR_ENV, DEFAULT_LOCAL_BANK_DATA_DIR))
    candidates = [
        configured / f"{symbol}.csv",
        Path(DEFAULT_LOCAL_BANK_DATA_DIR) / f"{symbol}.csv",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _candidate_urls(symbol: str) -> list[str]:
    template = os.getenv(URL_TEMPLATE_ENV, DEFAULT_URL_TEMPLATE).strip()
    if not template:
        raise ValueError(f"{URL_TEMPLATE_ENV} cannot be empty.")

    primary = _render_symbol_url(template=template, symbol=symbol)
    candidates = [primary]

    parsed = urlparse(primary)
    host = (parsed.hostname or "").lower()
    if host in LOCAL_HOSTS:
        for fallback_host in LOCAL_HOSTS:
            if fallback_host == host:
                continue
            alt = _replace_url_host(primary, fallback_host)
            if alt and alt not in candidates:
                candidates.append(alt)

    return candidates


def _render_symbol_url(template: str, symbol: str) -> str:
    encoded_symbol = quote(symbol, safe="")
    if "{symbol}" in template:
        return template.replace("{symbol}", encoded_symbol)

    parsed = urlparse(template)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items["symbol"] = symbol
    return urlunparse(parsed._replace(query=urlencode(query_items)))


def _replace_url_host(url: str, new_host: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.hostname:
        return None

    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{new_host}{port}"
    return urlunparse(parsed._replace(netloc=netloc))


def _extract_list_payload(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        for key in ("data", "result", "items", "history"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]

    return None


def _normalize_ohlcv_rows(
    symbol: str,
    raw_rows: list[dict[str, Any]],
    lookback_days: int,
) -> list[dict[str, Any]]:
    deduped_by_date: dict[str, dict[str, Any]] = {}

    for row in raw_rows:
        dt = _parse_row_date(row)
        close = _pick_float(row, ("closePrice", "close", "closingPrice"))
        if dt is None or close is None:
            continue

        open_price = _pick_float(row, ("openPrice", "open"))
        high_price = _pick_float(row, ("highPrice", "high"), default=close)
        low_price = _pick_float(row, ("lowPrice", "low"), default=close)
        volume = _pick_float(
            row,
            ("totalTradedQuantity", "volume", "totalQty", "traded_quantity"),
            default=0.0,
        )
        amount = _pick_float(
            row,
            ("totalTradedValue", "amount", "turnover", "traded_amount"),
            default=None,
        )

        deduped_by_date[dt.isoformat()] = {
            "date": dt.isoformat(),
            "bank": symbol,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close,
            "volume": volume,
            "amount": amount,
        }

    rows = sorted(deduped_by_date.values(), key=lambda item: item["date"])
    if lookback_days > 0 and len(rows) > lookback_days:
        rows = rows[-lookback_days:]

    previous_close: float | None = None
    normalized: list[dict[str, Any]] = []
    for row in rows:
        close = float(row["close"])
        open_price = row["open"] if row["open"] is not None else previous_close
        open_price = close if open_price is None else float(open_price)
        high = float(row["high"])
        low = float(row["low"])

        # Keep candles valid if source has inconsistent highs/lows.
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        normalized.append(
            {
                "date": row["date"],
                "bank": row["bank"],
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": float(row["volume"]),
                "amount": None if row["amount"] is None else float(row["amount"]),
            }
        )
        previous_close = close

    return normalized


def _build_nepse_rows(
    histories: dict[str, list[dict[str, Any]]],
    lookback_days: int,
) -> list[dict[str, Any]]:
    derived_rows = _build_nepse_rows_from_histories(histories)
    file_rows = _load_nepse_rows_from_json()

    by_date = {row["date"]: float(row["nepse_close"]) for row in derived_rows}
    for row in file_rows:
        by_date[row["date"]] = float(row["nepse_close"])

    nepse_rows = [
        {"date": dt, "nepse_close": nepse_close}
        for dt, nepse_close in sorted(by_date.items(), key=lambda item: item[0])
    ]
    if lookback_days > 0 and len(nepse_rows) > lookback_days:
        nepse_rows = nepse_rows[-lookback_days:]
    return nepse_rows


def _build_nepse_rows_from_histories(
    histories: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_date: dict[str, list[float]] = {}
    for rows in histories.values():
        for row in rows:
            by_date.setdefault(row["date"], []).append(float(row["close"]))

    return [
        {"date": dt, "nepse_close": float(fmean(closes))}
        for dt, closes in sorted(by_date.items(), key=lambda item: item[0])
        if closes
    ]


def _load_nepse_rows_from_json() -> list[dict[str, Any]]:
    path = _resolve_nepse_index_json_path()
    if not path:
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if not isinstance(payload, list):
        return []

    deduped: dict[str, float] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        dt = _parse_row_date(item)
        close = _pick_float(item, ("nepse_close", "close", "closePrice", "closingPrice"))
        if dt is None or close is None:
            continue
        deduped[dt.isoformat()] = float(close)

    return [
        {"date": dt, "nepse_close": close}
        for dt, close in sorted(deduped.items(), key=lambda item: item[0])
    ]


def _resolve_nepse_index_json_path() -> Path | None:
    configured = Path(os.getenv(NEPSE_INDEX_JSON_PATH_ENV, DEFAULT_NEPSE_INDEX_JSON_PATH))
    candidates = [
        configured,
        Path(DEFAULT_NEPSE_INDEX_JSON_PATH),
        Path("nepse_index.json"),
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _build_fundamentals(symbols: list[str]) -> dict[str, dict[str, float | None]]:
    defaults = {
        "car": _get_env_float("NEPSE_DEFAULT_CAR", 12.0),
        "npl": _get_env_float("NEPSE_DEFAULT_NPL", 2.0),
    }

    user_payload = os.getenv("NEPSE_FUNDAMENTALS_JSON", "").strip()
    custom: dict[str, dict[str, float | None]] = {}
    if user_payload:
        try:
            parsed = json.loads(user_payload)
            if isinstance(parsed, dict):
                for bank_symbol, values in parsed.items():
                    if not isinstance(values, dict):
                        continue
                    custom[str(bank_symbol).strip().upper()] = {
                        "car": _coerce_optional_float(values.get("car")),
                        "npl": _coerce_optional_float(values.get("npl")),
                    }
        except json.JSONDecodeError:
            custom = {}

    result: dict[str, dict[str, float | None]] = {}
    for bank_symbol in symbols:
        if bank_symbol in custom:
            car = custom[bank_symbol]["car"]
            npl = custom[bank_symbol]["npl"]
            result[bank_symbol] = {
                "car": defaults["car"] if car is None else car,
                "npl": defaults["npl"] if npl is None else npl,
            }
            continue
        result[bank_symbol] = dict(defaults)

    return result


def _parse_row_date(row: dict[str, Any]) -> date | None:
    raw = (
        row.get("businessDate")
        or row.get("published_date")
        or row.get("date")
        or row.get("tradeDate")
        or row.get("tradingDate")
    )
    if not raw:
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw

    as_text = _normalize_date_text(raw)
    if not as_text:
        return None
    try:
        return date.fromisoformat(as_text)
    except ValueError:
        return None


def _pick_float(
    row: dict[str, Any],
    keys: tuple[str, ...],
    default: float | None = None,
) -> float | None:
    for key in keys:
        if key not in row:
            continue
        value = _coerce_optional_float(row.get(key))
        if value is not None:
            return value
    return default


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    return as_float


def _normalize_date_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    as_text = str(value).strip()
    if not as_text:
        return None
    if "T" in as_text:
        as_text = as_text.split("T", 1)[0]

    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(as_text, pattern).date().isoformat()
        except ValueError:
            continue
    return None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)
