import json
import os
import subprocess
import sys
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests

# Binance USD-M Futures REST base URLs.
# Priority can be overridden by env `BINANCE_BASE_URLS` (comma-separated) or `BINANCE_BASE_URL`.
DEFAULT_BINANCE_BASE_URLS = [
    "https://fapi.binance.com",
    "https://www.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]

DEFAULT_HTTP_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_HTTP_SESSION = requests.Session()


# ---------------------------
# Generic helpers
# ---------------------------
def _to_float(x: Any):
    try:
        return float(x)
    except Exception:
        return None


def _iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_symbol(symbol: str) -> str:
    # Supports legacy user input like BTC_USDT, btc/usdt, BTC-USDT.
    s = "".join(ch for ch in str(symbol).strip().upper() if ch.isalnum())
    if not s:
        raise ValueError("symbol is empty")
    return s


def _interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "3m": 180,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    if interval not in mapping:
        raise ValueError(f"Unsupported interval: {interval}. Choose one of {list(mapping.keys())}")
    return mapping[interval]


def _safe_depth_limit(limit: int) -> int:
    allowed = [5, 10, 20, 50, 100, 500, 1000]
    n = int(limit)
    if n in allowed:
        return n
    for x in allowed:
        if n <= x:
            return x
    return 1000


def _get_binance_base_urls() -> List[str]:
    raw_list = os.getenv("BINANCE_BASE_URLS", "").strip()
    if raw_list:
        candidates = [x.strip().rstrip("/") for x in raw_list.split(",") if x.strip()]
    else:
        one = os.getenv("BINANCE_BASE_URL", "").strip()
        if one:
            candidates = [one.rstrip("/")]
        else:
            candidates = list(DEFAULT_BINANCE_BASE_URLS)

    # De-duplicate while preserving order.
    seen = set()
    out = []
    for url in candidates:
        if url and url not in seen:
            out.append(url)
            seen.add(url)
    return out


# ---------------------------
# HTTP helpers
# ---------------------------
def _http_get(url, params=None, timeout=10, headers=None):
    return _HTTP_SESSION.get(url, params=params, timeout=timeout, headers=headers)


def _curl_get_json(url, params=None, timeout=10):
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--max-time",
        str(int(timeout)),
        "-H",
        "Accept: application/json,text/plain,*/*",
        "-H",
        f"User-Agent: {DEFAULT_HTTP_HEADERS['User-Agent']}",
        "--get",
        url,
    ]
    for k, v in (params or {}).items():
        cmd.extend(["--data-urlencode", f"{k}={v}"])

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"curl failed rc={p.returncode}: {p.stderr.strip()[:180]}")

    body = (p.stdout or "").strip()
    if not body:
        raise RuntimeError("curl returned empty body")
    try:
        return json.loads(body)
    except Exception as e:
        raise RuntimeError(f"curl returned non-JSON: {body[:180]}") from e


def _get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
    last_err = None
    for _ in range(max(1, retries)):
        try:
            r = _http_get(url, params=params, timeout=timeout, headers=DEFAULT_HTTP_HEADERS)
            if r.status_code != 200:
                # Some runners get Cloudflare/challenge style responses where curl may still work.
                if r.status_code in (202, 403, 451):
                    try:
                        return _curl_get_json(url, params=params, timeout=timeout)
                    except Exception as curl_err:
                        last_err = RuntimeError(f"HTTP {r.status_code}: {r.text} | curl fallback failed: {curl_err}")
                        time.sleep(sleep)
                        continue
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text}")
                time.sleep(sleep)
                continue
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err


def _payload_matches_expect(data, expect) -> bool:
    if expect is None:
        return True
    if callable(expect):
        return bool(expect(data))
    if isinstance(expect, str):
        return isinstance(data, dict) and expect in data
    if isinstance(expect, (list, tuple, set)):
        return isinstance(data, dict) and all(k in data for k in expect)
    if isinstance(expect, type):
        return isinstance(data, expect)
    raise ValueError(f"Unsupported expect spec: {expect!r}")


def _binance_get(path: str, params=None, timeout=10, retries=3, sleep=0.25, expect=None):
    if not path.startswith("/"):
        path = "/" + path

    errors = []
    restricted_hits = 0
    bases = _get_binance_base_urls()

    for base in bases:
        url = f"{base}{path}"
        try:
            data = _get_json(url, params=params, timeout=timeout, retries=retries, sleep=sleep)
            if not _payload_matches_expect(data, expect):
                raise RuntimeError(f"Unexpected payload shape: {type(data).__name__} -> {str(data)[:220]}")
            return data
        except Exception as e:
            msg = str(e)
            errors.append(f"{base}: {msg}")
            if "HTTP 451" in msg or "restricted location" in msg.lower():
                restricted_hits += 1
            continue

    hint = ""
    if errors and restricted_hits == len(errors):
        hint = (
            " Binance endpoints rejected this runner location (HTTP 451). "
            "Use a non-restricted self-hosted runner or set BINANCE_BASE_URLS "
            "to a relay/proxy endpoint."
        )
    last = " | ".join(errors[-3:]) if errors else "unknown error"
    raise RuntimeError(f"All Binance base URLs failed.{hint} Last errors: {last}")


# ---------------------------
# Binance endpoints
# ---------------------------
def get_server_time_ms():
    data = _binance_get("/fapi/v1/time", expect="serverTime")
    return int(data["serverTime"])


def get_klines(symbol: str, interval: str, bars: int):
    symbol = _normalize_symbol(symbol)
    bars = max(1, int(bars))
    data = _binance_get(
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": bars},
        expect=list,
    )

    out = []
    for k in data[-bars:]:
        if not isinstance(k, list) or len(k) < 6:
            continue
        out.append(
            {
                "ts": _iso_from_ms(k[0]),
                "o": _to_float(k[1]),
                "h": _to_float(k[2]),
                "l": _to_float(k[3]),
                "c": _to_float(k[4]),
                "v": _to_float(k[5]),
            }
        )
    return out


def get_ticker_24h(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/ticker/24hr", params={"symbol": symbol}, expect=["symbol", "lastPrice"])


def get_book_ticker(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/ticker/bookTicker", params={"symbol": symbol}, expect=["bidPrice", "askPrice"])


def get_premium_index(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/premiumIndex", params={"symbol": symbol}, expect=["markPrice", "indexPrice"])


def get_funding_history(symbol: str, n: int = 24):
    symbol = _normalize_symbol(symbol)
    limit = max(1, min(1000, int(n)))
    data = _binance_get(
        "/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
        expect=list,
    )
    if not isinstance(data, list):
        return []
    # Keep latest first for quick consumers.
    return sorted(data, key=lambda x: int(x.get("fundingTime", 0)), reverse=True)[:n]


def get_open_interest(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/openInterest", params={"symbol": symbol}, expect="openInterest")


def get_exchange_info(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/exchangeInfo", params={"symbol": symbol}, expect="symbols")


def get_depth(symbol: str, limit: int = 20):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/depth", params={"symbol": symbol, "limit": _safe_depth_limit(limit)}, expect=["bids", "asks"])


def get_trades(symbol: str, limit: int = 100):
    symbol = _normalize_symbol(symbol)
    n = max(1, min(1000, int(limit)))
    return _binance_get("/fapi/v1/trades", params={"symbol": symbol, "limit": n}, expect=list)


def _extract_symbol_info(exchange_info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    symbols = exchange_info.get("symbols", []) if isinstance(exchange_info, dict) else []
    for item in symbols:
        if isinstance(item, dict) and item.get("symbol") == symbol:
            return item
    return {}


def build_market_info(contract: Dict[str, Any]) -> Dict[str, Any]:
    filters = {}
    for f in contract.get("filters", []) or []:
        if isinstance(f, dict) and f.get("filterType"):
            filters[f["filterType"]] = f

    price_filter = filters.get("PRICE_FILTER", {})
    lot_filter = filters.get("LOT_SIZE", {})

    tick_size = _to_float(price_filter.get("tickSize"))
    min_price = _to_float(price_filter.get("minPrice"))
    max_price = _to_float(price_filter.get("maxPrice"))

    step_size = _to_float(lot_filter.get("stepSize"))
    min_qty = _to_float(lot_filter.get("minQty"))
    max_qty = _to_float(lot_filter.get("maxQty"))

    return {
        "symbol": contract.get("symbol"),
        "pair": contract.get("pair"),
        "contractType": contract.get("contractType"),
        "status": contract.get("status"),
        "baseAsset": contract.get("baseAsset"),
        "quoteAsset": contract.get("quoteAsset"),
        "marginAsset": contract.get("marginAsset"),
        "onboardDate": contract.get("onboardDate"),
        "tickSize": tick_size,
        "minPrice": min_price,
        "maxPrice": max_price,
        "stepSize": step_size,
        "minQty": min_qty,
        "maxQty": max_qty,
        "pricePrecision": contract.get("pricePrecision"),
        "quantityPrecision": contract.get("quantityPrecision"),
        # Compatibility aliases for existing downstream prompt/agent.
        "priceUnit": tick_size,
        "volUnit": step_size,
        "minVol": min_qty,
        "contractSize": _to_float(contract.get("contractSize")) or 1.0,
        "priceScale": contract.get("pricePrecision"),
        "volScale": contract.get("quantityPrecision"),
        "maxLeverage": _to_float(contract.get("maxLeverage")),
        "filters": contract.get("filters", []),
    }


def build_ticker_snapshot(
    ticker_24h: Dict[str, Any],
    book_ticker: Dict[str, Any],
    premium_index: Dict[str, Any],
    open_interest: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "symbol": ticker_24h.get("symbol") or book_ticker.get("symbol") or premium_index.get("symbol"),
        "lastPrice": _to_float(ticker_24h.get("lastPrice")),
        "bid1": _to_float(book_ticker.get("bidPrice")),
        "ask1": _to_float(book_ticker.get("askPrice")),
        "volume24": _to_float(ticker_24h.get("volume")),
        "amount24": _to_float(ticker_24h.get("quoteVolume")),
        "high24Price": _to_float(ticker_24h.get("highPrice")),
        "lower24Price": _to_float(ticker_24h.get("lowPrice")),
        "riseFallRate": _to_float(ticker_24h.get("priceChangePercent")),
        "indexPrice": _to_float(premium_index.get("indexPrice")),
        "fairPrice": _to_float(premium_index.get("markPrice")),
        "fundingRate": _to_float(premium_index.get("lastFundingRate")),
        "holdVol": _to_float(open_interest.get("openInterest")),
        "nextFundingTime": premium_index.get("nextFundingTime"),
        "timestamp": ticker_24h.get("closeTime") or premium_index.get("time") or open_interest.get("time"),
    }


# ---------------------------
# Polling histories (Δ)
# ---------------------------
def sample_ticker_history(symbol: str, seconds: int = 60, step_sec: float = 2.0):
    symbol = _normalize_symbol(symbol)
    out = []
    end = time.time() + max(1, int(seconds))

    while time.time() < end:
        ticker_24h = get_ticker_24h(symbol)
        book_ticker = get_book_ticker(symbol)
        premium_index = get_premium_index(symbol)
        open_interest = get_open_interest(symbol)
        tk = build_ticker_snapshot(ticker_24h, book_ticker, premium_index, open_interest)

        now_ms = int(premium_index.get("time") or open_interest.get("time") or get_server_time_ms())
        out.append(
            {
                "t": now_ms,
                "lastPrice": tk.get("lastPrice"),
                "bid1": tk.get("bid1"),
                "ask1": tk.get("ask1"),
                "holdVol": tk.get("holdVol"),
                "fundingRate": tk.get("fundingRate"),
                "indexPrice": tk.get("indexPrice"),
                "fairPrice": tk.get("fairPrice"),
            }
        )
        time.sleep(step_sec)
    return out


def sample_depth_history(symbol: str, seconds: int = 60, step_sec: float = 2.0, levels: int = 20):
    symbol = _normalize_symbol(symbol)
    out = []
    end = time.time() + max(1, int(seconds))

    while time.time() < end:
        d = get_depth(symbol, limit=levels)
        out.append(
            {
                "timestamp": d.get("E") or d.get("T") or get_server_time_ms(),
                "version": d.get("lastUpdateId"),
                "bids": (d.get("bids", []) or [])[:levels],
                "asks": (d.get("asks", []) or [])[:levels],
            }
        )
        time.sleep(step_sec)
    return out


# ---------------------------
# Feature engineering helpers
# ---------------------------
def _sum_sizes(levels: List[List[Any]]) -> float:
    s = 0.0
    for lv in levels or []:
        if not isinstance(lv, list) or len(lv) < 2:
            continue
        sz = _to_float(lv[1])
        if sz is not None:
            s += sz
    return s


def _best_levels(orderbook: Dict[str, Any]) -> Tuple[float, float]:
    bids = orderbook.get("bids", []) or []
    asks = orderbook.get("asks", []) or []
    best_bid = _to_float(bids[0][0]) if bids and isinstance(bids[0], list) and len(bids[0]) >= 2 else None
    best_ask = _to_float(asks[0][0]) if asks and isinstance(asks[0], list) and len(asks[0]) >= 2 else None
    return best_bid, best_ask


def _near_band_levels(levels: List[List[Any]], p: float, band_pct: float, side: str) -> List[List[Any]]:
    out = []
    if p is None:
        return out
    lo = p * (1 - band_pct)
    hi = p * (1 + band_pct)
    for lv in levels or []:
        if not isinstance(lv, list) or len(lv) < 2:
            continue
        px = _to_float(lv[0])
        sz = _to_float(lv[1])
        if px is None or sz is None:
            continue
        if lo <= px <= hi:
            out.append([px, sz])
    return out


def _top_wall(levels: List[List[Any]]) -> Dict[str, Any]:
    best = {"price": None, "size": None}
    max_sz = -1.0
    for px, sz in levels:
        if sz > max_sz:
            max_sz = sz
            best = {"price": px, "size": sz}
    return best


def _price_within_tick_tolerance(prev_px: float, cur_px: float, tick_size: float, tick_tolerance_ticks: int) -> bool:
    if prev_px is None or cur_px is None:
        return False
    if tick_size is not None and tick_size > 0:
        tol = tick_size * max(0, int(tick_tolerance_ticks))
    else:
        tol = 1e-9
    return abs(cur_px - prev_px) <= (tol + 1e-12)


def _snapshot_mid_price(bids: List[List[Any]], asks: List[List[Any]], fallback_p: float):
    best_bid = _to_float(bids[0][0]) if bids and isinstance(bids[0], list) and len(bids[0]) >= 2 else None
    best_ask = _to_float(asks[0][0]) if asks and isinstance(asks[0], list) and len(asks[0]) >= 2 else None
    if best_bid is not None and best_ask is not None:
        return (best_bid + best_ask) / 2.0
    return best_bid if best_bid is not None else (best_ask if best_ask is not None else fallback_p)


def quantize_price(px: float, tick: float, mode: str = "round"):
    if px is None:
        return None
    tick_f = _to_float(tick)
    if tick_f is None or tick_f <= 0:
        return _to_float(px)

    p = Decimal(str(px))
    t = Decimal(str(tick_f))
    q = p / t

    if mode == "floor":
        q_int = q.to_integral_value(rounding=ROUND_FLOOR)
    elif mode == "ceil":
        q_int = q.to_integral_value(rounding=ROUND_CEILING)
    else:
        q_int = q.to_integral_value(rounding=ROUND_HALF_UP)

    return float(q_int * t)


def _align_price_to_tick(price: float, tick_size: float, mode: str = "nearest"):
    normalized_mode = "round" if mode in ("nearest", "round") else mode
    return quantize_price(price, tick_size, normalized_mode)


def build_tick_aligned_levels(entry_price: float, tick_size: float):
    entry = _align_price_to_tick(entry_price, tick_size, mode="nearest")
    if entry is None:
        return {"available": False, "tickSize": tick_size}

    long_stop_raw = entry * 0.99
    long_take_raw = entry * 1.02
    short_stop_raw = entry * 1.01
    short_take_raw = entry * 0.98

    return {
        "available": True,
        "tickSize": tick_size,
        "long": {
            "entry": entry,
            "stop1pct": _align_price_to_tick(long_stop_raw, tick_size, mode="floor"),
            "take2pct": _align_price_to_tick(long_take_raw, tick_size, mode="ceil"),
        },
        "short": {
            "entry": entry,
            "stop1pct": _align_price_to_tick(short_stop_raw, tick_size, mode="ceil"),
            "take2pct": _align_price_to_tick(short_take_raw, tick_size, mode="floor"),
        },
    }


def _valid_ohlc_rows(klines: List[Dict[str, Any]], lookback: int) -> List[Dict[str, float]]:
    rows = []
    tail = (klines or [])[-max(1, int(lookback)) :]
    for row in tail:
        h = _to_float(row.get("h"))
        l = _to_float(row.get("l"))
        c = _to_float(row.get("c"))
        if h is None or l is None or c is None:
            continue
        rows.append({"h": h, "l": l, "c": c})
    return rows


def _pivot_high_low_values(rows: List[Dict[str, float]], span: int = 2) -> Tuple[List[float], List[float]]:
    lows = []
    highs = []
    n = len(rows)
    span = max(1, int(span))
    if n < (span * 2 + 1):
        return lows, highs

    for i in range(span, n - span):
        low_window = [rows[j]["l"] for j in range(i - span, i + span + 1)]
        high_window = [rows[j]["h"] for j in range(i - span, i + span + 1)]
        cur_low = rows[i]["l"]
        cur_high = rows[i]["h"]

        if cur_low <= min(low_window):
            lows.append(cur_low)
        if cur_high >= max(high_window):
            highs.append(cur_high)

    return lows, highs


def _latest_swing_levels_5m(klines_5m: List[Dict[str, Any]], lookback: int = 60) -> Tuple[float, float]:
    rows = _valid_ohlc_rows(klines_5m, lookback=lookback)
    pivot_lows, pivot_highs = _pivot_high_low_values(rows, span=2)

    recent_rows = rows[-20:]
    fallback_low = min((x["l"] for x in recent_rows), default=None)
    fallback_high = max((x["h"] for x in recent_rows), default=None)

    swing_low = pivot_lows[-1] if pivot_lows else fallback_low
    swing_high = pivot_highs[-1] if pivot_highs else fallback_high
    return swing_low, swing_high


def _atr_from_klines(klines: List[Dict[str, Any]], period: int = 14, lookback: int = 60):
    period = max(1, int(period))
    rows = _valid_ohlc_rows(klines, lookback=lookback)
    if len(rows) < period:
        return None

    tr_values = []
    prev_close = None
    for row in rows:
        h = row["h"]
        l = row["l"]
        c = row["c"]
        if prev_close is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        tr_values.append(tr)
        prev_close = c

    if len(tr_values) < period:
        return None

    atr = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr = ((atr * (period - 1)) + tr) / period
    return atr


def compute_volatility_features(klines_1m: List[Dict[str, Any]], klines_5m: List[Dict[str, Any]], p: float):
    atr1m = _atr_from_klines(klines_1m, period=14, lookback=60)
    atr5m = _atr_from_klines(klines_5m, period=14, lookback=60)

    atr1m_pct = (atr1m / p * 100.0) if (atr1m is not None and p not in (None, 0)) else None
    atr5m_pct = (atr5m / p * 100.0) if (atr5m is not None and p not in (None, 0)) else None

    return {
        "available": atr1m is not None or atr5m is not None,
        "atr1m": atr1m,
        "atr1mPct": atr1m_pct,
        "atr5m": atr5m,
        "atr5mPct": atr5m_pct,
    }


def compute_micro_trend_features(klines_1m: List[Dict[str, Any]], lookback: int = 60):
    rows = _valid_ohlc_rows(klines_1m, lookback=lookback)
    swing_lows, swing_highs = _pivot_high_low_values(rows, span=2)

    trend = None
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        has_higher_high = swing_highs[-1] > swing_highs[-2]
        has_higher_low = swing_lows[-1] > swing_lows[-2]
        has_lower_high = swing_highs[-1] < swing_highs[-2]
        has_lower_low = swing_lows[-1] < swing_lows[-2]

        if has_higher_high and has_higher_low:
            trend = "UP"
        elif has_lower_high and has_lower_low:
            trend = "DOWN"
        else:
            trend = "RANGE"
    elif len(rows) >= 5:
        trend = "RANGE"

    return {
        "available": len(rows) >= 5,
        "lookback": lookback,
        "trend": trend,
        "lastSwingHigh": swing_highs[-1] if swing_highs else None,
        "lastSwingLow": swing_lows[-1] if swing_lows else None,
    }


def _entry_maker_safe(side: str, entry: float, bid: float, ask: float, tick_size: float) -> bool:
    if entry is None:
        return False

    tick = _to_float(tick_size)
    if side == "long":
        if ask is None:
            return False
        if tick is not None and tick > 0:
            return entry <= (ask - tick + 1e-12)
        return entry < ask

    if bid is None:
        return False
    if tick is not None and tick > 0:
        return entry >= (bid + tick - 1e-12)
    return entry > bid


def _candidate_risk_levels(side: str, entry: float, tick_size: float) -> Tuple[float, float]:
    if entry is None:
        return None, None
    if side == "long":
        stop = quantize_price(entry * 0.99, tick_size, mode="floor")
        take = quantize_price(entry * 1.02, tick_size, mode="ceil")
    else:
        stop = quantize_price(entry * 1.01, tick_size, mode="ceil")
        take = quantize_price(entry * 0.98, tick_size, mode="floor")
    return stop, take


def _build_level_candidate(
    name: str,
    side: str,
    entry: float,
    p: float,
    bid: float,
    ask: float,
    tick_size: float,
) -> Dict[str, Any]:
    if entry is None:
        return {}

    within_p08 = False
    if p not in (None, 0):
        within_p08 = abs(entry - p) / p <= 0.008

    stop, take = _candidate_risk_levels(side, entry, tick_size)
    return {
        "name": name,
        "entry": entry,
        "entryMakerSafe": _entry_maker_safe(side, entry, bid, ask, tick_size),
        "withinP08": within_p08,
        "stop1pct": stop,
        "take2pct": take,
    }


def compute_level_candidates(
    p: float,
    bid: float,
    ask: float,
    tick_size: float,
    orderbook_feat: Dict[str, Any],
    context_feat: Dict[str, Any],
    klines_5m: List[Dict[str, Any]],
):
    tick = _to_float(tick_size)
    near_band = (orderbook_feat or {}).get("nearBand", {}) if isinstance(orderbook_feat, dict) else {}
    range15 = (context_feat or {}).get("range15m", {}) if isinstance(context_feat, dict) else {}

    top_bid_wall_px = _to_float((near_band.get("topBidWall") or {}).get("price")) if isinstance(near_band, dict) else None
    top_ask_wall_px = _to_float((near_band.get("topAskWall") or {}).get("price")) if isinstance(near_band, dict) else None
    range_lo = _to_float(range15.get("lo")) if isinstance(range15, dict) else None
    range_hi = _to_float(range15.get("hi")) if isinstance(range15, dict) else None

    swing_low, swing_high = _latest_swing_levels_5m(klines_5m, lookback=60)

    long_candidates = []
    short_candidates = []

    if top_bid_wall_px is not None:
        raw = top_bid_wall_px + (tick if tick is not None and tick > 0 else 0.0)
        entry = quantize_price(raw, tick, mode="round")
        cand = _build_level_candidate("nearBand.topBidWall+1tick", "long", entry, p, bid, ask, tick)
        if cand:
            long_candidates.append(cand)

    if swing_low is not None:
        entry = quantize_price(swing_low, tick, mode="floor")
        cand = _build_level_candidate("swingLow5m", "long", entry, p, bid, ask, tick)
        if cand:
            long_candidates.append(cand)

    if range_lo is not None:
        raw = range_lo * 1.002
        entry = quantize_price(raw, tick, mode="floor")
        cand = _build_level_candidate("range15m_lo+0.2%", "long", entry, p, bid, ask, tick)
        if cand:
            long_candidates.append(cand)

    if top_ask_wall_px is not None:
        raw = top_ask_wall_px - (tick if tick is not None and tick > 0 else 0.0)
        entry = quantize_price(raw, tick, mode="round")
        cand = _build_level_candidate("nearBand.topAskWall-1tick", "short", entry, p, bid, ask, tick)
        if cand:
            short_candidates.append(cand)

    if swing_high is not None:
        entry = quantize_price(swing_high, tick, mode="ceil")
        cand = _build_level_candidate("swingHigh5m", "short", entry, p, bid, ask, tick)
        if cand:
            short_candidates.append(cand)

    if range_hi is not None:
        raw = range_hi * 0.998
        entry = quantize_price(raw, tick, mode="ceil")
        cand = _build_level_candidate("range15m_hi-0.2%", "short", entry, p, bid, ask, tick)
        if cand:
            short_candidates.append(cand)

    return {
        "available": bool(long_candidates or short_candidates),
        "tickSize": tick,
        "P": p,
        "bid": bid,
        "ask": ask,
        "long": long_candidates,
        "short": short_candidates,
    }


def _trade_sort_key(tr: Dict[str, Any]) -> int:
    for key in ("time", "T", "t", "timestamp"):
        v = _to_float(tr.get(key))
        if v is not None:
            return int(v)
    return 0


def _latest_trades_window(trades: List[Dict[str, Any]], n: int = 40) -> List[Dict[str, Any]]:
    if not trades:
        return []
    return sorted(trades, key=_trade_sort_key, reverse=True)[: max(1, int(n))]


def compute_orderbook_features(orderbook: Dict[str, Any], p: float, band_pct: float = 0.0015, topN: int = 20):
    bids = (orderbook.get("bids", []) or [])[:topN]
    asks = (orderbook.get("asks", []) or [])[:topN]

    bid_sum = _sum_sizes(bids)
    ask_sum = _sum_sizes(asks)
    ratio = (bid_sum / ask_sum) if ask_sum > 0 else None

    best_bid, best_ask = _best_levels(orderbook)
    spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
    spread_bps = (spread / p * 10000.0) if (spread is not None and p) else None

    near_bids = _near_band_levels(bids, p, band_pct, "bids")
    near_asks = _near_band_levels(asks, p, band_pct, "asks")

    top_bid_wall = _top_wall(near_bids) if near_bids else {"price": None, "size": None}
    top_ask_wall = _top_wall(near_asks) if near_asks else {"price": None, "size": None}

    return {
        "topN": topN,
        "bandPct": band_pct,
        "bidSumTopN": bid_sum,
        "askSumTopN": ask_sum,
        "bidAskRatioTopN": ratio,
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "spread": spread,
        "spreadBps": spread_bps,
        "nearBand": {
            "bidCount": len(near_bids),
            "askCount": len(near_asks),
            "topBidWall": top_bid_wall,
            "topAskWall": top_ask_wall,
        },
    }


def compute_depth_history_features(
    depth_snapshots: List[Dict[str, Any]],
    p: float,
    levels: int = 20,
    tick_size: float = None,
    tick_tolerance_ticks: int = 2,
    band_pct: float = 0.0015,
    min_samples: int = 6,
):
    if not depth_snapshots:
        return {"available": False}

    walls = []
    for snap in depth_snapshots:
        bids = (snap.get("bids", []) or [])[:levels]
        asks = (snap.get("asks", []) or [])[:levels]
        snap_mid = _snapshot_mid_price(bids, asks, p)
        near_bids = _near_band_levels(bids, snap_mid, band_pct, "bids")
        near_asks = _near_band_levels(asks, snap_mid, band_pct, "asks")
        bw = _top_wall(near_bids) if near_bids else {"price": None, "size": None}
        aw = _top_wall(near_asks) if near_asks else {"price": None, "size": None}
        walls.append({"t": snap.get("timestamp"), "midPrice": snap_mid, "bidWall": bw, "askWall": aw})

    def _persistence(wall_key: str):
        prices = [w[wall_key]["price"] for w in walls if w.get(wall_key, {}).get("price") is not None]
        sizes = [w[wall_key]["size"] for w in walls if w.get(wall_key, {}).get("size") is not None]
        if len(prices) < max(1, int(min_samples)):
            return {"samples": len(prices), "persistRate": None, "churnRate": None, "sizeCv": None}

        same = 0
        changes = 0
        for i in range(1, len(prices)):
            if _price_within_tick_tolerance(prices[i - 1], prices[i], tick_size, tick_tolerance_ticks):
                same += 1
            else:
                changes += 1

        persist_rate = same / (len(prices) - 1)
        churn_rate = changes / (len(prices) - 1)

        mean_sz = sum(sizes) / len(sizes) if sizes else 0.0
        if mean_sz > 0:
            var = sum((s - mean_sz) ** 2 for s in sizes) / len(sizes)
            std = var ** 0.5
            cv = std / mean_sz
        else:
            cv = None

        return {"samples": len(prices), "persistRate": persist_rate, "churnRate": churn_rate, "sizeCv": cv}

    bid_stats = _persistence("bidWall")
    ask_stats = _persistence("askWall")

    return {
        "available": True,
        "tickSizeUsed": tick_size,
        "tickToleranceTicks": tick_tolerance_ticks,
        "bandPct": band_pct,
        "minSamplesForPersist": min_samples,
        "bidWall": bid_stats,
        "askWall": ask_stats,
        # Compatibility aliases for downstream prompts expecting top* keys.
        "topBidWall": dict(bid_stats),
        "topAskWall": dict(ask_stats),
    }


def compute_trade_features(trades: List[Dict[str, Any]]):
    if not trades:
        return {"available": False}

    buy_vol = 0.0
    sell_vol = 0.0
    buy_cnt = 0
    sell_cnt = 0
    max_trade = {"price": None, "vol": None, "aggressorSide": None}

    for tr in trades:
        px = _to_float(tr.get("price") or tr.get("p"))
        v = _to_float(tr.get("qty") or tr.get("v") or tr.get("vol") or tr.get("volume"))
        if v is None:
            continue

        aggressor_side = None
        if "isBuyerMaker" in tr:
            aggressor_side = "SELL" if bool(tr.get("isBuyerMaker")) else "BUY"
        else:
            # Legacy fallback (MEXC style): T: 1=buy, 2=sell
            t = tr.get("T")
            if t == 1:
                aggressor_side = "BUY"
            elif t == 2:
                aggressor_side = "SELL"

        if max_trade["vol"] is None or v > max_trade["vol"]:
            max_trade = {"price": px, "vol": v, "aggressorSide": aggressor_side}

        if aggressor_side == "BUY":
            buy_vol += v
            buy_cnt += 1
        elif aggressor_side == "SELL":
            sell_vol += v
            sell_cnt += 1

    total_vol = buy_vol + sell_vol
    buy_ratio = (buy_vol / total_vol) if total_vol > 0 else None
    sell_ratio = (sell_vol / total_vol) if total_vol > 0 else None

    return {
        "available": True,
        "buyVol": buy_vol,
        "sellVol": sell_vol,
        "buyCnt": buy_cnt,
        "sellCnt": sell_cnt,
        "buyVolRatio": buy_ratio,
        "sellVolRatio": sell_ratio,
        "maxTrade": max_trade,
    }


def compute_oi_features(ticker_hist: List[Dict[str, Any]]):
    if not ticker_hist or len(ticker_hist) < 2:
        return {"available": False}

    def _series(key):
        vals = []
        for row in ticker_hist:
            v = _to_float(row.get(key))
            if v is not None:
                vals.append(v)
        return vals

    oi = _series("holdVol")
    last = _series("lastPrice")
    if len(oi) < 2 or len(last) < 2:
        return {"available": False}

    oi_change = oi[-1] - oi[0]
    oi_change_pct = (oi_change / oi[0] * 100.0) if oi[0] != 0 else None
    px_change = last[-1] - last[0]
    px_change_pct = (px_change / last[0] * 100.0) if last[0] != 0 else None

    chase_flag = None
    if oi_change_pct is not None and px_change_pct is not None:
        chase_flag = oi_change_pct > 0.3 and abs(px_change_pct) > 0.25

    return {
        "available": True,
        "windowSamples": len(ticker_hist),
        "oiStart": oi[0],
        "oiEnd": oi[-1],
        "oiChange": oi_change,
        "oiChangePct": oi_change_pct,
        "pxStart": last[0],
        "pxEnd": last[-1],
        "pxChangePct": px_change_pct,
        "leverageChaseFlag": chase_flag,
    }


def compute_context_features(klines_15m: List[Dict[str, Any]], klines_5m: List[Dict[str, Any]]):
    def _hlc(ks):
        highs = [_to_float(x.get("h")) for x in ks if _to_float(x.get("h")) is not None]
        lows = [_to_float(x.get("l")) for x in ks if _to_float(x.get("l")) is not None]
        closes = [_to_float(x.get("c")) for x in ks if _to_float(x.get("c")) is not None]
        return highs, lows, closes

    f = {"available": False}
    if not klines_15m or not klines_5m:
        return f

    h15, l15, c15 = _hlc(klines_15m[-20:])
    h5, l5, c5 = _hlc(klines_5m[-60:])

    if not h15 or not l15 or not c15 or not h5 or not l5 or not c5:
        return f

    r15_hi, r15_lo = max(h15), min(l15)
    last15 = c15[-1]
    pos15 = (last15 - r15_lo) / (r15_hi - r15_lo) if (r15_hi - r15_lo) > 0 else None

    r5_hi, r5_lo = max(h5), min(l5)
    last5 = c5[-1]
    pos5 = (last5 - r5_lo) / (r5_hi - r5_lo) if (r5_hi - r5_lo) > 0 else None

    mean15 = sum(c15) / len(c15)
    mean5 = sum(c5) / len(c5)

    return {
        "available": True,
        "range15m": {"hi": r15_hi, "lo": r15_lo, "pos01": pos15, "lastClose": last15, "meanClose": mean15},
        "range5m": {"hi": r5_hi, "lo": r5_lo, "pos01": pos5, "lastClose": last5, "meanClose": mean5},
    }


# ---------------------------
# Bundle builder
# ---------------------------
def make_latest(
    symbol: str,
    bars_1m: int = 240,
    bars_5m: int = 120,
    bars_15m: int = 80,
    depth_levels: int = 20,
    trades_n: int = 100,
    depth_commits_n: int = 60,  # kept for API compatibility, unused on Binance REST
    hist_seconds: int = 60,
    hist_step_sec: float = 3.0,
    enable_poll_history: bool = True,
):
    del depth_commits_n

    symbol = _normalize_symbol(symbol)
    now_ms = get_server_time_ms()

    ticker_24h = get_ticker_24h(symbol)
    book_ticker = get_book_ticker(symbol)
    premium_index = get_premium_index(symbol)
    funding_hist = get_funding_history(symbol, n=24)
    open_interest = get_open_interest(symbol)

    exchange_info = get_exchange_info(symbol)
    contract = _extract_symbol_info(exchange_info, symbol)

    depth = get_depth(symbol, limit=depth_levels)
    trades = get_trades(symbol, limit=trades_n)

    k1 = get_klines(symbol, "1m", bars_1m)
    k5 = get_klines(symbol, "5m", bars_5m)
    k15 = get_klines(symbol, "15m", bars_15m)

    ticker = build_ticker_snapshot(ticker_24h, book_ticker, premium_index, open_interest)

    ticker_hist = []
    depth_hist = []
    if enable_poll_history and hist_seconds > 0:
        ticker_hist = sample_ticker_history(symbol, seconds=hist_seconds, step_sec=hist_step_sec)
        depth_hist = sample_depth_history(symbol, seconds=hist_seconds, step_sec=hist_step_sec, levels=depth_levels)

    market_info = build_market_info(contract)

    metrics = {
        "openInterestNow": ticker.get("holdVol"),
        "oiHistory": ticker_hist,
        "fundingRateNow": ticker.get("fundingRate"),
        "fundingRateHistory": funding_hist,
    }

    p = ticker.get("lastPrice") or ticker.get("fairPrice") or ticker.get("indexPrice")
    if p is None and k1:
        p = _to_float(k1[-1].get("c"))
    bid = ticker.get("bid1") or _to_float(book_ticker.get("bidPrice"))
    ask = ticker.get("ask1") or _to_float(book_ticker.get("askPrice"))

    ob_feat = compute_orderbook_features(depth, p, band_pct=0.0015, topN=depth_levels) if p else {"available": False}
    if bid is None and isinstance(ob_feat, dict):
        bid = _to_float(ob_feat.get("bestBid"))
    if ask is None and isinstance(ob_feat, dict):
        ask = _to_float(ob_feat.get("bestAsk"))

    tick_size = _to_float(market_info.get("tickSize")) or _to_float(market_info.get("priceUnit"))
    ob_hist_feat = (
        compute_depth_history_features(
            depth_hist,
            p,
            levels=depth_levels,
            tick_size=tick_size,
            tick_tolerance_ticks=2,
            band_pct=0.0015,
            min_samples=6,
        )
        if p
        else {"available": False}
    )
    tr_feat = compute_trade_features(_latest_trades_window(trades, n=40))
    rr_levels = build_tick_aligned_levels(p, tick_size)
    oi_feat = compute_oi_features(ticker_hist)
    ctx_feat = compute_context_features(k15, k5)
    vol_feat = compute_volatility_features(k1, k5, p)
    micro_trend_feat = compute_micro_trend_features(k1, lookback=60)
    level_candidates = compute_level_candidates(
        p=p,
        bid=bid,
        ask=ask,
        tick_size=tick_size,
        orderbook_feat=ob_feat,
        context_feat=ctx_feat,
        klines_5m=k5,
    )

    features = {
        "P_used": p,
        "orderbook": ob_feat,
        "orderbookHistory": ob_hist_feat,
        "trades": tr_feat,
        "openInterest": oi_feat,
        "context": ctx_feat,
        "riskLevels": rr_levels,
        "levelCandidates": level_candidates,
        "volatility": vol_feat,
        "microTrend": micro_trend_feat,
    }

    bundle = {
        "meta": {
            "exchange": "BINANCE",
            "market": "USD_M_FUTURES",
            "symbol": symbol,
            "generated_at": _iso_from_ms(now_ms),
        },
        "marketInfo": market_info,
        "price": {
            "ticker": ticker,
            "ticker24h": ticker_24h,
            "bookTicker": book_ticker,
            "premiumIndex": premium_index,
            "openInterest": open_interest,
        },
        "metrics": metrics,
        "orderbook": depth,
        "orderbookHistory": {
            "depthCommits": [],
            "polled": {
                "seconds": hist_seconds if enable_poll_history else 0,
                "stepSec": hist_step_sec if enable_poll_history else 0,
                "levels": depth_levels,
                "snapshots": depth_hist,
            },
            "note": "Binance REST does not provide a depth commits endpoint. Use websocket for granular deltas.",
        },
        "trades": trades,
        "klines": {
            "1m": k1,
            "5m": k5,
            "15m": k15,
        },
        "riskLevels": rr_levels,
        "features": features,
    }

    return bundle


if __name__ == "__main__":
    # Usage:
    #   python collect_mexc_futures_latest.py BTCUSDT
    #   python collect_mexc_futures_latest.py BTCUSDT 240 120 80 60
    # args:
    #   symbol bars_1m bars_5m bars_15m hist_seconds
    symbol = _normalize_symbol(sys.argv[1]) if len(sys.argv) > 1 else "BTCUSDT"
    bars_1m = int(sys.argv[2]) if len(sys.argv) > 2 else 240
    bars_5m = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    bars_15m = int(sys.argv[4]) if len(sys.argv) > 4 else 80
    hist_seconds = int(sys.argv[5]) if len(sys.argv) > 5 else 60

    bundle = make_latest(
        symbol=symbol,
        bars_1m=bars_1m,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        depth_levels=20,
        trades_n=100,
        depth_commits_n=60,
        hist_seconds=hist_seconds,
        hist_step_sec=3.0,
        enable_poll_history=True,
    )

    with open("latest.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    print("Saved latest.json (BINANCE futures with features)")
