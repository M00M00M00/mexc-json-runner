import json
import os
import subprocess
import sys
import time
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
            r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HTTP_HEADERS)
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


def _binance_get(path: str, params=None, timeout=10, retries=3, sleep=0.25):
    if not path.startswith("/"):
        path = "/" + path

    errors = []
    restricted_hits = 0
    bases = _get_binance_base_urls()

    for base in bases:
        url = f"{base}{path}"
        try:
            return _get_json(url, params=params, timeout=timeout, retries=retries, sleep=sleep)
        except Exception as e:
            msg = str(e)
            errors.append(f"{base}: {msg}")
            if "HTTP 451" in msg:
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
    data = _binance_get("/fapi/v1/time")
    return int(data["serverTime"])


def get_klines(symbol: str, interval: str, bars: int):
    symbol = _normalize_symbol(symbol)
    bars = max(1, int(bars))
    data = _binance_get(
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": bars},
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
    return _binance_get("/fapi/v1/ticker/24hr", params={"symbol": symbol})


def get_book_ticker(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/ticker/bookTicker", params={"symbol": symbol})


def get_premium_index(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/premiumIndex", params={"symbol": symbol})


def get_funding_history(symbol: str, n: int = 24):
    symbol = _normalize_symbol(symbol)
    limit = max(1, min(1000, int(n)))
    data = _binance_get(
        "/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
    )
    if not isinstance(data, list):
        return []
    # Keep latest first for quick consumers.
    return sorted(data, key=lambda x: int(x.get("fundingTime", 0)), reverse=True)[:n]


def get_open_interest(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/openInterest", params={"symbol": symbol})


def get_exchange_info(symbol: str):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/exchangeInfo", params={"symbol": symbol})


def get_depth(symbol: str, limit: int = 20):
    symbol = _normalize_symbol(symbol)
    return _binance_get("/fapi/v1/depth", params={"symbol": symbol, "limit": _safe_depth_limit(limit)})


def get_trades(symbol: str, limit: int = 100):
    symbol = _normalize_symbol(symbol)
    n = max(1, min(1000, int(limit)))
    return _binance_get("/fapi/v1/trades", params={"symbol": symbol, "limit": n})


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


def compute_depth_history_features(depth_snapshots: List[Dict[str, Any]], p: float, levels: int = 20):
    if not depth_snapshots:
        return {"available": False}

    walls = []
    for snap in depth_snapshots:
        bids = (snap.get("bids", []) or [])[:levels]
        asks = (snap.get("asks", []) or [])[:levels]
        near_bids = _near_band_levels(bids, p, 0.0015, "bids")
        near_asks = _near_band_levels(asks, p, 0.0015, "asks")
        bw = _top_wall(near_bids) if near_bids else {"price": None, "size": None}
        aw = _top_wall(near_asks) if near_asks else {"price": None, "size": None}
        walls.append({"t": snap.get("timestamp"), "bidWall": bw, "askWall": aw})

    def _persistence(wall_key: str):
        prices = [w[wall_key]["price"] for w in walls if w.get(wall_key, {}).get("price") is not None]
        sizes = [w[wall_key]["size"] for w in walls if w.get(wall_key, {}).get("size") is not None]
        if len(prices) < 3:
            return {"samples": len(prices), "persistRate": None, "churnRate": None, "sizeCv": None}

        same = 0
        changes = 0
        for i in range(1, len(prices)):
            if prices[i] == prices[i - 1]:
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

    return {
        "available": True,
        "bidWall": _persistence("bidWall"),
        "askWall": _persistence("askWall"),
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
    hist_step_sec: float = 2.0,
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

    ob_feat = compute_orderbook_features(depth, p, band_pct=0.0015, topN=depth_levels) if p else {"available": False}
    ob_hist_feat = compute_depth_history_features(depth_hist, p, levels=depth_levels) if p else {"available": False}
    tr_feat = compute_trade_features(trades[:40])
    oi_feat = compute_oi_features(ticker_hist)
    ctx_feat = compute_context_features(k15, k5)

    features = {
        "P_used": p,
        "orderbook": ob_feat,
        "orderbookHistory": ob_hist_feat,
        "trades": tr_feat,
        "openInterest": oi_feat,
        "context": ctx_feat,
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
        hist_step_sec=2.0,
        enable_poll_history=True,
    )

    with open("latest.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    print("Saved latest.json (BINANCE futures with features)")
