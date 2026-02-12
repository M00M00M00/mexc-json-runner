import json
import time
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd

BASE = "https://api.mexc.com"


# ---------------------------
# HTTP helpers
# ---------------------------
def _get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
    last_err = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text}")
                time.sleep(sleep)
                continue
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(sleep)
    raise last_err


def _get_data(url, params=None):
    data = _get_json(url, params=params)
    if not data.get("success", False):
        raise RuntimeError(f"API failed: {data}")
    return data["data"]


def get_server_time_ms():
    return int(_get_data(f"{BASE}/api/v1/contract/ping"))


# ---------------------------
# Market endpoints
# ---------------------------
def _interval_to_seconds(interval: str) -> int:
    m = {
        "Min1": 60,
        "Min5": 300,
        "Min15": 900,
        "Min30": 1800,
        "Min60": 3600,
        "Hour4": 14400,
        "Day1": 86400,
    }
    if interval not in m:
        raise ValueError(f"Unsupported interval: {interval}. Choose one of {list(m.keys())}")
    return m[interval]


def get_klines(symbol: str, interval: str, bars: int):
    server_ms = get_server_time_ms()
    end_sec = server_ms // 1000
    step = _interval_to_seconds(interval)
    start_sec = end_sec - (bars * step)

    url = f"{BASE}/api/v1/contract/kline/{symbol}"
    params = {"interval": interval, "start": start_sec, "end": end_sec}
    resp = _get_json(url, params=params)
    if not resp.get("success", False):
        raise RuntimeError(f"Kline failed: {resp}")
    k = resp["data"]

    df = pd.DataFrame({
        "ts": pd.to_datetime(k["time"], unit="s", utc=True),
        "o": k.get("open", []),
        "h": k.get("high", []),
        "l": k.get("low", []),
        "c": k.get("close", []),
        "v": k.get("vol", []),
    })

    for col in ["o", "h", "l", "c", "v"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) > bars:
        df = df.iloc[-bars:].reset_index(drop=True)

    df["ts"] = df["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.to_dict(orient="records")


def get_ticker(symbol: str):
    data = _get_data(f"{BASE}/api/v1/contract/ticker", params={"symbol": symbol})
    if isinstance(data, list):
        for it in data:
            if it.get("symbol") == symbol:
                return it
        return {}
    if isinstance(data, dict) and data.get("symbol") == symbol:
        return data
    return data if isinstance(data, dict) else {}


def get_funding(symbol: str):
    return _get_data(f"{BASE}/api/v1/contract/funding_rate/{symbol}")


def get_funding_history(symbol: str, n: int = 24):
    page_size = max(20, min(1000, n))
    data = _get_data(
        f"{BASE}/api/v1/contract/funding_rate/history",
        params={"symbol": symbol, "page_num": 1, "page_size": page_size},
    )
    if isinstance(data, dict) and "resultList" in data and isinstance(data["resultList"], list):
        data["resultList"] = data["resultList"][:n]
    return data


def get_index(symbol: str):
    return _get_data(f"{BASE}/api/v1/contract/index_price/{symbol}")


def get_fair(symbol: str):
    return _get_data(f"{BASE}/api/v1/contract/fair_price/{symbol}")


def get_depth(symbol: str, limit: int = 20):
    return _get_data(f"{BASE}/api/v1/contract/depth/{symbol}", params={"limit": int(limit)})


def get_depth_commits(symbol: str, n: int = 60):
    return _get_data(f"{BASE}/api/v1/contract/depth_commits/{symbol}/{int(n)}")


def get_trades(symbol: str, limit: int = 100):
    return _get_data(f"{BASE}/api/v1/contract/deals/{symbol}", params={"limit": int(min(limit, 100))})


def get_contract_info(symbol: str):
    data = _get_data(f"{BASE}/api/v1/contract/detail", params={"symbol": symbol})
    if isinstance(data, dict) and data.get("symbol") == symbol:
        return data
    if isinstance(data, list):
        for it in data:
            if it.get("symbol") == symbol:
                return it
    return data if isinstance(data, dict) else {}


# ---------------------------
# Polling histories (Δ)
# ---------------------------
def sample_ticker_history(symbol: str, seconds: int = 60, step_sec: float = 2.0):
    out = []
    end = time.time() + max(1, seconds)
    while time.time() < end:
        tk = get_ticker(symbol)
        now_ms = get_server_time_ms()
        out.append({
            "t": int(now_ms),
            "lastPrice": tk.get("lastPrice"),
            "bid1": tk.get("bid1"),
            "ask1": tk.get("ask1"),
            "holdVol": tk.get("holdVol"),
            "fundingRate": tk.get("fundingRate"),
            "indexPrice": tk.get("indexPrice"),
            "fairPrice": tk.get("fairPrice"),
        })
        time.sleep(step_sec)
    return out


def sample_depth_history(symbol: str, seconds: int = 60, step_sec: float = 2.0, levels: int = 20):
    out = []
    end = time.time() + max(1, seconds)
    while time.time() < end:
        d = get_depth(symbol, limit=levels)
        out.append({
            "timestamp": d.get("timestamp"),
            "version": d.get("version"),
            "bids": d.get("bids", [])[:levels],
            "asks": d.get("asks", [])[:levels],
        })
        time.sleep(step_sec)
    return out


# ---------------------------
# Feature engineering helpers
# ---------------------------
def _to_float(x):
    try:
        return float(x)
    except Exception:
        return None


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
    """
    Filter levels within ±band_pct of price p.
    side is 'bids' or 'asks' (not used, but helpful for readability)
    """
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
    """
    Return the single largest wall in given levels by size.
    """
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
        }
    }


def compute_depth_history_features(depth_snapshots: List[Dict[str, Any]], p: float, levels: int = 20):
    """
    From polled depth snapshots (time series), approximate:
    - wall persistence: how often top wall stays at same/near price
    - wall churn: frequency of top wall price changes
    - size stability: coeff of variation (rough) for top wall size
    """
    if not depth_snapshots:
        return {"available": False}

    # For each snapshot, find top wall in near band
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
        # size CV (std/mean)
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
    """
    Use MEXC field T: 1=buy, 2=sell (aggressor-ish)
    """
    if not trades:
        return {"available": False}

    buy_vol = 0.0
    sell_vol = 0.0
    buy_cnt = 0
    sell_cnt = 0
    max_trade = {"price": None, "vol": None, "T": None}

    for tr in trades:
        px = _to_float(tr.get("p") or tr.get("price"))
        v = _to_float(tr.get("v") or tr.get("vol") or tr.get("volume"))
        t = tr.get("T")  # 1 or 2
        if v is None:
            continue

        if max_trade["vol"] is None or v > max_trade["vol"]:
            max_trade = {"price": px, "vol": v, "T": t}

        if t == 1:
            buy_vol += v
            buy_cnt += 1
        elif t == 2:
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
        "maxTrade": max_trade
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

    # crude "leverage chase" flag: OI increases while price moves fast
    chase_flag = None
    if oi_change_pct is not None and px_change_pct is not None:
        chase_flag = (oi_change_pct > 0.3 and abs(px_change_pct) > 0.25)

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
        "leverageChaseFlag": chase_flag
    }


def compute_context_features(klines_15m: List[Dict[str, Any]], klines_5m: List[Dict[str, Any]]):
    """
    Simple range/position metrics for LLM to use quickly.
    """
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

    # trend proxy: last close vs mean close
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
def make_latest(symbol: str,
                bars_1m: int = 240,
                bars_5m: int = 120,
                bars_15m: int = 80,
                depth_levels: int = 20,
                trades_n: int = 100,
                depth_commits_n: int = 60,
                hist_seconds: int = 60,
                hist_step_sec: float = 2.0,
                enable_poll_history: bool = True):
    now_ms = get_server_time_ms()

    ticker = get_ticker(symbol)
    funding = get_funding(symbol)
    funding_hist = get_funding_history(symbol, n=24)
    index_price = get_index(symbol)
    fair_price = get_fair(symbol)
    depth = get_depth(symbol, limit=depth_levels)
    depth_commits = get_depth_commits(symbol, n=depth_commits_n)
    trades = get_trades(symbol, limit=trades_n)
    contract = get_contract_info(symbol)

    k1 = get_klines(symbol, "Min1", bars_1m)
    k5 = get_klines(symbol, "Min5", bars_5m)
    k15 = get_klines(symbol, "Min15", bars_15m)

    ticker_hist = []
    depth_hist = []
    if enable_poll_history and hist_seconds > 0:
        ticker_hist = sample_ticker_history(symbol, seconds=hist_seconds, step_sec=hist_step_sec)
        depth_hist = sample_depth_history(symbol, seconds=hist_seconds, step_sec=hist_step_sec, levels=depth_levels)

    market_info = {
        "priceUnit": contract.get("priceUnit"),
        "volUnit": contract.get("volUnit"),
        "minVol": contract.get("minVol"),
        "contractSize": contract.get("contractSize"),
        "priceScale": contract.get("priceScale"),
        "volScale": contract.get("volScale"),
        "maxLeverage": contract.get("maxLeverage"),
    }

    metrics = {
        "openInterestNow": ticker.get("holdVol"),
        "oiHistory": ticker_hist,
        "fundingRateNow": ticker.get("fundingRate"),
        "fundingRateHistory": funding_hist,
    }

    # Determine P for feature calc
    P = _to_float(ticker.get("lastPrice")) or _to_float(ticker.get("fairPrice")) or _to_float(fair_price.get("fairPrice"))
    if P is None and k1:
        P = _to_float(k1[-1].get("c"))

    # ---- Features ----
    ob_feat = compute_orderbook_features(depth, P, band_pct=0.0015, topN=depth_levels) if P else {"available": False}
    ob_hist_feat = compute_depth_history_features(depth_hist, P, levels=depth_levels) if P else {"available": False}
    tr_feat = compute_trade_features(trades[:40])
    oi_feat = compute_oi_features(ticker_hist)
    ctx_feat = compute_context_features(k15, k5)

    features = {
        "P_used": P,
        "orderbook": ob_feat,
        "orderbookHistory": ob_hist_feat,
        "trades": tr_feat,
        "openInterest": oi_feat,
        "context": ctx_feat,
    }

    bundle = {
        "meta": {
            "exchange": "MEXC",
            "market": "FUTURES",
            "symbol": symbol,
            "generated_at": datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "marketInfo": market_info,
        "price": {
            "ticker": ticker,
            "funding": funding,
            "index": index_price,
            "fair": fair_price,
        },
        "metrics": metrics,

        "orderbook": depth,
        "orderbookHistory": {
            "depthCommits": depth_commits,
            "polled": {
                "seconds": hist_seconds if enable_poll_history else 0,
                "stepSec": hist_step_sec if enable_poll_history else 0,
                "levels": depth_levels,
                "snapshots": depth_hist,
            }
        },
        "trades": trades,

        "klines": {
            "1m": k1,
            "5m": k5,
            "15m": k15,
        },

        # ✅ LLM-friendly summary
        "features": features
    }

    return bundle


if __name__ == "__main__":
    # Usage:
    #   python collect_mexc_futures_latest.py BTC_USDT
    #   python collect_mexc_futures_latest.py BTC_USDT 240 120 80 60
    #
    # args:
    #   symbol bars_1m bars_5m bars_15m hist_seconds
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC_USDT"
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
        enable_poll_history=True
    )

    with open("latest.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))

    print("Saved latest.json (with features)")