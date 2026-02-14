import datetime as dt

import pytest

import collect_mexc_futures_latest as collector


def _is_tick_aligned(value, tick, eps=1e-9):
    units = round(value / tick)
    return abs(value - units * tick) < eps


def _build_linear_klines(n, start=100.0, step=0.1):
    rows = []
    px = start
    for i in range(n):
        o = px
        c = px + step
        h = max(o, c) + 0.2
        l = min(o, c) - 0.2
        rows.append({"ts": f"2024-01-01T00:{i:02d}:00Z", "o": o, "h": h, "l": l, "c": c, "v": 10.0 + i})
        px = c
    return rows


class DummyResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        return self._payload


def test_normalize_symbol_supports_legacy_mexc_style():
    assert collector._normalize_symbol("BTC_USDT") == "BTCUSDT"
    assert collector._normalize_symbol("btc/usdt") == "BTCUSDT"
    assert collector._normalize_symbol(" EthUsdt ") == "ETHUSDT"


def test_interval_to_seconds_binance():
    assert collector._interval_to_seconds("1m") == 60
    assert collector._interval_to_seconds("5m") == 300
    assert collector._interval_to_seconds("15m") == 900
    with pytest.raises(ValueError):
        collector._interval_to_seconds("Min1")


def test_get_json_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=10, headers=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResponse(500, {"code": -1}, text="internal")
        return DummyResponse(200, {"ok": True})

    monkeypatch.setattr(collector, "_http_get", fake_get)
    out = collector._get_json("https://example.com", retries=2, sleep=0)
    assert out == {"ok": True}
    assert calls["n"] == 2


def test_get_json_uses_curl_fallback_on_202(monkeypatch):
    def fake_get(url, params=None, timeout=10, headers=None):
        return DummyResponse(202, {"code": 0}, text="")

    monkeypatch.setattr(collector, "_http_get", fake_get)
    monkeypatch.setattr(collector, "_curl_get_json", lambda url, params=None, timeout=10: {"ok": True})

    out = collector._get_json("https://example.com", retries=1, sleep=0)
    assert out == {"ok": True}


def test_binance_get_fallbacks_on_451(monkeypatch):
    monkeypatch.setenv("BINANCE_BASE_URLS", "https://blocked.example,https://ok.example")
    calls = []

    def fake_get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
        calls.append(url)
        if url.startswith("https://blocked.example"):
            raise RuntimeError("HTTP 451: restricted location")
        return {"serverTime": 1700000000000}

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    out = collector._binance_get("/fapi/v1/time")

    assert out["serverTime"] == 1700000000000
    assert calls == [
        "https://blocked.example/fapi/v1/time",
        "https://ok.example/fapi/v1/time",
    ]


def test_binance_get_raises_helpful_error_when_all_451(monkeypatch):
    monkeypatch.setenv("BINANCE_BASE_URLS", "https://blocked1.example,https://blocked2.example")

    def fake_get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
        raise RuntimeError("HTTP 451: restricted location")

    monkeypatch.setattr(collector, "_get_json", fake_get_json)

    with pytest.raises(RuntimeError, match="runner location"):
        collector._binance_get("/fapi/v1/time")


def test_binance_get_skips_unexpected_payload_shape(monkeypatch):
    monkeypatch.setenv("BINANCE_BASE_URLS", "https://bad.example,https://ok.example")
    calls = []

    def fake_get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
        calls.append(url)
        if url.startswith("https://bad.example"):
            return {"code": 0, "msg": "Service unavailable from a restricted location"}
        return {"serverTime": 1700000000000}

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    out = collector._binance_get("/fapi/v1/time", expect="serverTime")

    assert out["serverTime"] == 1700000000000
    assert calls == [
        "https://bad.example/fapi/v1/time",
        "https://ok.example/fapi/v1/time",
    ]


def test_get_klines_transforms_binance_payload(monkeypatch):
    kline_payload = [
        [1700000000000, "100", "110", "90", "105", "11", 1700000059999, "0", 0, "0", "0", "0"],
        [1700000060000, "105", "120", "101", "118", "22", 1700000119999, "0", 0, "0", "0", "0"],
    ]

    def fake_get_json(url, params=None, timeout=10, retries=3, sleep=0.25):
        assert "/fapi/v1/klines" in url
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1m"
        assert params["limit"] == 2
        return kline_payload

    monkeypatch.setattr(collector, "_get_json", fake_get_json)
    rows = collector.get_klines("BTCUSDT", "1m", 2)

    assert len(rows) == 2
    assert rows[0]["ts"] == "2023-11-14T22:13:20Z"
    assert rows[0]["o"] == 100.0
    assert rows[1]["c"] == 118.0
    assert rows[1]["v"] == 22.0


def test_compute_trade_features_uses_binance_aggressor_side():
    trades = [
        {"price": "100", "qty": "2", "isBuyerMaker": False},  # taker buy
        {"price": "101", "qty": "3", "isBuyerMaker": True},   # taker sell
        {"price": "102", "qty": "5", "isBuyerMaker": False},
    ]
    out = collector.compute_trade_features(trades)

    assert out["available"] is True
    assert out["buyVol"] == 7.0
    assert out["sellVol"] == 3.0
    assert out["buyCnt"] == 2
    assert out["sellCnt"] == 1
    assert out["maxTrade"]["vol"] == 5.0
    assert out["maxTrade"]["aggressorSide"] == "BUY"


def test_latest_trades_window_sorts_by_time_desc():
    trades = [
        {"time": 1700000001000, "price": "100.0", "qty": "1"},
        {"time": 1700000003000, "price": "101.0", "qty": "1"},
        {"time": 1700000002000, "price": "102.0", "qty": "1"},
    ]
    out = collector._latest_trades_window(trades, n=2)
    assert [x["time"] for x in out] == [1700000003000, 1700000002000]


def test_trade_sort_key_priority_time_T_t_timestamp():
    trades = [
        {"timestamp": 10, "price": "1", "qty": "1"},
        {"t": 11, "price": "1", "qty": "1"},
        {"T": 12, "price": "1", "qty": "1"},
        {"time": 13, "price": "1", "qty": "1"},
    ]
    out = collector._latest_trades_window(trades, n=4)
    assert out[0].get("time") == 13
    assert out[1].get("T") == 12
    assert out[2].get("t") == 11
    assert out[3].get("timestamp") == 10


def test_depth_history_features_uses_tick_tolerance_and_snapshot_mid():
    snaps = [
        {
            "timestamp": 1,
            "bids": [["100.0", "50"], ["99.9", "10"]],
            "asks": [["100.1", "40"], ["100.2", "8"]],
        },
        {
            "timestamp": 2,
            "bids": [["100.1", "52"], ["100.0", "10"]],
            "asks": [["100.2", "41"], ["100.3", "8"]],
        },
        {
            "timestamp": 3,
            "bids": [["100.2", "54"], ["100.1", "10"]],
            "asks": [["100.3", "42"], ["100.4", "8"]],
        },
    ]
    # p is intentionally far; function should use per-snapshot mid, not this static p.
    out = collector.compute_depth_history_features(
        snaps,
        p=1000.0,
        levels=2,
        tick_size=0.1,
        tick_tolerance_ticks=2,
        band_pct=0.002,
        min_samples=3,
    )

    assert out["available"] is True
    assert out["bidWall"]["samples"] == 3
    assert out["askWall"]["samples"] == 3
    assert out["bidWall"]["persistRate"] == 1.0
    assert out["askWall"]["persistRate"] == 1.0
    assert out["tickSizeUsed"] == 0.1
    assert out["tickToleranceTicks"] == 2
    assert out["topBidWall"]["persistRate"] == out["bidWall"]["persistRate"]
    assert out["topAskWall"]["persistRate"] == out["askWall"]["persistRate"]


def test_depth_history_features_sets_persist_none_when_samples_lt_6():
    snaps = [
        {"timestamp": 1, "bids": [["100.0", "10"]], "asks": [["100.1", "10"]]},
        {"timestamp": 2, "bids": [["100.1", "10"]], "asks": [["100.2", "10"]]},
        {"timestamp": 3, "bids": [["100.2", "10"]], "asks": [["100.3", "10"]]},
        {"timestamp": 4, "bids": [["100.3", "10"]], "asks": [["100.4", "10"]]},
        {"timestamp": 5, "bids": [["100.4", "10"]], "asks": [["100.5", "10"]]},
    ]
    out = collector.compute_depth_history_features(
        snaps,
        p=100.0,
        levels=1,
        tick_size=0.1,
        tick_tolerance_ticks=2,
        band_pct=0.002,
        min_samples=6,
    )
    assert out["bidWall"]["samples"] == 5
    assert out["bidWall"]["persistRate"] is None
    assert out["askWall"]["persistRate"] is None
    assert out["topBidWall"]["persistRate"] is None


def test_extract_symbol_info_and_market_info():
    exchange_info = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "pair": "BTCUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "marginAsset": "USDT",
                "onboardDate": 1700000000000,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0.10", "maxPrice": "1000000"},
                    {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                ],
            }
        ]
    }

    contract = collector._extract_symbol_info(exchange_info, "BTCUSDT")
    market_info = collector.build_market_info(contract)

    assert contract["symbol"] == "BTCUSDT"
    assert market_info["tickSize"] == 0.1
    assert market_info["minQty"] == 0.001
    assert market_info["priceUnit"] == 0.1
    assert market_info["volUnit"] == 0.001


def test_build_ticker_snapshot_maps_binance_fields():
    ticker_24h = {
        "symbol": "BTCUSDT",
        "lastPrice": "30000.5",
        "volume": "1234.5",
        "quoteVolume": "9000000",
        "closeTime": 1700000123456,
    }
    book_ticker = {"bidPrice": "30000.4", "askPrice": "30000.6"}
    premium_index = {
        "markPrice": "30001.0",
        "indexPrice": "29999.9",
        "lastFundingRate": "0.0001",
    }
    open_interest = {"openInterest": "22222.2"}

    ticker = collector.build_ticker_snapshot(ticker_24h, book_ticker, premium_index, open_interest)

    assert ticker["lastPrice"] == 30000.5
    assert ticker["bid1"] == 30000.4
    assert ticker["ask1"] == 30000.6
    assert ticker["holdVol"] == 22222.2
    assert ticker["fundingRate"] == 0.0001
    assert ticker["fairPrice"] == 30001.0


def test_make_latest_builds_binance_bundle(monkeypatch):
    symbol = "BTC_USDT"

    monkeypatch.setattr(collector, "get_server_time_ms", lambda: 1700000000000)
    monkeypatch.setattr(
        collector,
        "get_ticker_24h",
        lambda s: {"symbol": "BTCUSDT", "lastPrice": "100.06", "volume": "10", "quoteVolume": "1000", "closeTime": 1700000000000},
    )
    monkeypatch.setattr(collector, "get_book_ticker", lambda s: {"symbol": "BTCUSDT", "bidPrice": "99.9", "askPrice": "100.1"})
    monkeypatch.setattr(
        collector,
        "get_premium_index",
        lambda s: {"symbol": "BTCUSDT", "markPrice": "100.2", "indexPrice": "100.0", "lastFundingRate": "0.0001", "nextFundingTime": 1700003600000},
    )
    monkeypatch.setattr(collector, "get_funding_history", lambda s, n=24: [{"symbol": "BTCUSDT", "fundingRate": "0.0001", "fundingTime": 1699990000000}])
    monkeypatch.setattr(collector, "get_open_interest", lambda s: {"symbol": "BTCUSDT", "openInterest": "20000", "time": 1700000000000})
    monkeypatch.setattr(
        collector,
        "get_exchange_info",
        lambda s: {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "pair": "BTCUSDT",
                    "contractType": "PERPETUAL",
                    "status": "TRADING",
                    "pricePrecision": 2,
                    "quantityPrecision": 3,
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "marginAsset": "USDT",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0.10", "maxPrice": "1000000"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
                    ],
                }
            ]
        },
    )
    monkeypatch.setattr(
        collector,
        "get_depth",
        lambda s, limit=20: {
            "lastUpdateId": 1,
            "E": 1700000000000,
            "T": 1700000000001,
            "bids": [["99.9", "10"], ["99.8", "9"]],
            "asks": [["100.1", "11"], ["100.2", "12"]],
        },
    )
    monkeypatch.setattr(
        collector,
        "get_trades",
        lambda s, limit=100: [
            {"id": 1, "price": "100", "qty": "1", "isBuyerMaker": False},
            {"id": 2, "price": "100.1", "qty": "2", "isBuyerMaker": True},
        ],
    )

    kline_rows = [
        {"ts": "2023-11-14T22:13:20Z", "o": 99.0, "h": 101.0, "l": 98.5, "c": 100.0, "v": 10.0},
        {"ts": "2023-11-14T22:14:20Z", "o": 100.0, "h": 102.0, "l": 99.5, "c": 101.0, "v": 12.0},
    ]

    def fake_klines(s, interval, bars):
        assert s == "BTCUSDT"
        assert interval in {"1m", "5m", "15m"}
        assert bars == 2
        return kline_rows

    monkeypatch.setattr(collector, "get_klines", fake_klines)
    monkeypatch.setattr(collector, "sample_ticker_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(collector, "sample_depth_history", lambda *args, **kwargs: [])

    bundle = collector.make_latest(
        symbol=symbol,
        bars_1m=2,
        bars_5m=2,
        bars_15m=2,
        depth_levels=2,
        trades_n=2,
        hist_seconds=0,
        enable_poll_history=False,
    )

    assert bundle["meta"]["exchange"] == "BINANCE"
    assert bundle["meta"]["symbol"] == "BTCUSDT"
    assert bundle["price"]["ticker"]["lastPrice"] == 100.06
    assert bundle["price"]["premiumIndex"]["markPrice"] == "100.2"
    assert bundle["marketInfo"]["tickSize"] == 0.1
    assert bundle["metrics"]["fundingRateNow"] == 0.0001
    assert bundle["orderbook"]["bids"][0][0] == "99.9"
    assert "orderbook" in bundle["features"]
    assert bundle["riskLevels"]["available"] is True
    assert bundle["riskLevels"]["long"]["entry"] == 100.1
    assert bundle["riskLevels"]["long"]["stop1pct"] == 99.0
    assert bundle["riskLevels"]["long"]["take2pct"] == 102.2
    assert bundle["riskLevels"]["short"]["stop1pct"] == 101.2
    assert bundle["riskLevels"]["short"]["take2pct"] == 98.0
    assert "levelCandidates" in bundle["features"]
    assert "volatility" in bundle["features"]
    assert "microTrend" in bundle["features"]


def test_iso_from_ms_utc():
    ts = collector._iso_from_ms(1700000000000)
    parsed = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    assert int(parsed.timestamp()) == 1700000000


def test_risk_levels_are_tick_aligned():
    out = collector.build_tick_aligned_levels(entry_price=68870.93, tick_size=0.1)
    assert out["available"] is True

    tick = 0.1
    vals = [
        out["long"]["entry"],
        out["long"]["stop1pct"],
        out["long"]["take2pct"],
        out["short"]["entry"],
        out["short"]["stop1pct"],
        out["short"]["take2pct"],
    ]
    for v in vals:
        units = round(v / tick)
        assert abs(v - units * tick) < 1e-9


def _sample_level_candidates():
    k5 = [
        {"ts": "2024-01-01T00:00:00Z", "o": 99.8, "h": 100.1, "l": 99.4, "c": 99.7, "v": 10},
        {"ts": "2024-01-01T00:05:00Z", "o": 99.7, "h": 100.0, "l": 99.3, "c": 99.8, "v": 11},
        {"ts": "2024-01-01T00:10:00Z", "o": 99.8, "h": 100.2, "l": 99.5, "c": 100.0, "v": 12},
        {"ts": "2024-01-01T00:15:00Z", "o": 100.0, "h": 100.6, "l": 99.8, "c": 100.4, "v": 13},
        {"ts": "2024-01-01T00:20:00Z", "o": 100.4, "h": 100.7, "l": 100.0, "c": 100.3, "v": 14},
        {"ts": "2024-01-01T00:25:00Z", "o": 100.3, "h": 100.8, "l": 100.1, "c": 100.6, "v": 15},
    ]
    return collector.compute_level_candidates(
        p=100.0,
        bid=99.9,
        ask=100.1,
        tick_size=0.1,
        orderbook_feat={
            "nearBand": {
                "topBidWall": {"price": 99.6, "size": 120.0},
                "topAskWall": {"price": 100.4, "size": 130.0},
            }
        },
        context_feat={"range15m": {"lo": 98.0, "hi": 102.0}},
        klines_5m=k5,
    )


def test_level_candidates_exist_and_have_required_fields():
    out = _sample_level_candidates()
    assert out["available"] is True
    assert out["long"]
    assert out["short"]

    required = {"name", "entry", "stop1pct", "take2pct", "withinP08", "entryMakerSafe"}
    for side in ("long", "short"):
        for item in out[side]:
            assert required.issubset(item.keys())


def test_level_candidates_tick_alignment_and_stop_take_rounding():
    out = _sample_level_candidates()
    tick = out["tickSize"]
    assert tick == 0.1

    for side in ("long", "short"):
        for item in out[side]:
            assert _is_tick_aligned(item["entry"], tick)
            assert _is_tick_aligned(item["stop1pct"], tick)
            assert _is_tick_aligned(item["take2pct"], tick)

            if side == "long":
                raw_stop = item["entry"] * 0.99
                raw_take = item["entry"] * 1.02
                assert item["stop1pct"] <= raw_stop + 1e-9
                assert raw_stop - item["stop1pct"] < tick + 1e-9
                assert item["take2pct"] >= raw_take - 1e-9
                assert item["take2pct"] - raw_take < tick + 1e-9
            else:
                raw_stop = item["entry"] * 1.01
                raw_take = item["entry"] * 0.98
                assert item["stop1pct"] >= raw_stop - 1e-9
                assert item["stop1pct"] - raw_stop < tick + 1e-9
                assert item["take2pct"] <= raw_take + 1e-9
                assert raw_take - item["take2pct"] < tick + 1e-9


def test_level_candidates_enforce_maker_safe_entries():
    out = _sample_level_candidates()
    tick = out["tickSize"]
    bid = out["bid"]
    ask = out["ask"]

    for item in out["long"]:
        assert item["entryMakerSafe"] is True
        assert item["entry"] <= ask - tick + 1e-9

    for item in out["short"]:
        assert item["entryMakerSafe"] is True
        assert item["entry"] >= bid + tick - 1e-9


def test_volatility_and_micro_trend_features_payload():
    k1 = _build_linear_klines(60, start=100.0, step=0.08)
    k5 = _build_linear_klines(60, start=100.0, step=0.25)

    vol = collector.compute_volatility_features(k1, k5, p=104.8)
    assert vol["available"] is True
    assert isinstance(vol["atr1m"], float)
    assert isinstance(vol["atr5m"], float)
    assert isinstance(vol["atr1mPct"], float)
    assert isinstance(vol["atr5mPct"], float)

    micro = collector.compute_micro_trend_features(k1, lookback=60)
    assert micro["trend"] in {"UP", "DOWN", "RANGE", None}
