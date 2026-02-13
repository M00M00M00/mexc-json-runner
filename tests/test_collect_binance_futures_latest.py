import datetime as dt

import pytest

import collect_mexc_futures_latest as collector


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

    monkeypatch.setattr(collector.requests, "get", fake_get)
    out = collector._get_json("https://example.com", retries=2, sleep=0)
    assert out == {"ok": True}
    assert calls["n"] == 2


def test_get_json_uses_curl_fallback_on_202(monkeypatch):
    def fake_get(url, params=None, timeout=10, headers=None):
        return DummyResponse(202, {"code": 0}, text="")

    monkeypatch.setattr(collector.requests, "get", fake_get)
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
        lambda s: {"symbol": "BTCUSDT", "lastPrice": "100", "volume": "10", "quoteVolume": "1000", "closeTime": 1700000000000},
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
    assert bundle["price"]["ticker"]["lastPrice"] == 100.0
    assert bundle["price"]["premiumIndex"]["markPrice"] == "100.2"
    assert bundle["marketInfo"]["tickSize"] == 0.1
    assert bundle["metrics"]["fundingRateNow"] == 0.0001
    assert bundle["orderbook"]["bids"][0][0] == "99.9"
    assert "orderbook" in bundle["features"]


def test_iso_from_ms_utc():
    ts = collector._iso_from_ms(1700000000000)
    parsed = dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    assert int(parsed.timestamp()) == 1700000000
