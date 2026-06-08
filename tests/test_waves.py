import pandas as pd

from trend_score.waves import classify_wave_levels, detect_waves


def price_frame(closes):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "ts_code": "000001.SH",
            "trade_date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "pre_close": closes,
            "change": 0.0,
            "pct_chg": 0.0,
            "vol": 1000.0,
            "amount": 10000.0,
        }
    )


def test_detect_waves_finds_reversals_above_minimum_threshold():
    df = price_frame([100, 105, 110, 107, 104, 115, 118, 116, 112])

    waves = detect_waves(df, symbol="000001.SH", min_reversal=4)

    assert waves[["direction", "start_price", "end_price", "points"]].to_dict("records") == [
        {"direction": "up", "start_price": 100.0, "end_price": 110.0, "points": 10.0},
        {"direction": "down", "start_price": 110.0, "end_price": 104.0, "points": 6.0},
        {"direction": "up", "start_price": 104.0, "end_price": 118.0, "points": 14.0},
        {"direction": "down", "start_price": 118.0, "end_price": 112.0, "points": 6.0},
    ]
    assert waves["days"].tolist() == [3, 3, 3, 3]


def test_detect_waves_handles_monotonic_series_as_one_wave():
    df = price_frame([100, 103, 106, 110])

    waves = detect_waves(df, symbol="000001.SH", min_reversal=4)

    assert len(waves) == 1
    assert waves.loc[0, "direction"] == "up"
    assert waves.loc[0, "points"] == 10.0


def test_classify_wave_levels_uses_symbol_specific_quantiles():
    waves = pd.DataFrame(
        {
            "symbol": ["A"] * 5 + ["B"] * 5,
            "direction": ["up"] * 10,
            "points": [10, 20, 30, 40, 50, 100, 200, 300, 400, 500],
        }
    )

    classified = classify_wave_levels(waves)

    assert classified[classified["symbol"] == "A"]["level"].tolist() == ["小", "小", "中", "大", "超大"]
    assert classified[classified["symbol"] == "B"]["level"].tolist() == ["小", "小", "中", "大", "超大"]
