import pandas as pd

from trend_score.waves import classify_wave_levels, detect_waves, _rolling_reversal_thresholds


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

    confirmed = waves[waves["status"] == "confirmed"].reset_index(drop=True)
    assert confirmed[["direction", "start_price", "end_price", "points"]].to_dict("records") == [
        {"direction": "up", "start_price": 100.0, "end_price": 110.0, "points": 10.0},
        {"direction": "down", "start_price": 110.0, "end_price": 104.0, "points": 6.0},
        {"direction": "up", "start_price": 104.0, "end_price": 118.0, "points": 14.0},
    ]
    assert confirmed["days"].tolist() == [5, 4, 5]
    assert waves.iloc[-1]["status"] == "open"
    assert waves.iloc[-1]["direction"] == "down"


def test_detect_waves_handles_monotonic_series_as_one_wave():
    df = price_frame([100, 103, 106, 110])

    waves = detect_waves(df, symbol="000001.SH", min_reversal=4)

    assert len(waves) == 1
    assert waves.loc[0, "direction"] == "up"
    assert waves.loc[0, "points"] == 10.0
    assert waves.loc[0, "status"] == "open"


def test_detect_waves_filters_one_day_noise_reversals_by_default():
    df = price_frame([100, 110, 104, 116, 112])

    waves = detect_waves(df, symbol="000001.SH", min_reversal=4)

    confirmed = waves[waves["status"] == "confirmed"].reset_index(drop=True)
    assert confirmed[["direction", "start_price", "end_price", "days", "status"]].to_dict("records") == [
        {"direction": "up", "start_price": 100.0, "end_price": 116.0, "days": 5, "status": "confirmed"}
    ]


def test_detect_waves_supports_percentage_reversal_threshold():
    low_price = price_frame([100, 110, 104, 116, 112])
    high_price = price_frame([3000, 3300, 3120, 3480, 3360])

    low_waves = detect_waves(low_price, symbol="LOW", min_reversal_pct=5)
    high_waves = detect_waves(high_price, symbol="HIGH", min_reversal_pct=5)

    assert low_waves[low_waves["status"] == "open"][["direction", "start_price", "end_price"]].to_dict("records") == [
        {"direction": "up", "start_price": 100.0, "end_price": 116.0}
    ]
    assert high_waves[high_waves["status"] == "open"][["direction", "start_price", "end_price"]].to_dict("records") == [
        {"direction": "up", "start_price": 3000.0, "end_price": 3480.0}
    ]


def test_detect_waves_uses_confirmation_date_for_confirmed_turns():
    df = price_frame([100, 105, 110, 108, 104])

    waves = detect_waves(df, symbol="000001.SH", min_reversal=4, min_wave_days=1)
    confirmed = waves[waves["status"] == "confirmed"].reset_index(drop=True)

    assert len(confirmed) == 1
    assert confirmed.loc[0, "direction"] == "up"
    assert confirmed.loc[0, "extreme_date"] == pd.Timestamp("2024-01-03")
    assert confirmed.loc[0, "end_date"] == pd.Timestamp("2024-01-05")
    assert confirmed.loc[0, "confirmation_date"] == pd.Timestamp("2024-01-05")
    assert confirmed.loc[0, "extreme_price"] == 110.0
    assert confirmed.loc[0, "end_price"] == 110.0


def test_detect_waves_does_not_confirm_turn_before_reversal_threshold_is_visible():
    partial = price_frame([100, 105, 110, 108])
    full = price_frame([100, 105, 110, 108, 104])

    partial_waves = detect_waves(partial, symbol="000001.SH", min_reversal=4, min_wave_days=1)
    full_waves = detect_waves(full, symbol="000001.SH", min_reversal=4, min_wave_days=1)

    assert partial_waves[partial_waves["status"] == "confirmed"].empty
    assert len(full_waves[full_waves["status"] == "confirmed"]) == 1


def test_rolling_reversal_thresholds_do_not_change_when_future_prices_are_appended():
    base = price_frame([100, 120, 160, 100, 90, 95])
    extended = price_frame([100, 120, 160, 100, 90, 95, 1000, 10, 1200])

    base_thresholds = _rolling_reversal_thresholds(base, 1.2)
    extended_thresholds = _rolling_reversal_thresholds(extended, 1.2).iloc[: len(base)]

    assert extended_thresholds.tolist() == base_thresholds.tolist()


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
