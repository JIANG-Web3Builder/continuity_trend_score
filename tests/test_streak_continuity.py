import pandas as pd

from trend_score.streaks import (
    STREAK_DOWN_LABEL,
    STREAK_UP_LABEL,
    detect_strict_streaks,
    score_streak_win_rates,
    streak_signal_outcomes,
)


def _prices(symbol: str, opens: list[float], closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(opens), freq="D")
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": dates,
            "open": opens,
            "high": [max(open_, close) + 1 for open_, close in zip(opens, closes)],
            "low": [min(open_, close) - 1 for open_, close in zip(opens, closes)],
            "close": closes,
            "pre_close": [closes[0], *closes[:-1]],
            "change": [0.0, *[closes[i] - closes[i - 1] for i in range(1, len(closes))]],
            "vol": 1000.0,
        }
    )


def test_detect_strict_streaks_uses_candle_body_not_close_to_close_path():
    prices = _prices("AAA", [100, 105, 106, 104, 103], [101, 104, 105, 103, 104])

    result = detect_strict_streaks(prices, min_days=2)

    assert result[["symbol", "direction", "continuity_label", "days", "status"]].to_dict("records") == [
        {"symbol": "AAA", "direction": "down", "continuity_label": STREAK_DOWN_LABEL, "days": 3, "status": "confirmed"}
    ]
    assert result.loc[0, "start_date"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "end_date"] == pd.Timestamp("2024-01-04")


def test_detect_strict_streaks_treats_doji_as_a_break():
    prices = _prices("AAA", [100, 101, 102, 103, 103, 104], [101, 102, 102, 104, 105, 106])

    result = detect_strict_streaks(prices, min_days=2)

    assert result[["direction", "continuity_label", "days", "status"]].to_dict("records") == [
        {"direction": "up", "continuity_label": STREAK_UP_LABEL, "days": 2, "status": "confirmed"},
        {"direction": "up", "continuity_label": STREAK_UP_LABEL, "days": 3, "status": "open"},
    ]


def test_score_streak_win_rates_summarizes_future_up_down_distribution_by_direction_and_length():
    prices = _prices(
        "AAA",
        [100, 101, 102, 103, 106, 105, 104, 101, 100, 95],
        [101, 102, 103, 104, 105, 104, 103, 100, 99, 96],
    )

    result = score_streak_win_rates(prices, min_days=3, horizons=(1, 3))

    assert result[
        [
            "direction",
            "continuity_label",
            "streak_days",
            "total_signals",
            "latest_start_date",
            "latest_end_date",
            "latest_streak_days",
            "signals_1d",
            "up_count_1d",
            "down_count_1d",
            "flat_count_1d",
            "win_rate_1d",
            "up_rate_1d",
            "down_rate_1d",
            "avg_return_1d",
            "signals_3d",
            "up_count_3d",
            "down_count_3d",
            "flat_count_3d",
            "win_rate_3d",
            "up_rate_3d",
            "down_rate_3d",
            "avg_return_3d",
        ]
    ].to_dict("records") == [
        {
            "direction": "up",
            "continuity_label": STREAK_UP_LABEL,
            "streak_days": 3,
            "total_signals": 1,
            "latest_start_date": pd.Timestamp("2024-01-01"),
            "latest_end_date": pd.Timestamp("2024-01-03"),
            "latest_streak_days": 3,
            "signals_1d": 1,
            "up_count_1d": 1,
            "down_count_1d": 0,
            "flat_count_1d": 0,
            "win_rate_1d": 100.0,
            "up_rate_1d": 100.0,
            "down_rate_1d": 0.0,
            "avg_return_1d": 0.97,
            "signals_3d": 1,
            "up_count_3d": 1,
            "down_count_3d": 0,
            "flat_count_3d": 0,
            "win_rate_3d": 100.0,
            "up_rate_3d": 100.0,
            "down_rate_3d": 0.0,
            "avg_return_3d": 0.97,
        },
        {
            "direction": "up",
            "continuity_label": STREAK_UP_LABEL,
            "streak_days": 4,
            "total_signals": 1,
            "latest_start_date": pd.Timestamp("2024-01-01"),
            "latest_end_date": pd.Timestamp("2024-01-04"),
            "latest_streak_days": 4,
            "signals_1d": 1,
            "up_count_1d": 1,
            "down_count_1d": 0,
            "flat_count_1d": 0,
            "win_rate_1d": 100.0,
            "up_rate_1d": 100.0,
            "down_rate_1d": 0.0,
            "avg_return_1d": 0.96,
            "signals_3d": 1,
            "up_count_3d": 0,
            "down_count_3d": 1,
            "flat_count_3d": 0,
            "win_rate_3d": 0.0,
            "up_rate_3d": 0.0,
            "down_rate_3d": 100.0,
            "avg_return_3d": -0.96,
        },
        {
            "direction": "down",
            "continuity_label": STREAK_DOWN_LABEL,
            "streak_days": 3,
            "total_signals": 1,
            "latest_start_date": pd.Timestamp("2024-01-05"),
            "latest_end_date": pd.Timestamp("2024-01-07"),
            "latest_streak_days": 3,
            "signals_1d": 1,
            "up_count_1d": 0,
            "down_count_1d": 1,
            "flat_count_1d": 0,
            "win_rate_1d": 100.0,
            "up_rate_1d": 0.0,
            "down_rate_1d": 100.0,
            "avg_return_1d": -2.91,
            "signals_3d": 1,
            "up_count_3d": 0,
            "down_count_3d": 1,
            "flat_count_3d": 0,
            "win_rate_3d": 100.0,
            "up_rate_3d": 0.0,
            "down_rate_3d": 100.0,
            "avg_return_3d": -6.8,
        },
        {
            "direction": "down",
            "continuity_label": STREAK_DOWN_LABEL,
            "streak_days": 4,
            "total_signals": 1,
            "latest_start_date": pd.Timestamp("2024-01-05"),
            "latest_end_date": pd.Timestamp("2024-01-08"),
            "latest_streak_days": 4,
            "signals_1d": 1,
            "up_count_1d": 0,
            "down_count_1d": 1,
            "flat_count_1d": 0,
            "win_rate_1d": 100.0,
            "up_rate_1d": 0.0,
            "down_rate_1d": 100.0,
            "avg_return_1d": -1.0,
            "signals_3d": 0,
            "up_count_3d": 0,
            "down_count_3d": 0,
            "flat_count_3d": 0,
            "win_rate_3d": None,
            "up_rate_3d": None,
            "down_rate_3d": None,
            "avg_return_3d": None,
        },
        {
            "direction": "down",
            "continuity_label": STREAK_DOWN_LABEL,
            "streak_days": 5,
            "total_signals": 1,
            "latest_start_date": pd.Timestamp("2024-01-05"),
            "latest_end_date": pd.Timestamp("2024-01-09"),
            "latest_streak_days": 5,
            "signals_1d": 1,
            "up_count_1d": 0,
            "down_count_1d": 1,
            "flat_count_1d": 0,
            "win_rate_1d": 100.0,
            "up_rate_1d": 0.0,
            "down_rate_1d": 100.0,
            "avg_return_1d": -3.03,
            "signals_3d": 0,
            "up_count_3d": 0,
            "down_count_3d": 0,
            "flat_count_3d": 0,
            "win_rate_3d": None,
            "up_rate_3d": None,
            "down_rate_3d": None,
            "avg_return_3d": None,
        },
    ]
    assert "trigger_rule" not in result.columns


def test_streak_signal_outcomes_lists_confirmed_ranges_at_or_above_minimum_with_future_returns():
    prices = _prices(
        "AAA",
        [100, 101, 102, 103, 106, 105, 104, 101, 100, 95],
        [101, 102, 103, 104, 105, 104, 103, 100, 99, 96],
    )

    result = streak_signal_outcomes(prices, min_days=3, horizons=(1, 3))

    assert result[
        [
            "direction",
            "start_date",
            "end_date",
            "streak_days",
            "end_close",
            "return_1d",
            "return_3d",
        ]
    ].to_dict("records") == [
        {
            "direction": "down",
            "start_date": pd.Timestamp("2024-01-05"),
            "end_date": pd.Timestamp("2024-01-09"),
            "streak_days": 5,
            "end_close": 99.0,
            "return_1d": -3.03,
            "return_3d": None,
        },
        {
            "direction": "up",
            "start_date": pd.Timestamp("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-04"),
            "streak_days": 4,
            "end_close": 104.0,
            "return_1d": 0.96,
            "return_3d": -0.96,
        },
    ]


def test_streak_signal_outcomes_keeps_full_longer_run_in_details():
    prices = _prices(
        "AAA",
        [100, 101, 102, 103, 104, 105],
        [101, 102, 103, 104, 103, 102],
    )

    result = streak_signal_outcomes(prices, min_days=3, horizons=(1,))

    assert result[["direction", "start_date", "end_date", "streak_days", "return_1d"]].to_dict("records") == [
        {
            "direction": "up",
            "start_date": pd.Timestamp("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-04"),
            "streak_days": 4,
            "return_1d": -0.96,
        }
    ]


def test_streak_functions_are_exported_from_package():
    import trend_score

    assert trend_score.detect_strict_streaks is detect_strict_streaks
    assert trend_score.score_streak_win_rates is score_streak_win_rates
    assert trend_score.streak_signal_outcomes is streak_signal_outcomes
