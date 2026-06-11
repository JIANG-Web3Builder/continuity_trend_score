import pandas as pd

from trend_score.candles import (
    NON_STRICT_DOWN_LABEL,
    NON_STRICT_UP_LABEL,
    STRICT_DOWN_LABEL,
    STRICT_UP_LABEL,
    detect_continuity_segments,
    detect_strict_runs,
    summarize_strict_runs,
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
            "pct_chg": 0.0,
            "vol": 1000.0,
            "amount": [close * 1000.0 for close in closes],
        }
    )


def test_detect_strict_runs_uses_price_path_and_requires_two_days():
    prices = _prices("AAA", [100, 100, 101, 102, 101, 100], [100, 101, 102, 101, 100, 99])

    result = detect_strict_runs(prices)

    assert result[["symbol", "direction", "continuity_label", "days", "status"]].to_dict("records") == [
        {"symbol": "AAA", "direction": "up", "continuity_label": STRICT_UP_LABEL, "days": 2, "status": "confirmed"},
        {"symbol": "AAA", "direction": "down", "continuity_label": STRICT_DOWN_LABEL, "days": 3, "status": "open"},
    ]
    assert (result["days"] >= 2).all()
    assert result.loc[0, "start_date"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "end_date"] == pd.Timestamp("2024-01-03")


def test_detect_strict_runs_does_not_count_positive_bodies_when_close_path_breaks():
    prices = _prices("AAA", [100, 98, 99, 101, 100], [101, 100, 102, 103, 104])

    result = detect_strict_runs(prices)

    assert result[["direction", "continuity_label", "days", "status"]].to_dict("records") == [
        {"direction": "up", "continuity_label": STRICT_UP_LABEL, "days": 3, "status": "open"},
    ]
    assert (result["days"] >= 2).all()


def test_detect_strict_runs_keeps_symbols_independent():
    prices = pd.concat(
        [
            _prices("AAA", [100, 100, 101, 102], [100, 101, 102, 103]),
            _prices("BBB", [50, 50, 49, 48], [50, 49, 48, 47]),
        ],
        ignore_index=True,
    )

    result = detect_strict_runs(prices)

    assert result[["symbol", "direction", "continuity_label", "days", "status"]].to_dict("records") == [
        {"symbol": "AAA", "direction": "up", "continuity_label": STRICT_UP_LABEL, "days": 3, "status": "open"},
        {"symbol": "BBB", "direction": "down", "continuity_label": STRICT_DOWN_LABEL, "days": 3, "status": "open"},
    ]


def test_detect_continuity_segments_adds_non_strict_interval_label_with_strict_runs():
    prices = _prices("AAA", [100, 98, 99, 101, 100], [101, 100, 102, 103, 104])

    result = detect_continuity_segments(prices)

    assert result[["continuity_label", "start_date", "end_date", "days"]].to_dict("records") == [
        {
            "continuity_label": NON_STRICT_UP_LABEL,
            "start_date": pd.Timestamp("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-05"),
            "days": 5,
        },
        {
            "continuity_label": STRICT_UP_LABEL,
            "start_date": pd.Timestamp("2024-01-03"),
            "end_date": pd.Timestamp("2024-01-05"),
            "days": 3,
        },
    ]
    assert (result["days"] >= 2).all()


def test_summarize_strict_runs_reports_current_and_longest_runs_by_interval():
    prices = pd.concat(
        [
            _prices("AAA", [100, 100, 101, 100, 99], [100, 101, 100, 99, 98]),
            _prices("BBB", [50, 50, 49, 50, 51], [50, 49, 50, 51, 50]),
        ],
        ignore_index=True,
    )

    result = summarize_strict_runs(prices, "2024-01-01", "2024-01-05")

    assert result[
        ["symbol", "continuity_label", "current_direction", "current_run_days", "longest_up_days", "longest_down_days"]
    ].to_dict("records") == [
        {
            "symbol": "AAA",
            "continuity_label": NON_STRICT_DOWN_LABEL,
            "current_direction": "down",
            "current_run_days": 3,
            "longest_up_days": 0,
            "longest_down_days": 3,
        },
        {
            "symbol": "BBB",
            "continuity_label": "none",
            "current_direction": "",
            "current_run_days": 0,
            "longest_up_days": 2,
            "longest_down_days": 0,
        },
    ]
    assert result.loc[0, "longest_direction"] == "down"
    assert result.loc[0, "longest_days"] == 3
