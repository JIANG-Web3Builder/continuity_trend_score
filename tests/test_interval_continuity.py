import pandas as pd

from trend_score.compare import score_interval_continuity


def _prices(symbol: str, closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    volumes = volumes or [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": dates,
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
            "pre_close": [closes[0], *closes[:-1]],
            "change": [0.0, *[closes[i] - closes[i - 1] for i in range(1, len(closes))]],
            "pct_chg": 0.0,
            "vol": volumes,
            "amount": [volume * close for volume, close in zip(volumes, closes)],
        }
    )


def test_interval_continuity_ranks_cleaner_stronger_move_first():
    prices = pd.concat(
        [
            _prices("AAA", [100, 103, 106, 109, 112], [100, 110, 120, 130, 140]),
            _prices("BBB", [100, 110, 96, 118, 106], [140, 90, 180, 80, 100]),
        ],
        ignore_index=True,
    )

    result = score_interval_continuity(prices, "2024-01-01", "2024-01-05")

    assert list(result["symbol"]) == ["AAA", "BBB"]
    assert result.loc[0, "direction"] == "up"
    assert result.loc[0, "interval_score"] > result.loc[1, "interval_score"]
    assert {
        "strength_score",
        "slope_score",
        "drawdown_score",
        "stability_score",
        "volume_score",
        "consistency_score",
        "trend_day_ratio",
        "adverse_day_ratio",
    }.issubset(result.columns)
    assert result.loc[0, "trend_day_ratio"] == 1.0
    assert result.loc[0, "consistency_score"] == 100.0


def test_interval_continuity_handles_single_symbol_short_interval():
    prices = _prices("AAA", [100, 101])

    result = score_interval_continuity(prices, "2024-01-01", "2024-01-02")

    assert len(result) == 1
    assert result.loc[0, "symbol"] == "AAA"
    assert result.loc[0, "interval_score"] >= 0
