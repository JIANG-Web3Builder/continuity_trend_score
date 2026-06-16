import pandas as pd

from trend_score.candles import NON_STRICT_UP_LABEL, STRICT_DOWN_LABEL, STRICT_UP_LABEL
from trend_score.compare import compare_two_waves, compare_waves, rank_waves, relative_path
from trend_score.scoring import score_review_waves, score_waves


def make_frame(symbol, closes, vols=None):
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    vols = vols or [1000 + i * 20 for i in range(len(closes))]
    return pd.DataFrame(
        {
            "ts_code": symbol,
            "trade_date": dates,
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "pre_close": closes,
            "change": 0.0,
            "pct_chg": 0.0,
            "vol": vols,
            "amount": [v * c for v, c in zip(vols, closes)],
        }
    )


def test_score_waves_rewards_low_drawdown_more_than_choppy_path():
    steady = make_frame("000001.SH", [100, 105, 110, 115, 120])
    choppy = make_frame("000001.SH", [100, 115, 104, 121, 120])
    prices = pd.concat([steady, choppy.assign(trade_date=lambda d: d["trade_date"] + pd.Timedelta(days=10))])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": steady.loc[0, "trade_date"],
                "end_date": steady.loc[4, "trade_date"],
                "start_price": 100.0,
                "end_price": 120.0,
                "points": 20.0,
                "pct_change": 20.0,
                "days": 5,
                "level": "500-800点",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": choppy.loc[0, "trade_date"] + pd.Timedelta(days=10),
                "end_date": choppy.loc[4, "trade_date"] + pd.Timedelta(days=10),
                "start_price": 100.0,
                "end_price": 120.0,
                "points": 20.0,
                "pct_change": 20.0,
                "days": 5,
                "level": "500-800点",
            },
        ]
    )

    scored = score_waves(prices, waves)

    assert scored.loc[0, "drawdown_score"] > scored.loc[1, "drawdown_score"]
    assert scored.loc[0, "total_score"] > scored.loc[1, "total_score"]
    assert {
        "continuity_label",
        "strength_score",
        "duration_score",
        "slope_score",
        "drawdown_score",
        "stability_score",
        "volume_score",
        "total_score",
        "historical_percentile",
    }.issubset(scored.columns)


def test_score_waves_labels_wave_continuity_from_price_path():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    closes = [100, 102, 104, 103, 106, 108, 106, 104]
    opens = [100, 101, 103, 104, 105, 107, 107, 105]
    prices = pd.DataFrame(
        {
            "ts_code": "AAA",
            "trade_date": dates,
            "open": opens,
            "high": [close + 1 for close in closes],
            "low": [close - 1 for close in closes],
            "close": closes,
            "pre_close": [100, *closes[:-1]],
            "change": [0, *[closes[i] - closes[i - 1] for i in range(1, len(closes))]],
            "pct_chg": 0.0,
            "vol": 1000.0,
            "amount": [close * 1000 for close in closes],
        }
    )
    waves = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "direction": "up",
                "start_date": dates[1],
                "end_date": dates[2],
                "start_price": 102.0,
                "end_price": 104.0,
                "points": 2.0,
                "pct_change": 1.96,
                "days": 2,
                "level": "300-500点",
            },
            {
                "symbol": "AAA",
                "direction": "up",
                "start_date": dates[1],
                "end_date": dates[5],
                "start_price": 102.0,
                "end_price": 108.0,
                "points": 6.0,
                "pct_change": 5.88,
                "days": 5,
                "level": "300-500点",
            },
            {
                "symbol": "AAA",
                "direction": "down",
                "start_date": dates[6],
                "end_date": dates[7],
                "start_price": 106.0,
                "end_price": 104.0,
                "points": 2.0,
                "pct_change": -1.89,
                "days": 2,
                "level": "300-500点",
            },
        ]
    )

    scored = score_waves(prices, waves)

    assert scored["continuity_label"].tolist() == [STRICT_UP_LABEL, NON_STRICT_UP_LABEL, STRICT_DOWN_LABEL]


def test_score_waves_percentile_scores_are_scoped_to_direction_and_level():
    prices = make_frame("000001.SH", [100, 105, 110, 115, 120, 118, 112, 106, 102, 98])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[2, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 3,
                "level": "500-800点",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[4, "trade_date"],
                "start_price": 100.0,
                "end_price": 120.0,
                "points": 20.0,
                "pct_change": 20.0,
                "days": 5,
                "level": "500-800点",
            },
            {
                "symbol": "000001.SH",
                "direction": "down",
                "start_date": prices.loc[5, "trade_date"],
                "end_date": prices.loc[9, "trade_date"],
                "start_price": 118.0,
                "end_price": 98.0,
                "points": 20.0,
                "pct_change": -16.95,
                "days": 5,
                "level": "300点以下",
            },
        ]
    )

    scored = score_waves(prices, waves)

    assert scored.loc[0, "strength_score"] == 100.0
    assert scored.loc[1, "strength_score"] == 100.0
    assert scored.loc[2, "strength_score"] == 100.0


def test_score_waves_excludes_open_waves_from_ranked_scoring():
    prices = make_frame("000001.SH", [100, 105, 110, 104, 98, 101])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[3, "trade_date"],
                "confirmation_date": prices.loc[3, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 4,
                "level": "large",
                "status": "confirmed",
            },
            {
                "symbol": "000001.SH",
                "direction": "down",
                "start_date": prices.loc[2, "trade_date"],
                "end_date": prices.loc[5, "trade_date"],
                "confirmation_date": pd.NaT,
                "start_price": 110.0,
                "end_price": 98.0,
                "points": 12.0,
                "pct_change": -10.91,
                "days": 4,
                "level": "large",
                "status": "open",
            },
        ]
    )

    scored = score_waves(prices, waves)

    assert scored["status"].tolist() == ["confirmed"]
    assert len(rank_waves(scored)) == 1


def test_score_waves_historical_percentile_uses_only_confirmed_history_to_date():
    prices = make_frame("000001.SH", [100, 110, 104, 100, 118, 111, 105, 135, 126])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[2, "trade_date"],
                "confirmation_date": prices.loc[2, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 3,
                "level": "large",
                "status": "confirmed",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[3, "trade_date"],
                "end_date": prices.loc[5, "trade_date"],
                "confirmation_date": prices.loc[5, "trade_date"],
                "start_price": 100.0,
                "end_price": 118.0,
                "points": 18.0,
                "pct_change": 18.0,
                "days": 3,
                "level": "large",
                "status": "confirmed",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[6, "trade_date"],
                "end_date": prices.loc[8, "trade_date"],
                "confirmation_date": prices.loc[8, "trade_date"],
                "start_price": 105.0,
                "end_price": 135.0,
                "points": 30.0,
                "pct_change": 28.57,
                "days": 3,
                "level": "large",
                "status": "confirmed",
            },
        ]
    )

    first_two = score_waves(prices, waves.iloc[:2])
    all_scored = score_waves(prices, waves)

    assert all_scored.loc[:1, "historical_percentile"].tolist() == first_two["historical_percentile"].tolist()


def test_score_review_waves_uses_full_sample_percentiles():
    prices = make_frame("000001.SH", [100, 110, 100, 120, 100, 130])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[1, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 2,
                "level": "500-800点",
                "status": "confirmed",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[2, "trade_date"],
                "end_date": prices.loc[3, "trade_date"],
                "start_price": 100.0,
                "end_price": 120.0,
                "points": 20.0,
                "pct_change": 20.0,
                "days": 2,
                "level": "500-800点",
                "status": "confirmed",
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[4, "trade_date"],
                "end_date": prices.loc[5, "trade_date"],
                "start_price": 100.0,
                "end_price": 130.0,
                "points": 30.0,
                "pct_change": 30.0,
                "days": 2,
                "level": "500-800点",
                "status": "confirmed",
            },
        ]
    )

    asof_scored = score_waves(prices, waves)
    review_scored = score_review_waves(prices, waves)

    assert asof_scored["strength_score"].tolist() == [100.0, 100.0, 100.0]
    assert review_scored["strength_score"].tolist() == [33.33, 66.67, 100.0]


def test_rank_waves_filters_by_direction_and_level():
    scored = pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "direction": ["up", "up", "down"],
            "level": ["500-800点", "300点以下", "500-800点"],
            "total_score": [80, 90, 70],
        }
    )

    ranked = rank_waves(scored, direction="up", level="500-800点")

    assert ranked["total_score"].tolist() == [80]


def test_relative_path_normalizes_wave_prices_to_start_at_100():
    prices = make_frame("000001.SH", [100, 110, 121])
    wave = {
        "symbol": "000001.SH",
        "start_date": prices.loc[0, "trade_date"],
        "end_date": prices.loc[2, "trade_date"],
    }

    path = relative_path(prices, wave, label="wave_a")

    assert path["wave"].unique().tolist() == ["wave_a"]
    assert path["relative_close"].round(2).tolist() == [100.0, 110.0, 121.0]


def test_compare_two_waves_returns_metrics_and_normalized_paths():
    prices = make_frame("000001.SH", [100, 105, 110, 120, 118, 125])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[2, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 3,
                "level": "300-500点",
                "total_score": 75.0,
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[3, "trade_date"],
                "end_date": prices.loc[5, "trade_date"],
                "start_price": 120.0,
                "end_price": 125.0,
                "points": 5.0,
                "pct_change": 4.17,
                "days": 3,
                "level": "300点以下",
                "total_score": 65.0,
            },
        ]
    )

    result = compare_two_waves(prices, waves, 0, 1)

    assert set(result.keys()) == {"metrics", "paths"}
    assert result["metrics"]["label"].tolist() == ["wave_0", "wave_1"]
    assert result["paths"]["wave"].nunique() == 2


def test_compare_waves_supports_more_than_two_waves():
    prices = make_frame("000001.SH", [100, 105, 110, 108, 101, 96, 98, 104, 111])
    waves = pd.DataFrame(
        [
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[0, "trade_date"],
                "end_date": prices.loc[2, "trade_date"],
                "start_price": 100.0,
                "end_price": 110.0,
                "points": 10.0,
                "pct_change": 10.0,
                "days": 3,
                "level": "500-800点",
                "total_score": 78.0,
            },
            {
                "symbol": "000001.SH",
                "direction": "down",
                "start_date": prices.loc[3, "trade_date"],
                "end_date": prices.loc[5, "trade_date"],
                "start_price": 108.0,
                "end_price": 96.0,
                "points": 12.0,
                "pct_change": -11.11,
                "days": 3,
                "level": "500-800点",
                "total_score": 81.0,
            },
            {
                "symbol": "000001.SH",
                "direction": "up",
                "start_date": prices.loc[6, "trade_date"],
                "end_date": prices.loc[8, "trade_date"],
                "start_price": 98.0,
                "end_price": 111.0,
                "points": 13.0,
                "pct_change": 13.27,
                "days": 3,
                "level": "1000点以上",
                "total_score": 85.0,
            },
        ]
    )

    result = compare_waves(prices, waves, [0, 1, 2])

    assert result["metrics"]["label"].tolist() == ["wave_0", "wave_1", "wave_2"]
    assert result["paths"]["wave"].nunique() == 3
