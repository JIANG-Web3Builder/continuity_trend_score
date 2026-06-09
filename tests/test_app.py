import importlib

import pandas as pd


def test_app_import_does_not_require_streamlit_dependency():
    app = importlib.import_module("app")

    assert hasattr(app, "run_dashboard")


def test_filter_scored_waves_by_date_preserves_precomputed_scores():
    app = importlib.import_module("app")
    scored = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "end_date": pd.to_datetime(["2024-01-10", "2024-02-10"]),
            "historical_percentile": [25.0, 75.0],
            "total_score": [60.0, 90.0],
        }
    )

    filtered = app.filter_scored_waves_by_date(scored, (pd.Timestamp("2024-02-05"), pd.Timestamp("2024-02-20")))

    assert filtered["historical_percentile"].tolist() == [75.0]
    assert filtered["total_score"].tolist() == [90.0]


def test_resolve_reversal_threshold_modes():
    app = importlib.import_module("app")

    assert app.resolve_reversal_threshold("自动", 0).points is None
    assert app.resolve_reversal_threshold("自动", 0).pct is None
    assert app.resolve_reversal_threshold("百分比", 3).pct == 3.0
    assert app.resolve_reversal_threshold("点数", 80).points == 80.0


def test_extend_latest_wave_for_chart_reaches_latest_price_date():
    app = importlib.import_module("app")
    prices = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"]),
            "close": [100.0, 120.0, 116.0],
        }
    )
    waves = pd.DataFrame(
        {
            "direction": ["up", "down"],
            "start_date": pd.to_datetime(["2024-01-01", "2023-12-01"]),
            "end_date": pd.to_datetime(["2024-01-10", "2023-12-10"]),
        }
    )

    display_waves = app.extend_latest_wave_for_chart(waves, prices)

    assert display_waves.loc[0, "end_date"] == pd.Timestamp("2024-01-20")
    assert waves.loc[0, "end_date"] == pd.Timestamp("2024-01-10")
