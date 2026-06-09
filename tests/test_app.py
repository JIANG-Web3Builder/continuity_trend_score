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
