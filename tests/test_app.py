import importlib

import pandas as pd


def make_price_frame():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "trade_date": dates,
            "open": [100.0, 105.0, 103.0, 108.0],
            "high": [106.0, 107.0, 110.0, 112.0],
            "low": [99.0, 101.0, 102.0, 106.0],
            "close": [105.0, 103.0, 108.0, 111.0],
            "vol": [1000.0, 1200.0, 900.0, 1500.0],
        }
    )


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


def test_wave_mode_options_are_chinese_and_default_to_asof():
    app = importlib.import_module("app")

    assert app.WAVE_MODE_OPTIONS == ["无未来函数模式", "复盘模式"]


def test_detect_waves_for_mode_switches_between_asof_and_review_dates():
    app = importlib.import_module("app")
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame(
        {
            "ts_code": "000001.SH",
            "trade_date": dates,
            "open": [100, 105, 110, 108, 104],
            "high": [101, 106, 111, 109, 105],
            "low": [99, 104, 109, 107, 103],
            "close": [100, 105, 110, 108, 104],
            "pre_close": [100, 100, 105, 110, 108],
            "vol": [1000] * 5,
        }
    )

    asof = app.detect_waves_for_mode(
        prices,
        "000001.SH",
        app.WAVE_MODE_OPTIONS[0],
        min_reversal=4,
        min_wave_days=1,
    )
    review = app.detect_waves_for_mode(
        prices,
        "000001.SH",
        app.WAVE_MODE_OPTIONS[1],
        min_reversal=4,
        min_wave_days=1,
    )

    assert asof[asof["status"] == "confirmed"].loc[0, "end_date"] == pd.Timestamp("2024-01-05")
    assert review.loc[0, "end_date"] == pd.Timestamp("2024-01-03")
    assert set(review["status"]) == {"confirmed"}


def test_extend_latest_wave_for_chart_does_not_extend_confirmed_wave_to_latest_date():
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
            "status": ["confirmed", "confirmed"],
        }
    )

    display_waves = app.extend_latest_wave_for_chart(waves, prices)

    assert display_waves.loc[0, "end_date"] == pd.Timestamp("2024-01-10")
    assert waves.loc[0, "end_date"] == pd.Timestamp("2024-01-10")


def test_split_waves_for_display_keeps_open_wave_out_of_confirmed_history():
    app = importlib.import_module("app")
    waves = pd.DataFrame(
        {
            "direction": ["up", "down"],
            "status": ["confirmed", "open"],
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "end_date": pd.to_datetime(["2024-01-05", "2024-01-10"]),
            "start_price": [100.0, 110.0],
            "end_price": [110.0, 104.0],
            "points": [10.0, 6.0],
            "pct_change": [10.0, -5.45],
            "days": [5, 6],
        }
    )

    confirmed, open_wave = app.split_waves_for_display(waves)

    assert confirmed["status"].tolist() == ["confirmed"]
    assert open_wave["status"] == "open"
    assert open_wave["points"] == 6.0


def test_chart_waves_for_display_includes_latest_open_wave_for_shading():
    app = importlib.import_module("app")
    ranked = pd.DataFrame(
        {
            "direction": ["up"],
            "status": ["confirmed"],
            "start_date": pd.to_datetime(["2024-01-01"]),
            "end_date": pd.to_datetime(["2024-01-05"]),
            "level": ["middle"],
        }
    )
    open_wave = pd.Series(
        {
            "direction": "down",
            "status": "open",
            "start_date": pd.Timestamp("2024-01-05"),
            "end_date": pd.Timestamp("2024-01-12"),
            "level": "middle",
            "days": 8,
        }
    )

    chart_waves = app.chart_waves_for_display(ranked, open_wave, direction="全部", level="全部")

    assert chart_waves["status"].tolist() == ["confirmed", "open"]
    assert chart_waves.iloc[-1]["end_date"] == pd.Timestamp("2024-01-12")


def test_extend_latest_wave_for_chart_keeps_open_wave_without_extending_confirmed_rows():
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
            "start_date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "end_date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
            "status": ["confirmed", "open"],
        }
    )

    display_waves = app.extend_latest_wave_for_chart(waves, prices)

    assert display_waves["status"].tolist() == ["confirmed", "open"]
    assert display_waves.loc[0, "end_date"] == pd.Timestamp("2024-01-10")


def test_wave_background_spans_do_not_overlap_when_waves_overlap():
    app = importlib.import_module("app")
    waves = pd.DataFrame(
        {
            "direction": ["up", "down", "up"],
            "start_date": pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-10"]),
            "end_date": pd.to_datetime(["2026-01-08", "2026-01-12", "2026-01-15"]),
        }
    )

    spans = app.wave_background_spans(waves)

    assert [(span["x0"], span["x1"], span["direction"]) for span in spans] == [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-04"), "up"),
        (pd.Timestamp("2026-01-04"), pd.Timestamp("2026-01-10"), "down"),
        (pd.Timestamp("2026-01-10"), pd.Timestamp("2026-01-15"), "up"),
    ]


def test_build_price_volume_figure_uses_candles_volume_and_range_slider():
    app = importlib.import_module("app")
    import plotly.graph_objects as go

    waves = pd.DataFrame(
        {
            "direction": ["up"],
            "start_date": pd.to_datetime(["2026-01-01"]),
            "end_date": pd.to_datetime(["2026-01-03"]),
        }
    )

    fig = app.build_price_volume_figure(go, make_price_frame(), waves, title="trial")

    assert [trace.type for trace in fig.data] == ["candlestick", "bar"]
    assert fig.layout.xaxis.rangeslider.visible is True
    assert fig.layout.height >= 620
    assert len(fig.layout.shapes) == 1


def test_open_wave_summary_items_keep_full_dates():
    app = importlib.import_module("app")
    open_wave = pd.Series(
        {
            "direction": "down",
            "start_date": pd.Timestamp("2026-05-25"),
            "end_date": pd.Timestamp("2026-06-05"),
            "points": 124.83,
            "pct_change": -3.01,
            "reversal_progress": 0.0,
        }
    )

    items = app.open_wave_summary_items(open_wave)

    assert ("起点", "2026-05-25") in items
    assert ("最新日期", "2026-06-05") in items


def test_format_display_dataframe_translates_score_columns_and_values():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "direction": ["up", "down"],
            "status": ["confirmed", "open"],
            "start_date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
            "total_score": [88.5, 66.0],
            "strength_score": [90.0, 55.0],
        }
    )

    display = app.format_display_dataframe(
        raw,
        ["direction", "status", "start_date", "total_score", "strength_score"],
    )

    assert display.columns.tolist() == ["方向", "状态", "起始日期", "总分", "强度分"]
    assert display["方向"].tolist() == ["上涨", "下跌"]
    assert display["状态"].tolist() == ["已确认", "未完成"]
    assert display["起始日期"].tolist() == ["2026-01-01", "2026-01-05"]


def test_format_display_dataframe_translates_strict_run_columns():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "continuity_label": ["严格连阳"],
            "current_direction": ["up"],
            "current_run_days": [3],
            "amplitude_pct": [6.25],
            "longest_up_days": [4],
            "longest_down_days": [2],
            "longest_direction": ["up"],
            "longest_days": [4],
        }
    )

    display = app.format_display_dataframe(raw)

    assert display.columns.tolist() == [
        "连续标签",
        "当前方向",
        "当前连续天数",
        "振幅(%)",
        "最长连阳",
        "最长连阴",
        "最长方向",
        "最长连续天数",
    ]


def test_strict_run_table_columns_drop_redundant_direction_and_status():
    app = importlib.import_module("app")

    assert app.strict_run_table_columns() == [
        "continuity_label",
        "start_date",
        "end_date",
        "days",
        "pct_change",
        "start_price",
        "end_price",
    ]


def test_strict_run_summary_table_columns_use_amplitude_instead_of_direction_columns():
    app = importlib.import_module("app")

    columns = app.strict_run_summary_table_columns()

    assert "current_direction" not in columns
    assert "longest_direction" not in columns
    assert "amplitude_pct" in columns


def test_strict_run_display_rows_keeps_full_history():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "end_date": pd.date_range("2024-01-01", periods=15, freq="D"),
            "days": [4] * 15,
        }
    )

    display = app.strict_run_display_rows(raw)

    assert len(display) == 15
    assert display["end_date"].tolist() == list(reversed(raw["end_date"].tolist()))


def test_strict_run_display_rows_only_keeps_strict_runs_longer_than_three_days():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "continuity_label": ["非严格连阳", "严格连阳", "严格连阴", "严格连阳"],
            "end_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "days": [8, 3, 4, 5],
        }
    )

    display = app.strict_run_display_rows(raw)

    assert display[["continuity_label", "days"]].to_dict("records") == [
        {"continuity_label": "严格连阳", "days": 5},
        {"continuity_label": "严格连阴", "days": 4},
    ]


def test_render_score_table_includes_continuity_label_column():
    app = importlib.import_module("app")

    class FakeStreamlit:
        def __init__(self):
            self.frame = None

        def subheader(self, _text):
            pass

        def info(self, _text):
            pass

        def dataframe(self, frame, **_kwargs):
            self.frame = frame

    waves = pd.DataFrame(
        {
            "direction": ["up"],
            "level": ["中"],
            "continuity_label": ["非严格连阳"],
            "start_date": pd.to_datetime(["2026-04-27"]),
            "end_date": pd.to_datetime(["2026-05-11"]),
            "points": [138.67],
            "pct_change": [3.39],
            "days": [8],
            "total_score": [80.0],
        }
    )
    fake_st = FakeStreamlit()

    app._render_score_table(fake_st, waves)

    assert app.DISPLAY_COLUMN_LABELS["continuity_label"] in fake_st.frame.columns


def test_analysis_tab_labels_include_strict_run_view():
    app = importlib.import_module("app")

    assert app.ANALYSIS_TAB_LABELS == ["波段评分表", "单个波段详情", "波段对比", "区间横向对比", "连阴连阳识别"]


def test_sort_option_label_is_chinese():
    app = importlib.import_module("app")

    assert app.sort_option_label("total_score") == "总分"
    assert app.sort_option_label("end_date") == "结束日期"


def test_render_analysis_sections_uses_tabs():
    app = importlib.import_module("app")

    class FakeTab:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeStreamlit:
        def __init__(self):
            self.labels = None

        def tabs(self, labels):
            self.labels = labels
            return [FakeTab() for _ in labels]

    fake_st = FakeStreamlit()
    calls = []

    app.render_analysis_sections(
        fake_st,
        renderers=[
            ("波段评分表", lambda: calls.append("table")),
            ("单个波段详情", lambda: calls.append("detail")),
            ("波段对比", lambda: calls.append("compare")),
            ("区间横向对比", lambda: calls.append("interval")),
        ],
    )

    assert fake_st.labels == ["波段评分表", "单个波段详情", "波段对比", "区间横向对比"]
    assert calls == ["table", "detail", "compare", "interval"]
