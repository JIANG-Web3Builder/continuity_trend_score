import importlib
from pathlib import Path

import pandas as pd


def app_source_path() -> Path:
    return Path(__file__).resolve().parents[1] / "app.py"


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


def test_app_does_not_import_wave_day_constant_from_waves_module():
    source = app_source_path().read_text(encoding="utf-8")

    assert "from trend_score.waves import MIN_WAVE_DAYS" not in source


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


def test_continuity_type_options_are_business_terms():
    app = importlib.import_module("app")

    assert app.CONTINUITY_TYPE_OPTIONS == ["区间连续性", "连阳连阴连续性"]


def test_detect_interval_waves_uses_review_history_and_asof_latest_wave():
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

    waves = app.detect_interval_waves(
        prices,
        "000001.SH",
        min_reversal=4,
        min_wave_days=1,
    )

    assert waves.loc[0, "status"] == "confirmed"
    assert waves.loc[0, "end_date"] == pd.Timestamp("2024-01-03")
    assert waves.loc[1, "status"] == "open"
    assert waves.loc[1, "start_date"] == pd.Timestamp("2024-01-03")


def test_detect_interval_waves_keeps_latest_short_wave_open_out_of_scored_history():
    app = importlib.import_module("app")
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame(
        {
            "ts_code": "000001.SH",
            "trade_date": dates,
            "open": [100, 102, 104, 106, 108, 110, 112, 114, 116, 112],
            "high": [101, 103, 105, 107, 109, 111, 113, 115, 117, 113],
            "low": [99, 101, 103, 105, 107, 109, 111, 113, 115, 111],
            "close": [100, 102, 104, 106, 108, 110, 112, 114, 116, 112],
            "pre_close": [100, 100, 102, 104, 106, 108, 110, 112, 114, 116],
            "vol": [1000] * 10,
        }
    )

    waves = app.detect_interval_waves(prices, "000001.SH", min_reversal=4, min_wave_days=1)
    confirmed, open_wave = app.split_waves_for_display(waves)
    scored = app.score_interval_waves(prices, confirmed)

    assert app.MIN_WAVE_DAYS == 10
    assert scored["status"].tolist() == ["confirmed"]
    assert open_wave["status"] == "open"
    assert int(open_wave["days"]) == 2


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
            "days": 10,
        }
    )

    chart_waves = app.chart_waves_for_display(ranked, open_wave, direction="全部", level="全部")

    assert chart_waves["status"].tolist() == ["confirmed", "open"]
    assert chart_waves.iloc[-1]["end_date"] == pd.Timestamp("2024-01-12")


def test_chart_waves_for_display_excludes_latest_open_wave_shorter_than_minimum():
    app = importlib.import_module("app")
    ranked = pd.DataFrame(
        {
            "direction": ["up"],
            "status": ["confirmed"],
            "start_date": pd.to_datetime(["2024-01-01"]),
            "end_date": pd.to_datetime(["2024-01-10"]),
            "level": ["middle"],
        }
    )
    open_wave = pd.Series(
        {
            "direction": "down",
            "status": "open",
            "start_date": pd.Timestamp("2024-01-10"),
            "end_date": pd.Timestamp("2024-01-18"),
            "level": "middle",
            "days": 9,
        }
    )

    chart_waves = app.chart_waves_for_display(ranked, open_wave, direction="全部", level="全部")

    assert chart_waves["status"].tolist() == ["confirmed"]


def test_chart_waves_for_display_uses_custom_minimum_for_streak_page():
    app = importlib.import_module("app")
    ranked = pd.DataFrame(columns=["direction", "status", "start_date", "end_date", "level"])
    open_wave = pd.Series(
        {
            "direction": "up",
            "status": "open",
            "start_date": pd.Timestamp("2024-01-10"),
            "end_date": pd.Timestamp("2024-01-12"),
            "level": "300点以下",
            "days": 3,
        }
    )

    chart_waves = app.chart_waves_for_display(ranked, open_wave, min_wave_days=3)

    assert chart_waves["status"].tolist() == ["open"]


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


def test_wave_background_spans_exclude_waves_shorter_than_minimum():
    app = importlib.import_module("app")
    waves = pd.DataFrame(
        {
            "direction": ["up", "down"],
            "start_date": pd.to_datetime(["2026-01-01", "2026-01-12"]),
            "end_date": pd.to_datetime(["2026-01-09", "2026-01-25"]),
            "days": [9, 14],
        }
    )

    spans = app.wave_background_spans(waves)

    assert [(span["x0"], span["x1"], span["direction"]) for span in spans] == [
        (pd.Timestamp("2026-01-12"), pd.Timestamp("2026-01-25"), "down"),
    ]


def test_wave_background_spans_respect_custom_minimum():
    app = importlib.import_module("app")
    waves = pd.DataFrame(
        {
            "direction": ["up", "down"],
            "start_date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
            "end_date": pd.to_datetime(["2026-01-03", "2026-01-07"]),
            "days": [3, 3],
        }
    )

    spans = app.wave_background_spans(waves, min_wave_days=3)

    assert [(span["x0"], span["x1"], span["direction"]) for span in spans] == [
        (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-03"), "up"),
        (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"), "down"),
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


def test_format_display_dataframe_formats_price_columns_to_two_decimals():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "close": [101.2345, 104.1],
            "end_close": [99.8765, 100],
            "start_price": [100.1234, 110.0],
            "end_price": [120.5678, 104.1],
        }
    )

    display = app.format_display_dataframe(raw)

    assert display["收盘价"].tolist() == ["101.23", "104.10"]
    assert display["结束收盘价"].tolist() == ["99.88", "100.00"]
    assert display["起点价格"].tolist() == ["100.12", "110.00"]
    assert display["终点价格"].tolist() == ["120.57", "104.10"]


def test_format_display_dataframe_translates_streak_columns():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "continuity_label": ["连阳连续性"],
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


def test_format_display_dataframe_translates_streak_outcome_distribution_columns():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "total_signals": [12],
            "latest_start_date": pd.to_datetime(["2026-05-08"]),
            "latest_end_date": pd.to_datetime(["2026-05-11"]),
            "latest_streak_days": [4],
            "signals_1d": [10],
            "up_count_1d": [6],
            "down_count_1d": [3],
            "flat_count_1d": [1],
            "up_rate_1d": [60.0],
            "down_rate_1d": [30.0],
            "avg_return_1d": [0.85],
            "return_1d": [1.2],
            "start_time": pd.to_datetime(["2026-05-08"]),
            "end_time": pd.to_datetime(["2026-05-11"]),
        }
    )

    display = app.format_display_dataframe(raw)

    assert display.columns.tolist() == [
        "样本数",
        "最近起始时间",
        "最近结束时间",
        "最近连续天数",
        "1日有效样本数",
        "1日上涨次数",
        "1日下跌次数",
        "1日持平次数",
        "1日上涨占比(%)",
        "1日下跌占比(%)",
        "1日平均涨跌幅(%)",
        "1日涨跌幅(%)",
        "起始时间",
        "结束时间",
    ]
    assert display["最近起始时间"].tolist() == ["2026-05-08"]
    assert display["最近结束时间"].tolist() == ["2026-05-11"]
    assert "触发条件" not in display.columns
    assert "1日延续胜率(%)" not in display.columns
    assert "1日延续成功" not in display.columns


def test_format_win_rate_summary_dataframe_labels_streak_days_by_direction():
    app = importlib.import_module("app")
    raw = pd.DataFrame(
        {
            "direction": ["up", "down", "up"],
            "streak_days": [4, 5, 10],
            "continuity_label": ["连阳连续性", "连阴连续性", "连阳连续性"],
        }
    )

    display = app._format_win_rate_summary_dataframe(raw)

    assert display["streak_days"].tolist() == ["四连阳", "五连阴", "十连阳"]
    assert raw["streak_days"].tolist() == [4, 5, 10]


def test_filter_waves_by_continuity_keeps_only_requested_labels():
    app = importlib.import_module("app")
    waves = pd.DataFrame(
        {
            "continuity_label": ["区间上涨连续性", "连阳连续性", "连阴连续性"],
            "total_score": [80.0, 90.0, 70.0],
        }
    )

    result = app.filter_waves_by_continuity(waves, {"连阳连续性", "连阴连续性"})

    assert result["continuity_label"].tolist() == ["连阳连续性", "连阴连续性"]


def test_detect_streak_waves_uses_minimum_continuous_days_as_wave_threshold():
    app = importlib.import_module("app")
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "ts_code": "AAA",
            "trade_date": dates,
            "open": [100, 101, 102, 104, 103, 102],
            "high": [102, 103, 104, 105, 104, 103],
            "low": [99, 100, 101, 103, 101, 100],
            "close": [101, 102, 103, 104, 102, 101],
            "vol": [1000] * 6,
        }
    )

    min_three = app.detect_streak_waves(prices, min_days=3)
    min_two = app.detect_streak_waves(prices, min_days=2)

    assert min_three[["direction", "days", "status", "points"]].to_dict("records") == [
        {"direction": "up", "days": 3, "status": "confirmed", "points": 3.0}
    ]
    assert min_two[["direction", "days", "status", "points"]].to_dict("records") == [
        {"direction": "up", "days": 3, "status": "confirmed", "points": 3.0},
        {"direction": "down", "days": 2, "status": "open", "points": 2.0},
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
            "level": ["300-500点"],
            "continuity_label": ["区间上涨连续性"],
            "start_date": pd.to_datetime(["2026-04-27"]),
            "end_date": pd.to_datetime(["2026-05-11"]),
            "confirmation_date": pd.to_datetime(["2026-05-12"]),
            "points": [138.67],
            "pct_change": [3.39],
            "days": [8],
            "total_score": [80.0],
        }
    )
    fake_st = FakeStreamlit()

    app._render_score_table(fake_st, waves)

    assert app.DISPLAY_COLUMN_LABELS["continuity_label"] in fake_st.frame.columns
    assert "确认日期" not in fake_st.frame.columns


def test_render_streak_win_rates_shows_summary_table_and_recent_signal_details_without_matrix_heading():
    app = importlib.import_module("app")
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    prices = pd.DataFrame(
        {
            "ts_code": "AAA",
            "trade_date": dates,
            "open": [100, 101, 102, 104, 103, 102, 98],
            "high": [102, 103, 104, 105, 104, 103, 100],
            "low": [99, 100, 101, 102, 101, 99, 97],
            "close": [101, 102, 103, 103, 102, 100, 99],
            "vol": [1000] * 7,
        }
    )

    class FakeStreamlit:
        def __init__(self):
            self.subheaders = []
            self.frames = []

        def subheader(self, text):
            self.subheaders.append(text)

        def info(self, _text):
            pass

        def dataframe(self, frame, **_kwargs):
            self.frames.append(frame)

    fake_st = FakeStreamlit()

    app._render_streak_win_rates(fake_st, prices, min_days=2)

    assert fake_st.subheaders == ["胜率统计", "信号明细表"]
    assert len(fake_st.frames) == 2
    assert "触发条件" not in fake_st.frames[0].columns
    assert fake_st.frames[0].columns.tolist() == [
        "连续标签",
        "连续天数",
        "样本数",
        "最近结束时间",
        "1日有效样本数",
        "1日胜率(%)",
        "1日平均涨跌幅(%)",
        "3日胜率(%)",
        "3日平均涨跌幅(%)",
    ]
    assert "观察周期" not in fake_st.frames[0].columns
    assert "1日延续胜率(%)" not in fake_st.frames[0].columns
    assert "3日延续胜率(%)" not in fake_st.frames[0].columns
    assert fake_st.frames[0]["连续天数"].tolist() == ["二连阳", "三连阳", "二连阴", "三连阴"]
    assert "信号日" not in fake_st.frames[1].columns
    assert "1日延续成功" not in fake_st.frames[1].columns
    assert fake_st.frames[1].columns.tolist() == [
        "起始时间",
        "结束时间",
        "方向",
        "连续标签",
        "连续天数",
        "结束收盘价",
        "1日涨跌幅(%)",
        "3日涨跌幅(%)",
        "5日涨跌幅(%)",
        "10日涨跌幅(%)",
    ]
    assert fake_st.frames[1]["连续天数"].tolist() == [3, 3]


def test_level_options_use_fixed_point_buckets():
    app = importlib.import_module("app")

    assert app.LEVEL_OPTIONS == ["全部", "300点以下", "300-500点", "500-800点", "800-1000点", "1000点以上"]


def test_wave_compare_label_uses_date_range_without_number_prefix():
    app = importlib.import_module("app")
    wave = pd.Series(
        {
            "direction": "up",
            "level": "300-500点",
            "start_date": pd.Timestamp("2026-03-19"),
            "end_date": pd.Timestamp("2026-03-23"),
            "total_score": 82.14,
        }
    )

    label = app._wave_compare_label(12, wave)

    assert label == "2026-03-19 -> 2026-03-23 · 上涨 · 300-500点 · 82.1"
    assert not label.startswith("W")


def test_interval_analysis_tab_labels_do_not_include_streak_win_rate():
    app = importlib.import_module("app")

    assert app.INTERVAL_ANALYSIS_TAB_LABELS == ["波段评分表", "单个波段详情", "波段对比", "区间横向对比"]
    assert "胜率统计" not in app.INTERVAL_ANALYSIS_TAB_LABELS


def test_streak_analysis_tab_labels_include_win_rate():
    app = importlib.import_module("app")

    assert app.STREAK_ANALYSIS_TAB_LABELS == ["波段评分表", "单个波段详情", "波段对比", "区间横向对比", "胜率统计"]


def test_continuity_type_selector_does_not_render_explanatory_caption():
    app = importlib.import_module("app")

    class FakeColumn:
        def segmented_control(self, _label, _options, default):
            return default

        def caption(self, text):
            raise AssertionError(f"unexpected caption: {text}")

    class FakeStreamlit:
        def columns(self, _spec):
            return [FakeColumn(), FakeColumn()]

    assert app._render_continuity_type_selector(FakeStreamlit()) == "区间连续性"


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
