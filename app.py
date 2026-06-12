from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trend_score.catalog import ASSET_GROUPS, asset_options, get_assets
from trend_score.candles import STRICT_DOWN_LABEL, STRICT_UP_LABEL, detect_strict_runs, summarize_strict_runs
from trend_score.compare import compare_waves, rank_waves, score_interval_continuity
from trend_score.data import available_symbol_files, display_name, load_symbol_data
from trend_score.scoring import score_review_waves, score_waves
from trend_score.waves import MIN_WAVE_DAYS, detect_review_waves, detect_waves


DEFAULT_DATA_DIR = Path("data")
WAVE_MODE_ASOF = "无未来函数模式"
WAVE_MODE_REVIEW = "复盘模式"
WAVE_MODE_OPTIONS = [WAVE_MODE_ASOF, WAVE_MODE_REVIEW]
DIRECTION_LABELS = {"全部": "全部", "up": "上涨", "down": "下跌"}
LEVEL_OPTIONS = ["全部", "小", "中", "大", "超大"]
GROUP_ORDER = ["index", "sector", "commodity"]
ANALYSIS_TAB_LABELS = ["波段评分表", "单个波段详情", "波段对比", "区间横向对比", "连阴连阳识别"]
UP_COLOR = "#d94b3d"
DOWN_COLOR = "#16a37f"
UP_BAND = "rgba(217, 75, 61, 0.10)"
DOWN_BAND = "rgba(22, 163, 127, 0.10)"
DISPLAY_COLUMN_LABELS = {
    "symbol": "代码",
    "name": "名称",
    "label": "标签",
    "wave": "波段",
    "step": "进程",
    "direction": "方向",
    "status": "状态",
    "level": "级别",
    "start_date": "起始日期",
    "end_date": "结束日期",
    "extreme_date": "极值日期",
    "confirmation_date": "确认日期",
    "trade_date": "交易日期",
    "start_price": "起点价格",
    "end_price": "终点价格",
    "extreme_price": "极值价格",
    "confirmation_price": "确认价",
    "close": "收盘价",
    "relative_close": "归一化价格",
    "points": "涨跌点数",
    "pct_change": "涨跌幅(%)",
    "days": "持续天数",
    "slope": "斜率",
    "max_adverse_pct": "最大逆向波动(%)",
    "trend_day_ratio": "顺势天数占比",
    "adverse_day_ratio": "逆势天数占比",
    "continuity_label": "连续标签",
    "current_direction": "当前方向",
    "current_run_days": "当前连续天数",
    "amplitude_pct": "振幅(%)",
    "longest_up_days": "最长连阳",
    "longest_down_days": "最长连阴",
    "longest_direction": "最长方向",
    "longest_days": "最长连续天数",
    "total_score": "总分",
    "historical_percentile": "历史分位(%)",
    "interval_score": "区间评分",
    "strength_score": "强度分",
    "duration_score": "持续分",
    "slope_score": "斜率分",
    "drawdown_score": "回撤控制分",
    "stability_score": "稳定性分",
    "volume_score": "量能配合分",
    "consistency_score": "方向一致性分",
}
STATUS_LABELS = {"confirmed": "已确认", "open": "未完成"}
SORT_LABELS = {
    "total_score": "总分",
    "end_date": "结束日期",
    "points": "涨跌点数",
    "days": "持续天数",
}


@dataclass(frozen=True)
class ReversalThreshold:
    points: float | None = None
    pct: float | None = None


def resolve_reversal_threshold(mode: str, value: float) -> ReversalThreshold:
    if mode == "百分比":
        return ReversalThreshold(pct=float(value) if value > 0 else None)
    if mode == "点数":
        return ReversalThreshold(points=float(value) if value > 0 else None)
    return ReversalThreshold()


def build_wave_scores(
    df: pd.DataFrame,
    symbol: str,
    min_reversal: float | None = None,
    min_reversal_pct: float | None = None,
) -> pd.DataFrame:
    waves = detect_waves(df, symbol=symbol, min_reversal=min_reversal, min_reversal_pct=min_reversal_pct)
    return score_waves(df, waves)


def detect_waves_for_mode(
    df: pd.DataFrame,
    symbol: str,
    wave_mode: str,
    min_reversal: float | None = None,
    min_reversal_pct: float | None = None,
    min_wave_days: int = MIN_WAVE_DAYS,
) -> pd.DataFrame:
    detector = detect_review_waves if wave_mode == WAVE_MODE_REVIEW else detect_waves
    return detector(
        df,
        symbol=symbol,
        min_reversal=min_reversal,
        min_reversal_pct=min_reversal_pct,
        min_wave_days=min_wave_days,
    )


def score_waves_for_mode(df: pd.DataFrame, waves: pd.DataFrame, wave_mode: str) -> pd.DataFrame:
    scorer = score_review_waves if wave_mode == WAVE_MODE_REVIEW else score_waves
    return scorer(df, waves)


def split_waves_for_display(waves: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return confirmed history and the latest open wave as separate display objects."""
    if waves.empty or "status" not in waves:
        return waves.copy(), None

    confirmed = waves[waves["status"] == "confirmed"].copy().reset_index(drop=True)
    open_waves = waves[waves["status"] == "open"].copy()
    if open_waves.empty:
        return confirmed, None
    latest_index = pd.to_datetime(open_waves["end_date"], errors="coerce").idxmax()
    return confirmed, open_waves.loc[latest_index].copy()


def filter_scored_waves_by_date(scored: pd.DataFrame, date_range) -> pd.DataFrame:
    """Filter already-scored waves to the selected display interval without recalculating ranks."""
    if scored.empty or not isinstance(date_range, tuple) or len(date_range) != 2:
        return scored.copy()
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask = (pd.to_datetime(scored["end_date"]) >= start_date) & (pd.to_datetime(scored["start_date"]) <= end_date)
    return scored.loc[mask].reset_index(drop=True)


def extend_latest_wave_for_chart(waves: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return wave rows for chart shading without extending confirmed rows."""
    if waves.empty:
        return waves.copy()

    return waves.copy().reset_index(drop=True)


def chart_waves_for_display(
    ranked_waves: pd.DataFrame,
    open_wave: pd.Series | None,
    direction: str = "全部",
    level: str = "全部",
) -> pd.DataFrame:
    """Add the latest open wave to chart shading when it matches active filters."""
    chart_waves = ranked_waves.copy()
    if open_wave is None:
        return chart_waves.reset_index(drop=True)
    if direction in {"up", "down"} and open_wave.get("direction") != direction:
        return chart_waves.reset_index(drop=True)
    all_level_labels = {"全部", LEVEL_OPTIONS[0]}
    if level not in all_level_labels and open_wave.get("level") != level:
        return chart_waves.reset_index(drop=True)
    if int(open_wave.get("days", 0) or 0) < MIN_WAVE_DAYS:
        return chart_waves.reset_index(drop=True)

    open_frame = pd.DataFrame([open_wave.to_dict()])
    return pd.concat([chart_waves, open_frame], ignore_index=True, sort=False)


def wave_background_spans(waves: pd.DataFrame) -> list[dict[str, object]]:
    """Return non-overlapping wave background spans ordered by start date."""
    if waves.empty:
        return []

    required = {"start_date", "end_date", "direction"}
    if not required.issubset(waves.columns):
        return []

    ordered = waves.copy()
    ordered["start_date"] = pd.to_datetime(ordered["start_date"], errors="coerce")
    ordered["end_date"] = pd.to_datetime(ordered["end_date"], errors="coerce")
    ordered = ordered.dropna(subset=["start_date", "end_date"]).sort_values(["start_date", "end_date"])
    if "days" in ordered.columns:
        days = pd.to_numeric(ordered["days"], errors="coerce").fillna(0)
        ordered = ordered.loc[days >= MIN_WAVE_DAYS]
    spans: list[dict[str, object]] = []
    rows = list(ordered.iterrows())
    for position, (_, wave) in enumerate(rows):
        start = pd.Timestamp(wave["start_date"])
        end = pd.Timestamp(wave["end_date"])
        if position + 1 < len(rows):
            next_start = pd.Timestamp(rows[position + 1][1]["start_date"])
            if next_start < end:
                end = next_start
        if end <= start:
            continue
        spans.append({"x0": start, "x1": end, "direction": wave["direction"]})
    return spans


def build_price_volume_figure(go, df: pd.DataFrame, waves: pd.DataFrame, title: str = "K线与波段区间"):
    """Build the main price workspace with K-line, volume, wave bands, and range slider."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.035,
    )
    candle_colors = _volume_colors(df)
    fig.add_trace(
        go.Candlestick(
            x=df["trade_date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing={"line": {"color": UP_COLOR}, "fillcolor": UP_COLOR},
            decreasing={"line": {"color": DOWN_COLOR}, "fillcolor": DOWN_COLOR},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["trade_date"],
            y=df["vol"],
            name="成交量",
            marker={"color": candle_colors, "opacity": 0.62},
        ),
        row=2,
        col=1,
    )
    for span in wave_background_spans(waves):
        color = UP_BAND if span["direction"] == "up" else DOWN_BAND
        fig.add_vrect(
            x0=span["x0"],
            x1=span["x1"],
            fillcolor=color,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )
    fig.update_layout(
        template="plotly_white",
        height=660,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        title=title,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172033"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1.0},
        hovermode="x unified",
        xaxis={"rangeslider": {"visible": True, "thickness": 0.08}, "showspikes": True},
        xaxis2={"showspikes": True},
        yaxis={"title": "价格", "gridcolor": "#edf1f6"},
        yaxis2={"title": "成交量", "gridcolor": "#edf1f6"},
    )
    fig.update_xaxes(showgrid=False)
    return fig


def _volume_colors(df: pd.DataFrame) -> list[str]:
    return [UP_COLOR if close >= open_ else DOWN_COLOR for open_, close in zip(df["open"], df["close"])]


def open_wave_summary_items(open_wave: pd.Series) -> list[tuple[str, str]]:
    direction = "上涨" if open_wave["direction"] == "up" else "下跌"
    start = pd.Timestamp(open_wave["start_date"]).strftime("%Y-%m-%d")
    end = pd.Timestamp(open_wave["end_date"]).strftime("%Y-%m-%d")
    return [
        ("方向", direction),
        ("起点", start),
        ("最新日期", end),
        ("当前点数", f"{float(open_wave['points']):.2f}"),
        ("当前涨跌幅", f"{float(open_wave['pct_change']):.2f}%"),
        ("反转确认进度", f"{float(open_wave.get('reversal_progress', 0.0)):.1f}%"),
    ]


def render_analysis_sections(st, renderers: list[tuple[str, object]]) -> None:
    tabs = st.tabs([label for label, _ in renderers])
    for tab, (_, render) in zip(tabs, renderers):
        with tab:
            render()


def sort_option_label(value: str) -> str:
    return SORT_LABELS.get(value, value)


def format_display_dataframe(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    display = _format_dates(df)
    if columns is not None:
        existing_columns = [column for column in columns if column in display.columns]
        display = display[existing_columns].copy()
    if "direction" in display:
        display["direction"] = display["direction"].map(lambda value: DIRECTION_LABELS.get(value, value))
    for direction_column in ["current_direction", "longest_direction"]:
        if direction_column in display:
            display[direction_column] = display[direction_column].map(lambda value: DIRECTION_LABELS.get(value, value))
    if "status" in display:
        display["status"] = display["status"].map(lambda value: STATUS_LABELS.get(value, value))
    return display.rename(columns={column: DISPLAY_COLUMN_LABELS.get(column, column) for column in display.columns})


def strict_run_table_columns() -> list[str]:
    return ["continuity_label", "start_date", "end_date", "days", "pct_change", "start_price", "end_price"]


def strict_run_summary_table_columns() -> list[str]:
    return [
        "symbol",
        "name",
        "continuity_label",
        "start_date",
        "end_date",
        "amplitude_pct",
        "current_run_days",
        "longest_up_days",
        "longest_down_days",
        "longest_days",
    ]


def strict_run_summary_interval(st, prices: pd.DataFrame, default_date_range, key: str):
    min_date = prices["trade_date"].min().date()
    max_date = prices["trade_date"].max().date()
    value = (min_date, max_date)
    if isinstance(default_date_range, tuple) and len(default_date_range) == 2:
        start = max(pd.Timestamp(default_date_range[0]).date(), min_date)
        end = min(pd.Timestamp(default_date_range[1]).date(), max_date)
        if start <= end:
            value = (start, end)

    interval = st.date_input(
        "横向统计区间",
        value=value,
        min_value=min_date,
        max_value=max_date,
        key=key,
    )
    if not isinstance(interval, tuple) or len(interval) != 2:
        return None
    return pd.Timestamp(interval[0]), pd.Timestamp(interval[1])


def strict_run_display_rows(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "end_date" not in runs:
        return runs.copy()
    result = runs.copy()
    if "continuity_label" in result:
        result = result[result["continuity_label"].isin([STRICT_UP_LABEL, STRICT_DOWN_LABEL])]
    if "days" in result:
        result = result[pd.to_numeric(result["days"], errors="coerce") > 3]
    return result.sort_values("end_date", ascending=False).reset_index(drop=True)


def run_dashboard() -> None:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import streamlit as st
    except ModuleNotFoundError as exc:
        missing = exc.name or "streamlit/plotly"
        raise RuntimeError(f"Missing dashboard dependency: {missing}. Run `pip install -r requirements.txt`.") from exc

    st.set_page_config(page_title="连续性趋势评分", layout="wide")
    _apply_theme(st)
    st.title("连续性系统 - 品种历史波段趋势识别与强度评分")
    wave_mode = _render_wave_mode_selector(st)

    data_dir = Path(st.sidebar.text_input("本地 CSV 目录", str(DEFAULT_DATA_DIR)))
    page_label_to_group = {ASSET_GROUPS[group]: group for group in GROUP_ORDER}
    group_label = st.sidebar.radio("页面", list(page_label_to_group), horizontal=True)
    group = page_label_to_group[group_label]
    threshold_mode = st.sidebar.selectbox("反转阈值模式", ["自动", "百分比", "点数"])
    if threshold_mode == "百分比":
        threshold_value = st.sidebar.number_input("最小反转幅度（%）", min_value=0.0, value=3.0, step=0.5)
    elif threshold_mode == "点数":
        threshold_value = st.sidebar.number_input("最小反转点数", min_value=0.0, value=0.0, step=10.0)
    else:
        threshold_value = 0.0
        st.sidebar.caption("自动模式会按 ATR、日波动和绝对波动估算阈值。")
    reversal_threshold = resolve_reversal_threshold(threshold_mode, threshold_value)

    _render_asset_group(st, px, go, data_dir, group, reversal_threshold, wave_mode)


def _render_wave_mode_selector(st) -> str:
    controls = st.columns([1.15, 3.2])
    if hasattr(controls[0], "segmented_control"):
        wave_mode = controls[0].segmented_control("识别模式", WAVE_MODE_OPTIONS, default=WAVE_MODE_ASOF)
    else:
        wave_mode = controls[0].radio("识别模式", WAVE_MODE_OPTIONS, horizontal=True)
    note = (
        "严格 as-of：只在反向阈值触发当天确认上一段波段，适合实盘观察。"
        if wave_mode == WAVE_MODE_ASOF
        else "复盘口径：使用事后 ZigZag 极值日和全样本分位，适合历史回看。"
    )
    controls[1].caption(note)
    return wave_mode


def _apply_theme(st) -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #667085;
            --line: #d9e0ea;
            --panel: #ffffff;
            --soft: #f5f7fa;
            --teal: #0f766e;
            --amber: #b7791f;
            --red: #b42318;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
            color: var(--ink);
        }
        [data-testid="stHeader"] {
            background: rgba(246, 248, 251, 0.92);
            border-bottom: 1px solid rgba(217, 224, 234, 0.75);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        .block-container {
            padding-top: 2.25rem;
            padding-bottom: 3.5rem;
            max-width: 1480px;
        }
        h1 {
            color: var(--ink);
            font-weight: 760;
            margin-bottom: 0.9rem;
        }
        h2, h3 {
            color: var(--ink);
            font-weight: 720;
        }
        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 760;
        }
        .open-wave-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 0.35rem 0 1rem;
        }
        .open-wave-item {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            min-width: 0;
        }
        .open-wave-label {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .open-wave-value {
            color: var(--ink);
            font-size: 1.02rem;
            font-weight: 720;
            line-height: 1.2;
            white-space: nowrap;
            overflow: visible;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            border-color: var(--line);
            border-radius: 8px;
            background: #ffffff;
        }
        [data-baseweb="tag"] {
            min-height: 32px;
            border-radius: 8px;
            background: #eef6f5;
            color: #0b4f4a;
            border: 1px solid #b9dfda;
            max-width: none;
        }
        [data-baseweb="tag"] span {
            white-space: nowrap;
            max-width: none;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            overflow: hidden;
            background: var(--panel);
        }
        .stPlotlyChart {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.25rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_asset_group(
    st,
    px,
    go,
    data_dir: Path,
    group: str,
    reversal_threshold: ReversalThreshold,
    wave_mode: str,
) -> None:
    st.header(ASSET_GROUPS[group])
    symbol_files = available_symbol_files(data_dir, group=group)
    if not symbol_files:
        _render_missing_data_help(st, data_dir, group)
        return

    options = {
        label: code
        for label, code in asset_options(group).items()
        if code in symbol_files
    }
    symbol_label = st.selectbox("品种", list(options), key=f"{group}_symbol")
    symbol = options[symbol_label]
    df = load_symbol_data(data_dir, symbol, group=group)
    if df.empty:
        st.warning(f"{symbol} 的 CSV 已找到，但没有可用数据。")
        return

    min_date = df["trade_date"].min().date()
    max_date = df["trade_date"].max().date()
    controls = st.columns([1.4, 1, 1, 1])
    date_range = controls[0].date_input(
        "日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{group}_date_range",
    )
    direction = controls[1].selectbox(
        "方向",
        ["全部", "up", "down"],
        format_func=lambda value: DIRECTION_LABELS.get(value, value),
        key=f"{group}_direction",
    )
    level = controls[2].selectbox("波段级别", LEVEL_OPTIONS, key=f"{group}_level")
    sort_by = controls[3].selectbox(
        "排序",
        ["total_score", "end_date", "points", "days"],
        format_func=sort_option_label,
        key=f"{group}_sort",
    )

    filtered_df = _filter_by_date(df, date_range)
    detected_waves = detect_waves_for_mode(
        df,
        symbol,
        wave_mode,
        min_reversal=reversal_threshold.points,
        min_reversal_pct=reversal_threshold.pct,
    )
    confirmed_waves, open_wave = split_waves_for_display(detected_waves)
    full_scored = score_waves_for_mode(df, confirmed_waves, wave_mode)
    scored = filter_scored_waves_by_date(full_scored, date_range)
    ranked = rank_waves(scored, direction=direction, level=level)
    if not ranked.empty:
        ascending = sort_by == "end_date"
        ranked = ranked.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    _render_summary_metrics(st, filtered_df, scored, ranked)
    chart_waves = chart_waves_for_display(ranked, open_wave, direction=direction, level=level)
    _render_price_chart(st, go, filtered_df, chart_waves, wave_mode)
    if wave_mode == WAVE_MODE_ASOF:
        _render_open_wave(st, open_wave)
    else:
        st.info("复盘模式使用事后极值日切分波段，不展示当前未完成波段。")
    render_analysis_sections(
        st,
        [
            (ANALYSIS_TAB_LABELS[0], lambda: _render_score_table(st, ranked)),
            (ANALYSIS_TAB_LABELS[1], lambda: _render_wave_detail(st, go, ranked)),
            (ANALYSIS_TAB_LABELS[2], lambda: _render_wave_compare(st, px, df, scored)),
            (ANALYSIS_TAB_LABELS[3], lambda: _render_interval_continuity(st, px, data_dir, group, symbol_files)),
            (ANALYSIS_TAB_LABELS[4], lambda: _render_strict_runs(st, data_dir, group, symbol_files, df, date_range)),
        ],
    )


def _filter_by_date(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        return df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)].reset_index(drop=True)
    return df


def _render_summary_metrics(st, df: pd.DataFrame, scored: pd.DataFrame, ranked: pd.DataFrame) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("识别波段", len(scored))
    metric_cols[1].metric("筛选后", len(ranked))
    metric_cols[2].metric("最高分", f"{ranked['total_score'].max():.1f}" if not ranked.empty else "-")
    metric_cols[3].metric("数据天数", len(df))


def _render_missing_data_help(st, data_dir: Path, group: str) -> None:
    group_dir = data_dir / group
    st.warning(f"没有在 `{group_dir}` 找到本地 CSV。看板只读取本地文件，不会在运行时联网拉取。")
    st.code(f"python scripts/refresh_data.py --group {group}")
    st.markdown("标准字段：")
    st.code("ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount")
    st.markdown("当前配置品种：")
    st.code("\n".join(f"{asset.ts_code}.csv  # {asset.name}" for asset in get_assets(group)))


def _render_price_chart(st, go, df: pd.DataFrame, waves: pd.DataFrame, wave_mode: str = WAVE_MODE_ASOF) -> None:
    chart_waves = extend_latest_wave_for_chart(waves, df)
    fig = build_price_volume_figure(go, df, chart_waves, title=f"K线、成交量与波段区间（{wave_mode}）")
    st.plotly_chart(fig, use_container_width=True)


def _render_open_wave(st, open_wave: pd.Series | None) -> None:
    st.subheader("当前未完成波段")
    if open_wave is None:
        st.info("当前没有未完成波段。")
        return

    cards = []
    for label, value in open_wave_summary_items(open_wave):
        cards.append(
            '<div class="open-wave-item">'
            f'<div class="open-wave-label">{label}</div>'
            f'<div class="open-wave-value">{value}</div>'
            "</div>"
        )
    st.markdown(f'<div class="open-wave-strip">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_score_table(st, waves: pd.DataFrame) -> None:
    st.subheader("波段评分表")
    if waves.empty:
        st.info("当前筛选条件下没有波段。")
        return
    columns = [
        "direction",
        "level",
        "continuity_label",
        "start_date",
        "end_date",
        "extreme_date",
        "confirmation_date",
        "points",
        "pct_change",
        "days",
        "total_score",
        "historical_percentile",
        "strength_score",
        "duration_score",
        "slope_score",
        "drawdown_score",
        "stability_score",
        "volume_score",
    ]
    st.dataframe(format_display_dataframe(waves, columns), use_container_width=True, hide_index=True)


def _render_wave_detail(st, go, waves: pd.DataFrame) -> None:
    st.subheader("单个波段详情")
    if waves.empty:
        return
    labels = [_wave_label(index, row) for index, row in waves.iterrows()]
    selected = st.selectbox("选择波段", labels)
    wave_index = int(selected.split(" | ", 1)[0])
    wave = waves.loc[wave_index]
    left, right = st.columns([1, 2])
    left.metric("总分", f"{wave['total_score']:.1f}")
    left.metric("历史分位", f"{wave['historical_percentile']:.1f}%")
    left.metric("涨跌点数", f"{wave['points']:.2f}")
    left.metric("持续天数", int(wave["days"]))
    fig = _wave_radar_chart(go, wave)
    right.plotly_chart(fig, use_container_width=True)


def _render_wave_compare(st, px, df: pd.DataFrame, waves: pd.DataFrame) -> None:
    st.subheader("波段对比")
    if len(waves) < 2:
        st.info("至少需要两个波段才能对比。")
        return

    controls = st.columns([1, 1, 2])
    compare_direction = controls[0].selectbox(
        "对比方向",
        ["全部", "up", "down"],
        format_func=lambda value: DIRECTION_LABELS.get(value, value),
        key="compare_direction",
    )
    compare_level = controls[1].selectbox("对比级别", LEVEL_OPTIONS, key="compare_level")
    compare_candidates = rank_waves(waves, direction=compare_direction, level=compare_level)
    if len(compare_candidates) < 2:
        st.info("当前对比筛选下至少需要两个波段。")
        return

    label_to_index = {
        _wave_compare_label(index, row): index
        for index, row in compare_candidates.iterrows()
    }
    labels = list(label_to_index)
    selected = controls[2].multiselect("选择波段", labels, default=labels[: min(3, len(labels))])
    if len(selected) < 2:
        st.info("请选择至少两个波段，支持多选。")
        return

    selected_indices = [label_to_index[label] for label in selected]
    comparison = compare_waves(df, compare_candidates, selected_indices)
    display_labels = {f"wave_{position}": selected[position] for position in range(len(selected))}
    comparison["paths"]["wave"] = comparison["paths"]["wave"].map(display_labels)
    comparison["metrics"]["label"] = comparison["metrics"]["label"].map(display_labels)

    st.plotly_chart(
        px.line(
            comparison["paths"],
            x="step",
            y="relative_close",
            color="wave",
            title="归一化走势（起点=100）",
            labels={"step": "进程", "relative_close": "归一化价格", "wave": "波段"},
            color_discrete_sequence=["#0f766e", "#b7791f", "#2563eb", "#b42318", "#7c3aed", "#475467", "#0891b2", "#c2410c"],
        ).update_layout(
            template="plotly_white",
            legend_title_text="波段",
            height=430,
            margin={"l": 20, "r": 20, "t": 56, "b": 20},
        ),
        use_container_width=True,
    )
    st.dataframe(format_display_dataframe(comparison["metrics"]), use_container_width=True, hide_index=True)


def _render_interval_continuity(st, px, data_dir: Path, group: str, symbol_files: dict[str, Path]) -> None:
    st.subheader("区间连续性横向对比")
    frames = []
    for symbol in symbol_files:
        try:
            frames.append(load_symbol_data(data_dir, symbol, group=group))
        except FileNotFoundError:
            continue
    if not frames:
        st.info("当前页面没有可参与区间对比的数据。")
        return
    prices = pd.concat(frames, ignore_index=True)
    min_date = prices["trade_date"].min().date()
    max_date = prices["trade_date"].max().date()
    default_start = prices["trade_date"].sort_values().drop_duplicates().iloc[max(0, prices["trade_date"].nunique() - 120)].date()
    interval = st.date_input(
        "对比区间",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        key=f"{group}_interval",
    )
    if not isinstance(interval, tuple) or len(interval) != 2:
        return
    ranked = score_interval_continuity(prices, pd.Timestamp(interval[0]), pd.Timestamp(interval[1]))
    if ranked.empty:
        st.info("所选区间内没有足够数据进行横向对比。")
        return
    ranked["name"] = ranked["symbol"].map(lambda code: display_name(code, group))
    chart_df = ranked[["name", "symbol", "interval_score"]].copy()
    chart_df["label"] = chart_df["name"] + " (" + chart_df["symbol"] + ")"
    st.plotly_chart(
        px.bar(
            chart_df,
            x="interval_score",
            y="label",
            orientation="h",
            title="区间连续性评分排名",
            range_x=[0, 100],
            color="interval_score",
            labels={"interval_score": "区间评分", "label": "品种"},
            color_continuous_scale=["#dfe9e8", "#0f766e"],
        ).update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            height=360,
            margin={"l": 20, "r": 20, "t": 52, "b": 20},
        ),
        use_container_width=True,
    )
    columns = [
        "symbol",
        "name",
        "direction",
        "start_date",
        "end_date",
        "pct_change",
        "points",
        "slope",
        "max_adverse_pct",
        "interval_score",
        "strength_score",
        "slope_score",
        "drawdown_score",
        "stability_score",
        "volume_score",
    ]
    st.dataframe(format_display_dataframe(ranked, columns), use_container_width=True, hide_index=True)


def _render_strict_runs(
    st,
    data_dir: Path,
    group: str,
    symbol_files: dict[str, Path],
    current_df: pd.DataFrame,
    date_range,
) -> None:
    st.subheader("连阴连阳识别")
    filtered_current = _filter_by_date(current_df, date_range)
    runs = detect_strict_runs(filtered_current)

    st.markdown("当前品种严格连阴连阳表（持续天数大于 3）")
    display_runs = strict_run_display_rows(runs)
    if display_runs.empty:
        st.info("当前日期范围内没有持续天数大于 3 的严格连阳或连阴段。")
    else:
        st.dataframe(format_display_dataframe(display_runs, strict_run_table_columns()), use_container_width=True, hide_index=True)

    frames = []
    for symbol in symbol_files:
        try:
            frames.append(load_symbol_data(data_dir, symbol, group=group))
        except FileNotFoundError:
            continue
    if not frames or not isinstance(date_range, tuple) or len(date_range) != 2:
        return

    prices = pd.concat(frames, ignore_index=True)
    summary_interval = strict_run_summary_interval(
        st,
        prices,
        date_range,
        key=f"{group}_strict_summary_interval",
    )
    if summary_interval is None:
        return

    summary = summarize_strict_runs(prices, summary_interval[0], summary_interval[1])
    st.markdown("所选区间横向统计")
    if summary.empty:
        st.info("所选区间内没有可统计的严格连续形态。")
        return

    summary["name"] = summary["symbol"].map(lambda code: display_name(code, group))
    st.dataframe(format_display_dataframe(summary, strict_run_summary_table_columns()), use_container_width=True, hide_index=True)


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in ["start_date", "end_date", "trade_date", "confirmation_date", "extreme_date"]:
        if column in display:
            display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
    return display


def _wave_radar_chart(go, wave: pd.Series):
    labels = ["强度", "斜率", "稳定性", "量能", "持续时间", "回撤控制"]
    values = [
        float(wave["strength_score"]),
        float(wave["slope_score"]),
        float(wave["stability_score"]),
        float(wave["volume_score"]),
        float(wave["duration_score"]),
        float(wave["drawdown_score"]),
    ]
    closed_labels = [*labels, labels[0]]
    closed_values = [*values, values[0]]
    start = pd.Timestamp(wave["start_date"]).strftime("%Y%m%d")
    end = pd.Timestamp(wave["end_date"]).strftime("%Y%m%d")

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=closed_values,
            theta=closed_labels,
            mode="lines+markers",
            fill="toself",
            name="维度得分",
            line={"color": "#0f766e", "width": 3},
            marker={"size": 7, "color": "#0f766e"},
            fillcolor="rgba(15, 118, 110, 0.26)",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"趋势评分: {float(wave['total_score']):.1f}分 ({start}-{end})",
        height=390,
        showlegend=False,
        margin={"l": 38, "r": 38, "t": 62, "b": 28},
        paper_bgcolor="#ffffff",
        font={"color": "#172033", "size": 13},
        polar={
            "bgcolor": "#ffffff",
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "tickvals": [0, 20, 40, 60, 80, 100],
                "tickfont": {"color": "#667085", "size": 11},
                "gridcolor": "#d9e0ea",
                "linecolor": "#d9e0ea",
            },
            "angularaxis": {
                "tickfont": {"color": "#475467", "size": 13},
                "gridcolor": "#d9e0ea",
                "linecolor": "#98a2b3",
            },
        },
    )
    return fig


def _wave_label(index: int, wave: pd.Series) -> str:
    start = pd.Timestamp(wave["start_date"]).strftime("%Y-%m-%d")
    end = pd.Timestamp(wave["end_date"]).strftime("%Y-%m-%d")
    direction = "上涨" if wave["direction"] == "up" else "下跌"
    return f"{index} | {start} -> {end} | {direction} | {wave['level']} | {wave['total_score']:.1f}"


def _wave_compare_label(index: int, wave: pd.Series) -> str:
    direction = "上涨" if wave["direction"] == "up" else "下跌"
    return f"W{index:03d} · {direction} · {wave['level']} · {wave['total_score']:.1f}"


if __name__ == "__main__":
    run_dashboard()
