from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trend_score.catalog import ASSET_GROUPS, asset_options, get_assets
from trend_score.compare import compare_waves, rank_waves, score_interval_continuity
from trend_score.data import available_symbol_files, display_name, load_symbol_data
from trend_score.scoring import score_waves
from trend_score.waves import detect_waves


DEFAULT_DATA_DIR = Path("data")
DIRECTION_LABELS = {"全部": "全部", "up": "上涨", "down": "下跌"}
LEVEL_OPTIONS = ["全部", "小", "中", "大", "超大"]
GROUP_ORDER = ["index", "sector", "commodity"]


def build_wave_scores(df: pd.DataFrame, symbol: str, min_reversal: float | None = None) -> pd.DataFrame:
    waves = detect_waves(df, symbol=symbol, min_reversal=min_reversal)
    return score_waves(df, waves)


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

    data_dir = Path(st.sidebar.text_input("本地 CSV 目录", str(DEFAULT_DATA_DIR)))
    page_label_to_group = {ASSET_GROUPS[group]: group for group in GROUP_ORDER}
    group_label = st.sidebar.radio("页面", list(page_label_to_group), horizontal=True)
    group = page_label_to_group[group_label]
    threshold = st.sidebar.number_input("最小反转点数（0=自适应）", min_value=0.0, value=0.0, step=10.0)
    min_reversal = threshold if threshold > 0 else None

    _render_asset_group(st, px, go, data_dir, group, min_reversal)


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
            padding: 0.35rem;
        }
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_asset_group(st, px, go, data_dir: Path, group: str, min_reversal: float | None) -> None:
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
    sort_by = controls[3].selectbox("排序", ["total_score", "end_date", "points", "days"], key=f"{group}_sort")

    filtered_df = _filter_by_date(df, date_range)
    scored = build_wave_scores(filtered_df, symbol, min_reversal=min_reversal)
    ranked = rank_waves(scored, direction=direction, level=level)
    if not ranked.empty:
        ascending = sort_by == "end_date"
        ranked = ranked.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    _render_summary_metrics(st, filtered_df, scored, ranked)
    _render_price_chart(st, go, filtered_df, ranked)
    _render_score_table(st, ranked)
    _render_wave_detail(st, px, ranked)
    _render_wave_compare(st, px, filtered_df, scored)
    _render_interval_continuity(st, px, data_dir, group, symbol_files)


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


def _render_price_chart(st, go, df: pd.DataFrame, waves: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["trade_date"],
            y=df["close"],
            mode="lines",
            name="close",
            line={"color": "#172033", "width": 1.8},
        )
    )
    for _, wave in waves.iterrows():
        color = "rgba(15, 118, 110, 0.14)" if wave["direction"] == "up" else "rgba(180, 35, 24, 0.14)"
        fig.add_vrect(
            x0=wave["start_date"],
            x1=wave["end_date"],
            fillcolor=color,
            line_width=0,
            annotation_text=f"{wave['level']} {wave['total_score']:.0f}",
            annotation_position="top left",
        )
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin={"l": 20, "r": 20, "t": 44, "b": 20},
        title="收盘价与波段标注",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#172033"},
        xaxis={"gridcolor": "#edf1f6"},
        yaxis={"gridcolor": "#edf1f6"},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_score_table(st, waves: pd.DataFrame) -> None:
    st.subheader("波段评分表")
    if waves.empty:
        st.info("当前筛选条件下没有波段。")
        return
    display = _format_dates(waves)
    columns = [
        "direction",
        "level",
        "start_date",
        "end_date",
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
    st.dataframe(display[columns], use_container_width=True, hide_index=True)


def _render_wave_detail(st, px, waves: pd.DataFrame) -> None:
    st.subheader("单个波段详情")
    if waves.empty:
        return
    labels = [_wave_label(index, row) for index, row in waves.iterrows()]
    selected = st.selectbox("选择波段", labels)
    wave_index = int(selected.split(" | ", 1)[0])
    wave = waves.loc[wave_index]
    dims = pd.DataFrame(
        {
            "维度": ["强度", "持续", "斜率", "回撤控制", "稳定性", "量能"],
            "得分": [
                wave["strength_score"],
                wave["duration_score"],
                wave["slope_score"],
                wave["drawdown_score"],
                wave["stability_score"],
                wave["volume_score"],
            ],
        }
    )
    left, right = st.columns([1, 2])
    left.metric("总分", f"{wave['total_score']:.1f}")
    left.metric("历史分位", f"{wave['historical_percentile']:.1f}%")
    left.metric("涨跌点数", f"{wave['points']:.2f}")
    left.metric("持续天数", int(wave["days"]))
    fig = px.bar(dims, x="维度", y="得分", range_y=[0, 100], title="维度拆解", color="维度")
    fig.update_layout(template="plotly_white", showlegend=False, height=330, margin={"l": 20, "r": 20, "t": 48, "b": 20})
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
            color_discrete_sequence=["#0f766e", "#b7791f", "#2563eb", "#b42318", "#7c3aed", "#475467", "#0891b2", "#c2410c"],
        ).update_layout(
            template="plotly_white",
            legend_title_text="波段",
            height=430,
            margin={"l": 20, "r": 20, "t": 56, "b": 20},
        ),
        use_container_width=True,
    )
    metrics = _format_dates(comparison["metrics"])
    st.dataframe(metrics, use_container_width=True, hide_index=True)


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
            color_continuous_scale=["#dfe9e8", "#0f766e"],
        ).update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            height=360,
            margin={"l": 20, "r": 20, "t": 52, "b": 20},
        ),
        use_container_width=True,
    )
    display = _format_dates(ranked)
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
    st.dataframe(display[columns], use_container_width=True, hide_index=True)


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in ["start_date", "end_date", "trade_date"]:
        if column in display:
            display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
    return display


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
