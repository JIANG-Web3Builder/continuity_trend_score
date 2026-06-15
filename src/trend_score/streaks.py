from __future__ import annotations

import pandas as pd


STREAK_UP_LABEL = "连阳连续性"
STREAK_DOWN_LABEL = "连阴连续性"

STREAK_COLUMNS = [
    "symbol",
    "direction",
    "continuity_label",
    "start_date",
    "end_date",
    "start_price",
    "end_price",
    "pct_change",
    "days",
    "status",
]

STREAK_SUMMARY_COLUMNS = [
    "symbol",
    "continuity_label",
    "start_date",
    "end_date",
    "amplitude_pct",
    "current_direction",
    "current_run_days",
    "longest_up_days",
    "longest_down_days",
    "longest_direction",
    "longest_days",
]

STREAK_WIN_RATE_COLUMNS = [
    "symbol",
    "direction",
    "continuity_label",
    "trigger_rule",
    "total_signals",
    "latest_start_date",
    "latest_end_date",
    "latest_streak_days",
    "signals_1d",
    "up_count_1d",
    "down_count_1d",
    "flat_count_1d",
    "up_rate_1d",
    "down_rate_1d",
    "avg_return_1d",
    "median_return_1d",
    "best_return_1d",
    "worst_return_1d",
    "signals_3d",
    "up_count_3d",
    "down_count_3d",
    "flat_count_3d",
    "up_rate_3d",
    "down_rate_3d",
    "avg_return_3d",
    "median_return_3d",
    "best_return_3d",
    "worst_return_3d",
    "signals_5d",
    "up_count_5d",
    "down_count_5d",
    "flat_count_5d",
    "up_rate_5d",
    "down_rate_5d",
    "avg_return_5d",
    "median_return_5d",
    "best_return_5d",
    "worst_return_5d",
    "signals_10d",
    "up_count_10d",
    "down_count_10d",
    "flat_count_10d",
    "up_rate_10d",
    "down_rate_10d",
    "avg_return_10d",
    "median_return_10d",
    "best_return_10d",
    "worst_return_10d",
]

STREAK_SIGNAL_OUTCOME_COLUMNS = [
    "symbol",
    "direction",
    "continuity_label",
    "start_date",
    "end_date",
    "streak_days",
    "end_close",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
]


def detect_strict_streaks(prices: pd.DataFrame, min_days: int = 2) -> pd.DataFrame:
    """Detect strict bullish/bearish streaks from candle body direction."""
    if prices.empty:
        return _empty_streaks()

    rows: list[dict[str, object]] = []
    min_days = max(int(min_days), 1)
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        rows.extend(_detect_symbol_streaks(str(symbol), group, min_days, include_index=False))

    if not rows:
        return _empty_streaks()
    return pd.DataFrame(rows, columns=STREAK_COLUMNS)


def summarize_strict_streaks(
    prices: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarize current and longest strict streaks inside one selected interval."""
    if prices.empty:
        return _empty_streak_summary()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows: list[dict[str, object]] = []
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        segment = group[(group["trade_date"] >= start) & (group["trade_date"] <= end)].sort_values("trade_date")
        if segment.empty:
            continue

        streaks = detect_strict_streaks(segment)
        symbol_streaks = streaks[streaks["symbol"] == str(symbol)] if not streaks.empty else streaks
        current = _current_streak(symbol_streaks)
        longest_up = _longest_days(symbol_streaks, "up")
        longest_down = _longest_days(symbol_streaks, "down")
        longest_direction = _longest_direction(longest_up, longest_down)
        label = _label(longest_direction) if longest_direction else ""
        rows.append(
            {
                "symbol": str(symbol),
                "continuity_label": label,
                "start_date": segment.iloc[0]["trade_date"],
                "end_date": segment.iloc[-1]["trade_date"],
                "amplitude_pct": _amplitude_pct(segment),
                "current_direction": current["direction"] if current is not None else "",
                "current_run_days": int(current["days"]) if current is not None else 0,
                "longest_up_days": longest_up,
                "longest_down_days": longest_down,
                "longest_direction": longest_direction,
                "longest_days": max(longest_up, longest_down),
            }
        )

    if not rows:
        return _empty_streak_summary()
    return pd.DataFrame(rows, columns=STREAK_SUMMARY_COLUMNS).sort_values(
        ["longest_days", "current_run_days", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def score_streak_win_rates(
    prices: pd.DataFrame,
    min_days: int = 2,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Summarize future up/down distribution for streaks at or above the selected threshold."""
    outcomes = streak_signal_outcomes(prices, min_days=min_days, horizons=horizons)
    if outcomes.empty:
        return _empty_win_rates()

    rows: list[dict[str, object]] = []
    for (symbol, direction, label), group in outcomes.groupby(["symbol", "direction", "continuity_label"], sort=False):
        row: dict[str, object] = {
            "symbol": symbol,
            "direction": direction,
            "continuity_label": label,
            "trigger_rule": f"{'连阳' if direction == 'up' else '连阴'}>={int(min_days)}天",
            "total_signals": int(len(group)),
            "latest_start_date": group.sort_values("end_date").iloc[-1]["start_date"],
            "latest_end_date": group["end_date"].max(),
            "latest_streak_days": int(group.sort_values("end_date").iloc[-1]["streak_days"]),
        }
        for horizon in horizons:
            returns = pd.to_numeric(group[f"return_{horizon}d"], errors="coerce").dropna()
            signals = int(len(returns))
            up_count = int((returns > 0).sum()) if signals else 0
            down_count = int((returns < 0).sum()) if signals else 0
            flat_count = int((returns == 0).sum()) if signals else 0
            row[f"signals_{horizon}d"] = signals
            row[f"up_count_{horizon}d"] = up_count
            row[f"down_count_{horizon}d"] = down_count
            row[f"flat_count_{horizon}d"] = flat_count
            row[f"up_rate_{horizon}d"] = round(up_count / signals * 100.0, 2) if signals else None
            row[f"down_rate_{horizon}d"] = round(down_count / signals * 100.0, 2) if signals else None
            row[f"avg_return_{horizon}d"] = round(float(returns.mean()), 2) if signals else None
            row[f"median_return_{horizon}d"] = round(float(returns.median()), 2) if signals else None
            row[f"best_return_{horizon}d"] = round(float(returns.max()), 2) if signals else None
            row[f"worst_return_{horizon}d"] = round(float(returns.min()), 2) if signals else None
        rows.append(row)

    if not rows:
        return _empty_win_rates()
    result = pd.DataFrame(rows)
    result = result.reindex(columns=STREAK_WIN_RATE_COLUMNS).sort_values(
        ["symbol", "direction"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return _restore_missing_values(result)


def streak_signal_outcomes(
    prices: pd.DataFrame,
    min_days: int = 2,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Return recent confirmed streak ranges and their forward returns."""
    if prices.empty:
        return _empty_signal_outcomes()

    rows: list[dict[str, object]] = []
    min_days = max(int(min_days), 1)
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        segment = group.sort_values("trade_date").reset_index(drop=True)
        closes = segment["close"].astype(float)
        for streak in _detect_symbol_streaks(str(symbol), segment, min_days, include_index=True):
            if streak["status"] != "confirmed":
                continue
            end_index = int(streak["_end_index"])
            end_close = float(streak["end_price"])
            row: dict[str, object] = {
                "symbol": str(symbol),
                "direction": str(streak["direction"]),
                "continuity_label": str(streak["continuity_label"]),
                "start_date": pd.Timestamp(streak["start_date"]),
                "end_date": pd.Timestamp(streak["end_date"]),
                "streak_days": int(streak["days"]),
                "end_close": end_close,
            }
            for horizon in horizons:
                return_column = f"return_{horizon}d"
                if horizon <= 0 or end_index + horizon >= len(segment):
                    row[return_column] = None
                    continue
                future_close = float(closes.iloc[end_index + horizon])
                raw_return = (future_close - end_close) / end_close * 100.0 if end_close else 0.0
                row[return_column] = round(float(raw_return), 2)
            rows.append(row)

    if not rows:
        return _empty_signal_outcomes()
    result = pd.DataFrame(rows).reindex(columns=STREAK_SIGNAL_OUTCOME_COLUMNS).sort_values(
        ["end_date", "symbol", "direction"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return _restore_missing_values(result)


def _detect_symbol_streaks(
    symbol: str,
    prices: pd.DataFrame,
    min_days: int,
    include_index: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    segment = prices.sort_values("trade_date").reset_index(drop=True)
    active_direction: str | None = None
    start_idx: int | None = None

    for idx, row in segment.iterrows():
        direction = _body_direction(row)
        if direction is None:
            if active_direction is not None and start_idx is not None:
                _append_streak(rows, symbol, segment, start_idx, idx - 1, active_direction, "confirmed", min_days, include_index)
            active_direction = None
            start_idx = None
            continue

        if active_direction is None:
            active_direction = direction
            start_idx = idx
        elif direction != active_direction:
            if start_idx is not None:
                _append_streak(rows, symbol, segment, start_idx, idx - 1, active_direction, "confirmed", min_days, include_index)
            active_direction = direction
            start_idx = idx

    if active_direction is not None and start_idx is not None:
        _append_streak(rows, symbol, segment, start_idx, len(segment) - 1, active_direction, "open", min_days, include_index)
    return rows


def _append_streak(
    rows: list[dict[str, object]],
    symbol: str,
    segment: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str,
    status: str,
    min_days: int,
    include_index: bool,
) -> None:
    if end_idx - start_idx + 1 < min_days:
        return
    row = _streak_row(symbol, segment, start_idx, end_idx, direction, status)
    if include_index:
        row["_end_index"] = end_idx
    rows.append(row)


def _streak_row(
    symbol: str,
    segment: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str,
    status: str,
) -> dict[str, object]:
    start = segment.loc[start_idx]
    end = segment.loc[end_idx]
    start_price = float(start["open"])
    end_price = float(end["close"])
    return {
        "symbol": symbol,
        "direction": direction,
        "continuity_label": _label(direction),
        "start_date": pd.Timestamp(start["trade_date"]),
        "end_date": pd.Timestamp(end["trade_date"]),
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": round((end_price - start_price) / start_price * 100.0, 6) if start_price else 0.0,
        "days": int(end_idx - start_idx + 1),
        "status": status,
    }


def _body_direction(row: pd.Series) -> str | None:
    delta = float(row["close"]) - float(row["open"])
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return None


def _label(direction: str) -> str:
    return STREAK_UP_LABEL if direction == "up" else STREAK_DOWN_LABEL


def _current_streak(streaks: pd.DataFrame) -> pd.Series | None:
    if streaks.empty:
        return None
    open_streaks = streaks[streaks["status"] == "open"]
    if open_streaks.empty:
        return None
    return open_streaks.sort_values("end_date").iloc[-1]


def _longest_days(streaks: pd.DataFrame, direction: str) -> int:
    if streaks.empty:
        return 0
    directed = streaks[streaks["direction"] == direction]
    if directed.empty:
        return 0
    return int(directed["days"].max())


def _longest_direction(longest_up: int, longest_down: int) -> str:
    if longest_up == 0 and longest_down == 0:
        return ""
    if longest_up >= longest_down:
        return "up"
    return "down"


def _amplitude_pct(segment: pd.DataFrame) -> float:
    base = float(segment.iloc[0]["close"])
    if base == 0:
        return 0.0
    high = float(segment["high"].max()) if "high" in segment else float(segment["close"].max())
    low = float(segment["low"].min()) if "low" in segment else float(segment["close"].min())
    return round((high - low) / base * 100.0, 6)


def _empty_streaks() -> pd.DataFrame:
    return pd.DataFrame(columns=STREAK_COLUMNS)


def _empty_streak_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=STREAK_SUMMARY_COLUMNS)


def _empty_win_rates() -> pd.DataFrame:
    return pd.DataFrame(columns=STREAK_WIN_RATE_COLUMNS)


def _empty_signal_outcomes() -> pd.DataFrame:
    return pd.DataFrame(columns=STREAK_SIGNAL_OUTCOME_COLUMNS)


def _restore_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).where(pd.notna(df), None)
