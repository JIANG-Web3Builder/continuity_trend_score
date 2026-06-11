from __future__ import annotations

import pandas as pd


STRICT_UP_LABEL = "\u4e25\u683c\u8fde\u9633"
STRICT_DOWN_LABEL = "\u4e25\u683c\u8fde\u9634"
NON_STRICT_UP_LABEL = "\u975e\u4e25\u683c\u8fde\u9633"
NON_STRICT_DOWN_LABEL = "\u975e\u4e25\u683c\u8fde\u9634"

STRICT_RUN_COLUMNS = [
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

STRICT_RUN_SUMMARY_COLUMNS = [
    "symbol",
    "continuity_label",
    "start_date",
    "end_date",
    "current_direction",
    "current_run_days",
    "longest_up_days",
    "longest_down_days",
    "longest_direction",
    "longest_days",
]


def detect_strict_runs(prices: pd.DataFrame, min_days: int = 2) -> pd.DataFrame:
    """Detect strict bullish/bearish runs from the close-to-close price path."""
    if prices.empty:
        return _empty_strict_runs()

    min_days = max(int(min_days), 1)
    rows: list[dict[str, object]] = []
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        segment = group.sort_values("trade_date").reset_index(drop=True)
        rows.extend(_detect_symbol_runs(str(symbol), segment, min_days))

    if not rows:
        return _empty_strict_runs()
    return pd.DataFrame(rows, columns=STRICT_RUN_COLUMNS)


def detect_continuity_segments(prices: pd.DataFrame, min_days: int = 2) -> pd.DataFrame:
    """Return one table containing non-strict interval rows and strict run rows."""
    if prices.empty:
        return _empty_strict_runs()

    min_days = max(int(min_days), 1)
    rows: list[dict[str, object]] = []
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        segment = group.sort_values("trade_date").reset_index(drop=True)
        non_strict = _non_strict_interval_row(str(symbol), segment, min_days)
        if non_strict is not None:
            rows.append(non_strict)
        rows.extend(_detect_symbol_runs(str(symbol), segment, min_days))

    if not rows:
        return _empty_strict_runs()
    return pd.DataFrame(rows, columns=STRICT_RUN_COLUMNS).sort_values(
        ["symbol", "start_date", "end_date", "continuity_label"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def summarize_strict_runs(
    prices: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarize strict continuity evidence inside one selected interval."""
    if prices.empty:
        return _empty_strict_run_summary()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows: list[dict[str, object]] = []
    for symbol, group in prices.groupby("ts_code", sort=False, dropna=False):
        segment = group[(group["trade_date"] >= start) & (group["trade_date"] <= end)].sort_values("trade_date")
        if segment.empty:
            continue

        runs = detect_strict_runs(segment)
        symbol_runs = runs[runs["symbol"] == str(symbol)] if not runs.empty else runs
        current = _current_run(symbol_runs)
        longest_up = _longest_days(symbol_runs, "up")
        longest_down = _longest_days(symbol_runs, "down")
        longest_direction = _longest_direction(longest_up, longest_down)
        rows.append(
            {
                "symbol": str(symbol),
                "continuity_label": _non_strict_label(segment),
                "start_date": segment.iloc[0]["trade_date"],
                "end_date": segment.iloc[-1]["trade_date"],
                "current_direction": current["direction"] if current is not None else "",
                "current_run_days": int(current["days"]) if current is not None else 0,
                "longest_up_days": longest_up,
                "longest_down_days": longest_down,
                "longest_direction": longest_direction,
                "longest_days": max(longest_up, longest_down),
            }
        )

    if not rows:
        return _empty_strict_run_summary()
    return pd.DataFrame(rows, columns=STRICT_RUN_SUMMARY_COLUMNS).sort_values(
        ["longest_days", "current_run_days", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def label_wave_continuity(segment: pd.DataFrame, direction: str) -> str:
    """Classify one scored wave as strict or non-strict continuity."""
    if direction not in {"up", "down"}:
        return "none"
    if len(segment) < 2:
        return _label(direction, strict=False)

    directions = segment.apply(_path_direction, axis=1)
    strict = bool(directions.notna().all() and (directions == direction).all())
    return _label(direction, strict=strict)


def _detect_symbol_runs(symbol: str, segment: pd.DataFrame, min_days: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    active_direction: str | None = None
    start_idx: int | None = None

    for idx, row in segment.iterrows():
        direction = _path_direction(row)
        if direction is None:
            if active_direction is not None and start_idx is not None:
                _append_run(rows, symbol, segment, start_idx, idx - 1, active_direction, "confirmed", min_days)
            active_direction = None
            start_idx = None
            continue

        if active_direction is None:
            active_direction = direction
            start_idx = idx
        elif direction != active_direction:
            if start_idx is not None:
                _append_run(rows, symbol, segment, start_idx, idx - 1, active_direction, "confirmed", min_days)
            active_direction = direction
            start_idx = idx

    if active_direction is not None and start_idx is not None:
        _append_run(rows, symbol, segment, start_idx, len(segment) - 1, active_direction, "open", min_days)
    return rows


def _append_run(
    rows: list[dict[str, object]],
    symbol: str,
    segment: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str,
    status: str,
    min_days: int,
) -> None:
    if end_idx - start_idx + 1 < min_days:
        return
    rows.append(_run_row(symbol, segment, start_idx, end_idx, direction, status, strict=True))


def _path_direction(row: pd.Series) -> str | None:
    if "change" in row and pd.notna(row["change"]):
        delta = float(row["change"])
    elif "pre_close" in row and pd.notna(row["pre_close"]):
        delta = float(row["close"]) - float(row["pre_close"])
    else:
        delta = float(row["close"]) - float(row["open"])
    if delta > 0:
        return "up"
    if delta < 0:
        return "down"
    return None


def _run_row(
    symbol: str,
    segment: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: str,
    status: str,
    strict: bool,
) -> dict[str, object]:
    start = segment.loc[start_idx]
    end = segment.loc[end_idx]
    start_price = _start_price(start, strict)
    end_price = float(end["close"])
    return {
        "symbol": symbol,
        "direction": direction,
        "continuity_label": _label(direction, strict),
        "start_date": pd.Timestamp(start["trade_date"]),
        "end_date": pd.Timestamp(end["trade_date"]),
        "start_price": start_price,
        "end_price": end_price,
        "pct_change": round((end_price - start_price) / start_price * 100.0, 6) if start_price else 0.0,
        "days": int(end_idx - start_idx + 1),
        "status": status,
    }


def _non_strict_interval_row(symbol: str, segment: pd.DataFrame, min_days: int) -> dict[str, object] | None:
    if len(segment) < min_days:
        return None
    first_close = float(segment.iloc[0]["close"])
    last_close = float(segment.iloc[-1]["close"])
    if last_close == first_close:
        return None
    direction = "up" if last_close > first_close else "down"
    return _run_row(symbol, segment, 0, len(segment) - 1, direction, "confirmed", strict=False)


def _start_price(row: pd.Series, strict: bool) -> float:
    if strict and "pre_close" in row and pd.notna(row["pre_close"]):
        return float(row["pre_close"])
    return float(row["close"])


def _label(direction: str, strict: bool) -> str:
    if strict:
        return STRICT_UP_LABEL if direction == "up" else STRICT_DOWN_LABEL
    return NON_STRICT_UP_LABEL if direction == "up" else NON_STRICT_DOWN_LABEL


def _current_run(runs: pd.DataFrame) -> pd.Series | None:
    if runs.empty:
        return None
    open_runs = runs[runs["status"] == "open"]
    if open_runs.empty:
        return None
    return open_runs.sort_values("end_date").iloc[-1]


def _longest_days(runs: pd.DataFrame, direction: str) -> int:
    if runs.empty:
        return 0
    directed = runs[runs["direction"] == direction]
    if directed.empty:
        return 0
    return int(directed["days"].max())


def _longest_direction(longest_up: int, longest_down: int) -> str:
    if longest_up == 0 and longest_down == 0:
        return ""
    if longest_up >= longest_down:
        return "up"
    return "down"


def _non_strict_label(segment: pd.DataFrame) -> str:
    first_close = float(segment.iloc[0]["close"])
    last_close = float(segment.iloc[-1]["close"])
    if last_close > first_close:
        return NON_STRICT_UP_LABEL
    if last_close < first_close:
        return NON_STRICT_DOWN_LABEL
    return "none"


def _empty_strict_runs() -> pd.DataFrame:
    return pd.DataFrame(columns=STRICT_RUN_COLUMNS)


def _empty_strict_run_summary() -> pd.DataFrame:
    return pd.DataFrame(columns=STRICT_RUN_SUMMARY_COLUMNS)
