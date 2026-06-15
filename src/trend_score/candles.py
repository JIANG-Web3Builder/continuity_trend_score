from __future__ import annotations

import pandas as pd

from trend_score.streaks import detect_strict_streaks, summarize_strict_streaks


STRICT_UP_LABEL = "连阳连续性"
STRICT_DOWN_LABEL = "连阴连续性"
NON_STRICT_UP_LABEL = "区间上涨连续性"
NON_STRICT_DOWN_LABEL = "区间下跌连续性"

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

def detect_strict_runs(prices: pd.DataFrame, min_days: int = 2) -> pd.DataFrame:
    """Backward-compatible wrapper for strict body-based streak detection."""
    return detect_strict_streaks(prices, min_days=min_days)


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
        rows.extend(detect_strict_streaks(segment, min_days=min_days).to_dict("records"))

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
    """Backward-compatible wrapper for strict body-based streak summaries."""
    return summarize_strict_streaks(prices, start_date, end_date)


def label_wave_continuity(segment: pd.DataFrame, direction: str) -> str:
    """Classify one scored wave by strict body continuity first, then interval continuity."""
    if direction in {"up", "down"} and _is_strict_body_segment(segment, direction):
        return _label(direction, strict=True)
    if direction == "up":
        return NON_STRICT_UP_LABEL
    if direction == "down":
        return NON_STRICT_DOWN_LABEL
    return "none"


def _is_strict_body_segment(segment: pd.DataFrame, direction: str) -> bool:
    if segment.empty:
        return False
    body_delta = segment["close"].astype(float) - segment["open"].astype(float)
    if direction == "up":
        return bool((body_delta > 0).all())
    return bool((body_delta < 0).all())


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
