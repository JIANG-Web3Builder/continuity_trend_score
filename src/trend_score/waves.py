from __future__ import annotations

import numpy as np
import pandas as pd


LEVEL_LABELS = ("小", "中", "大", "超大")


def detect_waves(
    prices: pd.DataFrame,
    symbol: str | None = None,
    min_reversal: float | None = None,
    min_reversal_pct: float | None = None,
    atr_multiplier: float = 1.2,
    min_wave_days: int = 3,
) -> pd.DataFrame:
    """Detect directional waves as they become confirmable from visible prices."""
    if prices.empty:
        return _empty_waves()

    df = prices.sort_values("trade_date").reset_index(drop=True).copy()
    symbol_value = symbol or str(df.loc[0, "ts_code"])
    threshold = None if min_reversal is None else max(float(min_reversal), 0.0)
    threshold_pct = None if min_reversal_pct is None else max(float(min_reversal_pct), 0.0)
    rolling_thresholds = None
    if threshold is None and threshold_pct is None:
        rolling_thresholds = _rolling_reversal_thresholds(df, atr_multiplier)

    rows = _asof_zigzag_waves(
        df,
        symbol_value,
        threshold=threshold,
        min_reversal_pct=threshold_pct,
        rolling_thresholds=rolling_thresholds,
        min_wave_days=min_wave_days,
    )
    return classify_wave_levels(pd.DataFrame(rows)) if rows else _empty_waves()


def classify_wave_levels(waves: pd.DataFrame) -> pd.DataFrame:
    """Classify waves by each symbol's own point-size quantiles."""
    if waves.empty:
        result = waves.copy()
        if "level" not in result.columns:
            result["level"] = pd.Series(dtype="object")
        return result

    if "status" not in waves.columns:
        return _classify_full_sample_levels(waves)

    result = waves.copy()
    result["level"] = ""
    for _, group in result.groupby("symbol", dropna=False, sort=False):
        history: list[float] = []
        for index, row in group.sort_values(_wave_order_columns(group)).iterrows():
            point = float(row.get("points", 0.0) or 0.0)
            level_history = history + [point]
            result.loc[index, "level"] = _level_for_history(level_history, point)
            if row.get("status") == "confirmed":
                history.append(point)
    return result


def _classify_full_sample_levels(waves: pd.DataFrame) -> pd.DataFrame:
    result = waves.copy()
    result["level"] = ""
    for symbol, index in result.groupby("symbol").groups.items():
        points = result.loc[index, "points"].astype(float)
        if len(points) < 4 or points.nunique() == 1:
            result.loc[index, "level"] = LEVEL_LABELS[1]
            continue

        q25, q50, q75 = points.quantile([0.25, 0.50, 0.75]).tolist()
        result.loc[index, "level"] = points.apply(lambda value: _level_for_points(value, q25, q50, q75))
    return result


def _asof_zigzag_waves(
    df: pd.DataFrame,
    symbol: str,
    threshold: float | None,
    min_reversal_pct: float | None,
    rolling_thresholds: pd.Series | None,
    min_wave_days: int,
) -> list[dict[str, object]]:
    prices = df["close"].astype(float).to_numpy()
    if len(prices) < 2:
        return []

    min_wave_days = max(int(min_wave_days), 1)
    pivot_idx = 0
    pivot_price = float(prices[0])
    extreme_idx = 0
    extreme_price = pivot_price
    trend: str | None = None
    rows: list[dict[str, object]] = []

    for idx in range(1, len(prices)):
        price = float(prices[idx])
        active_threshold = _threshold_for_index(idx, threshold, rolling_thresholds)

        if trend is None:
            if _up_move_reaches(pivot_price, price, active_threshold, min_reversal_pct):
                trend = "up"
                extreme_idx = idx
                extreme_price = price
            elif _down_move_reaches(pivot_price, price, active_threshold, min_reversal_pct):
                trend = "down"
                extreme_idx = idx
                extreme_price = price
            elif price < pivot_price:
                pivot_idx = idx
                pivot_price = price
                extreme_idx = idx
                extreme_price = price
            continue

        if trend == "up":
            if price > extreme_price:
                extreme_idx = idx
                extreme_price = price
            elif _down_move_reaches(extreme_price, price, active_threshold, min_reversal_pct):
                if _has_min_wave_days(pivot_idx, extreme_idx, min_wave_days):
                    rows.append(
                        _wave_row(
                            df,
                            symbol,
                            direction="up",
                            status="confirmed",
                            start_idx=pivot_idx,
                            extreme_idx=extreme_idx,
                            end_idx=idx,
                            threshold=active_threshold,
                            min_reversal_pct=min_reversal_pct,
                        )
                    )
                    pivot_idx = extreme_idx
                    pivot_price = extreme_price
                    trend = "down"
                    extreme_idx = idx
                    extreme_price = price
        else:
            if price < extreme_price:
                extreme_idx = idx
                extreme_price = price
            elif _up_move_reaches(extreme_price, price, active_threshold, min_reversal_pct):
                if _has_min_wave_days(pivot_idx, extreme_idx, min_wave_days):
                    rows.append(
                        _wave_row(
                            df,
                            symbol,
                            direction="down",
                            status="confirmed",
                            start_idx=pivot_idx,
                            extreme_idx=extreme_idx,
                            end_idx=idx,
                            threshold=active_threshold,
                            min_reversal_pct=min_reversal_pct,
                        )
                    )
                    pivot_idx = extreme_idx
                    pivot_price = extreme_price
                    trend = "up"
                    extreme_idx = idx
                    extreme_price = price

    latest_idx = len(prices) - 1
    if trend is None:
        if latest_idx != pivot_idx and float(prices[latest_idx]) != pivot_price:
            trend = "up" if float(prices[latest_idx]) > pivot_price else "down"
            extreme_idx = latest_idx
    if trend is not None and latest_idx != pivot_idx:
        rows.append(
            _wave_row(
                df,
                symbol,
                direction=trend,
                status="open",
                start_idx=pivot_idx,
                extreme_idx=extreme_idx,
                end_idx=latest_idx,
                threshold=_threshold_for_index(latest_idx, threshold, rolling_thresholds),
                min_reversal_pct=min_reversal_pct,
            )
        )
    return rows


def _wave_row(
    df: pd.DataFrame,
    symbol: str,
    direction: str,
    status: str,
    start_idx: int,
    extreme_idx: int,
    end_idx: int,
    threshold: float | None,
    min_reversal_pct: float | None,
) -> dict[str, object]:
    start_price = float(df.loc[start_idx, "close"])
    end_price = float(df.loc[extreme_idx, "close"])
    confirmation_price = float(df.loc[end_idx, "close"])
    delta = end_price - start_price
    latest_reversal = _current_reversal_from_extreme(direction, end_price, confirmation_price)
    reversal_progress = _reversal_progress(latest_reversal, end_price, threshold, min_reversal_pct)
    confirmation_date = pd.Timestamp(df.loc[end_idx, "trade_date"]) if status == "confirmed" else pd.NaT
    return {
        "symbol": symbol,
        "direction": direction,
        "status": status,
        "start_date": pd.Timestamp(df.loc[start_idx, "trade_date"]),
        "end_date": pd.Timestamp(df.loc[end_idx, "trade_date"]),
        "confirmation_date": confirmation_date,
        "extreme_date": pd.Timestamp(df.loc[extreme_idx, "trade_date"]),
        "start_price": start_price,
        "end_price": end_price,
        "extreme_price": end_price,
        "confirmation_price": confirmation_price,
        "points": round(abs(delta), 6),
        "pct_change": round(delta / start_price * 100, 6) if start_price else 0.0,
        "days": int(end_idx - start_idx + 1),
        "reversal_threshold": float(threshold or 0.0),
        "reversal_threshold_pct": float(min_reversal_pct or 0.0),
        "reversal_progress": reversal_progress,
    }


def _current_reversal_from_extreme(direction: str, extreme_price: float, current_price: float) -> float:
    if direction == "up":
        return max(extreme_price - current_price, 0.0)
    return max(current_price - extreme_price, 0.0)


def _reversal_progress(
    reversal_points: float,
    extreme_price: float,
    threshold: float | None,
    min_reversal_pct: float | None,
) -> float:
    if min_reversal_pct is not None:
        denominator = abs(extreme_price) * min_reversal_pct / 100.0
    else:
        denominator = float(threshold or 0.0)
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(reversal_points / denominator, 1.0)) * 100.0, 2)


def _wave_order_columns(group: pd.DataFrame) -> list[str]:
    columns = []
    if "confirmation_date" in group.columns:
        columns.append("confirmation_date")
    if "end_date" in group.columns:
        columns.append("end_date")
    return columns or list(group.index.names)


def _threshold_for_index(
    idx: int,
    threshold: float | None,
    rolling_thresholds: pd.Series | None,
) -> float | None:
    if rolling_thresholds is None:
        return threshold
    value = rolling_thresholds.iloc[idx]
    return float(value) if pd.notna(value) else 0.0


def _rolling_reversal_thresholds(df: pd.DataFrame, atr_multiplier: float) -> pd.Series:
    """Return per-row thresholds computed only from data visible through that row."""
    if df.empty:
        return pd.Series(dtype="float64")
    values = [_adaptive_reversal_threshold(df.iloc[: index + 1], atr_multiplier) for index in range(len(df))]
    return pd.Series(values, index=df.index, dtype="float64")


def _zigzag_pivots(
    prices: np.ndarray,
    threshold: float | None,
    min_reversal_pct: float | None = None,
    min_wave_days: int = 1,
) -> list[int]:
    if len(prices) < 2:
        return [0]

    min_wave_days = max(int(min_wave_days), 1)
    pivot_idx = 0
    pivot_price = float(prices[0])
    extreme_idx = 0
    extreme_price = pivot_price
    trend: str | None = None
    pivots = [0]

    for idx in range(1, len(prices)):
        price = float(prices[idx])
        if trend is None:
            if _up_move_reaches(pivot_price, price, threshold, min_reversal_pct):
                trend = "up"
                extreme_idx = idx
                extreme_price = price
            elif _down_move_reaches(pivot_price, price, threshold, min_reversal_pct):
                trend = "down"
                extreme_idx = idx
                extreme_price = price
            elif price < pivot_price:
                pivot_idx = idx
                pivot_price = price
                extreme_idx = idx
                extreme_price = price
                pivots[-1] = idx
            continue

        if trend == "up":
            if price > extreme_price:
                extreme_idx = idx
                extreme_price = price
            elif _down_move_reaches(extreme_price, price, threshold, min_reversal_pct):
                if pivots[-1] != extreme_idx and _has_min_wave_days(pivots[-1], extreme_idx, min_wave_days):
                    pivots.append(extreme_idx)
                    trend = "down"
                    extreme_idx = idx
                    extreme_price = price
        else:
            if price < extreme_price:
                extreme_idx = idx
                extreme_price = price
            elif _up_move_reaches(extreme_price, price, threshold, min_reversal_pct):
                if pivots[-1] != extreme_idx and _has_min_wave_days(pivots[-1], extreme_idx, min_wave_days):
                    pivots.append(extreme_idx)
                    trend = "up"
                    extreme_idx = idx
                    extreme_price = price

    if trend is not None and pivots[-1] != extreme_idx and _has_min_wave_days(pivots[-1], extreme_idx, min_wave_days):
        pivots.append(extreme_idx)
    elif trend is None and pivot_idx != len(prices) - 1:
        pivots.append(len(prices) - 1)

    if len(pivots) == 1 and pivots[0] != len(prices) - 1:
        pivots.append(len(prices) - 1)
    return pivots


def _has_min_wave_days(start_idx: int, end_idx: int, min_wave_days: int) -> bool:
    return abs(end_idx - start_idx) + 1 >= min_wave_days


def _up_move_reaches(start_price: float, end_price: float, threshold: float | None, threshold_pct: float | None) -> bool:
    if end_price <= start_price:
        return False
    if threshold_pct is not None:
        return _pct_move(start_price, end_price) >= threshold_pct
    return end_price - start_price >= float(threshold or 0.0)


def _down_move_reaches(start_price: float, end_price: float, threshold: float | None, threshold_pct: float | None) -> bool:
    if end_price >= start_price:
        return False
    if threshold_pct is not None:
        return _pct_move(start_price, end_price) >= threshold_pct
    return start_price - end_price >= float(threshold or 0.0)


def _pct_move(start_price: float, end_price: float) -> float:
    if start_price == 0:
        return 0.0
    return abs((end_price - start_price) / start_price * 100.0)


def _adaptive_reversal_threshold(df: pd.DataFrame, atr_multiplier: float) -> float:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    pre_close = df["pre_close"].astype(float).fillna(close.shift(1)).fillna(close)
    true_range = pd.concat([(high - low), (high - pre_close).abs(), (low - pre_close).abs()], axis=1).max(axis=1)
    atr_series = true_range.rolling(14, min_periods=3).mean()
    atr = atr_series.median() if atr_series.notna().any() else np.nan
    daily_std = close.pct_change().std()
    candidates = []
    if pd.notna(atr) and atr > 0:
        candidates.append(float(atr * atr_multiplier))
    if pd.notna(daily_std) and daily_std > 0:
        candidates.append(float(close.median() * daily_std * 2))
    daily_abs_series = close.diff().abs().rolling(20, min_periods=5).median()
    daily_abs_move = daily_abs_series.median() if daily_abs_series.notna().any() else np.nan
    if pd.notna(daily_abs_move) and daily_abs_move > 0:
        candidates.append(float(daily_abs_move * 3))
    if candidates:
        return max(candidates)
    return float(max(close.max() - close.min(), 0.0))


def _level_for_history(points: list[float], value: float) -> str:
    series = pd.Series(points, dtype="float64")
    if len(series) == 1 or series.nunique() == 1:
        return LEVEL_LABELS[1]
    q25, q50, q75 = series.quantile([0.25, 0.50, 0.75]).tolist()
    return _level_for_points(value, q25, q50, q75)


def _level_for_points(value: float, q25: float, q50: float, q75: float) -> str:
    if value <= q25:
        return LEVEL_LABELS[0]
    if value <= q50:
        return LEVEL_LABELS[1]
    if value <= q75:
        return LEVEL_LABELS[2]
    return LEVEL_LABELS[3]


def _empty_waves() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "direction",
            "status",
            "start_date",
            "end_date",
            "confirmation_date",
            "extreme_date",
            "start_price",
            "end_price",
            "extreme_price",
            "confirmation_price",
            "points",
            "pct_change",
            "days",
            "reversal_threshold",
            "reversal_threshold_pct",
            "reversal_progress",
            "level",
        ]
    )
