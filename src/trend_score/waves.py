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
    """Detect directional waves from local reversals in close prices."""
    if prices.empty:
        return _empty_waves()

    df = prices.sort_values("trade_date").reset_index(drop=True).copy()
    symbol_value = symbol or str(df.loc[0, "ts_code"])
    threshold = None if min_reversal is None else max(float(min_reversal), 0.0)
    threshold_pct = None if min_reversal_pct is None else max(float(min_reversal_pct), 0.0)
    if threshold is None and threshold_pct is None:
        threshold = max(_adaptive_reversal_threshold(df, atr_multiplier), 0.0)

    pivots = _zigzag_pivots(
        df["close"].astype(float).to_numpy(),
        threshold,
        min_reversal_pct=threshold_pct,
        min_wave_days=min_wave_days,
    )
    if len(pivots) < 2:
        return _empty_waves()

    rows = []
    for start_idx, end_idx in zip(pivots[:-1], pivots[1:]):
        start_price = float(df.loc[start_idx, "close"])
        end_price = float(df.loc[end_idx, "close"])
        delta = end_price - start_price
        if delta == 0:
            continue
        start_date = pd.Timestamp(df.loc[start_idx, "trade_date"])
        end_date = pd.Timestamp(df.loc[end_idx, "trade_date"])
        rows.append(
            {
                "symbol": symbol_value,
                "direction": "up" if delta > 0 else "down",
                "start_date": start_date,
                "end_date": end_date,
                "start_price": start_price,
                "end_price": end_price,
                "points": round(abs(delta), 6),
                "pct_change": round(delta / start_price * 100, 6) if start_price else 0.0,
                "days": int(end_idx - start_idx + 1),
            }
        )

    return classify_wave_levels(pd.DataFrame(rows)) if rows else _empty_waves()


def classify_wave_levels(waves: pd.DataFrame) -> pd.DataFrame:
    """Classify waves by each symbol's own historical point-size quantiles."""
    if waves.empty:
        result = waves.copy()
        if "level" not in result.columns:
            result["level"] = pd.Series(dtype="object")
        return result

    result = waves.copy()
    result["level"] = ""
    for symbol, index in result.groupby("symbol").groups.items():
        points = result.loc[index, "points"].astype(float)
        if len(points) < 4 or points.nunique() == 1:
            result.loc[index, "level"] = "中"
            continue

        q25, q50, q75 = points.quantile([0.25, 0.50, 0.75]).tolist()
        result.loc[index, "level"] = points.apply(lambda value: _level_for_points(value, q25, q50, q75))
    return result


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
    atr = true_range.rolling(14, min_periods=3).mean().median()
    daily_std = close.pct_change().std()
    candidates = []
    if pd.notna(atr) and atr > 0:
        candidates.append(float(atr * atr_multiplier))
    if pd.notna(daily_std) and daily_std > 0:
        candidates.append(float(close.median() * daily_std * 2))
    daily_abs_move = close.diff().abs().rolling(20, min_periods=5).median().median()
    if pd.notna(daily_abs_move) and daily_abs_move > 0:
        candidates.append(float(daily_abs_move * 3))
    if candidates:
        return max(candidates)
    return float(max(close.max() - close.min(), 0.0))


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
            "start_date",
            "end_date",
            "start_price",
            "end_price",
            "points",
            "pct_change",
            "days",
            "level",
        ]
    )
