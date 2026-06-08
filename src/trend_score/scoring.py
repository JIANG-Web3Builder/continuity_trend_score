from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "strength_score",
    "duration_score",
    "slope_score",
    "drawdown_score",
    "stability_score",
    "volume_score",
    "total_score",
    "historical_percentile",
]

DEFAULT_WEIGHTS = {
    "strength_score": 0.20,
    "duration_score": 0.15,
    "slope_score": 0.20,
    "drawdown_score": 0.20,
    "stability_score": 0.15,
    "volume_score": 0.10,
}


def score_waves(prices: pd.DataFrame, waves: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Score detected waves with balanced continuity dimensions."""
    if waves.empty:
        result = waves.copy()
        for column in SCORE_COLUMNS:
            result[column] = pd.Series(dtype="float")
        return result

    weights = weights or DEFAULT_WEIGHTS
    result = waves.copy().reset_index(drop=True)
    segments = [_segment(prices, wave) for _, wave in result.iterrows()]

    result["strength_score"] = _percentile_score(result["pct_change"].abs())
    result["duration_score"] = _percentile_score(result["days"])
    result["slope_score"] = _percentile_score(result["points"] / result["days"].clip(lower=1))
    result["drawdown_score"] = [_drawdown_score(segment, wave) for segment, (_, wave) in zip(segments, result.iterrows())]
    result["stability_score"] = [_stability_score(segment) for segment in segments]
    result["volume_score"] = [_volume_score(segment, wave) for segment, (_, wave) in zip(segments, result.iterrows())]

    total = sum(result[column].astype(float) * weight for column, weight in weights.items())
    result["total_score"] = total.round(2).clip(0, 100)
    result["historical_percentile"] = _group_percentile(result, "total_score")
    return result


def _segment(prices: pd.DataFrame, wave: pd.Series) -> pd.DataFrame:
    symbol = wave.get("symbol")
    mask = (
        (prices["ts_code"] == symbol)
        & (prices["trade_date"] >= pd.Timestamp(wave["start_date"]))
        & (prices["trade_date"] <= pd.Timestamp(wave["end_date"]))
    )
    return prices.loc[mask].sort_values("trade_date").reset_index(drop=True)


def _percentile_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(numeric) == 1:
        return pd.Series([100.0], index=values.index)
    return (numeric.rank(pct=True, method="average") * 100).round(2)


def _group_percentile(df: pd.DataFrame, column: str) -> pd.Series:
    keys = ["symbol", "direction", "level"]
    return df.groupby(keys, dropna=False)[column].rank(pct=True, method="average").mul(100).round(2)


def _drawdown_score(segment: pd.DataFrame, wave: pd.Series) -> float:
    if segment.empty or len(segment) < 2:
        return 50.0
    closes = segment["close"].astype(float)
    move = max(abs(float(wave.get("pct_change", 0.0))), 1.0)
    if wave["direction"] == "up":
        adverse = ((closes / closes.cummax()) - 1.0).min()
    else:
        adverse = ((closes / closes.cummin()) - 1.0).max()
    adverse_pct = abs(float(adverse) * 100)
    return round(max(0.0, 100.0 * (1.0 - min(adverse_pct / move, 1.0))), 2)


def _stability_score(segment: pd.DataFrame) -> float:
    if segment.empty or len(segment) < 3:
        return 50.0
    closes = segment["close"].astype(float).to_numpy()
    y = closes / closes[0] * 100
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total_var = float(((y - y.mean()) ** 2).sum())
    if total_var == 0:
        return 50.0
    residual_var = float(((y - fitted) ** 2).sum())
    r_squared = max(0.0, min(1.0, 1.0 - residual_var / total_var))
    return round(r_squared * 100, 2)


def _volume_score(segment: pd.DataFrame, wave: pd.Series) -> float:
    if segment.empty or len(segment) < 3 or "vol" not in segment:
        return 50.0
    volume = segment["vol"].astype(float)
    closes = segment["close"].astype(float)
    directional_price = closes if wave["direction"] == "up" else -closes
    corr = directional_price.rank().corr(volume.rank())
    if pd.isna(corr):
        return 50.0
    return round(float((corr + 1.0) / 2.0 * 100.0), 2)
