from __future__ import annotations

import numpy as np
import pandas as pd


def rank_waves(scored_waves: pd.DataFrame, direction: str | None = None, level: str | None = None) -> pd.DataFrame:
    result = scored_waves.copy()
    if direction and direction != "全部":
        result = result[result["direction"] == direction]
    if level and level != "全部":
        result = result[result["level"] == level]
    return result.sort_values("total_score", ascending=False).reset_index(drop=True)


def relative_path(prices: pd.DataFrame, wave: pd.Series | dict, label: str) -> pd.DataFrame:
    wave_series = pd.Series(wave)
    mask = (
        (prices["ts_code"] == wave_series["symbol"])
        & (prices["trade_date"] >= pd.Timestamp(wave_series["start_date"]))
        & (prices["trade_date"] <= pd.Timestamp(wave_series["end_date"]))
    )
    path = prices.loc[mask, ["trade_date", "close"]].sort_values("trade_date").reset_index(drop=True)
    if path.empty:
        return pd.DataFrame(columns=["wave", "step", "trade_date", "close", "relative_close"])
    start = float(path.loc[0, "close"])
    path["wave"] = label
    path["step"] = range(len(path))
    path["relative_close"] = path["close"].astype(float) / start * 100 if start else 0.0
    return path[["wave", "step", "trade_date", "close", "relative_close"]]


def compare_two_waves(prices: pd.DataFrame, scored_waves: pd.DataFrame, first_index: int, second_index: int) -> dict[str, pd.DataFrame]:
    return compare_waves(prices, scored_waves, [first_index, second_index])


def compare_waves(prices: pd.DataFrame, scored_waves: pd.DataFrame, indices: list[int]) -> dict[str, pd.DataFrame]:
    selected = [scored_waves.loc[index] for index in indices]
    labels = [f"wave_{position}" for position in range(len(selected))]
    metrics = pd.DataFrame([_metric_row(wave, label) for wave, label in zip(selected, labels)])
    paths = pd.concat(
        [relative_path(prices, wave, label) for wave, label in zip(selected, labels)],
        ignore_index=True,
    )
    return {"metrics": metrics, "paths": paths}


def score_interval_continuity(prices: pd.DataFrame, start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DataFrame:
    """Score every symbol's trend continuity inside one user-selected interval."""
    if prices.empty:
        return _empty_interval_scores()

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []
    for symbol, group in prices.groupby("ts_code"):
        segment = group[(group["trade_date"] >= start) & (group["trade_date"] <= end)].sort_values("trade_date")
        if len(segment) < 2:
            continue
        first_close = float(segment.iloc[0]["close"])
        last_close = float(segment.iloc[-1]["close"])
        signed_points = last_close - first_close
        pct_change = signed_points / first_close * 100 if first_close else 0.0
        direction = "up" if signed_points >= 0 else "down"
        days = int(len(segment))
        adverse_pct = _interval_adverse_pct(segment, direction)
        move = max(abs(pct_change), 1.0)
        drawdown_score = max(0.0, 100.0 * (1.0 - min(adverse_pct / move, 1.0)))
        rows.append(
            {
                "symbol": symbol,
                "direction": direction,
                "start_date": segment.iloc[0]["trade_date"],
                "end_date": segment.iloc[-1]["trade_date"],
                "start_price": first_close,
                "end_price": last_close,
                "points": abs(signed_points),
                "pct_change": pct_change,
                "days": days,
                "slope": abs(pct_change) / max(days - 1, 1),
                "max_adverse_pct": adverse_pct,
                "drawdown_score": round(drawdown_score, 2),
                "stability_score": _interval_stability_score(segment),
                "volume_score": _interval_volume_score(segment, direction),
            }
        )

    if not rows:
        return _empty_interval_scores()

    result = pd.DataFrame(rows)
    result["strength_score"] = _rank_score(result["pct_change"].abs())
    result["slope_score"] = _rank_score(result["slope"])
    result["duration_score"] = _rank_score(result["days"])
    result["interval_score"] = (
        result["strength_score"] * 0.20
        + result["duration_score"] * 0.10
        + result["slope_score"] * 0.20
        + result["drawdown_score"] * 0.20
        + result["stability_score"] * 0.20
        + result["volume_score"] * 0.10
    ).round(2)
    return result.sort_values("interval_score", ascending=False).reset_index(drop=True)


def _metric_row(wave: pd.Series, label: str) -> dict:
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
    ]
    row = {"label": label}
    for column in columns:
        if column in wave:
            row[column] = wave[column]
    return row


def _rank_score(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if len(numeric) == 1:
        return pd.Series([100.0], index=values.index)
    return numeric.rank(pct=True, method="average").mul(100).round(2)


def _interval_adverse_pct(segment: pd.DataFrame, direction: str) -> float:
    closes = segment["close"].astype(float)
    if direction == "up":
        adverse = ((closes / closes.cummax()) - 1.0).min()
    else:
        adverse = ((closes / closes.cummin()) - 1.0).max()
    return abs(float(adverse) * 100.0)


def _interval_stability_score(segment: pd.DataFrame) -> float:
    if len(segment) < 3:
        return 50.0
    closes = segment["close"].astype(float).to_numpy()
    if closes[0] == 0:
        return 50.0
    y = closes / closes[0] * 100
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total_var = float(((y - y.mean()) ** 2).sum())
    if total_var == 0:
        return 50.0
    residual_var = float(((y - fitted) ** 2).sum())
    return round(max(0.0, min(1.0, 1.0 - residual_var / total_var)) * 100.0, 2)


def _interval_volume_score(segment: pd.DataFrame, direction: str) -> float:
    if len(segment) < 3 or "vol" not in segment:
        return 50.0
    volume = pd.to_numeric(segment["vol"], errors="coerce")
    closes = segment["close"].astype(float)
    directional_price = closes if direction == "up" else -closes
    corr = directional_price.rank().corr(volume.rank())
    if pd.isna(corr):
        return 50.0
    return round(float((corr + 1.0) / 2.0 * 100.0), 2)


def _empty_interval_scores() -> pd.DataFrame:
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
            "slope",
            "max_adverse_pct",
            "strength_score",
            "duration_score",
            "slope_score",
            "drawdown_score",
            "stability_score",
            "volume_score",
            "interval_score",
        ]
    )
