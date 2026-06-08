from __future__ import annotations

from pathlib import Path

import pandas as pd

from trend_score.catalog import INDEX_ASSETS, assets_by_code, get_assets, group_data_dir


STANDARD_COLUMNS = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]

INDEX_SYMBOLS = {asset.ts_code: asset.name for asset in INDEX_ASSETS}

PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Load one local OHLCV CSV and normalize it to the project schema."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in STANDARD_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {', '.join(missing)}")

    df = df[STANDARD_COLUMNS].copy()
    df["trade_date"] = _parse_trade_date(df["trade_date"])
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["ts_code", "trade_date", *PRICE_COLUMNS])
    df = df.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return df


def available_symbol_files(data_dir: str | Path = "data", group: str = "index") -> dict[str, Path]:
    """Return CSV files keyed by symbol for one configured asset group."""
    root = group_data_dir(data_dir, group)
    if not root.exists():
        return {}

    files: dict[str, Path] = {}
    for asset in get_assets(group):
        symbol = asset.ts_code
        match = find_symbol_csv(root, symbol)
        if match is not None:
            files[symbol] = match
    return files


def find_symbol_csv(data_dir: str | Path, symbol: str) -> Path | None:
    root = Path(data_dir)
    candidates = [
        root / f"{symbol}.csv",
        root / f"{symbol.replace('.', '_')}.csv",
        root / f"{symbol.lower()}.csv",
        root / f"{symbol.replace('.', '_').lower()}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_symbol_data(data_dir: str | Path, symbol: str, group: str = "index") -> pd.DataFrame:
    root = group_data_dir(data_dir, group)
    csv_path = find_symbol_csv(root, symbol)
    if csv_path is None:
        raise FileNotFoundError(f"No local CSV found for {group}/{symbol} under {root}")
    df = load_ohlcv_csv(csv_path)
    return df[df["ts_code"] == symbol].reset_index(drop=True)


def display_name(symbol: str, group: str) -> str:
    return assets_by_code(group).get(symbol).name if symbol in assets_by_code(group) else symbol


def _parse_trade_date(values: pd.Series) -> pd.Series:
    as_text = values.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    parsed = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    fallback = pd.to_datetime(as_text, errors="coerce")
    return parsed.fillna(fallback)
