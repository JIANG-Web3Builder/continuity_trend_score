from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trend_score.catalog import AssetSpec, get_assets, save_manifest
from trend_score.data import STANDARD_COLUMNS


TUSHARE_API_URL = "http://api.tushare.pro"


def normalize_tushare_rows(asset: AssetSpec, rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize Tushare API rows into the dashboard's fixed OHLCV schema."""
    if not rows:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = pd.DataFrame(rows).copy()
    df["ts_code"] = df.get("ts_code", asset.ts_code)
    df["trade_date"] = df.get("trade_date")

    for column in ["open", "high", "low", "close"]:
        if column not in df:
            df[column] = pd.NA
    fallback_price = df["close"] if "close" in df else df.get("settle")
    for column in ["open", "high", "low"]:
        df[column] = df[column].fillna(fallback_price)

    if "pre_close" not in df:
        if "change" in df:
            df["pre_close"] = pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(df["change"], errors="coerce")
        else:
            df["pre_close"] = pd.to_numeric(df["close"], errors="coerce").shift(-1)
    df["pre_close"] = df["pre_close"].fillna(df["close"])

    if "change" not in df:
        if "change1" in df:
            df["change"] = df["change1"]
        else:
            df["change"] = pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(df["pre_close"], errors="coerce")

    if "pct_chg" not in df:
        if "pct_change" in df:
            df["pct_chg"] = df["pct_change"]
        else:
            pre_close = pd.to_numeric(df["pre_close"], errors="coerce").replace(0, pd.NA)
            df["pct_chg"] = pd.to_numeric(df["change"], errors="coerce") / pre_close * 100.0

    if "vol" not in df:
        df["vol"] = pd.NA
    if "amount" not in df:
        df["amount"] = pd.NA

    result = df.reindex(columns=STANDARD_COLUMNS).copy()
    result["trade_date"] = result["trade_date"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    numeric_columns = [column for column in STANDARD_COLUMNS if column not in {"ts_code", "trade_date"}]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["ts_code", "trade_date", "open", "high", "low", "close"])
    result = result.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    return result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def save_asset_csvs(data_dir: str | Path, asset_rows: Mapping[AssetSpec, list[dict[str, Any]]]) -> dict[str, Path]:
    """Write normalized rows to data/<group>/<ts_code>.csv and update manifest."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for asset, rows in asset_rows.items():
        df = normalize_tushare_rows(asset, rows)
        if df.empty:
            continue
        group_dir = root / asset.group
        group_dir.mkdir(parents=True, exist_ok=True)
        csv_path = group_dir / f"{asset.ts_code}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        written[asset.ts_code] = csv_path
    save_manifest(root)
    return written


def fetch_asset_rows(asset: AssetSpec, token: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "ts_code": asset.ts_code,
        "start_date": start_date,
        "end_date": end_date,
    }
    if asset.source == "fut_daily" and asset.exchange:
        params["exchange"] = asset.exchange
    if asset.source == "dc_daily" and asset.idx_type:
        params["idx_type"] = asset.idx_type
    return query_tushare(asset.source, token, params)


def query_tushare(api_name: str, token: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = json.dumps({"api_name": api_name, "token": token, "params": dict(params)}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        TUSHARE_API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    if data.get("code") not in (0, None):
        raise RuntimeError(f"Tushare {api_name} failed: {data.get('msg')}")
    return tushare_payload_to_rows(data.get("data", data))


def tushare_payload_to_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, dict) and "items" in payload and "fields" in payload:
        fields = payload["fields"]
        return [dict(zip(fields, item)) for item in payload["items"]]
    if isinstance(payload, dict) and "data" in payload:
        return tushare_payload_to_rows(payload["data"])
    return []


def refresh_assets(
    data_dir: str | Path = "data",
    groups: Iterable[str] | None = None,
    symbols: Iterable[str] | None = None,
    start_date: str = "20200101",
    end_date: str = "20260608",
    token: str | None = None,
    continue_on_error: bool = True,
) -> dict[str, Path]:
    token = token or resolve_tushare_token()
    selected_symbols = set(symbols or [])
    assets = [asset for group in (groups or ["index", "sector", "commodity"]) for asset in get_assets(group)]
    if selected_symbols:
        assets = [asset for asset in assets if asset.ts_code in selected_symbols]
    if not assets:
        return {}

    asset_rows: dict[AssetSpec, list[dict[str, Any]]] = {}
    for asset in assets:
        try:
            asset_rows[asset] = fetch_asset_rows(asset, token, start_date, end_date)
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"skip {asset.ts_code}: {exc}")
    return save_asset_csvs(data_dir, asset_rows)


def resolve_tushare_token() -> str:
    env_token = os.getenv("TUSHARE_TOKEN")
    if env_token:
        return env_token
    raise RuntimeError("Missing Tushare token. Set TUSHARE_TOKEN or pass --token.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh local OHLCV CSVs from Tushare.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--start-date", default="20200101")
    parser.add_argument("--end-date", default="20260608")
    parser.add_argument("--group", action="append", choices=["index", "sector", "commodity"], dest="groups")
    parser.add_argument("--symbol", action="append", dest="symbols")
    parser.add_argument("--token", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = refresh_assets(
        data_dir=args.data_dir,
        groups=args.groups,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        token=args.token,
        continue_on_error=not args.fail_fast,
    )
    print(f"wrote {len(written)} CSV files")


if __name__ == "__main__":
    main()
