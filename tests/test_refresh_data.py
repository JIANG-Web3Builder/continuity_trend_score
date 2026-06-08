import pandas as pd

from scripts.refresh_data import normalize_tushare_rows
from scripts.refresh_data import save_asset_csvs
from trend_score.catalog import AssetSpec
from trend_score.data import STANDARD_COLUMNS


def test_normalize_index_daily_rows_keeps_standard_schema():
    asset = AssetSpec("index", "000001.SH", "上证指数", "index_daily")
    rows = [
        {
            "ts_code": "000001.SH",
            "trade_date": "20240102",
            "open": 100,
            "high": 103,
            "low": 99,
            "close": 101,
            "pre_close": 100,
            "change": 1,
            "pct_chg": 1.0,
            "vol": 1000,
            "amount": 2000,
        }
    ]

    df = normalize_tushare_rows(asset, rows)

    assert list(df.columns) == STANDARD_COLUMNS
    assert df.loc[0, "pct_chg"] == 1.0


def test_normalize_sector_rows_maps_pct_change_and_missing_amount():
    asset = AssetSpec("sector", "886033.TI", "共封装光学(CPO)", "ths_daily")
    rows = [
        {
            "ts_code": "886033.TI",
            "trade_date": "20240102",
            "open": 100,
            "high": 103,
            "low": 99,
            "close": 101,
            "pre_close": 100,
            "change": 1,
            "pct_change": 1.0,
            "vol": 1000,
        }
    ]

    df = normalize_tushare_rows(asset, rows)

    assert pd.isna(df.loc[0, "amount"])
    assert df.loc[0, "pct_chg"] == 1.0


def test_normalize_futures_rows_uses_change1_and_settle_safe_price_fill():
    asset = AssetSpec("commodity", "RBL.SHF", "螺纹钢连续", "fut_daily", exchange="SHFE")
    rows = [
        {
            "ts_code": "RBL.SHF",
            "trade_date": "20240102",
            "pre_close": 100,
            "open": None,
            "high": None,
            "low": None,
            "close": 101,
            "settle": 101,
            "change1": 1,
            "vol": 1000,
            "amount": 2000,
        }
    ]

    df = normalize_tushare_rows(asset, rows)

    assert df.loc[0, "open"] == 101
    assert df.loc[0, "high"] == 101
    assert df.loc[0, "low"] == 101
    assert df.loc[0, "change"] == 1
    assert df.loc[0, "pct_chg"] == 1.0


def test_save_asset_csvs_writes_group_directories_and_manifest(tmp_path):
    asset = AssetSpec("index", "000001.SH", "上证指数", "index_daily")
    rows = [
        {
            "ts_code": "000001.SH",
            "trade_date": "20240102",
            "open": 100,
            "high": 103,
            "low": 99,
            "close": 101,
            "pre_close": 100,
            "change": 1,
            "pct_chg": 1.0,
            "vol": 1000,
            "amount": 2000,
        }
    ]

    written = save_asset_csvs(tmp_path, {asset: rows})

    csv_path = tmp_path / "index" / "000001.SH.csv"
    assert written == {asset.ts_code: csv_path}
    assert csv_path.exists()
    assert (tmp_path / "manifest.json").exists()
