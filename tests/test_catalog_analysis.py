import pandas as pd

from app import build_wave_scores
from trend_score.catalog import ASSET_GROUPS, AssetSpec, asset_options, get_assets, save_manifest
from trend_score.data import available_symbol_files


def test_catalog_has_index_sector_and_commodity_groups():
    assert set(ASSET_GROUPS) == {"index", "sector", "commodity"}
    assert {"000001.SH", "000300.SH", "000688.SH", "000852.SH", "399006.SZ"}.issubset(
        {asset.ts_code for asset in get_assets("index")}
    )
    assert any(asset.ts_code == "BK1128.DC" for asset in get_assets("sector"))
    assert any(asset.ts_code == "RBL.SHF" for asset in get_assets("commodity"))


def test_available_symbol_files_is_group_aware(tmp_path):
    group_dir = tmp_path / "sector"
    group_dir.mkdir()
    csv_path = group_dir / "BK1128.DC.csv"
    csv_path.write_text("placeholder", encoding="utf-8")

    files = available_symbol_files(tmp_path, group="sector")

    assert files == {"BK1128.DC": csv_path}


def test_asset_options_returns_label_to_code_mapping():
    options = asset_options("commodity")

    assert "螺纹钢连续 (RBL.SHF)" in options
    assert options["螺纹钢连续 (RBL.SHF)"] == "RBL.SHF"


def test_save_manifest_writes_asset_metadata(tmp_path):
    assets = [AssetSpec(group="index", ts_code="000001.SH", name="上证指数", source="index_daily")]

    manifest_path = save_manifest(tmp_path, assets)

    text = manifest_path.read_text(encoding="utf-8")
    assert "000001.SH" in text
    assert "index_daily" in text


def test_build_wave_scores_supports_non_index_assets():
    dates = pd.date_range("2024-01-01", periods=9, freq="D")
    df = pd.DataFrame(
        {
            "ts_code": "886033.TI",
            "trade_date": dates,
            "open": [100, 104, 108, 106, 102, 110, 118, 116, 112],
            "high": [101, 105, 109, 107, 103, 111, 119, 117, 113],
            "low": [99, 103, 107, 105, 101, 109, 117, 115, 111],
            "close": [100, 104, 108, 106, 102, 110, 118, 116, 112],
            "pre_close": [100, 100, 104, 108, 106, 102, 110, 118, 116],
            "change": 0.0,
            "pct_chg": 0.0,
            "vol": 1000.0,
            "amount": 10000.0,
        }
    )

    scored = build_wave_scores(df, "886033.TI", min_reversal=4)

    assert not scored.empty
    assert set(scored["symbol"]) == {"886033.TI"}
    assert "historical_percentile" in scored.columns
