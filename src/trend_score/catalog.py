from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ASSET_GROUPS = {
    "index": "指数",
    "sector": "板块",
    "commodity": "商品",
}


@dataclass(frozen=True)
class AssetSpec:
    group: str
    ts_code: str
    name: str
    source: str
    exchange: str | None = None
    idx_type: str | None = None
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.ts_code})"


INDEX_ASSETS = [
    AssetSpec("index", "000001.SH", "上证指数", "index_daily"),
    AssetSpec("index", "000300.SH", "沪深300", "index_daily"),
    AssetSpec("index", "000688.SH", "科创50", "index_daily"),
    AssetSpec("index", "000852.SH", "中证1000", "index_daily"),
    AssetSpec("index", "399006.SZ", "创业板指", "index_daily"),
]

SECTOR_ASSETS = [
    AssetSpec("sector", "BK1128.DC", "CPO概念", "dc_daily", idx_type="概念板块", note="东方财富概念板块"),
    AssetSpec("sector", "BK1202.DC", "房地产", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
    AssetSpec("sector", "BK1216.DC", "医药生物", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
    AssetSpec("sector", "BK0438.DC", "食品饮料", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
    AssetSpec("sector", "BK0900.DC", "新能源车", "dc_daily", idx_type="概念板块", note="东方财富概念板块"),
    AssetSpec("sector", "BK1203.DC", "非银金融", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
    AssetSpec("sector", "BK0737.DC", "软件开发", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
    AssetSpec("sector", "BK1215.DC", "通信", "dc_daily", idx_type="行业板块", note="东方财富行业板块"),
]

COMMODITY_ASSETS = [
    AssetSpec("commodity", "RBL.SHF", "螺纹钢连续", "fut_daily", exchange="SHFE"),
    AssetSpec("commodity", "SCL.INE", "原油连续", "fut_daily", exchange="INE"),
    AssetSpec("commodity", "AUL.SHF", "黄金连续", "fut_daily", exchange="SHFE"),
    AssetSpec("commodity", "AGL.SHF", "白银连续", "fut_daily", exchange="SHFE"),
]

ALL_ASSETS = [*INDEX_ASSETS, *SECTOR_ASSETS, *COMMODITY_ASSETS]


def get_assets(group: str | None = None) -> list[AssetSpec]:
    if group is None or group == "全部":
        return list(ALL_ASSETS)
    if group not in ASSET_GROUPS:
        raise ValueError(f"unknown asset group: {group}")
    return [asset for asset in ALL_ASSETS if asset.group == group]


def get_asset(group: str, ts_code: str) -> AssetSpec:
    for asset in get_assets(group):
        if asset.ts_code == ts_code:
            return asset
    raise KeyError(f"unknown asset: {group}/{ts_code}")


def asset_options(group: str) -> dict[str, str]:
    return {asset.label: asset.ts_code for asset in get_assets(group)}


def assets_by_code(group: str) -> dict[str, AssetSpec]:
    return {asset.ts_code: asset for asset in get_assets(group)}


def group_data_dir(data_dir: str | Path, group: str) -> Path:
    return Path(data_dir) / group


def save_manifest(data_dir: str | Path, assets: Iterable[AssetSpec] = ALL_ASSETS) -> Path:
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    payload = {
        "asset_groups": ASSET_GROUPS,
        "assets": [asdict(asset) for asset in assets],
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
