import pandas as pd
import pytest

from trend_score.data import STANDARD_COLUMNS, load_ohlcv_csv


def test_load_ohlcv_csv_sorts_deduplicates_and_normalizes_dates(tmp_path):
    csv_path = tmp_path / "000001.SH.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
                "000001.SH,20240103,102,108,101,107,101,6,5.94,1200,3000",
                "000001.SH,20240102,100,103,99,101,100,1,1.0,1000,2000",
                "000001.SH,20240103,103,109,102,108,101,7,6.93,1300,3300",
            ]
        ),
        encoding="utf-8",
    )

    df = load_ohlcv_csv(csv_path)

    assert list(df.columns) == STANDARD_COLUMNS
    assert df["trade_date"].dt.strftime("%Y%m%d").tolist() == ["20240102", "20240103"]
    assert df["close"].tolist() == [101.0, 108.0]
    assert df["vol"].tolist() == [1000.0, 1300.0]


def test_load_ohlcv_csv_rejects_missing_required_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,open,high,low,close",
                "000001.SH,20240102,100,103,99,101",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_ohlcv_csv(csv_path)


def test_load_ohlcv_csv_drops_rows_with_null_price_fields(tmp_path):
    csv_path = tmp_path / "nulls.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
                "000001.SH,20240102,100,103,99,101,100,1,1.0,1000,2000",
                "000001.SH,20240103,102,108,101,,101,6,5.94,1200,3000",
            ]
        ),
        encoding="utf-8",
    )

    df = load_ohlcv_csv(csv_path)

    assert len(df) == 1
    assert pd.Timestamp("2024-01-02") == df.loc[0, "trade_date"]
