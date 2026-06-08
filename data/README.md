# Local CSV Data

The Streamlit dashboard reads local CSV files only. It does not fetch Tushare data at runtime.

Data is grouped by asset type:

```text
data/index/
data/sector/
data/commodity/
```

Refresh local files from Tushare:

```text
python scripts/refresh_data.py --group index
python scripts/refresh_data.py --group sector
python scripts/refresh_data.py --group commodity
```

Required columns:

```text
ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
```

`trade_date` should use `YYYYMMDD`. Rows are sorted ascending by date when loaded; duplicate `ts_code + trade_date` rows keep the last row in the file.
