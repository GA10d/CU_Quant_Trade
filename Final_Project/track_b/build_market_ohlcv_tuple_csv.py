from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


TRACK_B_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRACK_B_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "market_data_full.csv"
CACHE_DIR = PROJECT_DIR / ".yfinance_cache"

TICKERS = ["GLD", "HYG", "LQD", "SPY", "TLT", "UUP", "^VIX"]
FIELDS = ["Close", "High", "Low", "Open", "Volume"]
START_DATE = "2015-01-01"


def format_tuple_cell(row: pd.Series) -> str:
    values = []
    for field in FIELDS:
        value = row.get(field)
        if pd.isna(value):
            values.append("null")
        elif field == "Volume":
            values.append(str(int(value)))
        else:
            values.append(f"{float(value):.6f}")
    return "(" + ", ".join(values) + ")"


def download_ohlcv_tuple_frame(auto_adjust: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(CACHE_DIR))

    raw = yf.download(
        tickers=TICKERS,
        start=START_DATE,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise RuntimeError("Yahoo download returned an empty DataFrame.")

    tuple_df = pd.DataFrame(index=raw.index)
    tuple_df.index.name = "Date"

    for ticker in TICKERS:
        ticker_frame = pd.DataFrame(index=raw.index)
        for field in FIELDS:
            if field not in raw.columns.get_level_values(0):
                raise KeyError(f"Field `{field}` is missing from Yahoo download result.")
            ticker_frame[field] = raw[field][ticker]
        tuple_df[ticker] = ticker_frame.apply(format_tuple_cell, axis=1)

    return tuple_df


def build_tuple_csv(output_path: Path = OUTPUT_PATH, auto_adjust: bool = False) -> Path:
    tuple_df = download_ohlcv_tuple_frame(auto_adjust=auto_adjust)
    tuple_df.to_csv(output_path)
    return output_path


if __name__ == "__main__":
    output_path = build_tuple_csv()
    print(f"Saved tuple OHLCV CSV to: {output_path}")
