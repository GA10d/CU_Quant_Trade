from __future__ import annotations

from pathlib import Path

from build_market_ohlcv_tuple_csv import DATA_DIR, build_tuple_csv


OUTPUT_PATH = Path(DATA_DIR) / "market_data_full_adjusted.csv"


if __name__ == "__main__":
    output_path = build_tuple_csv(output_path=OUTPUT_PATH, auto_adjust=True)
    print(f"Saved adjusted tuple OHLCV CSV to: {output_path}")
