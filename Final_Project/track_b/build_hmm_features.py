from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MARKET_DATA_PATH = DATA_DIR / "market_data.csv"
MACRO_DATA_PATH = DATA_DIR / "macro_data.csv"
OUTPUT_PATH = DATA_DIR / "hmm_features.csv"


def load_input_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    market_data = pd.read_csv(MARKET_DATA_PATH, index_col="Date", parse_dates=True)
    macro_data = pd.read_csv(MACRO_DATA_PATH, index_col="Date", parse_dates=True)

    market_data.index.name = "Date"
    macro_data.index.name = "Date"

    return market_data.sort_index(), macro_data.sort_index()


def add_feature_column(feature_store: list[pd.Series | pd.DataFrame], feature: pd.Series | pd.DataFrame) -> None:
    if isinstance(feature, pd.Series):
        if feature.notna().any():
            feature_store.append(feature)
        return

    valid_columns = [column for column in feature.columns if feature[column].notna().any()]
    if valid_columns:
        feature_store.append(feature[valid_columns])


def build_hmm_features(price_data: pd.DataFrame, macro_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    skipped_features: list[str] = []
    features: list[pd.Series | pd.DataFrame] = []

    price_data = price_data.rename(columns={"^VIX": "VIX"})
    returns = price_data.pct_change(fill_method=None)
    returns.columns = [f"{column}_ret" for column in returns.columns]

    for column in ["SPY_ret", "TLT_ret", "GLD_ret", "UUP_ret"]:
        if column in returns.columns and returns[column].notna().any():
            add_feature_column(features, returns[column])
        else:
            skipped_features.append(column)

    vol_features = pd.DataFrame(index=returns.index)
    for column in ["SPY_ret", "TLT_ret", "GLD_ret"]:
        if column in returns.columns and returns[column].notna().any():
            vol_features[f"{column}_vol_21d"] = returns[column].rolling(21).std() * np.sqrt(252)
        else:
            skipped_features.append(f"{column}_vol_21d")
    add_feature_column(features, vol_features)

    if "VIX" in price_data.columns and price_data["VIX"].notna().any():
        add_feature_column(features, price_data["VIX"].rename("VIX_level"))
    else:
        skipped_features.append("VIX_level")

    if {"HYG", "LQD"}.issubset(price_data.columns) and price_data["HYG"].notna().any() and price_data["LQD"].notna().any():
        credit_spread_proxy = np.log(price_data["HYG"] / price_data["LQD"]).rename("credit_spread_proxy")
        add_feature_column(features, credit_spread_proxy)
    else:
        skipped_features.append("credit_spread_proxy")

    corr_features = pd.DataFrame(index=returns.index)
    correlation_pairs = [
        ("SPY_ret", "TLT_ret"),
        ("SPY_ret", "GLD_ret"),
        ("SPY_ret", "UUP_ret"),
        ("HYG_ret", "LQD_ret"),
    ]
    for left, right in correlation_pairs:
        if left not in returns.columns or right not in returns.columns:
            skipped_features.append(f"corr_60d_{left.replace('_ret', '')}_{right.replace('_ret', '')}")
            continue
        if not returns[left].notna().any() or not returns[right].notna().any():
            skipped_features.append(f"corr_60d_{left.replace('_ret', '')}_{right.replace('_ret', '')}")
            continue
        corr_name = f"corr_60d_{left.replace('_ret', '')}_{right.replace('_ret', '')}"
        corr_features[corr_name] = returns[left].rolling(60).corr(returns[right])
    add_feature_column(features, corr_features)

    if {"DGS10", "DGS2"}.issubset(macro_data.columns):
        curve_slope = (macro_data["DGS10"] - macro_data["DGS2"]).rename("curve_slope_10y2y")
        add_feature_column(features, curve_slope)
    else:
        skipped_features.append("curve_slope_10y2y")

    hmm_features = pd.concat(features, axis=1)
    hmm_features = hmm_features.dropna(how="all", axis=1)
    hmm_features = hmm_features.dropna()
    hmm_features.index.name = "Date"

    return hmm_features, skipped_features


def main() -> None:
    market_data, macro_data = load_input_tables()
    hmm_features, skipped_features = build_hmm_features(market_data, macro_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    hmm_features.to_csv(OUTPUT_PATH)

    print(f"Saved HMM features to: {OUTPUT_PATH}")
    print(f"Rows: {len(hmm_features)}")
    print(f"Columns ({len(hmm_features.columns)}): {', '.join(hmm_features.columns)}")
    if skipped_features:
        unique_skipped = sorted(set(skipped_features))
        print(f"Skipped unavailable features: {', '.join(unique_skipped)}")


if __name__ == "__main__":
    main()
