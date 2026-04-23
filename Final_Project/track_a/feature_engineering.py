"""
Feature Engineering Module for Cross-Asset Regime Detection
Creates features for HMM regime detection from the unified dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineer for creating regime detection features.
    """
    
    def __init__(self, price_data: pd.DataFrame, macro_data: pd.DataFrame = None):
        """
        Initialize the feature engineer.
        
        Args:
            price_data: DataFrame with asset prices
            macro_data: DataFrame with macroeconomic indicators
        """
        self.price_data = price_data
        self.macro_data = macro_data
        self.features = None
        self.feature_names = None
    
    def calculate_returns(self, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate daily returns for all assets.
        
        Args:
            price_data: DataFrame with asset prices (uses self.price_data if None)
            
        Returns:
            DataFrame with daily returns
        """
        if price_data is None:
            price_data = self.price_data
        
        returns = price_data.pct_change()
        returns.columns = [f"{col}_ret" for col in returns.columns]
        
        return returns
    
    def calculate_volatility(self, returns: pd.DataFrame, windows: List[int] = [5, 10, 21, 63]) -> pd.DataFrame:
        """
        Calculate rolling volatility for returns.
        
        Args:
            returns: DataFrame with returns
            windows: List of rolling window sizes
            
        Returns:
            DataFrame with rolling volatility features
        """
        vol_features = pd.DataFrame(index=returns.index)
        
        for col in returns.columns:
            for window in windows:
                vol = returns[col].rolling(window).std() * np.sqrt(252)  # Annualized
                vol_features[f"{col}_vol_{window}d"] = vol
        
        return vol_features
    
    def calculate_momentum(self, price_data: pd.DataFrame, windows: List[int] = [5, 10, 21, 63]) -> pd.DataFrame:
        """
        Calculate momentum (cumulative returns) for various windows.
        
        Args:
            price_data: DataFrame with prices
            windows: List of lookback windows
            
        Returns:
            DataFrame with momentum features
        """
        momentum_features = pd.DataFrame(index=price_data.index)
        
        for col in price_data.columns:
            for window in windows:
                mom = price_data[col].pct_change(window)
                momentum_features[f"{col}_mom_{window}d"] = mom
        
        return momentum_features
    
    def calculate_correlations(self, returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """
        Calculate rolling correlations between key assets.
        
        Args:
            returns: DataFrame with returns
            window: Rolling window size
            
        Returns:
            DataFrame with rolling correlation features
        """
        corr_features = pd.DataFrame(index=returns.index)
        
        # Key correlation pairs for regime detection
        pairs = [
            ("SPY_ret", "TLT_ret"),  # Equity-Bond correlation
            ("SPY_ret", "GLD_ret"),  # Equity-Gold correlation
            ("SPY_ret", "UUP_ret"),  # Equity-Dollar correlation
            ("HYG_ret", "LQD_ret"),  # Credit spread proxy
        ]
        
        for col1, col2 in pairs:
            if col1 in returns.columns and col2 in returns.columns:
                # Rolling correlation
                rolling_corr = returns[col1].rolling(window).corr(returns[col2])
                
                # Clean up column names
                name1 = col1.replace("_ret", "")
                name2 = col2.replace("_ret", "")
                corr_features[f"corr_{window}d_{name1}_{name2}"] = rolling_corr
        
        return corr_features
    
    def calculate_credit_spread_proxy(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate credit spread proxy using HYG and LQD.
        
        Args:
            price_data: DataFrame with prices
            
        Returns:
            Series with credit spread proxy
        """
        if "HYG" in price_data.columns and "LQD" in price_data.columns:
            credit_spread = np.log(price_data["HYG"] / price_data["LQD"])
            return credit_spread.rename("credit_spread_proxy")
        
        return None
    
    def calculate_yield_curve_features(self, macro_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate yield curve features from macro data.
        
        Args:
            macro_data: DataFrame with macro indicators
            
        Returns:
            DataFrame with yield curve features
        """
        if macro_data is None:
            macro_data = self.macro_data
        
        if macro_data is None:
            return pd.DataFrame()
        
        yc_features = pd.DataFrame(index=macro_data.index)
        
        # Yield curve slope (10Y - 2Y)
        if "DGS10" in macro_data.columns and "DGS2" in macro_data.columns:
            yc_features["curve_slope_10y2y"] = macro_data["DGS10"] - macro_data["DGS2"]
        
        # Yield curve slope (10Y - 3M)
        if "DGS10" in macro_data.columns and "DGS3MO" in macro_data.columns:
            yc_features["curve_slope_10y3m"] = macro_data["DGS10"] - macro_data["DGS3MO"]
        
        # Fed funds rate changes
        if "FEDFUNDS" in macro_data.columns:
            yc_features["fedfunds_change"] = macro_data["FEDFUNDS"].diff()
            yc_features["fedfunds_mom"] = macro_data["FEDFUNDS"].pct_change()
        
        return yc_features
    
    def calculate_vix_features(self, price_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate VIX-related features.
        
        Args:
            price_data: DataFrame with prices (must include VIX)
            
        Returns:
            DataFrame with VIX features
        """
        if price_data is None:
            price_data = self.price_data
        
        vix_features = pd.DataFrame(index=price_data.index)
        
        if "VIX" in price_data.columns:
            vix = price_data["VIX"]
            
            # VIX level
            vix_features["VIX_level"] = vix
            
            # VIX changes
            vix_features["VIX_ret"] = vix.pct_change()
            vix_features["VIX_log_ret"] = np.log(vix).diff()
            
            # VIX rolling statistics
            vix_features["VIX_ma_10d"] = vix.rolling(10).mean()
            vix_features["VIX_ma_21d"] = vix.rolling(21).mean()
            vix_features["VIX_vol_21d"] = vix.rolling(21).std()
            
            # VIX percentiles (rolling)
            vix_features["VIX_pct_63d"] = vix.rolling(63).rank(pct=True)
        
        return vix_features
    
    def build_hmm_features(self, 
                           price_data: pd.DataFrame = None, 
                           macro_data: pd.DataFrame = None,
                           drop_na: bool = True) -> pd.DataFrame:
        """
        Build the feature set for HMM regime detection.
        
        Args:
            price_data: DataFrame with prices
            macro_data: DataFrame with macro indicators
            drop_na: Whether to drop rows with missing values
            
        Returns:
            DataFrame with HMM features
        """
        if price_data is None:
            price_data = self.price_data
        if macro_data is None:
            macro_data = self.macro_data
        
        print("Building HMM features...")
        
        # Calculate returns
        returns = self.calculate_returns(price_data)
        
        # Calculate volatility features (for key assets)
        key_returns = returns[["SPY_ret", "TLT_ret", "GLD_ret"]].copy() if "SPY_ret" in returns.columns else returns
        vol_features = self.calculate_volatility(key_returns, windows=[21])
        
        # Calculate correlations
        corr_features = self.calculate_correlations(returns, window=60)
        
        # Calculate credit spread proxy
        credit_spread = self.calculate_credit_spread_proxy(price_data)
        
        # Calculate VIX features
        vix_features = self.calculate_vix_features(price_data)
        
        # Calculate yield curve features
        yc_features = self.calculate_yield_curve_features(macro_data)
        
        # Combine all features
        feature_list = []
        
        # Key returns for HMM
        key_return_cols = ["SPY_ret", "TLT_ret", "GLD_ret", "UUP_ret"]
        for col in key_return_cols:
            if col in returns.columns:
                feature_list.append(returns[col])
        
        # Volatility
        feature_list.append(vol_features)
        
        # VIX
        feature_list.append(vix_features[["VIX_level"]])
        
        # Credit spread
        if credit_spread is not None:
            feature_list.append(credit_spread)
        
        # Correlations
        feature_list.append(corr_features)
        
        # Yield curve
        feature_list.append(yc_features[["curve_slope_10y2y"]] if not yc_features.empty else pd.DataFrame(index=price_data.index))
        
        # Combine
        features = pd.concat(feature_list, axis=1)
        
        # Drop NaN values
        if drop_na:
            original_len = len(features)
            features = features.dropna()
            print(f"  Dropped {original_len - len(features)} rows with missing values")
        
        self.features = features
        self.feature_names = list(features.columns)
        
        print(f"✓ Built {len(features.columns)} features over {len(features)} observations")
        
        return features
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """
        Get descriptive statistics for all features.
        
        Returns:
            DataFrame with feature statistics
        """
        if self.features is None:
            raise ValueError("Features not yet built. Call build_hmm_features() first.")
        
        stats = self.features.describe().T
        stats["missing"] = self.features.isnull().sum()
        stats["missing_pct"] = stats["missing"] / len(self.features) * 100
        
        return stats
    
    def normalize_features(self, method: str = "standard") -> Tuple[pd.DataFrame, object]:
        """
        Normalize features for HMM.
        
        Args:
            method: Normalization method ('standard' or 'minmax')
            
        Returns:
            Tuple of (normalized features, scaler object)
        """
        if self.features is None:
            raise ValueError("Features not yet built. Call build_hmm_features() first.")
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        normalized_values = scaler.fit_transform(self.features)
        normalized_features = pd.DataFrame(
            normalized_values, 
            index=self.features.index, 
            columns=self.features.columns
        )
        
        print(f"✓ Normalized features using {method} scaling")
        
        return normalized_features, scaler


def main():
    """
    Main function to demonstrate feature engineering.
    """
    print("=" * 60)
    print("Cross-Asset Regime Detection - Feature Engineering")
    print("=" * 60)
    print()
    
    # Load unified data
    print("Loading unified data...")
    unified_data = pd.read_csv("data/unified_data.csv", index_col=0, parse_dates=True)
    market_data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
    macro_data = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
    
    print(f"✓ Loaded unified data: {len(unified_data)} observations")
    print()
    
    # Create feature engineer
    engineer = FeatureEngineer(price_data=market_data, macro_data=macro_data)
    
    # Build HMM features
    features = engineer.build_hmm_features()
    
    # Get feature statistics
    print()
    print("=" * 60)
    print("Feature Statistics")
    print("=" * 60)
    stats = engineer.get_feature_statistics()
    print(stats.round(4))
    
    # Normalize features
    print()
    normalized, scaler = engineer.normalize_features(method="standard")
    
    # Save features
    print()
    features.to_csv("data/hmm_features.csv")
    normalized.to_csv("data/hmm_features_normalized.csv")
    print("✓ Saved features to data/hmm_features.csv")
    print("✓ Saved normalized features to data/hmm_features_normalized.csv")
    
    print()
    print("=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)
    
    return engineer, normalized


if __name__ == "__main__":
    engineer, features = main()