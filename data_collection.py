"""
Data Collection Module for Cross-Asset Regime Detection
This module handles data collection from multiple sources:
- yfinance: Market prices (equities, bonds, commodities, volatility)
- FRED: Macroeconomic indicators
- Ken French: Factor data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import requests

warnings.filterwarnings('ignore')

# Function to download Ken French data directly
def download_ken_french_data_direct(dataset_name: str = "F-F_Research_Data_5_Factors_2x3_daily") -> pd.DataFrame:
    """
    Download Ken French factor data directly from the website.
    """
    base_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    url = f"{base_url}{dataset_name}_CSV.zip"
    
    try:
        import io
        import zipfile
        
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get the CSV file name (same as zip name but with .csv)
            csv_name = f"{dataset_name}.csv"
            with z.open(csv_name) as f:
                # Read the CSV, skip the header rows
                content = f.read().decode('utf-8')
                
        # Parse the content
        lines = content.strip().split('\n')
        
        # Find where the actual data starts (after the header)
        data_lines = []
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    # Try to parse first column as date
                    int(parts[0])
                    data_lines.append(line)
                except ValueError:
                    continue
        
        # Create DataFrame from data lines
        from io import StringIO
        df = pd.read_csv(StringIO('\n'.join(data_lines)), header=None)
        
        # Set column names
        if '5_Factors' in dataset_name:
            df.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        else:
            df.columns = ['date'] + [f'col_{i}' for i in range(1, len(df.columns))]
        
        # Convert date
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        df = df.set_index('date')
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"  Warning: Could not download Ken French data: {e}")
        return None


class DataCollector:
    """
    Unified data collector for cross-asset regime detection.
    Handles data from multiple sources and provides a unified interface.
    """
    
    def __init__(self, start_date: str = "2015-01-01", end_date: Optional[str] = None):
        """
        Initialize the data collector.
        
        Args:
            start_date: Start date for data collection (YYYY-MM-DD format)
            end_date: End date for data collection (YYYY-MM-DD format), defaults to today
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # Define asset tickers and their descriptions
        self.market_tickers = {
            "SPY": "US Equities - S&P 500 ETF",
            "TLT": "Long-term US Treasuries - 20+ Year Treasury Bond ETF",
            "GLD": "Gold - SPDR Gold Shares ETF",
            "UUP": "US Dollar - Invesco DB US Dollar Index Bullish Fund ETF",
            "HYG": "High Yield Credit - iShares iBoxx High Yield Corporate Bond ETF",
            "LQD": "Investment Grade Credit - iShares iBoxx Investment Grade Corporate Bond ETF",
            "^VIX": "Volatility Index - CBOE Volatility Index",
        }
        
        # Define FRED macroeconomic series
        self.fred_series = {
            "GDP": "Real Gross Domestic Product",
            "UNRATE": "Unemployment Rate",
            "FEDFUNDS": "Effective Federal Funds Rate",
            "DGS10": "10-Year Treasury Constant Maturity Rate",
            "DGS2": "2-Year Treasury Constant Maturity Rate",
            "DGS3MO": "3-Month Treasury Constant Maturity Rate",
            "T10Y2Y": "10-Year Treasury Constant Maturity Minus 2-Year",
            "T10Y3M": "10-Year Treasury Constant Maturity Minus 3-Month",
            "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
            "INDPRO": "Industrial Production Index",
            "PAYEMS": "All Employees, Total Nonfarm",
        }
        
        # Storage for collected data
        self.market_data = None
        self.macro_data = None
        self.factor_data = None
        self.unified_data = None
    
    def download_market_data(self) -> pd.DataFrame:
        """
        Download market price data from yfinance.
        
        Returns:
            DataFrame with adjusted close prices for all market tickers
        """
        print(f"Downloading market data from {self.start_date} to {self.end_date}...")
        
        try:
            raw = yf.download(
                tickers=list(self.market_tickers.keys()),
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False,
            )
            
            # Extract adjusted close prices
            close = raw["Close"].copy()
            
            # Rename VIX for consistency
            if "^VIX" in close.columns:
                close = close.rename(columns={"^VIX": "VIX"})
            
            # Drop rows where all values are NaN
            close = close.dropna(how="all")
            
            self.market_data = close
            print(f"✓ Downloaded market data: {len(close)} observations, {len(close.columns)} assets")
            
            return close
            
        except Exception as e:
            print(f"✗ Error downloading market data: {e}")
            raise
    
    def download_fred_data(self) -> pd.DataFrame:
        """
        Download macroeconomic data from FRED.
        
        Returns:
            DataFrame with macroeconomic indicators
        """
        print(f"Downloading FRED macroeconomic data...")
        
        try:
            # Download all FRED series
            fred_dfs = []
            for series_id in self.fred_series.keys():
                try:
                    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                    df = pd.read_csv(url)
                    
                    # Standardize column names
                    date_col = next((col for col in ["DATE", "date", "observation_date"] 
                                   if col in df.columns), None)
                    if date_col is None:
                        continue
                    
                    value_col = next((col for col in df.columns if col != date_col), None)
                    if value_col is None:
                        continue
                    
                    df[date_col] = pd.to_datetime(df[date_col])
                    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
                    
                    df = df[[date_col, value_col]].rename(
                        columns={date_col: "date", value_col: series_id}
                    ).set_index("date")
                    
                    fred_dfs.append(df)
                    
                except Exception as e:
                    print(f"  Warning: Could not download {series_id}: {e}")
                    continue
            
            if not fred_dfs:
                raise ValueError("No FRED data could be downloaded")
            
            # Combine all series
            macro_df = pd.concat(fred_dfs, axis=1).sort_index()
            
            # Forward fill to align with market data frequency
            if self.market_data is not None:
                macro_df = macro_df.reindex(self.market_data.index).ffill()
            
            self.macro_data = macro_df
            print(f"✓ Downloaded FRED data: {len(macro_df)} observations, {len(macro_df.columns)} series")
            
            return macro_df
            
        except Exception as e:
            print(f"✗ Error downloading FRED data: {e}")
            raise
    
    def download_ken_french_data(self) -> pd.DataFrame:
        """
        Download factor data from Ken French Data Library.
        
        Returns:
            DataFrame with Fama-French factors
        """
        print(f"Downloading Ken French factor data...")
        
        try:
            # Download Fama-French 5-factor model
            ff5 = download_ken_french_data_direct("F-F_Research_Data_5_Factors_2x3_daily")
            
            if ff5 is not None:
                # Align with market data if available
                if self.market_data is not None:
                    ff5 = ff5.reindex(self.market_data.index).ffill()
                
                self.factor_data = ff5
                print(f"✓ Downloaded Ken French data: {len(ff5)} observations, {len(ff5.columns)} factors")
                return ff5
            else:
                print("  Note: Ken French data is optional for the main analysis")
                return None
            
        except Exception as e:
            print(f"✗ Error downloading Ken French data: {e}")
            # This is not critical, so we can continue without it
            print("  Note: Ken French data is optional for the main analysis")
            return None
    
    def create_unified_dataset(self) -> pd.DataFrame:
        """
        Create a unified dataset combining all data sources.
        
        Returns:
            DataFrame with all data sources aligned
        """
        print("Creating unified dataset...")
        
        # Ensure all data is downloaded
        if self.market_data is None:
            self.download_market_data()
        if self.macro_data is None:
            self.download_fred_data()
        
        # Start with market data
        unified = self.market_data.copy()
        
        # Add macro data
        if self.macro_data is not None:
            # Calculate additional macro features
            macro_features = self.macro_data.copy()
            
            # Yield curve features
            if "DGS10" in macro_features.columns and "DGS2" in macro_features.columns:
                macro_features["curve_slope_10y2y"] = macro_features["DGS10"] - macro_features["DGS2"]
            
            if "DGS10" in macro_features.columns and "DGS3MO" in macro_features.columns:
                macro_features["curve_slope_10y3m"] = macro_features["DGS10"] - macro_features["DGS3MO"]
            
            # Rate changes
            if "FEDFUNDS" in macro_features.columns:
                macro_features["fedfunds_change"] = macro_features["FEDFUNDS"].diff()
            
            if "curve_slope_10y2y" in macro_features.columns:
                macro_features["curve_slope_change"] = macro_features["curve_slope_10y2y"].diff()
            
            # Merge with unified dataset
            unified = pd.concat([unified, macro_features], axis=1)
        
        # Add factor data if available
        if self.factor_data is not None:
            unified = pd.concat([unified, self.factor_data], axis=1)
        
        # Remove rows with too many missing values
        max_missing_ratio = 0.3
        missing_ratio = unified.isnull().sum(axis=1) / len(unified.columns)
        unified = unified[missing_ratio <= max_missing_ratio]
        
        # Forward fill remaining missing values
        unified = unified.ffill().bfill()
        
        self.unified_data = unified
        print(f"✓ Created unified dataset: {len(unified)} observations, {len(unified.columns)} variables")
        
        return unified
    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of the collected data.
        
        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            "date_range": {
                "start": str(self.unified_data.index.min().date()) if self.unified_data is not None else None,
                "end": str(self.unified_data.index.max().date()) if self.unified_data is not None else None,
                "n_observations": len(self.unified_data) if self.unified_data is not None else 0,
            },
            "market_assets": {
                "count": len(self.market_data.columns) if self.market_data is not None else 0,
                "assets": list(self.market_data.columns) if self.market_data is not None else [],
            },
            "macro_series": {
                "count": len(self.macro_data.columns) if self.macro_data is not None else 0,
                "series": list(self.macro_data.columns) if self.macro_data is not None else [],
            },
            "factors": {
                "count": len(self.factor_data.columns) if self.factor_data is not None else 0,
                "factors": list(self.factor_data.columns) if self.factor_data is not None else [],
            },
            "total_variables": len(self.unified_data.columns) if self.unified_data is not None else 0,
        }
        
        return summary
    
    def save_data(self, output_dir: str = "data"):
        """
        Save all collected data to CSV files.
        
        Args:
            output_dir: Directory to save the data files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if self.market_data is not None:
            self.market_data.to_csv(f"{output_dir}/market_data.csv")
            print(f"✓ Saved market data to {output_dir}/market_data.csv")
        
        if self.macro_data is not None:
            self.macro_data.to_csv(f"{output_dir}/macro_data.csv")
            print(f"✓ Saved macro data to {output_dir}/macro_data.csv")
        
        if self.factor_data is not None:
            self.factor_data.to_csv(f"{output_dir}/factor_data.csv")
            print(f"✓ Saved factor data to {output_dir}/factor_data.csv")
        
        if self.unified_data is not None:
            self.unified_data.to_csv(f"{output_dir}/unified_data.csv")
            print(f"✓ Saved unified data to {output_dir}/unified_data.csv")


def main():
    """
    Main function to demonstrate data collection.
    """
    print("=" * 60)
    print("Cross-Asset Regime Detection - Data Collection")
    print("=" * 60)
    print()
    
    # Initialize data collector
    collector = DataCollector(start_date="2015-01-01")
    
    # Download all data
    collector.download_market_data()
    collector.download_fred_data()
    collector.download_ken_french_data()
    
    # Create unified dataset
    unified = collector.create_unified_dataset()
    
    # Print summary
    print()
    print("=" * 60)
    print("Data Summary")
    print("=" * 60)
    summary = collector.get_data_summary()
    
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Observations: {summary['date_range']['n_observations']}")
    print(f"Market Assets: {summary['market_assets']['count']}")
    print(f"Macro Series: {summary['macro_series']['count']}")
    print(f"Factors: {summary['factors']['count']}")
    print(f"Total Variables: {summary['total_variables']}")
    
    print()
    print("Market Assets:", summary['market_assets']['assets'])
    print("Macro Series:", summary['macro_series']['series'])
    if summary['factors']['factors']:
        print("Factors:", summary['factors']['factors'])
    
    # Save data
    print()
    collector.save_data()
    
    print()
    print("=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    
    return collector


if __name__ == "__main__":
    collector = main()