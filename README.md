# Cross-Asset Regime Detection & Portfolio Optimization

## Project Overview

This project develops a cross-asset regime detection framework using macroeconomic and financial data, and evaluates whether regime-aware portfolio strategies can outperform static allocation approaches.

### Research Questions

1. Can we reliably identify latent market regimes using macro-financial data?
2. Do these regimes correspond to economically meaningful states (e.g., risk-on, risk-off)?
3. Can regime-conditioned portfolio strategies achieve improved risk-adjusted returns compared to baseline strategies?

## Methodology

### Regime Detection

The primary approach uses a **Gaussian Hidden Markov Model (HMM)** to identify market regimes. The HMM provides a probabilistic framework for modeling time series with latent states and is well-suited for capturing regime-switching behavior in financial markets.

**Key Features:**
- Cross-asset features including returns, volatility, correlations, and macroeconomic indicators
- BIC-based model selection to determine optimal number of regimes
- Economic interpretation of regimes (Stress, Transition, Recovery, Risk-On)
- Transition probability analysis to understand regime dynamics

### Data Sources

1. **yfinance**: Market prices for equities, bonds, commodities, and volatility
   - SPY (S&P 500 ETF)
   - TLT (Long-term Treasury ETF)
   - GLD (Gold ETF)
   - UUP (US Dollar ETF)
   - HYG (High Yield Credit ETF)
   - LQD (Investment Grade Credit ETF)
   - VIX (Volatility Index)

2. **FRED**: Macroeconomic indicators
   - GDP, Unemployment Rate, Federal Funds Rate
   - Treasury yields (2Y, 10Y, 3M)
   - Yield curve spreads
   - CPI, Industrial Production, Employment

3. **Ken French Data Library**: Factor data
   - Fama-French 5-factor model
   - Momentum factor

### Portfolio Strategies

Four strategies are implemented and compared:

1. **60/40 Static Allocation**: Traditional 60% equities, 40% bonds
2. **Equal Weight**: Equal allocation across SPY, TLT, and GLD
3. **Regime-Conditioned**: Dynamic allocation based on detected regimes
   - Stress: 20% SPY, 50% TLT, 30% GLD (flight to safety)
   - Transition: 30% SPY, 40% TLT, 30% GLD (balanced)
   - Recovery: 50% SPY, 30% TLT, 20% GLD (risk-on)
   - Risk-On: 70% SPY, 20% TLT, 10% GLD (strong risk-on)
4. **Volatility Targeting**: SPY with volatility scaling to target 15% annual volatility

## Project Structure

```
.
├── data_collection.py          # Data collection from multiple sources
├── feature_engineering.py      # Feature engineering for regime detection
├── hmm_regime_detection.py     # HMM implementation and regime detection
├── portfolio_strategy.py       # Portfolio strategy implementation and backtesting
├── README.md                   # This file
├── data/                       # Data directory
│   ├── market_data.csv
│   ├── macro_data.csv
│   ├── factor_data.csv
│   ├── unified_data.csv
│   ├── hmm_features.csv
│   ├── hmm_features_normalized.csv
│   └── hmm_regimes.csv
└── outputs/                    # Output directory
```

## Installation

### Requirements

```bash
pip install numpy pandas scipy scikit-learn yfinance pandas-datareader hmmlearn matplotlib
```

### Usage

#### 1. Data Collection

```python
from data_collection import DataCollector

# Initialize collector
collector = DataCollector(start_date="2015-01-01")

# Download all data
collector.download_market_data()
collector.download_fred_data()
collector.download_ken_french_data()

# Create unified dataset
unified = collector.create_unified_dataset()

# Save data
collector.save_data()
```

#### 2. Feature Engineering

```python
from feature_engineering import FeatureEngineer
import pandas as pd

# Load data
market_data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
macro_data = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)

# Create feature engineer
engineer = FeatureEngineer(price_data=market_data, macro_data=macro_data)

# Build HMM features
features = engineer.build_hmm_features()

# Normalize features
normalized, scaler = engineer.normalize_features(method="standard")
```

#### 3. Regime Detection

```python
from hmm_regime_detection import HMMRegimeDetector
import pandas as pd

# Load features
features = pd.read_csv("data/hmm_features.csv", index_col=0, parse_dates=True)

# Select optimal number of states
detector = HMMRegimeDetector(random_state=42)
optimal_n_states, bic_summary = detector.select_optimal_states(
    features, 
    state_candidates=[2, 3, 4],
    n_restarts=5
)

# Fit final model
final_detector = HMMRegimeDetector(
    n_states=optimal_n_states,
    covariance_type="diag",
    n_iter=1000,
    random_state=42
)
final_detector.fit(features, n_restarts=10)

# Label regimes
final_detector.label_regimes(features)

# Plot results
market_data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
final_detector.plot_regimes(features, market_data)

# Save model
final_detector.save_model()
```

#### 4. Portfolio Strategy

```python
from portfolio_strategy import (
    StaticAllocation, 
    RegimeConditionedStrategy,
    compare_strategies,
    plot_strategy_comparison
)
import pandas as pd

# Load data
market_data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
regimes = pd.read_csv("data/hmm_regimes.csv", index_col=0, parse_dates=True)

# Calculate returns
returns = market_data.pct_change().dropna()

# Define regime-conditioned strategy
regime_weights = {
    'Stress': {'SPY': 0.2, 'TLT': 0.5, 'GLD': 0.3},
    'Transition': {'SPY': 0.3, 'TLT': 0.4, 'GLD': 0.3},
    'Recovery': {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2},
    'Risk-On': {'SPY': 0.7, 'TLT': 0.2, 'GLD': 0.1}
}

regime_strategy = RegimeConditionedStrategy(
    assets=['SPY', 'TLT', 'GLD'],
    regime_weights=regime_weights
)
regime_strategy.set_regimes(regimes['regime'])

# Backtest and compare
results = {
    '60/40': returns_60_40,
    'Equal Weight': returns_equal_weight,
    'Regime-Conditioned': returns_regime,
    'Vol Targeting': returns_vol_target
}

comparison = compare_strategies(results)
print(comparison)

plot_strategy_comparison(results)
```

## Key Results

### Regime Detection

The HMM identified **4 distinct market regimes**:

1. **Stress** (Risk Score: -3.13)
   - High volatility, low returns
   - Flight to safety behavior
   - Average duration: 63 days

2. **Transition** (Risk Score: 0.12)
   - Moderate volatility
   - Mixed market signals
   - Average duration: 1,011 days (longest)

3. **Recovery** (Risk Score: 3.21)
   - Improving conditions
   - Moderate risk-on behavior
   - Average duration: 78 days

4. **Risk-On** (Risk Score: 3.90)
   - Low volatility, strong returns
   - Bull market conditions
   - Average duration: 67 days

### Portfolio Performance

| Strategy | Annual Return | Volatility | Sharpe Ratio | Max Drawdown |
|----------|--------------|------------|--------------|--------------|
| Regime-Conditioned | 10.52% | 9.75% | **0.87** | -25.9% |
| Vol Targeting | 11.70% | 15.97% | 0.61 | -28.5% |
| Equal Weight | 9.19% | 9.71% | 0.75 | -22.6% |
| 60/40 | 8.15% | 11.27% | 0.53 | -27.2% |

**Key Findings:**
- The **Regime-Conditioned strategy** achieved the highest Sharpe ratio (0.87)
- It provided better risk-adjusted returns than all static allocation strategies
- Volatility targeting had the highest returns but with significantly higher volatility
- All strategies outperformed the traditional 60/40 portfolio

## Extensions

### Transformer-Based Representation Learning (Optional)

As an extension, a transformer encoder can be used to learn temporal representations of market data:

1. Rolling windows of market data (60-day sequences) are mapped into low-dimensional embeddings
2. These embeddings capture temporal patterns in a data-driven manner
3. Clustering on embeddings can identify regimes
4. Results can be compared with HMM-derived regimes for consistency

The existing `track_b_pipeline.py` in the project provides an implementation of this approach.

## References

1. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357-384.

2. Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

3. Ang, A., & Bekaert, G. (2002). International asset allocation with regime shifts. *Review of Financial Studies*, 15(4), 1137-1187.

## License

This project is for educational and research purposes.

## Contact

For questions or collaboration, please contact the project team.
