"""
Portfolio Strategy Module
Implements regime-conditioned portfolio strategies and baseline static allocations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001  # 10 bps
    rebalance_frequency: str = 'M'  # Monthly rebalancing
    risk_free_rate: float = 0.02  # Annual risk-free rate
    lookback_window: int = 252  # For volatility targeting


class PortfolioStrategy:
    """
    Base class for portfolio strategies.
    """
    
    def __init__(self, 
                 assets: List[str],
                 config: BacktestConfig = None):
        """
        Initialize portfolio strategy.
        
        Args:
            assets: List of asset tickers
            config: Backtest configuration
        """
        self.assets = assets
        self.config = config or BacktestConfig()
        self.weights = None
        self.portfolio_returns = None
        self.portfolio_values = None
        self.turnover = None
    
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute portfolio weights. To be overridden by subclasses.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame of portfolio weights
        """
        raise NotImplementedError
    
    def backtest(self, 
                 returns: pd.DataFrame,
                 weights_func: Optional[Callable] = None,
                 **kwargs) -> pd.DataFrame:
        """
        Backtest the portfolio strategy.
        
        Args:
            returns: DataFrame of asset returns
            weights_func: Optional custom weights function
            
        Returns:
            DataFrame with backtest results
        """
        print(f"Running backtest from {returns.index[0].date()} to {returns.index[-1].date()}...")
        
        # Compute weights
        if weights_func is not None:
            weights = weights_func(returns, **kwargs)
        else:
            weights = self.compute_weights(returns, **kwargs)
        
        # Align weights and returns
        common_dates = weights.index.intersection(returns.index)
        weights = weights.loc[common_dates]
        returns = returns.loc[common_dates]
        
        # Calculate portfolio returns
        portfolio_returns = (weights * returns).sum(axis=1)
        
        # Adjust for transaction costs
        if self.config.transaction_cost > 0:
            # Calculate turnover
            weight_changes = weights.diff().abs().sum(axis=1)
            transaction_costs = weight_changes * self.config.transaction_cost
            portfolio_returns = portfolio_returns - transaction_costs
        
        # Calculate portfolio values
        portfolio_values = (1 + portfolio_returns).cumprod() * self.config.initial_capital
        
        self.weights = weights
        self.portfolio_returns = portfolio_returns
        self.portfolio_values = portfolio_values
        self.turnover = weights.diff().abs().sum(axis=1)
        
        print(f"✓ Backtest complete: {len(returns)} periods")
        
        return pd.DataFrame({
            'returns': portfolio_returns,
            'values': portfolio_values,
            'turnover': self.turnover
        })


class StaticAllocation(PortfolioStrategy):
    """
    Static allocation strategy with fixed weights.
    """
    
    def __init__(self, 
                 assets: List[str],
                 fixed_weights: Dict[str, float],
                 config: BacktestConfig = None):
        """
        Initialize static allocation.
        
        Args:
            assets: List of asset tickers
            fixed_weights: Dictionary of fixed weights
            config: Backtest configuration
        """
        super().__init__(assets, config)
        self.fixed_weights = fixed_weights
    
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute fixed weights for all periods.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame with fixed weights
        """
        # Normalize weights to sum to 1
        total = sum(self.fixed_weights.values())
        normalized_weights = {k: v/total for k, v in self.fixed_weights.items()}
        
        # Create weights DataFrame
        weights = pd.DataFrame(
            normalized_weights,
            index=returns.index
        )
        
        # Rebalance at specified frequency
        if self.config.rebalance_frequency == 'M':
            # Keep weights only at first trading day of each month
            weights['month'] = weights.index.to_period('M')
            weights = weights.groupby('month').first()
            weights = weights.reindex(returns.index).ffill()
        
        return weights


class RegimeConditionedStrategy(PortfolioStrategy):
    """
    Regime-conditioned portfolio strategy.
    Adjusts allocation based on detected market regime.
    """
    
    def __init__(self,
                 assets: List[str],
                 regime_weights: Dict[str, Dict[str, float]],
                 config: BacktestConfig = None):
        """
        Initialize regime-conditioned strategy.
        
        Args:
            assets: List of asset tickers
            regime_weights: Dictionary mapping regimes to weight dictionaries
            config: Backtest configuration
        """
        super().__init__(assets, config)
        self.regime_weights = regime_weights
        self.regime_series = None
    
    def set_regimes(self, regime_series: pd.Series):
        """
        Set the regime labels for each date.
        
        Args:
            regime_series: Series with regime labels indexed by date
        """
        self.regime_series = regime_series
    
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute regime-conditioned weights.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame with regime-conditioned weights
        """
        if self.regime_series is None:
            raise ValueError("Regimes not set. Call set_regimes() first.")
        
        # Align regime series with returns
        common_dates = returns.index.intersection(self.regime_series.index)
        regimes = self.regime_series.loc[common_dates]
        
        # Initialize weights DataFrame
        weights = pd.DataFrame(index=returns.index, columns=self.assets)
        
        # Assign weights based on regime
        for date in returns.index:
            if date in regimes.index:
                regime = regimes[date]
                if regime in self.regime_weights:
                    w = self.regime_weights[regime]
                    for asset in self.assets:
                        weights.loc[date, asset] = w.get(asset, 0.0)
        
        # Fill any missing weights (forward fill)
        weights = weights.ffill().fillna(0)
        
        # Normalize weights
        weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
        
        # Rebalance at specified frequency
        if self.config.rebalance_frequency == 'M':
            # Keep weights only at first trading day of each month
            weights_copy = weights.copy()
            weights_copy['month'] = weights_copy.index.to_period('M')
            monthly_weights = weights_copy.groupby('month').first()
            weights = monthly_weights.reindex(returns.index).ffill()
        
        return weights


class VolatilityTargetingStrategy(PortfolioStrategy):
    """
    Volatility targeting strategy.
    Scales exposure to target a specific volatility level.
    """
    
    def __init__(self,
                 assets: List[str],
                 target_vol: float = 0.15,
                 base_weights: Dict[str, float] = None,
                 config: BacktestConfig = None):
        """
        Initialize volatility targeting strategy.
        
        Args:
            assets: List of asset tickers
            target_vol: Target annual volatility
            base_weights: Base portfolio weights
            config: Backtest configuration
        """
        super().__init__(assets, config)
        self.target_vol = target_vol
        self.base_weights = base_weights or {a: 1/len(assets) for a in assets}
    
    def compute_weights(self, returns: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute volatility-scaled weights.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            DataFrame with scaled weights
        """
        # Calculate rolling volatility
        lookback = self.config.lookback_window
        portfolio_vol = returns.rolling(lookback).std().mean(axis=1) * np.sqrt(252)
        
        # Calculate scaling factor
        scaling_factor = self.target_vol / portfolio_vol
        scaling_factor = scaling_factor.clip(0.1, 2.0)  # Limit leverage
        
        # Apply scaling to base weights
        weights = pd.DataFrame(
            {a: self.base_weights.get(a, 0) for a in self.assets},
            index=returns.index
        )
        
        weights = weights.mul(scaling_factor, axis=0)
        
        # Cap total exposure
        total_exposure = weights.sum(axis=1)
        weights = weights.div(total_exposure.clip(0.1, 2.0), axis=0)
        
        return weights


def calculate_performance_metrics(returns: pd.Series, 
                                  risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate performance metrics for a return series.
    
    Args:
        returns: Series of portfolio returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary with performance metrics
    """
    # Annualization factor
    n_periods = len(returns)
    years = n_periods / 252
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annualized return
    annual_return = (1 + total_return) ** (1/years) - 1
    
    # Volatility
    annual_vol = returns.std() * np.sqrt(252)
    
    # Sharpe ratio
    excess_return = annual_return - risk_free_rate
    sharpe_ratio = excess_return / annual_vol if annual_vol > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate
    }


def compare_strategies(results: Dict[str, pd.Series], 
                       risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Compare multiple strategy performances.
    
    Args:
        results: Dictionary mapping strategy names to return series
        risk_free_rate: Annual risk-free rate
        
    Returns:
        DataFrame with comparative metrics
    """
    comparison = []
    
    for name, returns in results.items():
        metrics = calculate_performance_metrics(returns, risk_free_rate)
        metrics['strategy'] = name
        comparison.append(metrics)
    
    df = pd.DataFrame(comparison)
    df = df.set_index('strategy')
    
    # Reorder columns
    columns = ['annual_return', 'annual_volatility', 'sharpe_ratio', 
               'sortino_ratio', 'max_drawdown', 'calmar_ratio', 'win_rate']
    df = df[columns]
    
    return df.sort_values('sharpe_ratio', ascending=False)


def plot_strategy_comparison(results: Dict[str, pd.Series], 
                            save_path: str = 'outputs/strategy_comparison.png'):
    """
    Plot comparison of multiple strategies.
    
    Args:
        results: Dictionary mapping strategy names to return series
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cumulative returns
    ax = axes[0, 0]
    for name, returns in results.items():
        cum_return = (1 + returns).cumprod()
        ax.plot(cum_return.index, cum_return.values, label=name, linewidth=1.5)
    
    ax.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Growth of $1')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Sharpe ratio
    ax = axes[0, 1]
    for name, returns in results.items():
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Drawdowns
    ax = axes[1, 0]
    for name, returns in results.items():
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        # Convert to numpy arrays to avoid type issues
        ax.fill_between(drawdown.index, drawdown.values.astype(float), 0, alpha=0.3, label=name)
    
    ax.set_title('Drawdowns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Drawdown')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics table
    ax = axes[1, 1]
    comparison = compare_strategies(results)
    
    # Create table
    table_data = []
    for idx, row in comparison.iterrows():
        table_data.append([
            f"{row['annual_return']:.2%}",
            f"{row['annual_volatility']:.2%}",
            f"{row['sharpe_ratio']:.2f}",
            f"{row['max_drawdown']:.2%}"
        ])
    
    table = ax.table(
        cellText=table_data,
        rowLabels=comparison.index,
        colLabels=['Return', 'Vol', 'Sharpe', 'MaxDD'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    ax.axis('off')
    ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {save_path}")
    plt.show()


def main():
    """
    Main function to demonstrate portfolio strategies.
    """
    print("=" * 60)
    print("Cross-Asset Regime Detection - Portfolio Strategy")
    print("=" * 60)
    print()
    
    # Load data
    print("Loading data...")
    market_data = pd.read_csv("data/market_data.csv", index_col=0, parse_dates=True)
    regimes = pd.read_csv("data/hmm_regimes.csv", index_col=0, parse_dates=True)
    
    # Calculate returns
    returns = market_data.pct_change().dropna()
    
    print(f"✓ Loaded data: {len(returns)} observations")
    print()
    
    # Define assets for portfolio
    assets = ['SPY', 'TLT', 'GLD']
    
    # Filter returns to common assets and align with regimes
    common_assets = [a for a in assets if a in returns.columns]
    returns_filtered = returns[common_assets]
    common_dates = returns_filtered.index.intersection(regimes.index)
    returns_filtered = returns_filtered.loc[common_dates]
    regimes_aligned = regimes.loc[common_dates]
    
    print("Defining strategies...")
    
    # Define strategy weights
    config = BacktestConfig()
    
    # 1. Static 60/40 Portfolio
    static_60_40_weights = pd.DataFrame({
        'SPY': 0.6, 'TLT': 0.4
    }, index=returns_filtered.index)
    
    # 2. Equal Weight Portfolio
    equal_weight_weights = pd.DataFrame({
        'SPY': 0.33, 'TLT': 0.33, 'GLD': 0.34
    }, index=returns_filtered.index)
    
    # 3. Regime-Conditioned Strategy
    regime_weights_map = {
        'Stress': {'SPY': 0.2, 'TLT': 0.5, 'GLD': 0.3},
        'Transition': {'SPY': 0.3, 'TLT': 0.4, 'GLD': 0.3},
        'Recovery': {'SPY': 0.5, 'TLT': 0.3, 'GLD': 0.2},
        'Risk-On': {'SPY': 0.7, 'TLT': 0.2, 'GLD': 0.1}
    }
    
    regime_weights = pd.DataFrame(index=returns_filtered.index, columns=common_assets)
    for date in returns_filtered.index:
        if date in regimes_aligned.index:
            regime = regimes_aligned.loc[date, 'regime']
            if regime in regime_weights_map:
                w = regime_weights_map[regime]
                for asset in common_assets:
                    regime_weights.loc[date, asset] = w.get(asset, 0.0)
    
    regime_weights = regime_weights.ffill().fillna(0)
    regime_weights = regime_weights.div(regime_weights.sum(axis=1), axis=0).fillna(0)
    
    # 4. Volatility Targeting (SPY only)
    spy_vol = returns_filtered['SPY'].rolling(252).std() * np.sqrt(252)
    vol_scaling = 0.15 / spy_vol
    vol_scaling = vol_scaling.clip(0.1, 2.0)
    vol_target_weights = pd.DataFrame({
        'SPY': vol_scaling
    }, index=returns_filtered.index).fillna(1.0)
    
    print("✓ Defined 4 strategies")
    print()
    
    # Calculate portfolio returns
    print("Calculating portfolio returns...")
    
    results = {}
    
    # 60/40
    results['60/40'] = (static_60_40_weights[['SPY', 'TLT']].reindex(returns_filtered.index) * 
                        returns_filtered[['SPY', 'TLT']]).sum(axis=1)
    
    # Equal weight
    results['Equal Weight'] = (equal_weight_weights[common_assets] * returns_filtered).sum(axis=1)
    
    # Regime-conditioned
    results['Regime-Conditioned'] = (regime_weights[common_assets] * returns_filtered).sum(axis=1)
    
    # Vol targeting (SPY only)
    results['Vol Targeting'] = (vol_target_weights['SPY'] * returns_filtered['SPY'])
    
    print(f"✓ Calculated {len(results)} strategy returns")
    print()
    
    # Compare strategies
    print("=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    comparison = compare_strategies(results)
    print(comparison.round(4))
    
    # Plot comparison
    print()
    plot_strategy_comparison(results)
    
    # Save results
    comparison.to_csv("outputs/strategy_comparison.csv")
    print("✓ Saved comparison to outputs/strategy_comparison.csv")
    
    print()
    print("=" * 60)
    print("Portfolio strategy analysis complete!")
    print("=" * 60)
    
    return results, comparison


if __name__ == "__main__":
    results, comparison = main()