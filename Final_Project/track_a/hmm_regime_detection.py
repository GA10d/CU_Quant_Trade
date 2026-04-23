"""
HMM Regime Detection Module
Implements Gaussian Hidden Markov Model for market regime detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional, Tuple
import warnings
import joblib

warnings.filterwarnings('ignore')


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.
    """
    
    def __init__(self, 
                 n_states: int = 3,
                 covariance_type: str = "diag",
                 n_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize the HMM regime detector.
        
        Args:
            n_states: Number of hidden states (regimes)
            covariance_type: Type of covariance matrix ('full', 'diag', 'spherical', 'tied')
            n_iter: Maximum number of EM iterations
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.model = None
        self.states = None
        self.state_probabilities = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.regime_names = None
    
    def fit(self, X: pd.DataFrame, n_restarts: int = 10) -> GaussianHMM:
        """
        Fit the HMM model with multiple restarts to avoid local optima.
        
        Args:
            X: Feature DataFrame
            n_restarts: Number of random restarts
            
        Returns:
            Best fitted HMM model
        """
        print(f"Fitting HMM with {self.n_states} states ({n_restarts} restarts)...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Multiple restarts to find best model
        best_model = None
        best_score = -np.inf
        
        for i in range(n_restarts):
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state + i,
                verbose=False
            )
            
            try:
                model.fit(X_scaled)
                score = model.score(X_scaled)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                print(f"  Restart {i+1} failed: {e}")
                continue
        
        if best_model is None:
            raise RuntimeError("All HMM fitting attempts failed")
        
        self.model = best_model
        
        # Predict states
        self.states = self.model.predict(X_scaled)
        self.state_probabilities = self.model.predict_proba(X_scaled)
        
        print(f"✓ HMM fitted successfully (log-likelihood: {best_score:.2f})")
        
        return best_model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict states for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predicted states
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        states = self.model.predict(X_scaled)
        
        return states
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """
        Get the transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        transmat = pd.DataFrame(
            self.model.transmat_,
            index=[f"State_{i}" for i in range(self.n_states)],
            columns=[f"State_{i}" for i in range(self.n_states)]
        )
        
        return transmat
    
    def get_state_durations(self) -> pd.DataFrame:
        """
        Calculate expected duration of each state.
        
        Returns:
            DataFrame with state durations
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Expected duration = 1 / (1 - diagonal element)
        durations = 1 / (1 - np.diag(self.model.transmat_))
        
        df = pd.DataFrame({
            'state': [f"State_{i}" for i in range(self.n_states)],
            'expected_duration_days': durations
        })
        
        return df
    
    def get_state_statistics(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each state.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            DataFrame with state statistics
        """
        if self.states is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_with_states = X.copy()
        X_with_states['state'] = self.states
        
        stats = X_with_states.groupby('state').agg(['mean', 'std'])
        
        return stats
    
    def label_regimes(self, X: pd.DataFrame) -> Dict[int, str]:
        """
        Label regimes based on economic interpretation.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary mapping state numbers to regime names
        """
        if self.states is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate state means
        X_with_states = X.copy()
        X_with_states['state'] = self.states
        
        state_means = X_with_states.groupby('state').mean()
        
        # Calculate risk score for each state
        # Higher risk score = more favorable market conditions
        risk_scores = {}
        
        for state in range(self.n_states):
            score = 0
            
            # Positive contributions
            if 'SPY_ret' in state_means.columns:
                score += state_means.loc[state, 'SPY_ret'] * 100  # Scale up
            
            if 'VIX_level' in state_means.columns:
                score -= state_means.loc[state, 'VIX_level'] / 10  # Lower VIX is better
            
            if 'SPY_ret_vol_21d' in state_means.columns:
                score -= state_means.loc[state, 'SPY_ret_vol_21d'] * 10  # Lower vol is better
            
            if 'credit_spread_proxy' in state_means.columns:
                score -= state_means.loc[state, 'credit_spread_proxy'] * 10  # Lower spread is better
            
            if 'curve_slope_10y2y' in state_means.columns:
                score += state_means.loc[state, 'curve_slope_10y2y'] * 2  # Steeper curve is better
            
            risk_scores[state] = score
        
        # Order states by risk score
        ordered_states = sorted(risk_scores.items(), key=lambda x: x[1])
        
        # Assign regime names based on ordering
        if self.n_states == 2:
            regime_names = {
                ordered_states[0][0]: "Risk-Off",
                ordered_states[1][0]: "Risk-On"
            }
        elif self.n_states == 3:
            regime_names = {
                ordered_states[0][0]: "Stress",
                ordered_states[1][0]: "Transition",
                ordered_states[2][0]: "Risk-On"
            }
        elif self.n_states == 4:
            regime_names = {
                ordered_states[0][0]: "Stress",
                ordered_states[1][0]: "Transition",
                ordered_states[2][0]: "Recovery",
                ordered_states[3][0]: "Risk-On"
            }
        else:
            regime_names = {
                state: f"Regime_{i}" for i, (state, _) in enumerate(ordered_states)
            }
        
        self.regime_names = regime_names
        
        print("Regime labels:")
        for state, name in regime_names.items():
            print(f"  State {state}: {name} (risk score: {risk_scores[state]:.2f})")
        
        return regime_names
    
    def select_optimal_states(self, 
                             X: pd.DataFrame, 
                             state_candidates: List[int] = [2, 3, 4, 5],
                             n_restarts: int = 5) -> Tuple[int, pd.DataFrame]:
        """
        Select optimal number of states using BIC.
        
        Args:
            X: Feature DataFrame
            state_candidates: List of state numbers to try
            n_restarts: Number of restarts per state count
            
        Returns:
            Tuple of (optimal_n_states, BIC summary DataFrame)
        """
        print("Selecting optimal number of states using BIC...")
        
        results = []
        
        for n_states in state_candidates:
            print(f"  Testing {n_states} states...")
            
            # Fit model
            detector = HMMRegimeDetector(
                n_states=n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
            
            try:
                detector.fit(X, n_restarts=n_restarts)
                
                # Calculate BIC
                log_likelihood = detector.model.score(detector.scaler.transform(X))
                n_params = detector._count_parameters()
                n_obs = len(X)
                bic = -2 * log_likelihood + n_params * np.log(n_obs)
                
                results.append({
                    'n_states': n_states,
                    'log_likelihood': log_likelihood,
                    'n_params': n_params,
                    'bic': bic
                })
                
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        summary = pd.DataFrame(results)
        
        # Select best by BIC
        best_n_states = int(summary.loc[summary['bic'].idxmin(), 'n_states'])
        
        print(f"✓ Optimal number of states: {best_n_states}")
        print()
        print("BIC Summary:")
        print(summary.round(2))
        
        return best_n_states, summary
    
    def _count_parameters(self) -> int:
        """
        Count the number of parameters in the HMM model.
        
        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0
        
        n_states = self.n_states
        n_features = len(self.feature_names)
        
        # Initial state probabilities (n_states - 1, as they sum to 1)
        n_startprob = n_states - 1
        
        # Transition matrix (n_states * (n_states - 1))
        n_transmat = n_states * (n_states - 1)
        
        # Emission parameters
        if self.covariance_type == 'full':
            n_means = n_states * n_features
            n_covars = n_states * n_features * (n_features + 1) // 2
        elif self.covariance_type == 'diag':
            n_means = n_states * n_features
            n_covars = n_states * n_features
        elif self.covariance_type == 'spherical':
            n_means = n_states * n_features
            n_covars = n_states
        elif self.covariance_type == 'tied':
            n_means = n_states * n_features
            n_covars = n_features * (n_features + 1) // 2
        else:
            raise ValueError(f"Unknown covariance type: {self.covariance_type}")
        
        total = n_startprob + n_transmat + n_means + n_covars
        
        return total
    
    def plot_regimes(self, X: pd.DataFrame, price_data: pd.DataFrame = None):
        """
        Plot regime detection results.
        
        Args:
            X: Feature DataFrame
            price_data: Optional price data for visualization
        """
        if self.states is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: State probabilities over time
        ax = axes[0]
        for state in range(self.n_states):
            regime_name = self.regime_names.get(state, f"State {state}")
            ax.plot(X.index, self.state_probabilities[:, state], 
                   label=regime_name, alpha=0.7, linewidth=1.5)
        
        ax.set_title('State Probabilities Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Most likely state over time
        ax = axes[1]
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_states))
        
        for state in range(self.n_states):
            mask = self.states == state
            regime_name = self.regime_names.get(state, f"State {state}")
            ax.scatter(X.index[mask], self.states[mask], 
                      c=[colors[state]], label=regime_name, s=10, alpha=0.6)
        
        ax.set_title('Most Likely State Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('State')
        ax.set_yticks(range(self.n_states))
        ax.set_yticklabels([self.regime_names.get(i, f"State {i}") for i in range(self.n_states)])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Price data with regime shading
        ax = axes[2]
        if price_data is not None and 'SPY' in price_data.columns:
            spy = price_data['SPY']
            ax.plot(spy.index, spy.values, color='black', linewidth=1.5, label='SPY')
            
            # Shade regimes
            for state in range(self.n_states):
                mask = self.states == state
                if mask.any():
                    regime_name = self.regime_names.get(state, f"State {state}")
                    ax.fill_between(X.index, spy.min(), spy.max(), 
                                   where=mask, alpha=0.2, 
                                   color=colors[state], label=regime_name)
            
            ax.set_title('SPY Price with Regime Shading', fontsize=12, fontweight='bold')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No price data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Price Data Not Available')
        
        plt.tight_layout()
        plt.savefig('outputs/hmm_regime_detection.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot to outputs/hmm_regime_detection.png")
        plt.show()
    
    def save_model(self, filepath: str = "outputs/hmm_model.joblib"):
        """
        Save the fitted model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'n_states': self.n_states,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'regime_names': self.regime_names
        }
        
        joblib.dump(model_data, filepath)
        print(f"✓ Saved model to {filepath}")
    
    def load_model(self, filepath: str = "outputs/hmm_model.joblib"):
        """
        Load a fitted model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.n_states = model_data['n_states']
        self.covariance_type = model_data['covariance_type']
        self.random_state = model_data['random_state']
        self.feature_names = model_data['feature_names']
        self.regime_names = model_data['regime_names']
        
        print(f"✓ Loaded model from {filepath}")


def main():
    """
    Main function to demonstrate HMM regime detection.
    """
    print("=" * 60)
    print("Cross-Asset Regime Detection - HMM Regime Detection")
    print("=" * 60)
    print()
    
    # Load features
    print("Loading features...")
    features = pd.read_csv("../data/hmm_features.csv", index_col=0, parse_dates=True)
    market_data = pd.read_csv("../data/market_data_full_adjusted.csv", index_col=0, parse_dates=True)
    
    print(f"✓ Loaded features: {len(features)} observations, {len(features.columns)} features")
    print()
    
    # Select optimal number of states
    detector = HMMRegimeDetector(random_state=42)
    optimal_n_states, bic_summary = detector.select_optimal_states(
        features, 
        state_candidates=[2, 3, 4],
        n_restarts=5
    )
    
    print()
    
    # Fit final model with optimal states
    print(f"Fitting final HMM with {optimal_n_states} states...")
    final_detector = HMMRegimeDetector(
        n_states=optimal_n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42
    )
    
    final_detector.fit(features, n_restarts=10)
    
    # Label regimes
    print()
    final_detector.label_regimes(features)
    
    # Get transition matrix
    print()
    print("=" * 60)
    print("Transition Matrix")
    print("=" * 60)
    transmat = final_detector.get_transition_matrix()
    print(transmat.round(3))
    
    # Get state durations
    print()
    print("=" * 60)
    print("Expected State Durations")
    print("=" * 60)
    durations = final_detector.get_state_durations()
    print(durations.round(1))
    
    # Get state statistics
    print()
    print("=" * 60)
    print("State Statistics")
    print("=" * 60)
    stats = final_detector.get_state_statistics(features)
    print(stats.round(4))
    
    # Plot results
    print()
    final_detector.plot_regimes(features, market_data)
    
    # Save model
    print()
    final_detector.save_model()
    
    # Save regime labels
    regime_series = pd.Series(
        [final_detector.regime_names[s] for s in final_detector.states],
        index=features.index,
        name='regime'
    )
    regime_series.to_csv("../data/hmm_regimes.csv")
    print("✓ Saved regime labels to ../data/hmm_regimes.csv")
    
    print()
    print("=" * 60)
    print("HMM regime detection complete!")
    print("=" * 60)
    
    return final_detector


if __name__ == "__main__":
    detector = main()