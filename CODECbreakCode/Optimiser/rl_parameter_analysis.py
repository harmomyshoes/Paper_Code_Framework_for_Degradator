import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd


class RLParameterAnalyzer:
    """
    Analyze which action parameters contribute most to high rewards in RL.
    Designed for continuous action spaces with multiple parameters.
    """
    
    def __init__(self, n_params: int = 28):
        """
        Initialize the analyzer.
        
        Args:
            n_params: Number of action parameters
        """
        self.n_params = n_params
        self.actions = None
        self.rewards = None
        self.states = None
        
    def load_data(self, actions: np.ndarray, rewards: np.ndarray, 
                  states: Optional[np.ndarray] = None):
        """
        Load trajectory data for analysis.
        
        Args:
            actions: Array of shape (n_samples, n_params)
            rewards: Array of shape (n_samples,)
            states: Optional array of shape (n_samples, state_dim)
        """
        assert actions.shape[1] == self.n_params, \
            f"Expected {self.n_params} parameters, got {actions.shape[1]}"
        assert len(actions) == len(rewards), \
            "Actions and rewards must have same length"
        
        self.actions = actions
        self.rewards = rewards
        self.states = states
        print(f"Loaded {len(actions)} samples with {self.n_params} action parameters")
        
    def analyze_high_reward_distribution(self, percentile: float = 90) -> pd.DataFrame:
        """
        Analyze the distribution of action parameters for high-reward episodes.
        
        Args:
            percentile: Percentile threshold for "high reward" (default: top 10%)
            
        Returns:
            DataFrame with mean, std, min, max for each parameter in high-reward cases
        """
        threshold = np.percentile(self.rewards, percentile)
        high_reward_mask = self.rewards >= threshold
        high_reward_actions = self.actions[high_reward_mask]
        
        print(f"\nHigh Reward Analysis (top {100-percentile}%)")
        print(f"Reward threshold: {threshold:.4f}")
        print(f"Number of high-reward samples: {high_reward_mask.sum()}")
        
        results = []
        for i in range(self.n_params):
            param_values = high_reward_actions[:, i]
            results.append({
                'parameter': i,
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'min': np.min(param_values),
                'max': np.max(param_values),
                'median': np.median(param_values)
            })
        
        df = pd.DataFrame(results)
        return df
    
    def parameter_sensitivity(self, policy_fn=None, state=None, 
                            n_samples: int = 1000, 
                            perturbation_scale: float = 0.1) -> pd.DataFrame:
        """
        Compute sensitivity of rewards to each parameter via perturbation analysis.
        
        Args:
            policy_fn: Function that takes state and returns action
            state: State to evaluate at (if None, uses mean from loaded data)
            n_samples: Number of perturbation samples per parameter
            perturbation_scale: Scale of perturbations
            
        Returns:
            DataFrame with sensitivity scores for each parameter
        """
        if policy_fn is None:
            # If no policy provided, use statistical sensitivity from data
            return self._statistical_sensitivity()
        
        if state is None and self.states is not None:
            state = np.mean(self.states, axis=0)
        
        # Get baseline action
        baseline_action = policy_fn(state)
        
        sensitivities = []
        for i in range(self.n_params):
            param_rewards = []
            for _ in range(n_samples):
                # Perturb parameter i
                perturbed_action = baseline_action.copy()
                perturbation = np.random.randn() * perturbation_scale
                perturbed_action[i] += perturbation
                
                # Evaluate (would need actual environment)
                # For now, this is a placeholder
                param_rewards.append(perturbation)  # Replace with actual reward
            
            sensitivity = np.std(param_rewards)
            sensitivities.append({
                'parameter': i,
                'sensitivity': sensitivity
            })
        
        return pd.DataFrame(sensitivities)
    
    def _statistical_sensitivity(self) -> pd.DataFrame:
        """
        Compute statistical correlation between parameters and rewards.
        """
        correlations = []
        for i in range(self.n_params):
            corr = np.corrcoef(self.actions[:, i], self.rewards)[0, 1]
            correlations.append({
                'parameter': i,
                'correlation': corr,
                'abs_correlation': abs(corr)
            })
        
        df = pd.DataFrame(correlations)
        df = df.sort_values('abs_correlation', ascending=False)
        return df
    
    def compare_distributions(self, percentile_low: float = 10, 
                             percentile_high: float = 90) -> pd.DataFrame:
        """
        Compare parameter distributions between low and high reward cases.
        
        Args:
            percentile_low: Lower percentile for "low reward"
            percentile_high: Upper percentile for "high reward"
            
        Returns:
            DataFrame comparing distributions with statistical tests
        """
        low_threshold = np.percentile(self.rewards, percentile_low)
        high_threshold = np.percentile(self.rewards, percentile_high)
        
        low_mask = self.rewards <= low_threshold
        high_mask = self.rewards >= high_threshold
        
        low_actions = self.actions[low_mask]
        high_actions = self.actions[high_mask]
        
        results = []
        for i in range(self.n_params):
            low_param = low_actions[:, i]
            high_param = high_actions[:, i]
            
            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(high_param, low_param)
            
            # KS test for distribution difference
            ks_stat, ks_pvalue = stats.ks_2samp(high_param, low_param)
            
            results.append({
                'parameter': i,
                'low_mean': np.mean(low_param),
                'low_std': np.std(low_param),
                'high_mean': np.mean(high_param),
                'high_std': np.std(high_param),
                'mean_difference': np.mean(high_param) - np.mean(low_param),
                't_statistic': t_stat,
                'p_value': p_value,
                'ks_statistic': ks_stat,
                'significant': p_value < 0.05
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('ks_statistic', ascending=False)
        return df
    
    def parameter_importance_ranking(self) -> pd.DataFrame:
        """
        Rank parameters by multiple importance metrics.
        
        Returns:
            DataFrame with combined importance ranking
        """
        # Get correlation-based importance
        corr_df = self._statistical_sensitivity()
        
        # Get distribution comparison
        comp_df = self.compare_distributions()
        
        # Combine metrics
        importance = []
        for i in range(self.n_params):
            corr_score = abs(corr_df[corr_df['parameter'] == i]['correlation'].values[0])
            ks_score = comp_df[comp_df['parameter'] == i]['ks_statistic'].values[0]
            p_value = comp_df[comp_df['parameter'] == i]['p_value'].values[0]
            
            # Combined importance score (weighted average)
            combined_score = 0.5 * corr_score + 0.5 * ks_score
            
            importance.append({
                'parameter': i,
                'correlation_score': corr_score,
                'distribution_difference': ks_score,
                'statistical_significance': p_value,
                'combined_importance': combined_score
            })
        
        df = pd.DataFrame(importance)
        df = df.sort_values('combined_importance', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        return df
    
    def plot_parameter_distributions(self, top_k: int = 10, 
                                     percentile: float = 90,
                                     save_path: Optional[str] = None):
        """
        Plot distributions of top-k most important parameters.
        
        Args:
            top_k: Number of top parameters to plot
            percentile: Percentile for high reward threshold
            save_path: Optional path to save figure
        """
        importance = self.parameter_importance_ranking()
        top_params = importance.head(top_k)['parameter'].values
        
        threshold = np.percentile(self.rewards, percentile)
        high_mask = self.rewards >= threshold
        low_mask = self.rewards < threshold
        
        n_cols = 3
        n_rows = (top_k + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if top_k > 1 else [axes]
        
        for idx, param_idx in enumerate(top_params):
            ax = axes[idx]
            
            high_values = self.actions[high_mask, param_idx]
            low_values = self.actions[low_mask, param_idx]
            
            ax.hist(low_values, bins=30, alpha=0.5, label='Low Reward', 
                   density=True, color='red')
            ax.hist(high_values, bins=30, alpha=0.5, label='High Reward', 
                   density=True, color='green')
            
            ax.axvline(np.mean(high_values), color='green', 
                      linestyle='--', linewidth=2, label='High Mean')
            ax.axvline(np.mean(low_values), color='red', 
                      linestyle='--', linewidth=2, label='Low Mean')
            
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Density')
            ax.set_title(f'Parameter {param_idx} (Rank #{idx+1})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(top_k, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """
        Plot correlation between parameters and rewards.
        
        Args:
            save_path: Optional path to save figure
        """
        correlations = [np.corrcoef(self.actions[:, i], self.rewards)[0, 1] 
                       for i in range(self.n_params)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if c < 0 else 'green' for c in correlations]
        bars = ax.bar(range(self.n_params), correlations, color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Correlation with Reward')
        ax.set_title('Parameter-Reward Correlations')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if abs(corr) > 0.1:  # Only label significant correlations
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{corr:.2f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive text report of the analysis.
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 80)
        report.append("RL PARAMETER IMPORTANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"\nDataset: {len(self.actions)} samples, {self.n_params} parameters")
        report.append(f"Reward range: [{self.rewards.min():.4f}, {self.rewards.max():.4f}]")
        report.append(f"Mean reward: {self.rewards.mean():.4f} ± {self.rewards.std():.4f}")
        
        # Parameter importance ranking
        report.append("\n" + "-" * 80)
        report.append("PARAMETER IMPORTANCE RANKING")
        report.append("-" * 80)
        importance = self.parameter_importance_ranking()
        report.append(importance.head(10).to_string(index=False))
        
        # High reward distribution
        report.append("\n" + "-" * 80)
        report.append("HIGH REWARD PARAMETER DISTRIBUTIONS (Top 10%)")
        report.append("-" * 80)
        high_dist = self.analyze_high_reward_distribution(percentile=90)
        top_10_params = importance.head(10)['parameter'].values
        report.append(high_dist[high_dist['parameter'].isin(top_10_params)].to_string(index=False))
        
        # Distribution comparison
        report.append("\n" + "-" * 80)
        report.append("STATISTICAL COMPARISON (High vs Low Reward)")
        report.append("-" * 80)
        comparison = self.compare_distributions()
        report.append(comparison.head(10).to_string(index=False))
        
        report.append("\n" + "=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text


# Example usage function
def example_usage():
    """
    Demonstrate how to use the analyzer with synthetic data.
    """
    print("Generating synthetic RL data...")
    
    # Generate synthetic data
    n_samples = 10000
    n_params = 28
    
    # Create synthetic actions
    actions = np.random.randn(n_samples, n_params)
    
    # Create synthetic rewards that depend on certain parameters
    # Let's say parameters 0, 5, and 10 are most important
    rewards = (
        2.0 * actions[:, 0] +          # Parameter 0 has positive effect
        -1.5 * actions[:, 5] +          # Parameter 5 has negative effect
        1.0 * actions[:, 10] +          # Parameter 10 has positive effect
        0.3 * actions[:, 15] +          # Parameter 15 has small effect
        np.random.randn(n_samples) * 0.5  # Add noise
    )
    
    # Initialize analyzer
    analyzer = RLParameterAnalyzer(n_params=n_params)
    analyzer.load_data(actions, rewards)
    
    # Run analyses
    print("\n" + "="*80)
    print("1. PARAMETER IMPORTANCE RANKING")
    print("="*80)
    importance = analyzer.parameter_importance_ranking()
    print(importance.head(10))
    
    print("\n" + "="*80)
    print("2. HIGH REWARD PARAMETER DISTRIBUTIONS")
    print("="*80)
    high_reward_dist = analyzer.analyze_high_reward_distribution(percentile=90)
    print(high_reward_dist.head(10))
    
    print("\n" + "="*80)
    print("3. DISTRIBUTION COMPARISON (High vs Low Reward)")
    print("="*80)
    comparison = analyzer.compare_distributions()
    print(comparison.head(10))
    
    # Generate visualizations
    print("\n" + "="*80)
    print("4. GENERATING VISUALIZATIONS")
    print("="*80)
    analyzer.plot_correlation_heatmap(save_path='/home/claude/correlation_heatmap.png')
    analyzer.plot_parameter_distributions(top_k=9, save_path='/home/claude/top_parameters.png')
    
    # Generate report
    print("\n" + "="*80)
    print("5. COMPREHENSIVE REPORT")
    print("="*80)
    report = analyzer.generate_report(save_path='/home/claude/analysis_report.txt')
    print(report)
    
    return analyzer


if __name__ == "__main__":
    analyzer = example_usage()
