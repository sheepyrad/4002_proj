"""
Advanced Imputation Strategies for Causal Analysis

This module provides multiple imputation methods that better preserve
the variance and distribution of the original data compared to simple
income-group median imputation.

Methods implemented:
1. MICE (Multiple Imputation by Chained Equations) via IterativeImputer
2. KNN Imputation - uses similar countries
3. Hybrid approach - MICE with income group stratification
4. Multiple Imputation with uncertainty quantification
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from prepare_imputed_data import (
    load_panel_data, 
    merge_additional_variables,
    MAX_ANALYSIS_YEAR
)

RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis'


def load_data_for_imputation():
    """Load and prepare data for advanced imputation"""
    df = load_panel_data(filter_year=True)
    df = merge_additional_variables(df)
    return df


def mice_imputation(df, n_imputations=5, max_iter=10, random_state=42):
    """
    Multiple Imputation by Chained Equations (MICE).
    
    Creates multiple imputed datasets and can be used for:
    1. Single best imputation (mean of imputations)
    2. Multiple imputation inference (Rubin's rules)
    
    Parameters:
    -----------
    df : DataFrame
        Data with missing values
    n_imputations : int
        Number of imputed datasets to create
    max_iter : int
        Number of iterations for each imputation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    list of DataFrames, one per imputation
    """
    print(f"\n--- MICE Imputation (n={n_imputations}, max_iter={max_iter}) ---")
    
    # Variables to impute (continuous only)
    impute_vars = ['GDPpc', 'HealthExpPC', 'PolStability', 'BirthRate', 
                   'PopDensity', 'MCV1', 'MCV2', 'NetMigration', 
                   'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    
    # Filter to existing columns
    impute_vars = [v for v in impute_vars if v in df.columns]
    
    # Also include auxiliary variables that help prediction
    aux_vars = ['Year', 'HIC']  # Year trend, income indicator
    all_vars = impute_vars + [v for v in aux_vars if v in df.columns]
    
    # Prepare data matrix
    X = df[all_vars].copy()
    
    # Track original missing
    missing_mask = X[impute_vars].isna()
    n_missing_total = missing_mask.sum().sum()
    print(f"  Total missing values to impute: {n_missing_total}")
    
    imputed_datasets = []
    
    for i in range(n_imputations):
        print(f"  Creating imputation {i+1}/{n_imputations}...")
        
        # Set bounds for each variable
        # Variables that are percentages should be bounded 0-100
        min_vals = [0] * len(all_vars)  # Default min is 0
        max_vals = [np.inf] * len(all_vars)  # Default max is inf
        
        percentage_vars = ['MCV1', 'MCV2', 'VaccineHesitancy', 'UrbanPop']
        for j, var in enumerate(all_vars):
            if var in percentage_vars:
                max_vals[j] = 100
        
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state + i,  # Different seed for each
            sample_posterior=True,  # CRITICAL: adds randomness to preserve variance
            min_value=min_vals,
            max_value=max_vals,
            verbose=0
        )
        
        X_imputed = imputer.fit_transform(X)
        
        # Create imputed dataframe
        df_imp = df.copy()
        for j, var in enumerate(all_vars):
            df_imp[var] = X_imputed[:, j]
        
        # Recalculate log variables
        df_imp['LogGDPpc'] = np.log1p(df_imp['GDPpc'])
        df_imp['LogPopDensity'] = np.log1p(df_imp['PopDensity'])
        df_imp['LogHealthExpPC'] = np.log1p(df_imp['HealthExpPC'])
        
        imputed_datasets.append(df_imp)
    
    return imputed_datasets


def knn_imputation(df, n_neighbors=5, weights='distance'):
    """
    K-Nearest Neighbors Imputation.
    
    Imputes missing values using values from similar observations.
    Better preserves local patterns and relationships.
    
    Parameters:
    -----------
    df : DataFrame
        Data with missing values
    n_neighbors : int
        Number of neighbors to use
    weights : str
        'uniform' or 'distance' weighting
    
    Returns:
    --------
    DataFrame with imputed values
    """
    print(f"\n--- KNN Imputation (k={n_neighbors}, weights={weights}) ---")
    
    impute_vars = ['GDPpc', 'HealthExpPC', 'PolStability', 'BirthRate', 
                   'PopDensity', 'MCV1', 'MCV2', 'NetMigration', 
                   'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    impute_vars = [v for v in impute_vars if v in df.columns]
    
    # Include auxiliary variables
    aux_vars = ['Year', 'HIC']
    all_vars = impute_vars + [v for v in aux_vars if v in df.columns]
    
    X = df[all_vars].copy()
    
    # Standardize for KNN (important for distance calculation)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.median()))  # Temp fill for scaling
    X_scaled = pd.DataFrame(X_scaled, columns=all_vars, index=X.index)
    
    # Put NaN back
    for var in impute_vars:
        X_scaled.loc[df[var].isna(), var] = np.nan
    
    # KNN imputation
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_imputed_scaled = imputer.fit_transform(X_scaled)
    
    # Inverse transform
    X_imputed = scaler.inverse_transform(X_imputed_scaled)
    
    # Create result dataframe
    df_imp = df.copy()
    for j, var in enumerate(all_vars):
        df_imp[var] = X_imputed[:, j]
    
    # Recalculate log variables
    df_imp['LogGDPpc'] = np.log1p(df_imp['GDPpc'].clip(lower=0))
    df_imp['LogPopDensity'] = np.log1p(df_imp['PopDensity'].clip(lower=0))
    df_imp['LogHealthExpPC'] = np.log1p(df_imp['HealthExpPC'].clip(lower=0))
    
    print(f"  Imputation complete")
    
    return df_imp


def stratified_mice_imputation(df, n_imputations=5, max_iter=10, random_state=42):
    """
    MICE imputation stratified by income group.
    
    Performs separate MICE imputation within each income group to preserve
    income-group specific patterns and relationships.
    
    This is a hybrid approach that:
    1. Maintains income-group specific distributions
    2. Uses MICE to preserve variance within groups
    3. Uses auxiliary variables for better prediction
    """
    print(f"\n--- Stratified MICE Imputation ---")
    
    income_groups = df['IncomeGroup'].dropna().unique()
    print(f"  Income groups: {list(income_groups)}")
    
    imputed_datasets = []
    
    for i in range(n_imputations):
        print(f"\n  Imputation {i+1}/{n_imputations}:")
        
        df_imp = df.copy()
        
        for ig in income_groups:
            mask = df['IncomeGroup'] == ig
            df_ig = df[mask].copy()
            
            if len(df_ig) < 30:
                print(f"    {ig}: too few observations, using global imputation")
                continue
            
            impute_vars = ['GDPpc', 'HealthExpPC', 'PolStability', 'BirthRate', 
                          'PopDensity', 'MCV1', 'MCV2', 'NetMigration', 
                          'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
            impute_vars = [v for v in impute_vars if v in df_ig.columns]
            
            # Add Year as auxiliary
            all_vars = impute_vars + ['Year']
            X = df_ig[all_vars].copy()
            
            # Check if there's anything to impute
            n_missing = X[impute_vars].isna().sum().sum()
            if n_missing == 0:
                continue
            
            # Set bounds for variables
            min_vals = [0] * len(all_vars)
            max_vals = [np.inf] * len(all_vars)
            percentage_vars = ['MCV1', 'MCV2', 'VaccineHesitancy', 'UrbanPop']
            for j, var in enumerate(all_vars):
                if var in percentage_vars:
                    max_vals[j] = 100
            
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state + i * 100 + hash(ig) % 100,
                sample_posterior=True,
                min_value=min_vals,
                max_value=max_vals,
                verbose=0
            )
            
            try:
                X_imputed = imputer.fit_transform(X)
                
                for j, var in enumerate(all_vars):
                    df_imp.loc[mask, var] = X_imputed[:, j]
                
                print(f"    {ig}: {n_missing} values imputed")
            except Exception as e:
                print(f"    {ig}: imputation failed ({e}), using median")
        
        # Handle any remaining missing (countries without income group)
        for var in impute_vars:
            if df_imp[var].isna().sum() > 0:
                df_imp[var] = df_imp[var].fillna(df_imp[var].median())
        
        # Recalculate log variables
        df_imp['LogGDPpc'] = np.log1p(df_imp['GDPpc'].clip(lower=0))
        df_imp['LogPopDensity'] = np.log1p(df_imp['PopDensity'].clip(lower=0))
        df_imp['LogHealthExpPC'] = np.log1p(df_imp['HealthExpPC'].clip(lower=0))
        
        imputed_datasets.append(df_imp)
    
    return imputed_datasets


def pool_multiple_imputations(imputed_datasets, var):
    """
    Pool estimates from multiple imputed datasets using Rubin's rules.
    
    For a parameter estimate θ:
    - Pooled estimate: mean of estimates across imputations
    - Within-imputation variance: mean of variances
    - Between-imputation variance: variance of estimates
    - Total variance: W + (1 + 1/m) * B
    
    Returns pooled mean and pooled standard error.
    """
    estimates = [df[var].mean() for df in imputed_datasets]
    variances = [df[var].var() / len(df) for df in imputed_datasets]
    
    m = len(imputed_datasets)
    
    # Pooled estimate
    pooled_mean = np.mean(estimates)
    
    # Within-imputation variance
    W = np.mean(variances)
    
    # Between-imputation variance
    B = np.var(estimates, ddof=1)
    
    # Total variance (Rubin's rules)
    total_var = W + (1 + 1/m) * B
    pooled_se = np.sqrt(total_var)
    
    return pooled_mean, pooled_se


def compare_imputation_methods(df, output_dir=None):
    """
    Compare different imputation methods and visualize results.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import seaborn as sns
    
    if output_dir is None:
        output_dir = RESULTS_DIR / 'imputation_comparison'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("COMPARING IMPUTATION METHODS")
    print("="*60)
    
    # Original data
    df_original = df.copy()
    
    # Method 1: Simple median (current approach)
    print("\n1. Simple Income-Group Median Imputation...")
    from prepare_imputed_data import impute_by_income_group
    df_median = df.copy()
    for var in ['VaccineHesitancy', 'HealthExpPC', 'MCV2']:
        if var in df.columns and df[var].isna().sum() > 0:
            df_median[var] = impute_by_income_group(df_median, var, method='median')
    
    # Method 2: KNN
    print("\n2. KNN Imputation...")
    df_knn = knn_imputation(df.copy(), n_neighbors=5)
    
    # Method 3: MICE (single)
    print("\n3. MICE Imputation...")
    mice_datasets = mice_imputation(df.copy(), n_imputations=1, max_iter=10)
    df_mice = mice_datasets[0]
    
    # Method 4: Stratified MICE
    print("\n4. Stratified MICE Imputation...")
    strat_datasets = stratified_mice_imputation(df.copy(), n_imputations=1, max_iter=10)
    df_strat = strat_datasets[0]
    
    # Compare distributions for key variables
    compare_vars = ['VaccineHesitancy', 'HealthExpPC', 'MCV2']
    compare_vars = [v for v in compare_vars if v in df.columns]
    
    for var in compare_vars:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        methods = [
            ('Original (non-missing)', df_original[var].dropna(), '#3498db'),
            ('Median Imputation', df_median[var], '#e74c3c'),
            ('KNN Imputation', df_knn[var], '#2ecc71'),
            ('MICE Imputation', df_mice[var], '#9b59b6'),
            ('Stratified MICE', df_strat[var], '#f39c12'),
        ]
        
        # Row 1: Overall distributions
        ax = axes[0, 0]
        for name, data, color in methods:
            if len(data.dropna()) > 0:
                sns.kdeplot(data=data.dropna(), ax=ax, label=name, color=color, linewidth=2)
        ax.set_title(f'{var}: All Methods Comparison')
        ax.set_xlabel(var)
        ax.legend(fontsize=8)
        
        # Histograms for each method
        for i, (name, data, color) in enumerate(methods[1:], 1):  # Skip original
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            ax.hist(data.dropna(), bins=30, alpha=0.7, color=color, edgecolor='black')
            
            # Overlay original
            ax.hist(df_original[var].dropna(), bins=30, alpha=0.3, color='blue', 
                   edgecolor='blue', label='Original')
            
            n_imputed = df_original[var].isna().sum()
            ax.set_title(f'{name}\n({n_imputed} values imputed)')
            ax.set_xlabel(var)
            ax.legend()
        
        # Summary statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        stats_text = f"Summary Statistics for {var}:\n\n"
        stats_text += f"{'Method':<25} {'Mean':>8} {'Std':>8} {'Median':>8}\n"
        stats_text += "-" * 55 + "\n"
        
        for name, data, _ in methods:
            mean = data.mean()
            std = data.std()
            median = data.median()
            stats_text += f"{name:<25} {mean:>8.2f} {std:>8.2f} {median:>8.2f}\n"
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Imputation Method Comparison: {var}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f'{var}_imputation_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {var}_imputation_comparison.png")
    
    # Save summary statistics
    summary_rows = []
    all_dfs = {
        'Original (non-missing)': df_original,
        'Median Imputation': df_median,
        'KNN Imputation': df_knn,
        'MICE Imputation': df_mice,
        'Stratified MICE': df_strat
    }
    
    for var in compare_vars:
        for method_name, method_df in all_dfs.items():
            if method_name == 'Original (non-missing)':
                data = method_df[var].dropna()
            else:
                data = method_df[var]
            
            summary_rows.append({
                'Variable': var,
                'Method': method_name,
                'Mean': data.mean(),
                'Std': data.std(),
                'Median': data.median(),
                'Min': data.min(),
                'Max': data.max()
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'imputation_method_comparison.csv', index=False)
    print(f"\n  Summary saved to: imputation_method_comparison.csv")
    
    return {
        'median': df_median,
        'knn': df_knn,
        'mice': df_mice,
        'stratified_mice': df_strat
    }


def get_best_imputed_data(method='stratified_mice', n_imputations=5):
    """
    Get analysis-ready data using the specified imputation method.
    
    Parameters:
    -----------
    method : str
        'median' - simple income-group median (fast, but biased)
        'knn' - K-nearest neighbors (good for local patterns)
        'mice' - Multiple imputation (best for inference)
        'stratified_mice' - MICE within income groups (recommended)
    n_imputations : int
        Number of imputations for MICE methods
    
    Returns:
    --------
    If method is 'mice' or 'stratified_mice': list of imputed DataFrames
    Otherwise: single imputed DataFrame
    """
    df = load_data_for_imputation()
    
    if method == 'median':
        from prepare_imputed_data import prepare_imputed_data
        df_imputed, _ = prepare_imputed_data(verbose=False, save_stats=False)
        return df_imputed
    
    elif method == 'knn':
        return knn_imputation(df, n_neighbors=5)
    
    elif method == 'mice':
        return mice_imputation(df, n_imputations=n_imputations)
    
    elif method == 'stratified_mice':
        return stratified_mice_imputation(df, n_imputations=n_imputations)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    """Run imputation comparison"""
    df = load_data_for_imputation()
    results = compare_imputation_methods(df)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
For your causal analysis, consider:

1. **For VaccineHesitancy (84% missing)**:
   - Use MICE or Stratified MICE to preserve variance
   - Consider sensitivity analysis: run analysis with different methods
   - Report uncertainty due to imputation

2. **For HealthExpPC (16-21% missing)**:
   - KNN or MICE are both good choices
   - Less sensitive to imputation method due to lower missingness

3. **For MCV2 (24-39% missing in LMIC/LIC)**:
   - Stratified MICE recommended to preserve income-group patterns
   - Country-level patterns matter for vaccine coverage

4. **Best Practice**:
   - Run analysis with multiple imputation (e.g., 5 imputations)
   - Pool results using Rubin's rules
   - Report both point estimates and imputation uncertainty
""")


if __name__ == '__main__':
    main()

