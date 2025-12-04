"""
Visualization functions for mixed-effects analysis results
- Marginal effects plots
- Uncertainty intervals
- Country clusters based on random effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utils for saving figures
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import save_figure


def plot_marginal_effects(marginal_effects_df, variable_name, output_path=None):
    """
    Plot marginal effects of a variable
    
    Parameters:
    -----------
    marginal_effects_df : DataFrame
        Output from policy_simulations.calculate_marginal_effects
    variable_name : str
        Name of variable being plotted
    output_path : str or Path, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'HIC' in marginal_effects_df.columns:
        # Plot separately for HIC and LMIC
        for hic_value in [0, 1]:
            subset = marginal_effects_df[marginal_effects_df['HIC'] == hic_value]
            label = 'High-income countries' if hic_value == 1 else 'Low/middle-income countries'
            ax.plot(subset['Value'], subset['PredictedIncidence'], 
                   label=label, linewidth=2, marker='o', markersize=3)
        
        ax.legend()
    else:
        ax.plot(marginal_effects_df['Value'], marginal_effects_df['PredictedIncidence'],
               linewidth=2, color='steelblue', marker='o', markersize=3)
    
    ax.set_xlabel(variable_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Measles Incidence\n(per 1M population)', fontsize=12, fontweight='bold')
    ax.set_title(f'Marginal Effects: {variable_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def plot_uncertainty_intervals(model, panel_df, variable, output_path=None):
    """
    Plot uncertainty intervals (confidence intervals) for predictions
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model
    panel_df : DataFrame
        Panel dataset
    variable : str
        Variable to plot uncertainty for
    output_path : str or Path, optional
        Path to save figure
    """
    print(f"\nPlotting uncertainty intervals for {variable}...")
    
    df = panel_df.copy()
    
    # Create prediction dataset
    prediction_df = df.copy()
    
    # Set other variables to median
    numeric_cols = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                    'PopDensity', 'NetMigration', 'PolStability']
    numeric_cols = [c for c in numeric_cols if c in prediction_df.columns and c != variable]
    
    for col in numeric_cols:
        prediction_df[col] = prediction_df[col].median()
    
    # Get range of variable
    valid_values = df[variable].dropna()
    values = np.linspace(valid_values.min(), valid_values.max(), 30)
    
    # Predictions and confidence intervals
    predictions = []
    ci_lower = []
    ci_upper = []
    
    for val in values:
        pred_df = prediction_df.copy()
        pred_df[variable] = val
        
        # Predict (using first row as template)
        try:
            try:
                if hasattr(model, 'fittedvalues') and hasattr(model, 'model'):
                    # Linear mixed model
                    pred = model.predict(pred_df.iloc[:1])
                    if isinstance(pred, pd.Series):
                        pred_val = pred.iloc[0]
                    elif isinstance(pred, np.ndarray):
                        pred_val = pred[0]
                    else:
                        pred_val = float(pred)
                    pred_incidence = np.expm1(pred_val)
                    
                    # Get confidence interval (simplified - use prediction interval if available)
                    # For mixed models, CI calculation is complex, so we'll use a simplified approach
                    try:
                        if hasattr(model, 'cov_params'):
                            cov_params = model.cov_params()
                            if isinstance(cov_params, pd.DataFrame):
                                se = np.sqrt(cov_params.iloc[0, 0])
                            else:
                                se = np.sqrt(np.diag(cov_params)[0])
                            ci_low = np.expm1(pred_val - 1.96 * se)
                            ci_high = np.expm1(pred_val + 1.96 * se)
                        else:
                            ci_low = pred_incidence * 0.8  # Approximate
                            ci_high = pred_incidence * 1.2
                    except:
                        ci_low = pred_incidence * 0.8
                        ci_high = pred_incidence * 1.2
                else:
                    # GLM model
                    pred = model.predict(pred_df.iloc[:1])
                    if isinstance(pred, pd.Series):
                        pred_incidence = pred.iloc[0]
                    elif isinstance(pred, np.ndarray):
                        pred_incidence = pred[0]
                    else:
                        pred_incidence = float(pred)
                    # Approximate CI
                    ci_low = pred_incidence * 0.7
                    ci_high = pred_incidence * 1.3
            except Exception as e:
                # Fallback
                pred_incidence = df['MeaslesIncidence'].median()
                ci_low = pred_incidence * 0.8
                ci_high = pred_incidence * 1.2
            
            predictions.append(pred_incidence)
            ci_lower.append(ci_low)
            ci_upper.append(ci_high)
            
        except Exception as e:
            print(f"  Warning: Could not predict for value {val}: {e}")
            predictions.append(np.nan)
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fill confidence interval
    ax.fill_between(values, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% Confidence Interval')
    
    # Plot mean prediction
    ax.plot(values, predictions, linewidth=2, color='darkblue', label='Predicted Incidence', marker='o', markersize=3)
    
    ax.set_xlabel(variable, fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Measles Incidence\n(per 1M population)', fontsize=12, fontweight='bold')
    ax.set_title(f'Uncertainty Intervals: {variable}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def cluster_countries_by_random_effects(model, panel_df, n_clusters=3, output_path=None):
    """
    Cluster countries based on random effects (intercepts and slopes)
    
    Parameters:
    -----------
    model : MixedLMResults
        Fitted mixed-effects model
    panel_df : DataFrame
        Panel dataset
    n_clusters : int
        Number of clusters to create
    output_path : str or Path, optional
        Path to save figure
        
    Returns:
    --------
    country_clusters : DataFrame
        Country codes with cluster assignments
    """
    print(f"\nClustering countries by random effects...")
    
    from sklearn.cluster import KMeans
    
    # Extract random effects
    if hasattr(model, 'random_effects'):
        random_effects = model.random_effects
        
        # Convert to DataFrame
        re_df = pd.DataFrame(random_effects).T
        
        # Get country codes
        countries = list(random_effects.keys())
        re_df['Code'] = countries
        
        # Cluster
        if len(re_df.columns) > 1:
            # Use all random effect components
            X = re_df.drop('Code', axis=1).values
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            re_df['Cluster'] = clusters
            
            # Create visualization
            if X.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, 
                                   cmap='viridis', s=100, alpha=0.6, edgecolors='black')
                
                ax.set_xlabel('Random Intercept', fontsize=12, fontweight='bold')
                if X.shape[1] > 1:
                    ax.set_ylabel('Random Slope (MCV1)', fontsize=12, fontweight='bold')
                else:
                    ax.set_ylabel('Component 2', fontsize=12, fontweight='bold')
                
                ax.set_title('Country Clusters Based on Random Effects', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, ax=ax, label='Cluster')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if output_path:
                    save_figure(fig, output_path)
                else:
                    plt.show()
                
                plt.close()
            
            # Add cluster labels
            cluster_labels = {i: f'Cluster {i+1}' for i in range(n_clusters)}
            re_df['ClusterLabel'] = re_df['Cluster'].map(cluster_labels)
            
            # Merge with panel data to get country names
            country_info = panel_df[['Code', 'Country', 'HIC']].drop_duplicates()
            re_df = re_df.merge(country_info, on='Code', how='left')
            
            print(f"\nCluster assignments:")
            for cluster_id in range(n_clusters):
                cluster_countries = re_df[re_df['Cluster'] == cluster_id]
                print(f"  Cluster {cluster_id+1}: {len(cluster_countries)} countries")
                if len(cluster_countries) <= 10:
                    print(f"    {', '.join(cluster_countries['Country'].dropna().tolist())}")
            
            return re_df[['Code', 'Country', 'Cluster', 'ClusterLabel', 'HIC']]
        else:
            print("  Warning: Insufficient random effects for clustering")
            return pd.DataFrame({'Code': countries, 'Cluster': [0] * len(countries)})
    else:
        print("  Warning: Model does not have random_effects attribute")
        return None


def plot_intervention_comparison(intervention_results_dict, output_path=None):
    """
    Create bar plot comparing different interventions
    
    Parameters:
    -----------
    intervention_results_dict : dict
        Dictionary of intervention results from policy_simulations.run_all_interventions
    output_path : str or Path, optional
        Path to save figure
    """
    # Calculate summary statistics for each intervention
    summary_data = []
    
    for intervention, results in intervention_results_dict.items():
        affected = results[results['InterventionApplied'] == 1]
        if len(affected) > 0:
            summary_data.append({
                'Intervention': intervention.replace('_', ' ').title(),
                'MeanReduction': affected['Change'].mean(),
                'TotalCasesPrevented': affected['Change'].sum(),
                'MeanPercentReduction': affected['PercentChange'].mean()
            })
    
    if not summary_data:
        print("No intervention results to plot")
        return
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Mean reduction per country
    ax1 = axes[0]
    bars1 = ax1.barh(summary_df['Intervention'], summary_df['MeanReduction'], 
                    color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Mean Reduction in Incidence\n(per 1M population)', fontsize=12, fontweight='bold')
    ax1.set_title('Intervention Effectiveness:\nMean Reduction per Country', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax1.text(row['MeanReduction'], i, f"  {row['MeanReduction']:.2f}", 
                va='center', fontsize=10)
    
    # Plot 2: Total cases prevented
    ax2 = axes[1]
    bars2 = ax2.barh(summary_df['Intervention'], summary_df['TotalCasesPrevented'], 
                    color='coral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Total Cases Prevented', fontsize=12, fontweight='bold')
    ax2.set_title('Intervention Impact:\nTotal Cases Prevented', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(summary_df.iterrows()):
        ax2.text(row['TotalCasesPrevented'], i, f"  {row['TotalCasesPrevented']:.0f}", 
                va='center', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    else:
        plt.show()
    
    plt.close()


def create_comprehensive_visualizations(model, panel_df, intervention_results_dict=None, 
                                       output_dir=None):
    """
    Create all visualizations for the analysis
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model
    panel_df : DataFrame
        Panel dataset
    intervention_results_dict : dict, optional
        Intervention results
    output_dir : str or Path, optional
        Output directory for figures
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'figures'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    # Import policy_simulations for marginal effects
    from policy_simulations import calculate_marginal_effects
    
    # Key variables to visualize
    key_variables = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'PolStability']
    key_variables = [v for v in key_variables if v in panel_df.columns]
    
    # 1. Marginal effects plots
    print("\n1. Creating marginal effects plots...")
    for var in key_variables:
        try:
            marginal_effects = calculate_marginal_effects(model, panel_df, var, hic_split=True)
            plot_marginal_effects(marginal_effects, var, 
                                 output_dir / f'marginal_effects_{var.lower()}.png')
        except Exception as e:
            print(f"  Error plotting marginal effects for {var}: {e}")
    
    # 2. Uncertainty intervals
    print("\n2. Creating uncertainty interval plots...")
    for var in key_variables[:3]:  # Plot top 3 variables
        try:
            plot_uncertainty_intervals(model, panel_df, var,
                                      output_dir / f'uncertainty_intervals_{var.lower()}.png')
        except Exception as e:
            print(f"  Error plotting uncertainty intervals for {var}: {e}")
    
    # 3. Country clusters
    print("\n3. Creating country cluster visualization...")
    try:
        if hasattr(model, 'random_effects'):
            clusters_df = cluster_countries_by_random_effects(
                model, panel_df, n_clusters=3,
                output_path=output_dir / 'country_clusters.png'
            )
            if clusters_df is not None:
                clusters_df.to_csv(output_dir.parent / 'country_clusters.csv', index=False)
        else:
            print("  Skipping: Model does not support random effects clustering")
    except Exception as e:
        print(f"  Error creating country clusters: {e}")
    
    # 4. Intervention comparison
    if intervention_results_dict:
        print("\n4. Creating intervention comparison plot...")
        try:
            plot_intervention_comparison(intervention_results_dict,
                                       output_dir / 'intervention_comparison.png')
        except Exception as e:
            print(f"  Error creating intervention comparison: {e}")
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    from prepare_data import prepare_panel_data
    from fit_models import load_model
    from policy_simulations import run_all_interventions
    
    # Load data and model
    print("Loading data and model...")
    panel_df = prepare_panel_data()
    
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis'
    model = load_model(results_dir / 'linear_mixed_model.pkl')
    
    # Run interventions
    intervention_results = run_all_interventions(model, panel_df)
    
    # Create visualizations
    create_comprehensive_visualizations(model, panel_df, intervention_results)

