"""
Plot Distribution of Variables Before and After Imputation by Income Group

This script creates visualizations to compare variable distributions
before and after income-group based imputation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import seaborn as sns
from pathlib import Path
from prepare_imputed_data import (
    load_panel_data, 
    merge_additional_variables,
    MAX_ANALYSIS_YEAR
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis' / 'imputation_plots'


def load_data_before_imputation():
    """Load and merge data without imputation"""
    data_path = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'panel_data.csv'
    df = pd.read_csv(data_path)
    
    # Filter years
    df = df[df['Year'] <= MAX_ANALYSIS_YEAR]
    
    # Merge additional variables
    df = merge_additional_variables(df)
    
    return df


def load_data_after_imputation():
    """Load data with hybrid imputation (KNN for VaccineHesitancy, median for others)"""
    from prepare_imputed_data import prepare_imputed_data
    
    # Use hybrid imputation: KNN for VaccineHesitancy, median for others
    df_imputed, _ = prepare_imputed_data(imputation_method='hybrid', verbose=False, save_stats=False)
    
    return df_imputed


def plot_variable_distributions(df_before, df_after, variable, output_dir):
    """
    Plot distribution of a variable before and after imputation for each income group.
    
    Creates:
    1. Overall comparison (all data)
    2. By income group comparison
    """
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_short = ['HIC', 'UMIC', 'LMIC', 'LIC']
    
    if variable not in df_before.columns:
        print(f"  Skipping {variable}: not in dataframe")
        return
    
    # Count missing before imputation
    n_missing_before = df_before[variable].isna().sum()
    n_total = len(df_before)
    
    if n_missing_before == 0:
        print(f"  Skipping {variable}: no missing values")
        return
    
    # Create figure with subplots: 1 overall + 4 income groups
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Color scheme
    color_before = '#3498db'  # Blue
    color_after = '#e74c3c'   # Red
    color_imputed = '#2ecc71'  # Green for imputed values only
    
    # Plot 1: Overall distribution
    ax = axes[0]
    
    # Before (excluding NaN)
    data_before = df_before[variable].dropna()
    data_after = df_after[variable].dropna()
    
    if len(data_before) > 0:
        sns.kdeplot(data=data_before, ax=ax, color=color_before, label=f'Before (n={len(data_before)})', linewidth=2)
    if len(data_after) > 0:
        sns.kdeplot(data=data_after, ax=ax, color=color_after, label=f'After (n={len(data_after)})', linewidth=2, linestyle='--')
    
    ax.set_title(f'Overall Distribution\n({n_missing_before}/{n_total} = {100*n_missing_before/n_total:.1f}% imputed)')
    ax.set_xlabel(variable)
    ax.legend()
    
    # Plot 2-5: By income group
    for i, (ig, ig_short) in enumerate(zip(income_groups, income_short)):
        ax = axes[i + 1]
        
        # Filter by income group
        mask_before = df_before['IncomeGroup'] == ig
        mask_after = df_after['IncomeGroup'] == ig
        
        data_ig_before = df_before.loc[mask_before, variable].dropna()
        data_ig_after = df_after.loc[mask_after, variable]
        
        n_ig_total = mask_before.sum()
        n_ig_missing = df_before.loc[mask_before, variable].isna().sum()
        
        if len(data_ig_before) > 0:
            sns.kdeplot(data=data_ig_before, ax=ax, color=color_before, 
                       label=f'Before (n={len(data_ig_before)})', linewidth=2)
        if len(data_ig_after) > 0:
            sns.kdeplot(data=data_ig_after, ax=ax, color=color_after, 
                       label=f'After (n={len(data_ig_after)})', linewidth=2, linestyle='--')
        
        # Add imputed values as rug plot if there are any
        if n_ig_missing > 0:
            # Identify imputed values (values in after but were NaN in before)
            imputed_mask = mask_after & df_before[variable].isna()
            imputed_values = df_after.loc[imputed_mask, variable]
            if len(imputed_values) > 0:
                ax.axvline(imputed_values.median(), color=color_imputed, linestyle=':', 
                          label=f'Imputed median', linewidth=2)
        
        ax.set_title(f'{ig_short}\n({n_ig_missing}/{n_ig_total} = {100*n_ig_missing/n_ig_total:.1f}% imputed)')
        ax.set_xlabel(variable)
        ax.legend(fontsize=8)
    
    # Hide the 6th subplot (if any)
    axes[5].axis('off')
    
    # Add summary text in the 6th subplot area
    summary_text = f"Variable: {variable}\n\n"
    summary_text += "Missing Data by Income Group:\n"
    for ig, ig_short in zip(income_groups, income_short):
        mask = df_before['IncomeGroup'] == ig
        n_total_ig = mask.sum()
        n_missing_ig = df_before.loc[mask, variable].isna().sum()
        pct = 100 * n_missing_ig / n_total_ig if n_total_ig > 0 else 0
        summary_text += f"  {ig_short}: {n_missing_ig}/{n_total_ig} ({pct:.1f}%)\n"
    
    axes[5].text(0.1, 0.5, summary_text, transform=axes[5].transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Distribution of {variable} Before vs After Imputation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{variable}_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path.name}")


def plot_histogram_comparison(df_before, df_after, variable, output_dir):
    """
    Plot histogram comparison (alternative to KDE for discrete/sparse data).
    """
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_short = ['HIC', 'UMIC', 'LMIC', 'LIC']
    
    if variable not in df_before.columns:
        return
    
    n_missing_before = df_before[variable].isna().sum()
    if n_missing_before == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ig, ig_short) in enumerate(zip(income_groups, income_short)):
        ax = axes[i]
        
        mask = df_before['IncomeGroup'] == ig
        data_before = df_before.loc[mask, variable].dropna()
        data_after = df_after.loc[mask, variable]
        
        n_total = mask.sum()
        n_missing = df_before.loc[mask, variable].isna().sum()
        
        # Determine bins
        all_data = pd.concat([data_before, data_after]).dropna()
        if len(all_data) == 0:
            continue
        bins = np.linspace(all_data.min(), all_data.max(), 20)
        
        ax.hist(data_before, bins=bins, alpha=0.5, label=f'Before (n={len(data_before)})', 
               color='#3498db', edgecolor='black')
        ax.hist(data_after, bins=bins, alpha=0.5, label=f'After (n={len(data_after)})', 
               color='#e74c3c', edgecolor='black')
        
        ax.set_title(f'{ig_short} ({n_missing}/{n_total} = {100*n_missing/n_total:.1f}% imputed)')
        ax.set_xlabel(variable)
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.suptitle(f'Histogram: {variable} Before vs After Imputation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f'{variable}_histogram.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_boxplot_comparison(df_before, df_after, variable, output_dir):
    """
    Plot boxplot comparison by income group.
    """
    if variable not in df_before.columns:
        return
    
    n_missing_before = df_before[variable].isna().sum()
    if n_missing_before == 0:
        return
    
    # Prepare data for boxplot
    df_before_plot = df_before[['IncomeGroup', variable]].copy()
    df_before_plot['Stage'] = 'Before'
    
    df_after_plot = df_after[['IncomeGroup', variable]].copy()
    df_after_plot['Stage'] = 'After'
    
    df_combined = pd.concat([df_before_plot, df_after_plot], ignore_index=True)
    
    # Order income groups
    income_order = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    df_combined['IncomeGroup'] = pd.Categorical(df_combined['IncomeGroup'], categories=income_order, ordered=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.boxplot(data=df_combined, x='IncomeGroup', y=variable, hue='Stage', ax=ax,
               palette={'Before': '#3498db', 'After': '#e74c3c'})
    
    ax.set_title(f'Boxplot: {variable} Before vs After Imputation by Income Group', fontsize=14, fontweight='bold')
    ax.set_xlabel('Income Group')
    ax.set_ylabel(variable)
    ax.legend(title='Stage')
    
    # Rotate x-labels for readability
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{variable}_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_heatmap(df_before, output_dir):
    """
    Create a heatmap showing missing data percentage by variable and income group.
    """
    variables = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'PolStability', 
                'BirthRate', 'PopDensity', 'NetMigration', 'UrbanPop', 
                'HouseholdSize', 'VaccineHesitancy']
    
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    
    # Build matrix
    missing_matrix = []
    for var in variables:
        if var not in df_before.columns:
            continue
        row = []
        for ig in income_groups:
            mask = df_before['IncomeGroup'] == ig
            n_total = mask.sum()
            n_missing = df_before.loc[mask, var].isna().sum()
            pct = 100 * n_missing / n_total if n_total > 0 else 0
            row.append(pct)
        missing_matrix.append(row)
    
    # Filter variables that exist
    existing_vars = [v for v in variables if v in df_before.columns]
    
    # Create DataFrame
    missing_df = pd.DataFrame(missing_matrix, index=existing_vars, 
                             columns=['HIC', 'UMIC', 'LMIC', 'LIC'])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(missing_df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': '% Missing (Imputed)'}, vmin=0, vmax=100)
    
    ax.set_title('Missing Data Percentage by Variable and Income Group\n(Before Imputation)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Income Group')
    ax.set_ylabel('Variable')
    
    plt.tight_layout()
    
    output_path = output_dir / 'missing_data_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    missing_df.to_csv(output_dir / 'missing_data_matrix.csv')
    
    print(f"  Saved: missing_data_heatmap.png")
    print(f"  Saved: missing_data_matrix.csv")


def plot_imputation_effect_summary(df_before, df_after, output_dir):
    """
    Create summary plot showing mean/median shift due to imputation.
    """
    variables = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'PolStability', 
                'BirthRate', 'PopDensity', 'VaccineHesitancy']
    
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_short = ['HIC', 'UMIC', 'LMIC', 'LIC']
    
    # Calculate mean shift for each variable and income group
    results = []
    for var in variables:
        if var not in df_before.columns:
            continue
        for ig, ig_short in zip(income_groups, income_short):
            mask = df_before['IncomeGroup'] == ig
            
            mean_before = df_before.loc[mask, var].mean()
            mean_after = df_after.loc[mask, var].mean()
            
            median_before = df_before.loc[mask, var].median()
            median_after = df_after.loc[mask, var].median()
            
            n_missing = df_before.loc[mask, var].isna().sum()
            n_total = mask.sum()
            pct_imputed = 100 * n_missing / n_total if n_total > 0 else 0
            
            # Calculate percentage change
            mean_change = ((mean_after - mean_before) / mean_before * 100) if mean_before != 0 else 0
            median_change = ((median_after - median_before) / median_before * 100) if median_before != 0 else 0
            
            results.append({
                'Variable': var,
                'Income_Group': ig_short,
                'Mean_Before': mean_before,
                'Mean_After': mean_after,
                'Mean_Change_Pct': mean_change,
                'Median_Before': median_before,
                'Median_After': median_after,
                'Median_Change_Pct': median_change,
                'Pct_Imputed': pct_imputed
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'imputation_effect_summary.csv', index=False)
    
    # Plot mean change heatmap
    pivot_mean = results_df.pivot(index='Variable', columns='Income_Group', values='Mean_Change_Pct')
    pivot_mean = pivot_mean[['HIC', 'UMIC', 'LMIC', 'LIC']]  # Reorder columns
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean change
    ax = axes[0]
    sns.heatmap(pivot_mean, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax,
               cbar_kws={'label': '% Change in Mean'})
    ax.set_title('Mean Change (%) After Imputation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Income Group')
    ax.set_ylabel('Variable')
    
    # Median change
    pivot_median = results_df.pivot(index='Variable', columns='Income_Group', values='Median_Change_Pct')
    pivot_median = pivot_median[['HIC', 'UMIC', 'LMIC', 'LIC']]
    
    ax = axes[1]
    sns.heatmap(pivot_median, annot=True, fmt='.1f', cmap='RdBu_r', center=0, ax=ax,
               cbar_kws={'label': '% Change in Median'})
    ax.set_title('Median Change (%) After Imputation', fontsize=12, fontweight='bold')
    ax.set_xlabel('Income Group')
    ax.set_ylabel('Variable')
    
    plt.suptitle('Effect of Imputation on Variable Distributions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'imputation_effect_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: imputation_effect_summary.csv")
    print(f"  Saved: imputation_effect_heatmap.png")
    
    return results_df


def main():
    """Run all imputation distribution plots."""
    print("="*60)
    print("PLOTTING IMPUTATION DISTRIBUTIONS")
    print("="*60)
    
    # Create output directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {RESULTS_DIR}")
    
    # Load data
    print("\nLoading data before imputation...")
    df_before = load_data_before_imputation()
    print(f"  Loaded: {len(df_before)} observations, {df_before['Code'].nunique()} countries")
    
    print("\nLoading data after imputation...")
    df_after = load_data_after_imputation()
    print(f"  Loaded: {len(df_after)} observations, {df_after['Code'].nunique()} countries")
    
    # Variables to plot
    variables = ['MCV1', 'MCV2', 'GDPpc', 'LogGDPpc', 'HealthExpPC', 
                'PolStability', 'BirthRate', 'PopDensity', 'LogPopDensity',
                'NetMigration', 'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    
    # 1. Summary heatmap
    print("\n--- Creating Summary Heatmap ---")
    plot_summary_heatmap(df_before, RESULTS_DIR)
    
    # 2. Imputation effect summary
    print("\n--- Creating Imputation Effect Summary ---")
    plot_imputation_effect_summary(df_before, df_after, RESULTS_DIR)
    
    # 3. Distribution plots for each variable
    print("\n--- Creating Distribution Plots ---")
    for var in variables:
        print(f"\nProcessing: {var}")
        plot_variable_distributions(df_before, df_after, var, RESULTS_DIR)
        plot_boxplot_comparison(df_before, df_after, var, RESULTS_DIR)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nAll plots saved to: {RESULTS_DIR}")
    
    # List created files
    print("\nCreated files:")
    for f in sorted(RESULTS_DIR.glob('*.png')):
        print(f"  - {f.name}")
    for f in sorted(RESULTS_DIR.glob('*.csv')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()

