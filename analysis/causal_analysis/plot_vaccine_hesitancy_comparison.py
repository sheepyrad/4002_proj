"""
Compare VaccineHesitancy Imputation Methods

This script compares different imputation methods for VaccineHesitancy:
1. Original (non-missing data only)
2. Income-group median imputation
3. KNN imputation

Saves comparison plots to results/causal_analysis/imputation_plots/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

from prepare_imputed_data import (
    load_panel_data, 
    merge_additional_variables,
    impute_by_income_group,
    knn_impute_all,
    MAX_ANALYSIS_YEAR
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis' / 'imputation_plots'


def load_data():
    """Load data before any imputation"""
    df = load_panel_data(filter_year=True)
    df = merge_additional_variables(df)
    return df


def create_vaccine_hesitancy_comparison():
    """Create comparison plot for VaccineHesitancy imputation methods"""
    
    print("="*60)
    print("VACCINE HESITANCY IMPUTATION METHOD COMPARISON")
    print("="*60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    print("\nLoading data...")
    df_original = load_data()
    
    # Method 1: Original (non-missing only)
    original_data = df_original['VaccineHesitancy'].dropna()
    n_original = len(original_data)
    n_total = len(df_original)
    n_missing = n_total - n_original
    print(f"  Original: {n_original} observations ({100*n_original/n_total:.1f}% non-missing)")
    print(f"  Missing: {n_missing} observations ({100*n_missing/n_total:.1f}%)")
    
    # Method 2: Income-group median imputation
    print("\nApplying income-group median imputation...")
    df_median = df_original.copy()
    df_median['VaccineHesitancy'] = impute_by_income_group(df_median, 'VaccineHesitancy', method='median')
    median_data = df_median['VaccineHesitancy']
    
    # Method 3: KNN imputation
    print("\nApplying KNN imputation...")
    df_knn = df_original.copy()
    df_knn, _ = knn_impute_all(df_knn, ['VaccineHesitancy'])
    knn_data = df_knn['VaccineHesitancy']
    
    # Income groups
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    income_short = ['HIC', 'UMIC', 'LMIC', 'LIC']
    
    # =========================================
    # Plot 1: Overall Distribution Comparison
    # =========================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Colors
    color_original = '#3498db'
    color_median = '#e74c3c'
    color_knn = '#2ecc71'
    
    # Overall KDE comparison
    ax = axes[0, 0]
    sns.kdeplot(data=original_data, ax=ax, color=color_original, 
                label=f'Original (n={n_original})', linewidth=2.5)
    sns.kdeplot(data=median_data, ax=ax, color=color_median, 
                label=f'Median Imputed (n={n_total})', linewidth=2, linestyle='--')
    sns.kdeplot(data=knn_data, ax=ax, color=color_knn, 
                label=f'KNN Imputed (n={n_total})', linewidth=2, linestyle=':')
    ax.set_title('Overall Distribution Comparison', fontsize=12, fontweight='bold')
    ax.set_xlabel('Vaccine Hesitancy (%)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.set_xlim(-5, 60)
    
    # By income group
    for i, (ig, ig_short) in enumerate(zip(income_groups, income_short)):
        row = (i + 1) // 3
        col = (i + 1) % 3
        ax = axes[row, col]
        
        mask = df_original['IncomeGroup'] == ig
        
        orig_ig = df_original.loc[mask, 'VaccineHesitancy'].dropna()
        median_ig = df_median.loc[mask, 'VaccineHesitancy']
        knn_ig = df_knn.loc[mask, 'VaccineHesitancy']
        
        n_missing_ig = df_original.loc[mask, 'VaccineHesitancy'].isna().sum()
        n_total_ig = mask.sum()
        pct_missing = 100 * n_missing_ig / n_total_ig if n_total_ig > 0 else 0
        
        if len(orig_ig) > 1:
            sns.kdeplot(data=orig_ig, ax=ax, color=color_original, 
                       label=f'Original (n={len(orig_ig)})', linewidth=2.5)
        sns.kdeplot(data=median_ig, ax=ax, color=color_median, 
                   label=f'Median (n={n_total_ig})', linewidth=2, linestyle='--')
        sns.kdeplot(data=knn_ig, ax=ax, color=color_knn, 
                   label=f'KNN (n={n_total_ig})', linewidth=2, linestyle=':')
        
        ax.set_title(f'{ig_short}\n({pct_missing:.1f}% imputed)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Vaccine Hesitancy (%)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.set_xlim(-5, 60)
    
    # Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "IMPUTATION SUMMARY\n" + "="*30 + "\n\n"
    summary_text += f"Total observations: {n_total}\n"
    summary_text += f"Missing values: {n_missing} ({100*n_missing/n_total:.1f}%)\n\n"
    summary_text += "Statistics by Method:\n" + "-"*30 + "\n"
    summary_text += f"{'Method':<15} {'Mean':>8} {'Std':>8} {'Median':>8}\n"
    summary_text += f"{'Original':<15} {original_data.mean():>8.2f} {original_data.std():>8.2f} {original_data.median():>8.2f}\n"
    summary_text += f"{'Median Imp.':<15} {median_data.mean():>8.2f} {median_data.std():>8.2f} {median_data.median():>8.2f}\n"
    summary_text += f"{'KNN Imp.':<15} {knn_data.mean():>8.2f} {knn_data.std():>8.2f} {knn_data.median():>8.2f}\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('VaccineHesitancy: Imputation Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = RESULTS_DIR / 'VaccineHesitancy_method_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path.name}")
    
    # =========================================
    # Plot 2: Histogram Comparison
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    bins = np.linspace(0, 55, 25)
    
    # Original
    ax = axes[0]
    ax.hist(original_data, bins=bins, color=color_original, edgecolor='black', alpha=0.7)
    ax.axvline(original_data.mean(), color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {original_data.mean():.1f}')
    ax.axvline(original_data.median(), color='blue', linestyle=':', linewidth=2, label=f'Median: {original_data.median():.1f}')
    ax.set_title(f'Original Data\n(n={n_original}, {100*n_missing/n_total:.0f}% missing)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Vaccine Hesitancy (%)')
    ax.set_ylabel('Count')
    ax.legend()
    
    # Median imputed
    ax = axes[1]
    ax.hist(median_data, bins=bins, color=color_median, edgecolor='black', alpha=0.7)
    ax.axvline(median_data.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {median_data.mean():.1f}')
    ax.axvline(median_data.median(), color='red', linestyle=':', linewidth=2, label=f'Median: {median_data.median():.1f}')
    ax.set_title(f'Median Imputation\n(Std: {median_data.std():.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Vaccine Hesitancy (%)')
    ax.set_ylabel('Count')
    ax.legend()
    
    # KNN imputed
    ax = axes[2]
    ax.hist(knn_data, bins=bins, color=color_knn, edgecolor='black', alpha=0.7)
    ax.axvline(knn_data.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {knn_data.mean():.1f}')
    ax.axvline(knn_data.median(), color='green', linestyle=':', linewidth=2, label=f'Median: {knn_data.median():.1f}')
    ax.set_title(f'KNN Imputation\n(Std: {knn_data.std():.2f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Vaccine Hesitancy (%)')
    ax.set_ylabel('Count')
    ax.legend()
    
    plt.suptitle('VaccineHesitancy: Histogram Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = RESULTS_DIR / 'VaccineHesitancy_histogram_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path.name}")
    
    # =========================================
    # Plot 3: Boxplot by Income Group
    # =========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    for ax, (data, title, color) in zip(axes, [
        (df_original, 'Original', color_original),
        (df_median, 'Median Imputation', color_median),
        (df_knn, 'KNN Imputation', color_knn)
    ]):
        # Prepare data for boxplot
        plot_data = []
        labels = []
        for ig, ig_short in zip(income_groups, income_short):
            mask = data['IncomeGroup'] == ig
            values = data.loc[mask, 'VaccineHesitancy'].dropna()
            if len(values) > 0:
                plot_data.append(values)
                labels.append(ig_short)
        
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Income Group')
        ax.set_ylabel('Vaccine Hesitancy (%)')
        ax.set_ylim(0, 55)
    
    plt.suptitle('VaccineHesitancy by Income Group: Method Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = RESULTS_DIR / 'VaccineHesitancy_boxplot_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path.name}")
    
    # =========================================
    # Save Summary Statistics
    # =========================================
    summary_rows = []
    
    for ig, ig_short in zip(income_groups + ['All'], income_short + ['ALL']):
        if ig == 'All':
            mask = pd.Series([True] * len(df_original))
        else:
            mask = df_original['IncomeGroup'] == ig
        
        orig = df_original.loc[mask, 'VaccineHesitancy'].dropna()
        med = df_median.loc[mask, 'VaccineHesitancy']
        knn = df_knn.loc[mask, 'VaccineHesitancy']
        
        n_miss = df_original.loc[mask, 'VaccineHesitancy'].isna().sum()
        n_tot = mask.sum()
        
        summary_rows.append({
            'Income_Group': ig_short,
            'N_Total': n_tot,
            'N_Missing': n_miss,
            'Pct_Missing': 100 * n_miss / n_tot if n_tot > 0 else 0,
            'Original_Mean': orig.mean() if len(orig) > 0 else np.nan,
            'Original_Std': orig.std() if len(orig) > 0 else np.nan,
            'Original_Median': orig.median() if len(orig) > 0 else np.nan,
            'Median_Imp_Mean': med.mean(),
            'Median_Imp_Std': med.std(),
            'Median_Imp_Median': med.median(),
            'KNN_Mean': knn.mean(),
            'KNN_Std': knn.std(),
            'KNN_Median': knn.median(),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / 'VaccineHesitancy_imputation_comparison.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path.name}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\n{'Income':<6} {'Miss%':>6} {'Orig.Mean':>10} {'Med.Mean':>10} {'KNN Mean':>10} {'Orig.Std':>10} {'Med.Std':>10} {'KNN.Std':>10}")
    print("-"*76)
    for _, row in summary_df.iterrows():
        print(f"{row['Income_Group']:<6} {row['Pct_Missing']:>5.1f}% "
              f"{row['Original_Mean']:>10.2f} {row['Median_Imp_Mean']:>10.2f} {row['KNN_Mean']:>10.2f} "
              f"{row['Original_Std']:>10.2f} {row['Median_Imp_Std']:>10.2f} {row['KNN_Std']:>10.2f}")
    
    print("\n" + "="*60)
    print("KEY OBSERVATION:")
    print("="*60)
    orig_std = original_data.std()
    median_std = median_data.std()
    knn_std = knn_data.std()
    print(f"  Original Std: {orig_std:.2f}")
    print(f"  Median Imputation Std: {median_std:.2f} (Variance {'REDUCED' if median_std < orig_std else 'preserved'} by {100*(orig_std-median_std)/orig_std:.1f}%)")
    print(f"  KNN Imputation Std: {knn_std:.2f} (Variance {'REDUCED' if knn_std < orig_std else 'preserved'} by {100*(orig_std-knn_std)/orig_std:.1f}%)")
    
    return summary_df


if __name__ == '__main__':
    create_vaccine_hesitancy_comparison()


