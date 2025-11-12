"""
Analysis script for Vaccine Coverage (1st and 2nd dose) data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """Load vaccine coverage data"""
    # Handle parsing errors - some lines may be malformed
    # Check pandas version for compatibility
    pd_version = pd.__version__
    major_version = int(pd_version.split('.')[0])
    
    if major_version >= 2:
        # pandas 2.0+ uses on_bad_lines
        df1 = pd.read_csv(DATA_DIR / '1st_dose_WUENIC.csv', on_bad_lines='skip')
        df2 = pd.read_csv(DATA_DIR / '2nd_dose_WUENIC.csv', on_bad_lines='skip')
    elif major_version >= 1:
        # pandas 1.3.0+ supports on_bad_lines
        try:
            df1 = pd.read_csv(DATA_DIR / '1st_dose_WUENIC.csv', on_bad_lines='skip')
            df2 = pd.read_csv(DATA_DIR / '2nd_dose_WUENIC.csv', on_bad_lines='skip')
        except TypeError:
            # Fallback for older pandas 1.x
            df1 = pd.read_csv(DATA_DIR / '1st_dose_WUENIC.csv', error_bad_lines=False, warn_bad_lines=True)
            df2 = pd.read_csv(DATA_DIR / '2nd_dose_WUENIC.csv', error_bad_lines=False, warn_bad_lines=True)
    else:
        # pandas < 1.0
        df1 = pd.read_csv(DATA_DIR / '1st_dose_WUENIC.csv', error_bad_lines=False, warn_bad_lines=True)
        df2 = pd.read_csv(DATA_DIR / '2nd_dose_WUENIC.csv', error_bad_lines=False, warn_bad_lines=True)
    
    # Filter for MCV1 and MCV2
    df1_mcv = df1[df1['ANTIGEN'] == 'MCV1'].copy() if 'ANTIGEN' in df1.columns else df1.copy()
    df2_mcv = df2[df2['ANTIGEN'] == 'MCV2'].copy() if 'ANTIGEN' in df2.columns else df2.copy()
    
    return df1_mcv, df2_mcv

def analyze_vaccine_coverage():
    """Perform comprehensive analysis of vaccine coverage data"""
    print("="*60)
    print("VACCINE COVERAGE ANALYSIS")
    print("="*60)
    
    # Load data (already filtered for MCV1 and MCV2)
    df1_mcv, df2_mcv = load_data()
    print(f"\n1st Dose Data shape: {df1_mcv.shape}")
    print(f"2nd Dose Data shape: {df2_mcv.shape}")
    print(f"\n1st Dose Columns: {df1_mcv.columns.tolist()}")
    print(f"\n2nd Dose Columns: {df2_mcv.columns.tolist()}")
    
    # Basic statistics
    print(f"\n1st Dose Coverage Statistics:")
    print(df1_mcv['COVERAGE'].describe())
    print(f"\n2nd Dose Coverage Statistics:")
    print(df2_mcv['COVERAGE'].describe())
    print(f"\nYear range (1st dose): {df1_mcv['YEAR'].min()} - {df1_mcv['YEAR'].max()}")
    print(f"Year range (2nd dose): {df2_mcv['YEAR'].min()} - {df2_mcv['YEAR'].max()}")
    print(f"Unique countries (1st dose): {df1_mcv['NAME'].nunique()}")
    print(f"Unique countries (2nd dose): {df2_mcv['NAME'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution of 1st dose coverage
    axes[0, 0].hist(df1_mcv['COVERAGE'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='blue')
    axes[0, 0].set_title('Distribution of 1st Dose Coverage (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Coverage (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Distribution of 2nd dose coverage
    axes[0, 1].hist(df2_mcv['COVERAGE'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[0, 1].set_title('Distribution of 2nd Dose Coverage (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Coverage (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Coverage over time (average)
    coverage1_by_year = df1_mcv.groupby('YEAR')['COVERAGE'].mean()
    coverage2_by_year = df2_mcv.groupby('YEAR')['COVERAGE'].mean()
    axes[1, 0].plot(coverage1_by_year.index, coverage1_by_year.values, linewidth=2, marker='o', label='1st Dose', color='blue')
    axes[1, 0].plot(coverage2_by_year.index, coverage2_by_year.values, linewidth=2, marker='s', label='2nd Dose', color='green')
    axes[1, 0].set_title('Average Vaccine Coverage Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Average Coverage (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Latest year coverage comparison
    latest_year = max(df1_mcv['YEAR'].max(), df2_mcv['YEAR'].max())
    latest_1st = df1_mcv[df1_mcv['YEAR'] == latest_year].nlargest(10, 'COVERAGE')
    latest_2nd = df2_mcv[df2_mcv['YEAR'] == latest_year].nlargest(10, 'COVERAGE')
    
    # Combine for comparison
    comparison = pd.merge(
        latest_1st[['NAME', 'COVERAGE']].rename(columns={'COVERAGE': '1st Dose'}),
        latest_2nd[['NAME', 'COVERAGE']].rename(columns={'COVERAGE': '2nd Dose'}),
        on='NAME', how='outer'
    ).fillna(0)
    
    x = np.arange(len(comparison.head(10)))
    width = 0.35
    axes[1, 1].bar(x - width/2, comparison.head(10)['1st Dose'], width, label='1st Dose', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, comparison.head(10)['2nd Dose'], width, label='2nd Dose', color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Country')
    axes[1, 1].set_ylabel('Coverage (%)')
    axes[1, 1].set_title(f'Top 10 Countries by Coverage ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(comparison.head(10)['NAME'], rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'vaccine_coverage_analysis.png')
    plt.close()
    
    return df1_mcv, df2_mcv

if __name__ == '__main__':
    df1, df2 = analyze_vaccine_coverage()

