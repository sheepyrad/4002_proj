"""
Analysis script for Vaccine Hesitancy data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """Load vaccine hesitancy data"""
    df = pd.read_csv(DATA_DIR / 'Vaccine_hesitancy.csv')
    return df

def analyze_vaccine_hesitancy():
    """Perform comprehensive analysis of vaccine hesitancy data"""
    print("="*60)
    print("VACCINE HESITANCY ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get vaccine column name
    vaccine_col = [col for col in df.columns if 'vaccine' in col.lower() or 'disagree' in col.lower()][0]
    print(f"\nVaccine hesitancy column: {vaccine_col}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[vaccine_col].describe())
    print(f"\nMissing values: {df[vaccine_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[vaccine_col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 0].set_title('Distribution of Vaccine Disagreement (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Share that Disagrees Vaccines are Effective (%)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Over time
    vaccine_by_year = df.groupby('Year')[vaccine_col].mean()
    axes[0, 1].plot(vaccine_by_year.index, vaccine_by_year.values, linewidth=2, marker='o', color='orange')
    axes[0, 1].set_title('Average Vaccine Disagreement Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Disagreement (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Countries with highest disagreement (latest year)
    latest_year = df['Year'].max()
    latest_vaccine = df[df['Year'] == latest_year].nlargest(10, vaccine_col)
    axes[1, 0].barh(latest_vaccine['Entity'], latest_vaccine[vaccine_col], color='orange')
    axes[1, 0].set_title(f'Top 10 Countries by Vaccine Disagreement ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Disagreement (%)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Countries with lowest disagreement
    lowest_vaccine = df[df['Year'] == latest_year].nsmallest(10, vaccine_col)
    axes[1, 1].barh(lowest_vaccine['Entity'], lowest_vaccine[vaccine_col], color='lightgreen')
    axes[1, 1].set_title(f'Top 10 Countries with Lowest Vaccine Disagreement ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Disagreement (%)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'vaccine_hesitancy_analysis.png')
    plt.close()
    
    return df, vaccine_col

if __name__ == '__main__':
    df, vaccine_col = analyze_vaccine_hesitancy()

