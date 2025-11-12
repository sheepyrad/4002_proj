"""
Analysis script for Political Stability Index data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir, convert_wide_to_long

def load_data():
    """Load political stability index data"""
    df_wide = pd.read_csv(DATA_DIR / 'Political_Stability_idx.csv')
    # Convert from wide to long format
    df = convert_wide_to_long(
        df_wide,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_name='Political Stability Index'
    )
    df = df[['Entity', 'Code', 'Year', 'Political Stability Index']].dropna(subset=['Political Stability Index'])
    return df

def analyze_political_stability():
    """Perform comprehensive analysis of political stability index data"""
    print("="*60)
    print("POLITICAL STABILITY INDEX ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    political_stability_col = 'Political Stability Index'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[political_stability_col].describe())
    print(f"\nMissing values: {df[political_stability_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[political_stability_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='darkviolet')
    axes[0, 0].set_title('Distribution of Political Stability Index', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Political Stability Index (Percentile Rank)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global trend over time
    political_stability_by_year = df.groupby('Year')[political_stability_col].mean()
    axes[0, 1].plot(political_stability_by_year.index, political_stability_by_year.values, linewidth=2, marker='o', color='darkviolet')
    axes[0, 1].set_title('Average Political Stability Index Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Political Stability Index')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Countries with highest political stability (latest year)
    latest_year = df['Year'].max()
    latest_stability = df[df['Year'] == latest_year].nlargest(10, political_stability_col)
    axes[1, 0].barh(latest_stability['Entity'], latest_stability[political_stability_col], color='darkviolet')
    axes[1, 0].set_title(f'Top 10 Countries by Political Stability ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Political Stability Index')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Countries with lowest political stability
    lowest_stability = df[df['Year'] == latest_year].nsmallest(10, political_stability_col)
    axes[1, 1].barh(lowest_stability['Entity'], lowest_stability[political_stability_col], color='maroon')
    axes[1, 1].set_title(f'Top 10 Countries with Lowest Political Stability ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Political Stability Index')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'political_stability_analysis.png')
    plt.close()
    
    return df, political_stability_col

if __name__ == '__main__':
    df, political_stability_col = analyze_political_stability()

