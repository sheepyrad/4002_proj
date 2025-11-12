"""
Analysis script for Population Density data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir, convert_wide_to_long

def load_data():
    """Load population density data"""
    df_wide = pd.read_csv(DATA_DIR / 'Pop_density.csv')
    # Convert from wide to long format
    df = convert_wide_to_long(
        df_wide,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_name='Population Density'
    )
    df = df[['Entity', 'Code', 'Year', 'Population Density']].dropna(subset=['Population Density'])
    return df

def analyze_pop_density():
    """Perform comprehensive analysis of population density data"""
    print("="*60)
    print("POPULATION DENSITY ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    pop_density_col = 'Population Density'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[pop_density_col].describe())
    print(f"\nMissing values: {df[pop_density_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[pop_density_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='forestgreen')
    axes[0, 0].set_title('Distribution of Population Density', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Population Density (people per sq. km)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global trend over time
    pop_density_by_year = df.groupby('Year')[pop_density_col].mean()
    axes[0, 1].plot(pop_density_by_year.index, pop_density_by_year.values, linewidth=2, marker='o', color='forestgreen')
    axes[0, 1].set_title('Average Population Density Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Population Density (people/sq. km)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Countries with highest population density (latest year)
    latest_year = df['Year'].max()
    latest_density = df[df['Year'] == latest_year].nlargest(10, pop_density_col)
    axes[1, 0].barh(latest_density['Entity'], latest_density[pop_density_col], color='forestgreen')
    axes[1, 0].set_title(f'Top 10 Countries by Population Density ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Population Density (people/sq. km)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Countries with lowest population density
    lowest_density = df[df['Year'] == latest_year].nsmallest(10, pop_density_col)
    axes[1, 1].barh(lowest_density['Entity'], lowest_density[pop_density_col], color='lightgreen')
    axes[1, 1].set_title(f'Top 10 Countries with Lowest Population Density ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Population Density (people/sq. km)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'pop_density_analysis.png')
    plt.close()
    
    return df, pop_density_col

if __name__ == '__main__':
    df, pop_density_col = analyze_pop_density()

