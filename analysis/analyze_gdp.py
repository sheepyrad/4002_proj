"""
Analysis script for GDP per capita data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """Load GDP per capita data"""
    df = pd.read_csv(DATA_DIR / 'GDP_per_cap.csv')
    return df

def analyze_gdp():
    """Perform comprehensive analysis of GDP per capita data"""
    print("="*60)
    print("GDP PER CAPITA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get GDP column name
    gdp_col = [col for col in df.columns if 'GDP' in col][0]
    print(f"\nGDP column: {gdp_col}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[gdp_col].describe())
    print(f"\nMissing values: {df[gdp_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution of GDP per capita
    axes[0, 0].hist(df[gdp_col].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of GDP per Capita', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('GDP per Capita (PPP, constant 2021 international $)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log scale distribution
    axes[0, 1].hist(np.log10(df[gdp_col].dropna()), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of GDP per Capita (Log Scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Log10(GDP per Capita)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # GDP over time (average across all countries)
    gdp_by_year = df.groupby('Year')[gdp_col].mean()
    axes[1, 0].plot(gdp_by_year.index, gdp_by_year.values, linewidth=2, marker='o')
    axes[1, 0].set_title('Average GDP per Capita Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Average GDP per Capita')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Top 10 countries by latest GDP
    latest_year = df['Year'].max()
    latest_gdp = df[df['Year'] == latest_year].nlargest(10, gdp_col)
    axes[1, 1].barh(latest_gdp['Entity'], latest_gdp[gdp_col])
    axes[1, 1].set_title(f'Top 10 Countries by GDP per Capita ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('GDP per Capita')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'gdp_analysis.png')
    plt.close()
    
    return df, gdp_col

if __name__ == '__main__':
    df, gdp_col = analyze_gdp()

