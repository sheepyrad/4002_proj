"""
Analysis script for Birth Rate data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir, convert_wide_to_long

def load_data():
    """Load birth rate data"""
    df_wide = pd.read_csv(DATA_DIR / 'Birth_rate.csv')
    # Convert from wide to long format
    df = convert_wide_to_long(
        df_wide,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_name='Birth Rate'
    )
    df = df[['Entity', 'Code', 'Year', 'Birth Rate']].dropna(subset=['Birth Rate'])
    return df

def analyze_birth_rate():
    """Perform comprehensive analysis of birth rate data"""
    print("="*60)
    print("BIRTH RATE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    birth_rate_col = 'Birth Rate'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[birth_rate_col].describe())
    print(f"\nMissing values: {df[birth_rate_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[birth_rate_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 0].set_title('Distribution of Birth Rate', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Birth Rate (per 1,000 people)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global trend over time
    birth_rate_by_year = df.groupby('Year')[birth_rate_col].mean()
    axes[0, 1].plot(birth_rate_by_year.index, birth_rate_by_year.values, linewidth=2, marker='o', color='coral')
    axes[0, 1].set_title('Average Birth Rate Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Birth Rate (per 1,000)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Countries with highest birth rate (latest year)
    latest_year = df['Year'].max()
    latest_birth = df[df['Year'] == latest_year].nlargest(10, birth_rate_col)
    axes[1, 0].barh(latest_birth['Entity'], latest_birth[birth_rate_col], color='coral')
    axes[1, 0].set_title(f'Top 10 Countries by Birth Rate ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Birth Rate (per 1,000)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Countries with lowest birth rate
    lowest_birth = df[df['Year'] == latest_year].nsmallest(10, birth_rate_col)
    axes[1, 1].barh(lowest_birth['Entity'], lowest_birth[birth_rate_col], color='lightblue')
    axes[1, 1].set_title(f'Top 10 Countries with Lowest Birth Rate ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Birth Rate (per 1,000)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'birth_rate_analysis.png')
    plt.close()
    
    return df, birth_rate_col

if __name__ == '__main__':
    df, birth_rate_col = analyze_birth_rate()

