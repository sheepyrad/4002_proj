"""
Analysis script for Net Migration data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir, convert_wide_to_long

def load_data():
    """Load net migration data"""
    df_wide = pd.read_csv(DATA_DIR / 'Net_migration.csv')
    # Convert from wide to long format
    df = convert_wide_to_long(
        df_wide,
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        value_name='Net Migration'
    )
    df = df[['Entity', 'Code', 'Year', 'Net Migration']].dropna(subset=['Net Migration'])
    return df

def analyze_net_migration():
    """Perform comprehensive analysis of net migration data"""
    print("="*60)
    print("NET MIGRATION ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    net_migration_col = 'Net Migration'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[net_migration_col].describe())
    print(f"\nMissing values: {df[net_migration_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[net_migration_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Distribution of Net Migration', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Net Migration')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global trend over time
    net_migration_by_year = df.groupby('Year')[net_migration_col].mean()
    axes[0, 1].plot(net_migration_by_year.index, net_migration_by_year.values, linewidth=2, marker='o', color='steelblue')
    axes[0, 1].set_title('Average Net Migration Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Net Migration')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Countries with highest net migration (latest year)
    latest_year = df['Year'].max()
    latest_migration = df[df['Year'] == latest_year].nlargest(10, net_migration_col)
    axes[1, 0].barh(latest_migration['Entity'], latest_migration[net_migration_col], color='steelblue')
    axes[1, 0].set_title(f'Top 10 Countries by Net Migration ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Net Migration')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Countries with lowest net migration (most negative)
    lowest_migration = df[df['Year'] == latest_year].nsmallest(10, net_migration_col)
    axes[1, 1].barh(lowest_migration['Entity'], lowest_migration[net_migration_col], color='crimson')
    axes[1, 1].set_title(f'Top 10 Countries with Lowest Net Migration ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Net Migration')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'net_migration_analysis.png')
    plt.close()
    
    return df, net_migration_col

if __name__ == '__main__':
    df, net_migration_col = analyze_net_migration()

