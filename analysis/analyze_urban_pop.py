"""
Analysis script for Urban Population data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """Load urban population data"""
    df = pd.read_csv(DATA_DIR / 'Urban_pop.csv')
    return df

def analyze_urban_pop():
    """Perform comprehensive analysis of urban population data"""
    print("="*60)
    print("URBAN POPULATION ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get urban population column name
    urban_col = [col for col in df.columns if 'urban' in col.lower()][0]
    print(f"\nUrban population column: {urban_col}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[urban_col].describe())
    print(f"\nMissing values: {df[urban_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[urban_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='purple')
    axes[0, 0].set_title('Distribution of Urban Population (%)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Urban Population (% of total)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Global trend over time
    urban_by_year = df.groupby('Year')[urban_col].mean()
    axes[0, 1].plot(urban_by_year.index, urban_by_year.values, linewidth=2, marker='o', color='purple')
    axes[0, 1].set_title('Average Urban Population Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Urban Population (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Most urbanized countries (latest year)
    latest_year = df['Year'].max()
    latest_urban = df[df['Year'] == latest_year].nlargest(10, urban_col)
    axes[1, 0].barh(latest_urban['Entity'], latest_urban[urban_col], color='purple')
    axes[1, 0].set_title(f'Top 10 Most Urbanized Countries ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Urban Population (%)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Least urbanized countries
    lowest_urban = df[df['Year'] == latest_year].nsmallest(10, urban_col)
    axes[1, 1].barh(lowest_urban['Entity'], lowest_urban[urban_col], color='brown')
    axes[1, 1].set_title(f'Top 10 Least Urbanized Countries ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Urban Population (%)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    save_figure(fig, 'urban_pop_analysis.png')
    plt.close()
    
    return df, urban_col

if __name__ == '__main__':
    df, urban_col = analyze_urban_pop()

