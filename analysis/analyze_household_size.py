"""
Analysis script for Household Size data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir, extract_year_from_date

def load_data():
    """Load household size data"""
    df = pd.read_csv(DATA_DIR / 'Household_size.csv')
    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.replace('\ufeff', '')
    # Remove empty columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.dropna(axis=1, how='all')
    
    # Parse reference date to extract year
    df['Year'] = df['Reference date (dd/mm/yyyy)'].apply(extract_year_from_date)
    
    # Get household size column
    hh_size_col = 'Average household size (number of members)'
    df[hh_size_col] = pd.to_numeric(df[hh_size_col], errors='coerce')
    
    return df

def analyze_household_size():
    """Perform comprehensive analysis of household size data"""
    print("="*60)
    print("HOUSEHOLD SIZE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    hh_size_col = 'Average household size (number of members)'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[hh_size_col].describe())
    print(f"\nMissing values: {df[hh_size_col].isnull().sum()}")
    print(f"Unique countries: {df['Country or area'].nunique()}")
    print(f"Data source categories: {df['Data source category'].value_counts()}")
    
    # Filter out invalid household sizes
    valid_data = df[df[hh_size_col].notna() & (df[hh_size_col] > 0) & (df[hh_size_col] < 20)]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of household sizes
    axes[0, 0].hist(valid_data[hh_size_col], bins=40, edgecolor='black', alpha=0.7, color='teal')
    axes[0, 0].set_title('Distribution of Average Household Size', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Average Household Size (members)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Household size over time (global average)
    if valid_data['Year'].notna().sum() > 0:
        hh_by_year = valid_data.groupby('Year')[hh_size_col].mean()
        axes[0, 1].plot(hh_by_year.index, hh_by_year.values, linewidth=2, marker='o', color='teal')
        axes[0, 1].set_title('Global Average Household Size Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Average Household Size')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Household size by data source
    source_means = valid_data.groupby('Data source category')[hh_size_col].mean().sort_values(ascending=False)
    axes[0, 2].barh(range(len(source_means)), source_means.values, color='teal')
    axes[0, 2].set_yticks(range(len(source_means)))
    axes[0, 2].set_yticklabels(source_means.index)
    axes[0, 2].set_title('Average Household Size by Data Source', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Average Household Size')
    axes[0, 2].grid(True, alpha=0.3, axis='x')
    
    # 4. Countries with largest household sizes (most recent data)
    if valid_data['Year'].notna().sum() > 0:
        latest_year_hh = valid_data['Year'].max()
        latest_hh_data = valid_data[valid_data['Year'] == latest_year_hh]
        latest_hh_data = latest_hh_data.sort_values('Year').drop_duplicates(subset='Country or area', keep='last')
        largest_hh = latest_hh_data.nlargest(10, hh_size_col)
        axes[1, 0].barh(largest_hh['Country or area'], largest_hh[hh_size_col], color='teal')
        axes[1, 0].set_title(f'Top 10 Countries by Household Size ({latest_year_hh})', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Average Household Size')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Countries with smallest household sizes
    if valid_data['Year'].notna().sum() > 0:
        smallest_hh = latest_hh_data.nsmallest(10, hh_size_col)
        axes[1, 1].barh(smallest_hh['Country or area'], smallest_hh[hh_size_col], color='lightblue')
        axes[1, 1].set_title(f'Top 10 Countries with Smallest Households ({latest_year_hh})', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Average Household Size')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Box plot by data source
    source_data = [valid_data[valid_data['Data source category'] == source][hh_size_col].dropna() 
                   for source in valid_data['Data source category'].value_counts().head(5).index]
    axes[1, 2].boxplot(source_data, labels=valid_data['Data source category'].value_counts().head(5).index)
    axes[1, 2].set_title('Household Size Distribution by Data Source', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Average Household Size')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'household_size_analysis.png')
    plt.close()
    
    return df, hh_size_col

if __name__ == '__main__':
    df, hh_size_col = analyze_household_size()

