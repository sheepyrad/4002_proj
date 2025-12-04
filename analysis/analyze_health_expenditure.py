"""
Analysis script for Healthcare Spending data (Per Capita, PPP)

Updated to use healthcare_spending.csv which contains:
- Current health expenditure per capita, PPP (current international $)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """
    Load healthcare spending data (per capita, PPP).
    
    Returns DataFrame with columns: Entity, Code, Year, and health expenditure per capita.
    """
    df = pd.read_csv(DATA_DIR / 'healthcare_spending.csv')
    return df

def analyze_health_expenditure():
    """Perform comprehensive analysis of health expenditure data"""
    print("="*60)
    print("HEALTHCARE SPENDING ANALYSIS (Per Capita, PPP)")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get health expenditure column name
    health_col = [col for col in df.columns if 'health' in col.lower() or 'expenditure' in col.lower()][0]
    print(f"\nHealth expenditure column: {health_col}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[health_col].describe())
    print(f"\nMissing values: {df[health_col].isnull().sum()}")
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Entity'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution
    axes[0, 0].hist(df[health_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[0, 0].set_title('Distribution of Health Expenditure (Per Capita, PPP $)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Health Expenditure ($ per capita)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Over time
    health_by_year = df.groupby('Year')[health_col].mean()
    axes[0, 1].plot(health_by_year.index, health_by_year.values, linewidth=2, marker='o', color='green')
    axes[0, 1].set_title('Average Health Expenditure Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Average Health Expenditure ($ per capita)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top countries by latest health expenditure
    latest_year = df['Year'].max()
    latest_health = df[df['Year'] == latest_year].nlargest(10, health_col)
    axes[1, 0].barh(latest_health['Entity'], latest_health[health_col], color='green')
    axes[1, 0].set_title(f'Top 10 Countries by Health Expenditure ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Health Expenditure ($ per capita)')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Box plot by decade
    df['Decade'] = (df['Year'] // 10) * 10
    recent_data = df[df['Decade'] >= 2000]
    axes[1, 1].boxplot([recent_data[recent_data['Decade'] == d][health_col].dropna() 
                        for d in sorted(recent_data['Decade'].unique())],
                       labels=sorted(recent_data['Decade'].unique()))
    axes[1, 1].set_title('Health Expenditure Distribution by Decade', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Decade')
    axes[1, 1].set_ylabel('Health Expenditure ($ per capita)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'health_expenditure_analysis.png')
    plt.close()
    
    return df, health_col

if __name__ == '__main__':
    df, health_col = analyze_health_expenditure()

