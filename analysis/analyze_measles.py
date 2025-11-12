"""
Analysis script for Measles Reporting Data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir

def load_data():
    """Load measles reporting data"""
    df = pd.read_csv(DATA_DIR / 'Measles_reporting_data.csv')
    # Clean column names (remove BOM if present)
    df.columns = df.columns.str.replace('\ufeff', '')
    return df

def analyze_measles():
    """Perform comprehensive analysis of measles reporting data"""
    print("="*60)
    print("MEASLES REPORTING DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get column names
    cases_col = 'Total confirmed  measles cases'
    incidence_col = 'Measles incidence rate per 1\'000\'000  total population'
    lab_col = 'Lab confirmed'
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df[cases_col].describe())
    print(f"\nMissing values:")
    print(df.isnull().sum())
    print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"Unique countries: {df['Member State'].nunique()}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of total confirmed cases
    axes[0, 0].hist(df[cases_col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='red')
    axes[0, 0].set_title('Distribution of Total Confirmed Measles Cases', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Total Confirmed Cases')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Measles cases over time (global)
    cases_by_year = df.groupby('Year')[cases_col].sum()
    axes[0, 1].plot(cases_by_year.index, cases_by_year.values, linewidth=2, marker='o', color='red')
    axes[0, 1].set_title('Global Measles Cases Over Time', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('Total Cases')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Incidence rate over time (average)
    incidence_by_year = df.groupby('Year')[incidence_col].mean()
    axes[0, 2].plot(incidence_by_year.index, incidence_by_year.values, linewidth=2, marker='o', color='orange')
    axes[0, 2].set_title('Average Measles Incidence Rate Over Time', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Year')
    axes[0, 2].set_ylabel('Incidence Rate (per 1M population)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Top 10 countries by total cases (latest year)
    latest_year = df['Year'].max()
    latest_measles = df[df['Year'] == latest_year].nlargest(10, cases_col)
    axes[1, 0].barh(latest_measles['Member State'], latest_measles[cases_col], color='red')
    axes[1, 0].set_title(f'Top 10 Countries by Measles Cases ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Total Cases')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Top 10 countries by incidence rate (latest year)
    latest_incidence = df[df['Year'] == latest_year].nlargest(10, incidence_col)
    axes[1, 1].barh(latest_incidence['Member State'], latest_incidence[incidence_col], color='orange')
    axes[1, 1].set_title(f'Top 10 Countries by Incidence Rate ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Incidence Rate (per 1M)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # 6. Cases by region (latest year)
    region_cases = df[df['Year'] == latest_year].groupby('Region')[cases_col].sum().sort_values(ascending=False)
    axes[1, 2].bar(range(len(region_cases)), region_cases.values, color='crimson')
    axes[1, 2].set_xticks(range(len(region_cases)))
    axes[1, 2].set_xticklabels(region_cases.index, rotation=45, ha='right')
    axes[1, 2].set_title(f'Measles Cases by Region ({latest_year})', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('Total Cases')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'measles_analysis.png')
    plt.close()
    
    # Additional analysis: Confirmation types breakdown
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Lab confirmed vs other types over time
    lab_by_year = df.groupby('Year')[lab_col].sum()
    epi_by_year = df.groupby('Year')['Epidemiologically linked'].sum()
    clinical_by_year = df.groupby('Year')['Clinically compatible'].sum()
    
    axes[0].plot(lab_by_year.index, lab_by_year.values, label='Lab Confirmed', linewidth=2, marker='o')
    axes[0].plot(epi_by_year.index, epi_by_year.values, label='Epidemiologically Linked', linewidth=2, marker='s')
    axes[0].plot(clinical_by_year.index, clinical_by_year.values, label='Clinically Compatible', linewidth=2, marker='^')
    axes[0].set_title('Measles Cases by Confirmation Type Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Number of Cases')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap: Cases by region and year (recent years)
    recent_years = sorted(df['Year'].unique())[-10:]  # Last 10 years
    region_year = df[df['Year'].isin(recent_years)].pivot_table(
        values=cases_col, index='Region', columns='Year', aggfunc='sum', fill_value=0
    )
    sns.heatmap(region_year, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Total Cases'})
    axes[1].set_title('Measles Cases Heatmap: Region vs Year', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Region')
    
    plt.tight_layout()
    save_figure(fig, 'measles_detailed_analysis.png')
    plt.close()
    
    return df, cases_col, incidence_col

if __name__ == '__main__':
    df, cases_col, incidence_col = analyze_measles()

