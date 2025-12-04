"""
Data preparation for mixed-effects negative binomial analysis
Merges all datasets into a panel format suitable for mixed-effects modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils import DATA_DIR
from analyze_measles import load_data as load_measles
from analyze_gdp import load_data as load_gdp
from analyze_health_expenditure import load_data as load_health_exp
from analyze_birth_rate import load_data as load_birth_rate
from analyze_pop_density import load_data as load_pop_density
from analyze_net_migration import load_data as load_net_migration
from analyze_political_stability import load_data as load_political_stability
from analyze_vaccine_coverage import load_data as load_vaccine_coverage


def load_income_groups():
    """Load income group classifications"""
    income_df = pd.read_csv(DATA_DIR / 'incomegroups.csv')
    income_map = dict(zip(income_df['Country.Code'], income_df['IncomeGroup']))
    return income_map


def prepare_panel_data():
    """
    Prepare panel dataset for mixed-effects analysis
    
    Returns:
    --------
    panel_df : DataFrame
        Panel dataset with columns:
        - Code: ISO country code
        - Year: Year
        - Country: Country name
        - MeaslesCases: Total confirmed measles cases
        - MeaslesIncidence: Measles incidence rate per 1M population
        - Population: Total population
        - MCV1: First dose vaccine coverage (%)
        - MCV2: Second dose vaccine coverage (%)
        - GDPpc: GDP per capita
        - HealthExpPC: Health expenditure per capita (PPP $)
        - BirthRate: Birth rate
        - PopDensity: Population density
        - NetMigration: Net migration
        - PolStability: Political stability index
        - HIC: High-income country indicator (1 if High income, 0 otherwise)
    """
    print("="*60)
    print("PREPARING PANEL DATA FOR MIXED-EFFECTS ANALYSIS")
    print("="*60)
    
    # Load measles data
    print("\n1. Loading measles data...")
    measles_df = load_measles()
    measles_df.columns = measles_df.columns.str.replace('\ufeff', '')
    
    cases_col = 'Total confirmed  measles cases'
    incidence_col = 'Measles incidence rate per 1\'000\'000  total population'
    pop_col = 'Total population'
    
    # Prepare measles panel
    measles_panel = measles_df[[
        'ISO country code', 'Year', 'Member State', 
        cases_col, incidence_col, pop_col
    ]].copy()
    measles_panel = measles_panel.rename(columns={
        'ISO country code': 'Code',
        'Member State': 'Country',
        cases_col: 'MeaslesCases',
        incidence_col: 'MeaslesIncidence',
        pop_col: 'Population'
    })
    
    # Convert incidence to rate per 1M (already in this format, but ensure numeric)
    measles_panel['MeaslesIncidence'] = pd.to_numeric(
        measles_panel['MeaslesIncidence'], errors='coerce'
    )
    measles_panel['MeaslesCases'] = pd.to_numeric(
        measles_panel['MeaslesCases'], errors='coerce'
    )
    measles_panel['Population'] = pd.to_numeric(
        measles_panel['Population'], errors='coerce'
    )
    
    print(f"   Loaded {len(measles_panel)} observations")
    print(f"   Countries: {measles_panel['Code'].nunique()}")
    print(f"   Years: {measles_panel['Year'].min()} - {measles_panel['Year'].max()}")
    
    # Load vaccine coverage data
    print("\n2. Loading vaccine coverage data...")
    vaccine_coverage_1st, vaccine_coverage_2nd = load_vaccine_coverage()
    
    # Prepare vaccine coverage panels
    mcv1_panel = vaccine_coverage_1st[
        vaccine_coverage_1st['ANTIGEN'] == 'MCV1'
    ][['CODE', 'YEAR', 'COVERAGE']].copy()
    mcv1_panel = mcv1_panel.rename(columns={
        'CODE': 'Code',
        'YEAR': 'Year',
        'COVERAGE': 'MCV1'
    })
    mcv1_panel['MCV1'] = pd.to_numeric(mcv1_panel['MCV1'], errors='coerce')
    
    mcv2_panel = vaccine_coverage_2nd[
        vaccine_coverage_2nd['ANTIGEN'] == 'MCV2'
    ][['CODE', 'YEAR', 'COVERAGE']].copy()
    mcv2_panel = mcv2_panel.rename(columns={
        'CODE': 'Code',
        'YEAR': 'Year',
        'COVERAGE': 'MCV2'
    })
    mcv2_panel['MCV2'] = pd.to_numeric(mcv2_panel['MCV2'], errors='coerce')
    
    print(f"   MCV1: {len(mcv1_panel)} observations")
    print(f"   MCV2: {len(mcv2_panel)} observations")
    
    # Merge vaccine coverage
    panel_df = measles_panel.merge(
        mcv1_panel, on=['Code', 'Year'], how='left'
    )
    panel_df = panel_df.merge(
        mcv2_panel, on=['Code', 'Year'], how='left'
    )
    
    # Load socioeconomic data
    print("\n3. Loading socioeconomic data...")
    
    # GDP per capita
    gdp_df = load_gdp()
    gdp_col = [col for col in gdp_df.columns if 'GDP' in col][0]
    gdp_panel = gdp_df[['Code', 'Year', gdp_col]].copy()
    gdp_panel = gdp_panel.rename(columns={gdp_col: 'GDPpc'})
    gdp_panel['GDPpc'] = pd.to_numeric(gdp_panel['GDPpc'], errors='coerce')
    panel_df = panel_df.merge(gdp_panel, on=['Code', 'Year'], how='left')
    
    # Health expenditure (per capita, PPP $)
    health_df = load_health_exp()
    health_col = [col for col in health_df.columns 
                  if 'health' in col.lower() or 'expenditure' in col.lower()][0]
    health_panel = health_df[['Code', 'Year', health_col]].copy()
    health_panel = health_panel.rename(columns={health_col: 'HealthExpPC'})
    health_panel['HealthExpPC'] = pd.to_numeric(
        health_panel['HealthExpPC'], errors='coerce'
    )
    panel_df = panel_df.merge(health_panel, on=['Code', 'Year'], how='left')
    
    # Birth rate
    birth_rate_df = load_birth_rate()
    birth_panel = birth_rate_df[['Code', 'Year', 'Birth Rate']].copy()
    birth_panel = birth_panel.rename(columns={'Birth Rate': 'BirthRate'})
    birth_panel['BirthRate'] = pd.to_numeric(birth_panel['BirthRate'], errors='coerce')
    panel_df = panel_df.merge(birth_panel, on=['Code', 'Year'], how='left')
    
    # Population density
    pop_density_df = load_pop_density()
    pop_dens_panel = pop_density_df[['Code', 'Year', 'Population Density']].copy()
    pop_dens_panel = pop_dens_panel.rename(columns={'Population Density': 'PopDensity'})
    pop_dens_panel['PopDensity'] = pd.to_numeric(
        pop_dens_panel['PopDensity'], errors='coerce'
    )
    panel_df = panel_df.merge(pop_dens_panel, on=['Code', 'Year'], how='left')
    
    # Net migration
    net_migration_df = load_net_migration()
    mig_panel = net_migration_df[['Code', 'Year', 'Net Migration']].copy()
    mig_panel = mig_panel.rename(columns={'Net Migration': 'NetMigration'})
    mig_panel['NetMigration'] = pd.to_numeric(mig_panel['NetMigration'], errors='coerce')
    panel_df = panel_df.merge(mig_panel, on=['Code', 'Year'], how='left')
    
    # Political stability
    pol_stab_df = load_political_stability()
    pol_panel = pol_stab_df[['Code', 'Year', 'Political Stability Index']].copy()
    pol_panel = pol_panel.rename(columns={'Political Stability Index': 'PolStability'})
    pol_panel['PolStability'] = pd.to_numeric(pol_panel['PolStability'], errors='coerce')
    panel_df = panel_df.merge(pol_panel, on=['Code', 'Year'], how='left')
    
    # Load income groups
    print("\n4. Adding income group classifications...")
    income_map = load_income_groups()
    panel_df['IncomeGroup'] = panel_df['Code'].map(income_map)
    panel_df['HIC'] = (panel_df['IncomeGroup'] == 'High income').astype(int)
    
    print(f"   High-income countries: {panel_df['HIC'].sum()} observations")
    print(f"   Low/middle-income countries: {(panel_df['HIC'] == 0).sum()} observations")
    
    # Sort by country and year
    panel_df = panel_df.sort_values(['Code', 'Year']).reset_index(drop=True)
    
    # Create log-transformed variables for analysis
    panel_df['LogIncidence'] = np.log1p(panel_df['MeaslesIncidence'])  # log(1 + x) to handle zeros
    panel_df['LogGDPpc'] = np.log1p(panel_df['GDPpc'])
    panel_df['LogPopDensity'] = np.log1p(panel_df['PopDensity'])
    panel_df['LogHealthExpPC'] = np.log1p(panel_df['HealthExpPC'])  # Log of health expenditure per capita
    
    # Summary statistics
    print("\n5. Data summary:")
    print(f"   Total observations: {len(panel_df)}")
    print(f"   Countries: {panel_df['Code'].nunique()}")
    print(f"   Years: {panel_df['Year'].min()} - {panel_df['Year'].max()}")
    print(f"\n   Missing values:")
    missing = panel_df.isnull().sum()
    missing = missing[missing > 0]
    for col, count in missing.items():
        pct = 100 * count / len(panel_df)
        print(f"     {col}: {count} ({pct:.1f}%)")
    
    print(f"\n   Variable ranges:")
    numeric_cols = ['MeaslesIncidence', 'MCV1', 'MCV2', 'GDPpc', 
                    'HealthExpPC', 'BirthRate', 'PopDensity', 
                    'NetMigration', 'PolStability']
    for col in numeric_cols:
        if col in panel_df.columns:
            valid = panel_df[col].dropna()
            if len(valid) > 0:
                print(f"     {col}: [{valid.min():.2f}, {valid.max():.2f}]")
    
    return panel_df


def save_panel_data(panel_df, output_path=None):
    """Save prepared panel data to CSV"""
    if output_path is None:
        output_path = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'panel_data.csv'
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False)
    print(f"\nPanel data saved to: {output_path}")
    return output_path


if __name__ == '__main__':
    panel_df = prepare_panel_data()
    save_panel_data(panel_df)

