"""
Data Preparation with KNN Imputation

This module prepares data for causal analysis by:
1. Loading and merging additional variables (Urban pop, Household size, Vaccine hesitancy)
2. Imputing missing values using K-Nearest Neighbors (KNN) imputation
3. Standardizing continuous variables
4. Creating analysis-ready datasets with more countries

Imputation Strategy:
- Use KNN imputation which finds similar countries based on available data
- Better preserves variance and distribution compared to median imputation
- Uses income group and year as auxiliary features for better matching
- Track imputation rates for transparency
- Exclude years with excessive missing data (e.g., 2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import re

DATA_DIR = Path(__file__).parent.parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis'

# Maximum year to include in analysis (2025 has 100% missing data)
MAX_ANALYSIS_YEAR = 2024

# KNN imputation parameters
KNN_NEIGHBORS = 5
KNN_WEIGHTS = 'distance'  # Weight by distance (closer neighbors have more influence)


def load_panel_data(filter_year=True):
    """
    Load the prepared panel data.
    
    Parameters:
    -----------
    filter_year : bool
        If True, exclude years beyond MAX_ANALYSIS_YEAR (2025 has 100% missing data)
    """
    data_path = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'panel_data.csv'
    df = pd.read_csv(data_path)
    
    if filter_year:
        original_years = df['Year'].nunique()
        original_rows = len(df)
        df = df[df['Year'] <= MAX_ANALYSIS_YEAR]
        filtered_years = df['Year'].nunique()
        filtered_rows = len(df)
        print(f"\n[DATA FILTER] Excluded years > {MAX_ANALYSIS_YEAR}")
        print(f"  Years: {original_years} -> {filtered_years} (removed {original_years - filtered_years})")
        print(f"  Rows: {original_rows} -> {filtered_rows} (removed {original_rows - filtered_rows})")
    
    return df


def load_urban_population():
    """Load urban population percentage data"""
    urban_df = pd.read_csv(DATA_DIR / 'Urban_pop.csv')
    urban_df = urban_df.rename(columns={
        'Code': 'Code',
        'Year': 'Year',
        'Urban population (% of total population)': 'UrbanPop'
    })
    urban_df = urban_df[['Code', 'Year', 'UrbanPop']].dropna()
    urban_df['UrbanPop'] = pd.to_numeric(urban_df['UrbanPop'], errors='coerce')
    return urban_df


def load_household_size():
    """
    Load household size data.
    Data is sparse (survey-based), so we'll use latest available per country.
    """
    household_df = pd.read_csv(DATA_DIR / 'Household_size.csv')
    
    # Parse the reference date to get year
    def extract_year(date_str):
        if pd.isna(date_str):
            return None
        # Try to find 4-digit year
        match = re.search(r'(\d{4})', str(date_str))
        if match:
            return int(match.group(1))
        # Try dd/mm/yy format
        match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', str(date_str))
        if match:
            year = int(match.group(3))
            if year < 100:
                year = 2000 + year if year < 50 else 1900 + year
            return year
        return None
    
    household_df['Year'] = household_df['Reference date (dd/mm/yyyy)'].apply(extract_year)
    household_df = household_df.rename(columns={
        'ISO Code': 'ISOCode',
        'Average household size (number of members)': 'HouseholdSize'
    })
    
    # Get ISO3 code mapping (ISO Code is numeric, need to convert)
    # We'll merge by country name instead
    household_df['HouseholdSize'] = pd.to_numeric(household_df['HouseholdSize'], errors='coerce')
    household_df = household_df[['Country or area', 'Year', 'HouseholdSize']].dropna()
    household_df = household_df.rename(columns={'Country or area': 'Country'})
    
    # Get most recent value per country
    household_latest = household_df.sort_values('Year', ascending=False).groupby('Country').first().reset_index()
    
    return household_latest[['Country', 'HouseholdSize']]


def load_vaccine_hesitancy():
    """
    Load vaccine hesitancy data.
    Data is sparse - will use income group imputation for missing.
    """
    hesitancy_df = pd.read_csv(DATA_DIR / 'Vaccine_hesitancy.csv')
    hesitancy_df = hesitancy_df.rename(columns={
        'Code': 'Code',
        'Year': 'Year',
        'Share that disagrees vaccines are effective': 'VaccineHesitancy'
    })
    hesitancy_df = hesitancy_df[['Code', 'Year', 'VaccineHesitancy']].dropna()
    hesitancy_df['VaccineHesitancy'] = pd.to_numeric(hesitancy_df['VaccineHesitancy'], errors='coerce')
    return hesitancy_df


def merge_additional_variables(df):
    """
    Merge additional variables into the panel data.
    """
    print("\n--- MERGING ADDITIONAL VARIABLES ---")
    
    # 1. Urban Population
    urban_df = load_urban_population()
    df = df.merge(urban_df, on=['Code', 'Year'], how='left')
    n_urban = df['UrbanPop'].notna().sum()
    print(f"  Urban Population: {n_urban}/{len(df)} observations matched")
    
    # 2. Household Size (by country name, latest available)
    household_df = load_household_size()
    # Need to match by country name
    df = df.merge(household_df, on='Country', how='left')
    n_household = df['HouseholdSize'].notna().sum()
    print(f"  Household Size: {n_household}/{len(df)} observations matched (latest per country)")
    
    # 3. Vaccine Hesitancy (sparse)
    hesitancy_df = load_vaccine_hesitancy()
    df = df.merge(hesitancy_df, on=['Code', 'Year'], how='left')
    n_hesitancy = df['VaccineHesitancy'].notna().sum()
    print(f"  Vaccine Hesitancy: {n_hesitancy}/{len(df)} observations matched (sparse data)")
    
    return df


def log_missing_data_by_year_and_income(df, variables, verbose=True):
    """
    Log detailed missing data statistics by year and income group.
    
    Parameters:
    -----------
    df : DataFrame
        Panel data with 'Year' and 'IncomeGroup' columns
    variables : list
        Variables to check for missing data
    verbose : bool
        Print detailed statistics
    
    Returns:
    --------
    dict with missing data statistics
    """
    stats = {
        'by_year': {},
        'by_income_group': {},
        'by_year_income': {},
        'by_variable_year': {}
    }
    
    years = sorted(df['Year'].unique())
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    
    if verbose:
        print("\n" + "="*80)
        print("MISSING DATA ANALYSIS (BEFORE IMPUTATION)")
        print("="*80)
    
    # --- By Year ---
    if verbose:
        print("\n" + "-"*80)
        print("MISSING DATA BY YEAR")
        print("-"*80)
        print(f"{'Year':<6} {'Total':<6} " + " ".join([f"{v:<12}" for v in variables[:6]]))
        print("-"*80)
    
    for year in years:
        year_df = df[df['Year'] == year]
        total = len(year_df)
        year_stats = {'total': total}
        
        row_str = f"{year:<6} {total:<6} "
        for var in variables:
            if var in df.columns:
                missing = year_df[var].isna().sum()
                pct = 100 * missing / total if total > 0 else 0
                year_stats[var] = {'missing': missing, 'total': total, 'pct': pct}
                row_str += f"{missing}/{total} ({pct:4.1f}%) "
        
        stats['by_year'][year] = year_stats
        if verbose and len(variables) <= 6:
            print(row_str)
    
    if verbose:
        print("\n" + "-"*80)
        print("MISSING DATA BY INCOME GROUP")
        print("-"*80)
    
    # --- By Income Group ---
    for ig in income_groups:
        ig_df = df[df['IncomeGroup'] == ig]
        total = len(ig_df)
        if total == 0:
            continue
        
        ig_stats = {'total': total, 'n_countries': ig_df['Code'].nunique()}
        
        if verbose:
            print(f"\n{ig} ({ig_stats['n_countries']} countries, {total} obs):")
        
        for var in variables:
            if var in df.columns:
                missing = ig_df[var].isna().sum()
                pct = 100 * missing / total if total > 0 else 0
                ig_stats[var] = {'missing': missing, 'total': total, 'pct': pct}
                if verbose:
                    print(f"  {var:<20}: {missing:4d}/{total:4d} missing ({pct:5.1f}%)")
        
        stats['by_income_group'][ig] = ig_stats
    
    # --- By Variable and Year (for CSV export) ---
    for var in variables:
        if var not in df.columns:
            continue
        stats['by_variable_year'][var] = {}
        for year in years:
            year_df = df[df['Year'] == year]
            total = len(year_df)
            missing = year_df[var].isna().sum()
            pct = 100 * missing / total if total > 0 else 0
            stats['by_variable_year'][var][year] = {'missing': missing, 'total': total, 'pct': pct}
    
    return stats


def impute_by_income_group(df, var, method='median'):
    """
    Impute missing values using income group statistics.
    (Legacy function - kept for comparison, use knn_impute_all instead)
    
    Parameters:
    -----------
    df : DataFrame
        Panel data with 'IncomeGroup' column
    var : str
        Variable to impute
    method : str
        'median' or 'mean'
    
    Returns:
    --------
    Series with imputed values
    """
    result = df[var].copy()
    
    # Calculate income group statistics
    if method == 'median':
        group_stats = df.groupby('IncomeGroup')[var].median()
    else:
        group_stats = df.groupby('IncomeGroup')[var].mean()
    
    # Global fallback
    global_stat = df[var].median() if method == 'median' else df[var].mean()
    
    # Impute by income group
    for idx in df[df[var].isna()].index:
        income_group = df.loc[idx, 'IncomeGroup']
        if pd.notna(income_group) and income_group in group_stats.index and pd.notna(group_stats[income_group]):
            result.loc[idx] = group_stats[income_group]
        else:
            result.loc[idx] = global_stat
    
    return result


def knn_impute_all(df, impute_vars, n_neighbors=KNN_NEIGHBORS, weights=KNN_WEIGHTS):
    """
    Impute all variables using K-Nearest Neighbors.
    
    KNN imputation finds similar observations based on non-missing features
    and uses their values to impute missing ones. This better preserves
    the variance and relationships between variables.
    
    Parameters:
    -----------
    df : DataFrame
        Panel data with missing values
    impute_vars : list
        Variables to impute
    n_neighbors : int
        Number of neighbors to use (default: 5)
    weights : str
        'uniform' or 'distance' weighting
    
    Returns:
    --------
    DataFrame with imputed values, dict of imputation statistics
    """
    print(f"\n--- KNN Imputation (k={n_neighbors}, weights={weights}) ---")
    
    # Filter to existing columns
    impute_vars = [v for v in impute_vars if v in df.columns]
    
    # Auxiliary variables that help find similar countries
    aux_vars = ['Year', 'HIC']
    aux_vars = [v for v in aux_vars if v in df.columns]
    
    all_vars = impute_vars + aux_vars
    
    # Track missing before imputation
    imputation_stats = {}
    for var in impute_vars:
        n_missing = df[var].isna().sum()
        imputation_stats[var] = {
            'original_missing': n_missing,
            'original_pct': 100 * n_missing / len(df)
        }
    
    # Prepare data matrix
    X = df[all_vars].copy()
    
    # Standardize for KNN (important for distance calculation)
    # First, fill NaN with median temporarily to fit scaler
    X_for_scaling = X.copy()
    for col in X_for_scaling.columns:
        median_val = X_for_scaling[col].median()
        if pd.isna(median_val):
            median_val = 0
        X_for_scaling[col] = X_for_scaling[col].fillna(median_val)
    
    scaler = StandardScaler()
    scaler.fit(X_for_scaling)
    
    # Scale the data manually (preserving NaN)
    X_scaled = X.copy()
    for i, col in enumerate(all_vars):
        col_mean = scaler.mean_[i]
        col_std = scaler.scale_[i] if scaler.scale_[i] > 0 else 1  # Avoid division by zero
        X_scaled[col] = (X[col] - col_mean) / col_std
    
    # KNN imputation on scaled data
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_imputed_scaled = imputer.fit_transform(X_scaled.values)
    
    # Inverse transform to original scale
    X_imputed = pd.DataFrame(X_imputed_scaled, columns=all_vars, index=df.index)
    for i, col in enumerate(all_vars):
        col_mean = scaler.mean_[i]
        col_std = scaler.scale_[i] if scaler.scale_[i] > 0 else 1
        X_imputed[col] = X_imputed[col] * col_std + col_mean
    
    # Apply results to dataframe
    df_result = df.copy()
    for var in impute_vars:
        # Only fill missing values
        missing_mask = df[var].isna()
        df_result.loc[missing_mask, var] = X_imputed.loc[missing_mask, var]
        
        # Clip to valid ranges for percentage variables
        if var in ['MCV1', 'MCV2', 'VaccineHesitancy', 'UrbanPop']:
            df_result[var] = df_result[var].clip(lower=0, upper=100)
        elif var in ['GDPpc', 'HealthExpPC', 'PopDensity', 'HouseholdSize']:
            df_result[var] = df_result[var].clip(lower=0)
        
        # Track imputation by income group
        imputation_stats[var]['by_income_group'] = {}
        for ig in df['IncomeGroup'].dropna().unique():
            ig_mask = (df['IncomeGroup'] == ig) & missing_mask
            imputation_stats[var]['by_income_group'][ig] = ig_mask.sum()
        
        imputation_stats[var]['imputed'] = missing_mask.sum()
        imputation_stats[var]['remaining_missing'] = df_result[var].isna().sum()
    
    # Print summary
    print(f"  Imputed {len(impute_vars)} variables using {n_neighbors} nearest neighbors")
    for var in impute_vars:
        stats = imputation_stats[var]
        print(f"    {var}: {stats['original_missing']} values imputed ({stats['original_pct']:.1f}%)")
    
    return df_result, imputation_stats


def impute_by_country_year(df, var, method='median'):
    """
    Impute using country-specific median first, then income group, then global.
    
    This preserves country-level patterns where possible.
    """
    result = df[var].copy()
    
    # Country-level statistics
    country_stats = df.groupby('Code')[var].median() if method == 'median' else df.groupby('Code')[var].mean()
    
    # Income group statistics
    income_stats = df.groupby('IncomeGroup')[var].median() if method == 'median' else df.groupby('IncomeGroup')[var].mean()
    
    # Global fallback
    global_stat = df[var].median() if method == 'median' else df[var].mean()
    
    for idx in df[df[var].isna()].index:
        code = df.loc[idx, 'Code']
        income_group = df.loc[idx, 'IncomeGroup']
        
        # Try country median first
        if code in country_stats.index and pd.notna(country_stats[code]):
            result.loc[idx] = country_stats[code]
        # Then income group
        elif pd.notna(income_group) and income_group in income_stats.index and pd.notna(income_stats[income_group]):
            result.loc[idx] = income_stats[income_group]
        # Finally global
        else:
            result.loc[idx] = global_stat
    
    return result


def prepare_imputed_data(imputation_method='hybrid', verbose=True, save_stats=True):
    """
    Prepare data with imputation for causal analysis.
    
    Parameters:
    -----------
    imputation_method : str
        'hybrid' - KNN for VaccineHesitancy (high missingness), median for others (default)
        'knn' - K-Nearest Neighbors imputation for all variables
        'income_group' - use income group median for all (legacy)
        'country_first' - try country median first, then income group (legacy)
    verbose : bool
        Print progress and statistics
    save_stats : bool
        Save imputation statistics to CSV files
    
    Returns:
    --------
    DataFrame with imputed values and standardized variables
    """
    df = load_panel_data(filter_year=True)  # Filter out years > MAX_ANALYSIS_YEAR
    
    if verbose:
        print("\n" + "="*60)
        print("DATA PREPARATION WITH IMPUTATION")
        print("="*60)
        print(f"\nOriginal data: {len(df)} observations, {df['Code'].nunique()} countries")
        print(f"Year range: {df['Year'].min()} - {df['Year'].max()}")
        print(f"Imputation method: {imputation_method}")
    
    # Merge additional variables
    df = merge_additional_variables(df)
    
    # Variables to check for missing data
    check_vars = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'PolStability', 
                  'BirthRate', 'PopDensity', 'NetMigration', 'UrbanPop', 
                  'HouseholdSize', 'VaccineHesitancy']
    check_vars = [v for v in check_vars if v in df.columns]
    
    # Log missing data BEFORE imputation
    missing_stats_before = log_missing_data_by_year_and_income(df, check_vars, verbose=verbose)
    
    imputation_stats = {
        'by_variable': {},
        'by_income_group': {},
        'by_year': {}
    }
    
    if imputation_method == 'knn':
        # KNN imputation for ALL variables
        impute_vars = ['GDPpc', 'HealthExpPC', 'PolStability', 'BirthRate', 
                       'PopDensity', 'MCV1', 'MCV2', 'NetMigration', 
                       'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
        df, knn_stats = knn_impute_all(df, impute_vars)
        imputation_stats['by_variable'] = knn_stats
        
        if verbose:
            print("\n" + "-"*60)
            print("IMPUTATION SUMMARY (KNN for all)")
            print("-"*60)
            for var, stats in knn_stats.items():
                print(f"  {var:20s}: {stats['original_missing']:4d} missing ({stats['original_pct']:5.1f}%) -> "
                      f"{stats['imputed']:4d} imputed, {stats['remaining_missing']:4d} remaining")
    
    elif imputation_method == 'hybrid':
        # HYBRID: KNN for VaccineHesitancy, income-group median for others
        if verbose:
            print("\n" + "-"*60)
            print("HYBRID IMPUTATION: KNN for VaccineHesitancy, Median for others")
            print("-"*60)
        
        # Step 1: Income-group median for most variables
        vars_median = {
            'GDPpc': 'income_group',
            'HealthExpPC': 'income_group',
            'PolStability': 'income_group',
            'BirthRate': 'income_group',
            'PopDensity': 'country_first',
            'MCV1': 'country_first',
            'MCV2': 'country_first',
            'NetMigration': 'country_first',
            'UrbanPop': 'income_group',
            'HouseholdSize': 'income_group',
        }
        
        if verbose:
            print("\n  [Median Imputation]")
        
        for var, strategy in vars_median.items():
            if var not in df.columns:
                continue
                
            original_missing = df[var].isna().sum()
            original_pct = 100 * original_missing / len(df)
            
            if original_missing == 0:
                continue
            
            # Track missing by income group
            ig_missing_before = {}
            for ig in df['IncomeGroup'].unique():
                if pd.notna(ig):
                    ig_df = df[df['IncomeGroup'] == ig]
                    ig_missing_before[ig] = ig_df[var].isna().sum()
            
            # Apply imputation
            if strategy == 'income_group':
                df[var] = impute_by_income_group(df, var, method='median')
            else:
                df[var] = impute_by_country_year(df, var, method='median')
            
            new_missing = df[var].isna().sum()
            imputed_count = original_missing - new_missing
            
            # Track by income group
            ig_imputed = {}
            for ig in df['IncomeGroup'].unique():
                if pd.notna(ig):
                    ig_df = df[df['IncomeGroup'] == ig]
                    ig_missing_after = ig_df[var].isna().sum()
                    ig_imputed[ig] = ig_missing_before.get(ig, 0) - ig_missing_after
            
            imputation_stats['by_variable'][var] = {
                'original_missing': original_missing,
                'original_pct': original_pct,
                'imputed': imputed_count,
                'remaining_missing': new_missing,
                'by_income_group': ig_imputed,
                'method': 'median'
            }
            
            if verbose:
                print(f"    {var:20s}: {original_missing:4d} missing -> {imputed_count:4d} imputed (median)")
        
        # Step 2: KNN for VaccineHesitancy (high missingness - 84%)
        if 'VaccineHesitancy' in df.columns and df['VaccineHesitancy'].isna().sum() > 0:
            if verbose:
                print("\n  [KNN Imputation for VaccineHesitancy]")
            
            df, knn_stats = knn_impute_all(df, ['VaccineHesitancy'])
            imputation_stats['by_variable']['VaccineHesitancy'] = knn_stats.get('VaccineHesitancy', {})
            imputation_stats['by_variable']['VaccineHesitancy']['method'] = 'knn'
            
            stats = knn_stats.get('VaccineHesitancy', {})
            if verbose and stats:
                print(f"    VaccineHesitancy  : {stats['original_missing']:4d} missing -> {stats['imputed']:4d} imputed (KNN)")
    
    else:
        # Legacy: variable-by-variable median imputation
        vars_to_impute = {
            'GDPpc': 'income_group',
            'HealthExpPC': 'income_group',
            'PolStability': 'income_group',
            'BirthRate': 'income_group',
            'PopDensity': 'country_first',
            'MCV1': 'country_first',
            'MCV2': 'country_first',
            'NetMigration': 'country_first',
            'UrbanPop': 'income_group',
            'HouseholdSize': 'income_group',
            'VaccineHesitancy': 'income_group',
        }
        
        if verbose:
            print("\n" + "-"*60)
            print("IMPUTATION BY VARIABLE (Median Method)")
            print("-"*60)
        
        for var, strategy in vars_to_impute.items():
            if var not in df.columns:
                continue
                
            original_missing = df[var].isna().sum()
            original_pct = 100 * original_missing / len(df)
            
            if original_missing == 0:
                continue
            
            # Track missing by income group before imputation
            ig_missing_before = {}
            for ig in df['IncomeGroup'].unique():
                if pd.notna(ig):
                    ig_df = df[df['IncomeGroup'] == ig]
                    ig_missing_before[ig] = ig_df[var].isna().sum()
            
            # Choose imputation function
            if strategy == 'income_group' or imputation_method == 'income_group':
                df[var] = impute_by_income_group(df, var, method='median')
            else:
                df[var] = impute_by_country_year(df, var, method='median')
            
            new_missing = df[var].isna().sum()
            imputed_count = original_missing - new_missing
            
            # Track imputed by income group
            ig_imputed = {}
            for ig in df['IncomeGroup'].unique():
                if pd.notna(ig):
                    ig_df = df[df['IncomeGroup'] == ig]
                    ig_missing_after = ig_df[var].isna().sum()
                    ig_imputed[ig] = ig_missing_before.get(ig, 0) - ig_missing_after
            
            imputation_stats['by_variable'][var] = {
                'original_missing': original_missing,
                'original_pct': original_pct,
                'imputed': imputed_count,
                'remaining_missing': new_missing,
                'by_income_group': ig_imputed
            }
            
            if verbose:
                print(f"  {var:20s}: {original_missing:4d} missing ({original_pct:5.1f}%) -> "
                      f"{imputed_count:4d} imputed, {new_missing:4d} remaining")
    
    # Summarize imputation by income group
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    if verbose:
        print("\n" + "-"*60)
        print("IMPUTATION SUMMARY BY INCOME GROUP")
        print("-"*60)
    
    for ig in income_groups:
        ig_df = df[df['IncomeGroup'] == ig]
        total_obs = len(ig_df)
        if total_obs == 0:
            continue
        
        total_imputed = 0
        for var, var_stats in imputation_stats['by_variable'].items():
            total_imputed += var_stats['by_income_group'].get(ig, 0)
        
        imputation_stats['by_income_group'][ig] = {
            'total_observations': total_obs,
            'n_countries': ig_df['Code'].nunique(),
            'total_imputed_values': total_imputed
        }
        
        if verbose:
            print(f"\n  {ig} ({ig_df['Code'].nunique()} countries, {total_obs} obs):")
            for var, var_stats in imputation_stats['by_variable'].items():
                ig_imp = var_stats['by_income_group'].get(ig, 0)
                if ig_imp > 0:
                    print(f"    {var:<20}: {ig_imp:4d} values imputed")
    
    # Summarize imputation by year
    if verbose:
        print("\n" + "-"*60)
        print("IMPUTATION PERCENTAGE BY YEAR AND VARIABLE")
        print("-"*60)
    
    years = sorted(df['Year'].unique())
    year_stats_rows = []
    
    for year in years:
        year_df = df[df['Year'] == year]
        year_total = len(year_df)
        year_row = {'Year': year, 'Total_Obs': year_total}
        
        for var in check_vars:
            if var in missing_stats_before['by_variable_year']:
                before_stats = missing_stats_before['by_variable_year'][var].get(year, {})
                original_missing = before_stats.get('missing', 0)
                imputed_pct = 100 * original_missing / year_total if year_total > 0 else 0
                year_row[f'{var}_imputed_pct'] = imputed_pct
        
        year_stats_rows.append(year_row)
        imputation_stats['by_year'][year] = year_row
    
    if verbose and year_stats_rows:
        print(f"\n{'Year':<6} " + " ".join([f"{v[:8]:<10}" for v in check_vars[:6]]))
        print("-"*80)
        high_imputation_years = []
        for row in year_stats_rows:
            row_str = f"{row['Year']:<6} "
            year_has_high_imputation = False
            for var in check_vars[:6]:
                pct = row.get(f'{var}_imputed_pct', 0)
                if pct >= 50:  # Flag variables with >50% imputation
                    row_str += f"{pct:>7.1f}%* "
                    if var in ['MCV1', 'MCV2', 'PolStability', 'BirthRate', 'GDPpc']:
                        year_has_high_imputation = True
                else:
                    row_str += f"{pct:>7.1f}%  "
            print(row_str)
            if year_has_high_imputation:
                high_imputation_years.append(row['Year'])
        
        if high_imputation_years:
            print("\n" + "!"*60)
            print("WARNING: Years with >50% imputation on key variables (*)")
            print("!"*60)
            for y in high_imputation_years:
                print(f"  Year {y}: Heavy imputation required for some key variables")
    
    # Recalculate log variables after imputation
    if 'GDPpc' in df.columns:
        df['LogGDPpc'] = np.log1p(df['GDPpc'].clip(lower=0))
    if 'PopDensity' in df.columns:
        df['LogPopDensity'] = np.log1p(df['PopDensity'].clip(lower=0))
    if 'HealthExpPC' in df.columns:
        df['LogHealthExpPC'] = np.log1p(df['HealthExpPC'].clip(lower=0))
    
    # Check complete cases now (core variables only - new vars may still have missing)
    core_vars = ['MCV1', 'LogIncidence', 'LogGDPpc', 'LogHealthExpPC', 
                 'PolStability', 'HIC', 'LogPopDensity', 'BirthRate']
    
    # Extended variables (may have more missing)
    extended_vars = core_vars + ['NetMigration', 'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    
    complete_core = df[core_vars + ['Code', 'Year']].dropna()
    complete_extended = df[extended_vars + ['Code', 'Year']].dropna()
    
    if verbose:
        print("\n" + "-"*60)
        print("COMPLETE CASES AFTER IMPUTATION")
        print("-"*60)
        print(f"  Core variables: {len(complete_core)} obs, {complete_core['Code'].nunique()} countries")
        print(f"  Extended (with new vars): {len(complete_extended)} obs, {complete_extended['Code'].nunique()} countries")
        print(f"  Year range: {complete_core['Year'].min()} - {complete_core['Year'].max()}")
    
    # Save imputation statistics to CSV
    if save_stats:
        save_imputation_stats(imputation_stats, missing_stats_before, df)
    
    return df, imputation_stats


def save_imputation_stats(imputation_stats, missing_stats_before, df):
    """
    Save imputation statistics to CSV files for reference.
    
    Creates:
    - imputation_by_variable.csv: Overall imputation stats per variable
    - imputation_by_income_group.csv: Imputation breakdown by income group
    - imputation_by_year.csv: Missing data percentage by year and variable
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. By Variable
    var_rows = []
    for var, stats in imputation_stats['by_variable'].items():
        row = {
            'Variable': var,
            'Original_Missing': stats['original_missing'],
            'Original_Missing_Pct': stats['original_pct'],
            'Imputed': stats['imputed'],
            'Remaining_Missing': stats['remaining_missing']
        }
        # Add by income group
        for ig, count in stats['by_income_group'].items():
            if pd.notna(ig) and isinstance(ig, str):
                row[f'Imputed_{ig.replace(" ", "_")}'] = count
        var_rows.append(row)
    
    if var_rows:
        pd.DataFrame(var_rows).to_csv(RESULTS_DIR / 'imputation_by_variable.csv', index=False)
        print(f"\n[SAVED] {RESULTS_DIR / 'imputation_by_variable.csv'}")
    
    # 2. By Income Group
    ig_rows = []
    income_groups = ['High income', 'Upper middle income', 'Lower middle income', 'Low income']
    for ig in income_groups:
        if ig not in imputation_stats['by_income_group']:
            continue
        ig_stats = imputation_stats['by_income_group'][ig]
        ig_df = df[df['IncomeGroup'] == ig]
        
        row = {
            'Income_Group': ig,
            'N_Countries': ig_stats['n_countries'],
            'Total_Observations': ig_stats['total_observations'],
            'Total_Imputed_Values': ig_stats['total_imputed_values']
        }
        
        # Add per-variable imputation counts
        for var, var_stats in imputation_stats['by_variable'].items():
            row[f'{var}_Imputed'] = var_stats['by_income_group'].get(ig, 0)
        
        ig_rows.append(row)
    
    if ig_rows:
        pd.DataFrame(ig_rows).to_csv(RESULTS_DIR / 'imputation_by_income_group.csv', index=False)
        print(f"[SAVED] {RESULTS_DIR / 'imputation_by_income_group.csv'}")
    
    # 3. By Year (missing percentage before imputation)
    year_rows = []
    for year, year_stats in imputation_stats['by_year'].items():
        row = {'Year': year, 'Total_Observations': year_stats.get('Total_Obs', 0)}
        for key, val in year_stats.items():
            if key.endswith('_imputed_pct'):
                row[key] = val
        year_rows.append(row)
    
    if year_rows:
        pd.DataFrame(year_rows).to_csv(RESULTS_DIR / 'imputation_by_year.csv', index=False)
        print(f"[SAVED] {RESULTS_DIR / 'imputation_by_year.csv'}")


def standardize_variables(df, vars_to_standardize=None):
    """
    Standardize continuous variables (z-score normalization).
    
    Parameters:
    -----------
    df : DataFrame
    vars_to_standardize : list
        Variables to standardize. If None, use defaults.
    
    Returns:
    --------
    DataFrame with standardized variables (suffix '_std')
    """
    if vars_to_standardize is None:
        vars_to_standardize = ['LogGDPpc', 'LogHealthExpPC', 'PolStability', 
                               'BirthRate', 'LogPopDensity', 'MCV1', 'MCV2']
    
    df_out = df.copy()
    scaler = StandardScaler()
    
    for var in vars_to_standardize:
        if var in df.columns:
            # Fit on non-missing values
            valid_mask = df[var].notna()
            if valid_mask.sum() > 0:
                df_out[f'{var}_std'] = np.nan
                df_out.loc[valid_mask, f'{var}_std'] = scaler.fit_transform(
                    df.loc[valid_mask, [var]]
                ).flatten()
    
    return df_out


def get_analysis_ready_data(verbose=True, use_extended_vars=True):
    """
    Get fully prepared data for causal analysis with imputation and standardization.
    
    Parameters:
    -----------
    verbose : bool
        Print progress
    use_extended_vars : bool
        If True, include NetMigration, UrbanPop, HouseholdSize, VaccineHesitancy
    """
    # Impute
    df, imputation_stats = prepare_imputed_data(verbose=verbose)
    
    # Standardize all continuous variables
    vars_to_std = ['LogGDPpc', 'LogHealthExpPC', 'PolStability', 
                   'BirthRate', 'LogPopDensity', 'MCV1', 'MCV2',
                   'NetMigration', 'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    df = standardize_variables(df, vars_to_standardize=vars_to_std)
    
    # Core analysis variables
    core_vars = ['MCV1', 'LogIncidence', 'LogGDPpc', 'LogHealthExpPC', 
                 'PolStability', 'HIC', 'LogPopDensity', 'BirthRate',
                 'Code', 'Year', 'Country', 'IncomeGroup']
    
    # Extended variables
    extended_vars = core_vars + ['NetMigration', 'UrbanPop', 'HouseholdSize', 'VaccineHesitancy']
    
    if use_extended_vars:
        analysis_vars = extended_vars
    else:
        analysis_vars = core_vars
    
    complete_df = df[analysis_vars].dropna()
    
    if verbose:
        print("\n" + "="*60)
        print("FINAL ANALYSIS DATASET")
        print("="*60)
        print(f"\nTotal observations: {len(complete_df)}")
        print(f"Countries: {complete_df['Code'].nunique()}")
        print(f"Year range: {complete_df['Year'].min()} - {complete_df['Year'].max()}")
        
        print("\nCountries by Income Group:")
        income_counts = complete_df.groupby('IncomeGroup')['Code'].nunique()
        for group, count in income_counts.items():
            print(f"  {group}: {count} countries")
        
        print("\nVariables included:")
        for var in analysis_vars:
            if var not in ['Code', 'Year', 'Country', 'IncomeGroup']:
                print(f"  - {var}")
    
    return df, complete_df, imputation_stats


def compare_before_after_imputation():
    """Compare sample composition before and after imputation"""
    # Before imputation
    df_original = load_panel_data()
    analysis_vars = ['MCV1', 'LogIncidence', 'LogGDPpc', 'LogHealthExpPC', 
                     'PolStability', 'HIC', 'LogPopDensity', 'BirthRate']
    before = df_original[analysis_vars + ['Code', 'IncomeGroup']].dropna()
    
    # After imputation
    df_imputed, complete_df, _ = get_analysis_ready_data(verbose=False)
    
    print("="*60)
    print("COMPARISON: BEFORE vs AFTER IMPUTATION")
    print("="*60)
    
    print("\n--- BEFORE IMPUTATION ---")
    print(f"Observations: {len(before)}")
    print(f"Countries: {before['Code'].nunique()}")
    print("\nBy Income Group:")
    before_income = before.groupby('IncomeGroup')['Code'].nunique()
    for group in ['High income', 'Upper middle income', 'Lower middle income', 'Low income']:
        if group in before_income.index:
            print(f"  {group}: {before_income[group]} countries")
        else:
            print(f"  {group}: 0 countries")
    
    print("\n--- AFTER IMPUTATION ---")
    print(f"Observations: {len(complete_df)}")
    print(f"Countries: {complete_df['Code'].nunique()}")
    print("\nBy Income Group:")
    after_income = complete_df.groupby('IncomeGroup')['Code'].nunique()
    for group in ['High income', 'Upper middle income', 'Lower middle income', 'Low income']:
        if group in after_income.index:
            print(f"  {group}: {after_income[group]} countries")
        else:
            print(f"  {group}: 0 countries")
    
    print("\n--- NEW COUNTRIES ADDED ---")
    before_countries = set(before['Code'].unique())
    after_countries = set(complete_df['Code'].unique())
    new_countries = after_countries - before_countries
    
    if new_countries:
        new_df = complete_df[complete_df['Code'].isin(new_countries)][['Code', 'Country', 'IncomeGroup']].drop_duplicates()
        print(f"\n{len(new_countries)} new countries added:")
        for _, row in new_df.sort_values('Country').iterrows():
            print(f"  {row['Code']} - {row['Country']} ({row['IncomeGroup']})")
    else:
        print("No new countries added")
    
    return before, complete_df


if __name__ == '__main__':
    # Run comparison
    compare_before_after_imputation()

