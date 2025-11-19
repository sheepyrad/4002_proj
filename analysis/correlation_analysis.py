"""
Spearman Rank Correlation Analysis: Measles Incidence vs Various Factors

This script performs Spearman rank correlation analysis between measles incidence
rate and various socioeconomic and health factors.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import warnings

from utils import (
    DATA_DIR, RESULTS_DIR, save_figure, ensure_results_dir,
    convert_wide_to_long, extract_year_from_date
)

warnings.filterwarnings('ignore')

# Import analysis functions
from analyze_gdp import load_data as load_gdp
from analyze_health_expenditure import load_data as load_health_exp
from analyze_vaccine_hesitancy import load_data as load_vaccine_hesitancy
from analyze_urban_pop import load_data as load_urban_pop
from analyze_measles import load_data as load_measles
from analyze_birth_rate import load_data as load_birth_rate
from analyze_net_migration import load_data as load_net_migration
from analyze_pop_density import load_data as load_pop_density
from analyze_political_stability import load_data as load_political_stability
from analyze_household_size import load_data as load_household_size
from analyze_vaccine_coverage import load_data as load_vaccine_coverage

def load_all_data():
    """Load all datasets"""
    print("Loading all datasets...")
    
    # Load measles data
    measles_df = load_measles()
    cases_col = 'Total confirmed  measles cases'
    incidence_col = 'Measles incidence rate per 1\'000\'000  total population'
    
    # Load other datasets
    gdp_df = load_gdp()
    health_df = load_health_exp()
    vaccine_hesitancy_df = load_vaccine_hesitancy()
    urban_df = load_urban_pop()
    birth_rate_df = load_birth_rate()
    net_migration_df = load_net_migration()
    pop_density_df = load_pop_density()
    political_stability_df = load_political_stability()
    household_size_df = load_household_size()
    vaccine_coverage_1st, vaccine_coverage_2nd = load_vaccine_coverage()
    
    # Get column names
    gdp_col = [col for col in gdp_df.columns if 'GDP' in col][0]
    health_col = [col for col in health_df.columns if 'health' in col.lower() or 'expenditure' in col.lower()][0]
    vaccine_hesitancy_col = [col for col in vaccine_hesitancy_df.columns if 'vaccine' in col.lower() or 'disagree' in col.lower()][0]
    urban_col = [col for col in urban_df.columns if 'urban' in col.lower()][0]
    birth_rate_col = 'Birth Rate'
    net_migration_col = 'Net Migration'
    pop_density_col = 'Population Density'
    political_stability_col = 'Political Stability Index'
    hh_size_col = 'Average household size (number of members)'
    
    return {
        'measles': (measles_df, cases_col, incidence_col),
        'gdp': (gdp_df, gdp_col),
        'health': (health_df, health_col),
        'vaccine_hesitancy': (vaccine_hesitancy_df, vaccine_hesitancy_col),
        'urban': (urban_df, urban_col),
        'birth_rate': (birth_rate_df, birth_rate_col),
        'net_migration': (net_migration_df, net_migration_col),
        'pop_density': (pop_density_df, pop_density_col),
        'political_stability': (political_stability_df, political_stability_col),
        'household_size': (household_size_df, hh_size_col),
        'vaccine_coverage_1st': (vaccine_coverage_1st, 'COVERAGE'),
        'vaccine_coverage_2nd': (vaccine_coverage_2nd, 'COVERAGE')
    }

def find_best_year_for_analysis(data_dict):
    """Find the year with best data coverage across all datasets"""
    measles_df, _, incidence_col = data_dict['measles']
    
    # Get years from measles data
    measles_years = set(measles_df['Year'].unique())
    
    # Get years from other datasets
    other_years = {}
    for key, value in data_dict.items():
        if key != 'measles':
            df = value[0]
            if 'Year' in df.columns:
                other_years[key] = set(df['Year'].unique())
            elif key in ['vaccine_coverage_1st', 'vaccine_coverage_2nd']:
                # Vaccine coverage uses 'YEAR' column
                if 'YEAR' in df.columns:
                    other_years[key] = set(df['YEAR'].unique())
    
    # Find overlap years
    overlap_years = measles_years.copy()
    for years in other_years.values():
        overlap_years = overlap_years.intersection(years)
    
    if not overlap_years:
        # Fallback: use most recent measles year
        return measles_df['Year'].max()
    
    # Find year with most countries
    best_year = None
    max_countries = 0
    
    for year in sorted(overlap_years, reverse=True):
        measles_codes = set(measles_df[measles_df['Year'] == year]['ISO country code'].dropna().unique())
        count = len(measles_codes)
        
        # Check other datasets
        for key, value in data_dict.items():
            if key != 'measles':
                df = value[0]
                if key in ['vaccine_coverage_1st', 'vaccine_coverage_2nd']:
                    if 'YEAR' in df.columns and 'CODE' in df.columns:
                        year_col = 'YEAR'
                        code_col = 'CODE'
                    else:
                        continue
                else:
                    # Check for both Year and Code columns
                    if 'Year' in df.columns and 'Code' in df.columns:
                        year_col = 'Year'
                        code_col = 'Code'
                    elif 'Year' in df.columns:
                        # Some datasets might not have Code column, skip intersection
                        continue
                    else:
                        continue
                
                codes = set(df[df[year_col] == year][code_col].dropna().unique())
                measles_codes = measles_codes.intersection(codes)
                count = len(measles_codes)
        
        if count > max_countries:
            max_countries = count
            best_year = year
    
    return best_year if best_year else max(overlap_years)

def load_income_groups():
    """Load income groups mapping"""
    income_path = Path(__file__).parent.parent / 'data' / 'incomegroups.csv'
    income_df = pd.read_csv(income_path)
    # Clean column names (remove quotes if present)
    income_df.columns = income_df.columns.str.strip('"')
    # Filter out empty income groups and create mapping
    income_df = income_df[income_df['IncomeGroup'].notna() & (income_df['IncomeGroup'] != '')]
    # Create mapping from Code to IncomeGroup
    income_map = dict(zip(income_df['Country.Code'], income_df['IncomeGroup']))
    return income_map


def impute_by_income_group(correlation_df, income_map):
    """
    Impute missing values in numeric columns using mean within income group.
    Only imputes if there's at least one non-null value in the income group.
    """
    # Add income group column
    correlation_df['IncomeGroup'] = correlation_df['Code'].map(income_map)
    
    # Get numeric columns (excluding Code, Year, and IncomeGroup itself)
    exclude_cols = {'Code', 'Year', 'IncomeGroup'}
    numeric_cols = [col for col in correlation_df.columns 
                   if col not in exclude_cols and correlation_df[col].dtype in ['float64', 'int64']]
    
    imputed_count = 0
    for col in numeric_cols:
        missing_before = correlation_df[col].isna().sum()
        if missing_before == 0:
            continue
        
        # Group by income group and calculate mean
        income_means = correlation_df.groupby('IncomeGroup')[col].mean()
        
        # Impute missing values
        for idx, row in correlation_df[correlation_df[col].isna()].iterrows():
            income_group = row['IncomeGroup']
            if pd.notna(income_group) and income_group in income_means.index:
                mean_value = income_means[income_group]
                if pd.notna(mean_value):
                    correlation_df.at[idx, col] = mean_value
                    imputed_count += 1
        
        missing_after = correlation_df[col].isna().sum()
        if missing_before > missing_after:
            print(f"  Imputed {missing_before - missing_after} missing values for {col} using income group means")
    
    if imputed_count > 0:
        print(f"  Total imputed values: {imputed_count}")
    
    # Remove IncomeGroup column (temporary)
    correlation_df = correlation_df.drop(columns=['IncomeGroup'])
    
    return correlation_df


def prepare_correlation_data(data_dict, analysis_year, income_group_filter=None):
    """Prepare merged dataset for correlation analysis with income-group-based imputation
    
    Args:
        data_dict: Dictionary of datasets
        analysis_year: Year to analyze
        income_group_filter: Optional income group to filter by (e.g., 'Low income', 'High income')
    """
    measles_df, _, incidence_col = data_dict['measles']
    
    # Start with measles data
    correlation_df = measles_df[measles_df['Year'] == analysis_year][
        ['ISO country code', incidence_col]
    ].copy()
    correlation_df = correlation_df.rename(columns={'ISO country code': 'Code'})
    correlation_df = correlation_df.dropna(subset=[incidence_col])
    
    print(f"\nStarting with {len(correlation_df)} countries from measles data")
    
    # Load income groups for imputation and filtering
    try:
        income_map = load_income_groups()
        print(f"  Loaded income groups for {len(income_map)} countries")
    except Exception as e:
        print(f"  Warning: Could not load income groups: {e}")
        income_map = {}
    
    # Filter by income group if specified
    if income_map and income_group_filter:
        correlation_df['IncomeGroup'] = correlation_df['Code'].map(income_map)
        before_filter = len(correlation_df)
        correlation_df = correlation_df[correlation_df['IncomeGroup'] == income_group_filter].copy()
        after_filter = len(correlation_df)
        print(f"  Filtered to {income_group_filter} countries: {before_filter} -> {after_filter} countries")
        if after_filter == 0:
            print(f"  WARNING: No {income_group_filter} countries found! Continuing with all countries.")
            correlation_df = measles_df[measles_df['Year'] == analysis_year][
                ['ISO country code', incidence_col]
            ].copy()
            correlation_df = correlation_df.rename(columns={'ISO country code': 'Code'})
            correlation_df = correlation_df.dropna(subset=[incidence_col])
            correlation_df['IncomeGroup'] = correlation_df['Code'].map(income_map)
        else:
            # Drop IncomeGroup column temporarily (will be re-added for imputation)
            correlation_df = correlation_df.drop(columns=['IncomeGroup'])
    
    # Merge with other datasets
    merge_info = [
        ('gdp', 'Code', 'Year'),
        ('health', 'Code', 'Year'),
        ('vaccine_hesitancy', 'Code', 'Year'),
        ('urban', 'Code', 'Year'),
        ('birth_rate', 'Code', 'Year'),
        ('net_migration', 'Code', 'Year'),
        ('pop_density', 'Code', 'Year'),
        ('political_stability', 'Code', 'Year'),
    ]
    
    for key, code_col, year_col in merge_info:
        if key in data_dict:
            df, value_col = data_dict[key]
            # Check if required columns exist
            if year_col not in df.columns or code_col not in df.columns or value_col not in df.columns:
                print(f"  Skipping {key}: missing required columns")
                continue
            
            # Check if data exists for this year
            df_year = df[df[year_col] == analysis_year][[code_col, value_col]].copy()
            if len(df_year) == 0:
                print(f"  Skipping {key}: no data for year {analysis_year}")
                continue
            
            correlation_df = correlation_df.merge(
                df_year, on='Code', how='left'
            )
            matched = correlation_df[value_col].notna().sum()
            print(f"  Merged {key}: {matched}/{len(correlation_df)} countries matched")
    
    # Handle household size (use most recent per country)
    if 'household_size' in data_dict:
        hh_df, hh_col = data_dict['household_size']
        # Get most recent household size per country
        hh_df_clean = hh_df[hh_df[hh_col].notna()].copy()
        if 'ISO Code' in hh_df_clean.columns:
            # Need to convert ISO numeric codes to ISO3
            try:
                import country_converter as coco
                cc = coco.CountryConverter()
                iso_numeric_codes = hh_df_clean['ISO Code'].astype(int).astype(str).tolist()
                iso3_codes = cc.convert(names=iso_numeric_codes, src='UNcode', to='ISO3', not_found=None)
                hh_df_clean['Code'] = iso3_codes if isinstance(iso3_codes, (list, pd.Series)) else [iso3_codes] * len(hh_df_clean)
                hh_df_clean = hh_df_clean[
                    hh_df_clean['Code'].notna() & 
                    (hh_df_clean['Code'] != 'not found') &
                    (hh_df_clean['Code'].str.len() == 3)
                ].copy()
                
                if 'Year' in hh_df_clean.columns:
                    hh_latest = hh_df_clean.sort_values('Year').drop_duplicates(subset='Code', keep='last')
                    correlation_df = correlation_df.merge(
                        hh_latest[['Code', hh_col]], on='Code', how='left'
                    )
                    matched = correlation_df[hh_col].notna().sum()
                    print(f"  Merged household_size: {matched}/{len(correlation_df)} countries matched")
            except Exception as e:
                print(f"  Could not merge household_size: {e}")
    
    # Handle vaccine coverage (use YEAR column)
    for key in ['vaccine_coverage_1st', 'vaccine_coverage_2nd']:
        if key in data_dict:
            df, value_col = data_dict[key]
            # Check if required columns exist
            if 'YEAR' not in df.columns or 'CODE' not in df.columns or value_col not in df.columns:
                print(f"  Skipping {key}: missing required columns")
                continue
            
            # Check if data exists for this year
            df_year = df[df['YEAR'] == analysis_year][['CODE', value_col]].copy()
            if len(df_year) == 0:
                print(f"  Skipping {key}: no data for year {analysis_year}")
                continue
            
            df_year = df_year.rename(columns={'CODE': 'Code'})
            coverage_col = f'{key}_{value_col}'
            df_year = df_year.rename(columns={value_col: coverage_col})
            correlation_df = correlation_df.merge(
                df_year, on='Code', how='left'
            )
            matched = correlation_df[coverage_col].notna().sum()
            print(f"  Merged {key}: {matched}/{len(correlation_df)} countries matched")
    
    # Add Year column for reference
    correlation_df['Year'] = analysis_year
    
    # Impute missing values using income group means
    if income_map:
        print(f"\nImputing missing values using income group means...")
        correlation_df = impute_by_income_group(correlation_df, income_map)
    
    return correlation_df

def calculate_spearman_correlations(correlation_df, incidence_col):
    """Calculate Spearman rank correlations with measles incidence"""
    metrics = {}
    
    # Identify metric columns (exclude Code and incidence_col)
    exclude_cols = ['Code', incidence_col]
    for col in correlation_df.columns:
        if col not in exclude_cols:
            metrics[col] = col
    
    correlations = {}
    pvalues = {}
    sample_sizes = {}
    
    for metric_name, metric_col in metrics.items():
        data_subset = correlation_df[[incidence_col, metric_col]].dropna()
        if len(data_subset) > 2:
            corr_result = spearmanr(data_subset[incidence_col], data_subset[metric_col])
            correlations[metric_name] = corr_result.statistic
            pvalues[metric_name] = corr_result.pvalue
            sample_sizes[metric_name] = len(data_subset)
    
    return correlations, pvalues, sample_sizes

def create_correlation_matrix_heatmap(correlation_df, incidence_col, analysis_year, income_group=None, output_path=None):
    """Create a full correlation matrix heatmap for all variables
    
    Args:
        correlation_df: DataFrame with correlation data
        incidence_col: Name of the incidence column
        analysis_year: Year being analyzed
        income_group: Optional income group name for title
        output_path: Optional path to save the figure (Path object)
    
    Returns:
        None (saves figure to file)
    """
    # Prepare income group label for titles
    if income_group:
        income_label = f"({income_group}, "
        income_suffix = f"_{income_group.lower().replace(' ', '_')}"
    else:
        income_label = "(All Countries, "
        income_suffix = "_all"
    
    # Prepare heatmap data
    heatmap_data = correlation_df[[incidence_col]].copy()
    # Get all numeric columns except Code, Year, and incidence_col
    exclude_cols = {'Code', 'Year', incidence_col}
    for col in correlation_df.columns:
        if col not in exclude_cols and correlation_df[col].dtype in ['float64', 'int64']:
            heatmap_data[col] = correlation_df[col]
    
    # Explicitly exclude Year and Code columns if they exist
    heatmap_data = heatmap_data.drop(columns=['Year', 'Code'], errors='ignore')
    
    # Calculate correlation matrix
    corr_matrix = heatmap_data.corr(method='spearman')
    
    # Calculate p-values for all pairwise correlations
    pvalue_matrix = pd.DataFrame(
        index=corr_matrix.index,
        columns=corr_matrix.columns,
        dtype=float
    )
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            data_subset = heatmap_data[[col1, col2]].dropna()
            if len(data_subset) > 2:
                try:
                    corr_result = spearmanr(data_subset[col1], data_subset[col2])
                    # Ensure pvalue is a scalar float - handle both array and scalar returns
                    pval = corr_result.pvalue
                    if isinstance(pval, (np.ndarray, list)):
                        pval = float(pval[0]) if len(pval) > 0 else np.nan
                    else:
                        pval = float(pval)
                    pvalue_matrix.loc[col1, col2] = pval
                except Exception:
                    pvalue_matrix.loc[col1, col2] = np.nan
            else:
                pvalue_matrix.loc[col1, col2] = np.nan
    
    # Create annotation matrix with correlation, p-value, and significance
    annot_matrix = corr_matrix.copy().astype(str)
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle - empty
                annot_matrix.loc[col1, col2] = ''
            elif i == j:  # Diagonal - just correlation
                annot_matrix.loc[col1, col2] = f'{corr_matrix.loc[col1, col2]:.3f}'
            else:  # Lower triangle - correlation + p-value + significance
                corr_val = corr_matrix.loc[col1, col2]
                pval = pvalue_matrix.loc[col1, col2]
                sig = format_significance(pval)
                
                # Format p-value nicely
                if pd.isna(pval):
                    pval_str = 'p=NA'
                elif pval < 0.001:
                    pval_str = 'p<0.001'
                elif pval < 0.01:
                    pval_str = f'p={pval:.3f}'
                else:
                    pval_str = f'p={pval:.3f}'
                
                # Combine correlation, p-value, and significance symbol
                if sig and sig != 'ns':
                    annot_matrix.loc[col1, col2] = f'{corr_val:.3f}\n{pval_str}\n{sig}'
                else:
                    annot_matrix.loc[col1, col2] = f'{corr_val:.3f}\n{pval_str}'
    
    # Create mask for upper triangle
    mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Identify non-significant correlations (p >= 0.05) in lower triangle
    nonsig_cells = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i > j:  # Lower triangle only
                pval = pvalue_matrix.loc[col1, col2]
                if pd.isna(pval) or pval >= 0.05:
                    nonsig_cells.append((i, j, col1, col2))
    
    fig, ax = plt.subplots(figsize=(12, 10))
    # Plot heatmap with upper triangle masked (show all lower triangle values)
    # All correlations are colored so they map to the colorbar
    sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                mask=mask_upper, annot_kws={'fontsize': 8})
    
    # Get the positions of the heatmap cells
    yticks = ax.get_yticks()
    xticks = ax.get_xticks()
    
    # Explicitly fill upper triangle with white
    if len(yticks) > 0 and len(xticks) > 0:
        cell_height = yticks[1] - yticks[0] if len(yticks) > 1 else 1
        cell_width = xticks[1] - xticks[0] if len(xticks) > 1 else 1
        
        # Fill upper triangle with white rectangles
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Upper triangle
                    rect = plt.Rectangle((xticks[j] - cell_width/2, yticks[i] - cell_height/2),
                                        cell_width, cell_height,
                                        facecolor='white', edgecolor='white', linewidth=1,
                                        zorder=1)
                    ax.add_patch(rect)
    
    # Overlay semi-transparent pattern for non-significant correlations
    # This allows the underlying color to show through so values map to the colorbar
    if len(yticks) > 0 and len(xticks) > 0 and len(nonsig_cells) > 0:
        cell_height = yticks[1] - yticks[0] if len(yticks) > 1 else 1
        cell_width = xticks[1] - xticks[0] if len(xticks) > 1 else 1
        
        for i, j, col1, col2 in nonsig_cells:
            # Add very light grey overlay with diagonal hatching pattern
            # Low alpha allows underlying correlation color to show through
            # so the value is still mapped to the colorbar scale
            rect = plt.Rectangle((xticks[j] - cell_width/2, yticks[i] - cell_height/2),
                                cell_width, cell_height,
                                facecolor='lightgrey', edgecolor='grey', linewidth=1.5,
                                hatch='///', alpha=0.3, zorder=2)
            ax.add_patch(rect)
    
    ax.set_title(f'Spearman Correlation Matrix: All Variables\n{income_label}Year {analysis_year})', 
                 fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        save_figure(fig, f'spearman_correlation_matrix{income_suffix}.png')
    plt.close()

def format_significance(pval):
    """Format p-value significance"""
    if pd.isna(pval):
        return ''
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    elif pval < 0.1:
        return '.'
    else:
        return 'ns'

def create_correlation_visualizations(correlation_df, incidence_col, correlations, pvalues, analysis_year, income_group=None):
    """Create visualizations for correlation analysis
    
    Args:
        income_group: Optional income group name for filename and title (e.g., 'Low income', 'High income')
    """
    
    # Prepare results dataframe with better metric names
    metric_display_names = {
        'GDP per capita, PPP (constant 2021 international $)': 'GDP per Capita',
        'Public health expenditure as a share of GDP': 'Health Expenditure (% GDP)',
        'Share that disagrees vaccines are effective': 'Vaccine Hesitancy (%)',
        'Urban population (% of total population)': 'Urban Population (%)',
        'Birth Rate': 'Birth Rate',
        'Net Migration': 'Net Migration',
        'Population Density': 'Population Density',
        'Political Stability Index': 'Political Stability Index',
        'Average household size (number of members)': 'Household Size',
        'vaccine_coverage_1st_COVERAGE': 'Vaccine Coverage 1st Dose (%)',
        'vaccine_coverage_2nd_COVERAGE': 'Vaccine Coverage 2nd Dose (%)'
    }
    
    results_data = []
    for metric_name in correlations.keys():
        display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
        results_data.append({
            'Metric': display_name,
            'Correlation': correlations[metric_name],
            'P-value': pvalues[metric_name],
            'Significance': format_significance(pvalues[metric_name])
        })
    
    corr_results = pd.DataFrame(results_data)
    corr_results['Abs_Correlation'] = corr_results['Correlation'].abs()
    corr_results = corr_results.sort_values('Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])
    
    # Prepare income group label for titles and filenames
    if income_group:
        income_label = f"({income_group}, "
        income_suffix = f"_{income_group.lower().replace(' ', '_')}"
    else:
        income_label = "(All Countries, "
        income_suffix = "_all"
    
    # 1. Bar plot of correlations
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    colors = ['red' if x < 0 else 'blue' for x in corr_results['Correlation']]
    bars = ax.barh(corr_results['Metric'], corr_results['Correlation'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Spearman Rank Correlation Coefficient', fontsize=12)
    ax.set_title(f'Spearman Correlation with Measles Incidence Rate\n{income_label}Year {analysis_year})', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()
    
    # Add significance indicators
    for i, (bar, sig) in enumerate(zip(bars, corr_results['Significance'])):
        if sig and sig != 'ns':
            x_pos = bar.get_width()
            if x_pos < 0:
                x_pos = x_pos - 0.05
            else:
                x_pos = x_pos + 0.05
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, sig, 
                    va='center', fontsize=10, fontweight='bold')
    
    fig.text(0.5, 0.02, 'Significance: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    save_figure(fig, f'spearman_correlation_barplot{income_suffix}.png')
    plt.close()
    
    # 2. Scatter plots
    n_metrics = len(correlations)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    # Create reverse mapping from display name to actual column name
    display_to_col = {v: k for k, v in metric_display_names.items()}
    # Also create direct mapping for columns that exist
    for col in correlation_df.columns:
        if col != 'Code' and col != incidence_col:
            if col not in display_to_col.values():
                # Create display name
                display_name = col.replace('_', ' ').title()
                display_to_col[display_name] = col
    
    for metric_name in corr_results['Metric']:
        # Find actual column name
        actual_col = display_to_col.get(metric_name)
        
        if actual_col and actual_col in correlation_df.columns and plot_idx < len(axes):
            plot_data = correlation_df[[incidence_col, actual_col]].dropna()
            
            if len(plot_data) > 2:
                ax = axes[plot_idx]
                ax.scatter(plot_data[actual_col], plot_data[incidence_col], alpha=0.6, s=50, color='crimson')
                ax.set_xlabel(metric_name, fontsize=11)
                ax.set_ylabel('Measles Incidence Rate (per 1M)', fontsize=11)
                
                corr_val = correlations.get(actual_col, np.nan)
                pval = pvalues.get(actual_col, np.nan)
                sig = format_significance(pval)
                
                if not pd.isna(pval):
                    title = f'{metric_name} vs Measles Incidence Rate\n{income_label}ρ={corr_val:.3f}, p={pval:.4f}{sig})'
                else:
                    title = f'{metric_name} vs Measles Incidence Rate\n{income_label}ρ={corr_val:.3f})'
                
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add trend line
                if len(plot_data) > 1:
                    z = np.polyfit(plot_data[actual_col], plot_data[incidence_col], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(plot_data[actual_col].min(), plot_data[actual_col].max(), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
                
                plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_figure(fig, f'spearman_correlation_scatterplots{income_suffix}.png')
    plt.close()
    
    # 3. Correlation heatmap with significance values (lower triangle only)
    # Use the reusable function
    create_correlation_matrix_heatmap(
        correlation_df, incidence_col, analysis_year, income_group,
        output_path=None  # Will use default save_figure naming
    )
    # Rename the file to match the expected name
    ensure_results_dir()
    if income_group:
        old_name = RESULTS_DIR / f'spearman_correlation_matrix_{income_group.lower().replace(" ", "_")}.png'
        new_name = RESULTS_DIR / f'spearman_correlation_heatmap_{income_group.lower().replace(" ", "_")}.png'
    else:
        old_name = RESULTS_DIR / 'spearman_correlation_matrix_all.png'
        new_name = RESULTS_DIR / 'spearman_correlation_heatmap_all.png'
    if old_name.exists():
        old_name.rename(new_name)
    
    return corr_results

def run_single_analysis(data_dict, analysis_year, income_group_filter=None):
    """Run correlation analysis for a single income group
    
    Args:
        data_dict: Dictionary of datasets
        analysis_year: Year to analyze
        income_group_filter: Optional income group to filter by (e.g., 'Low income', 'High income')
    
    Returns:
        corr_results: DataFrame with correlation results
    """
    income_label = income_group_filter if income_group_filter else "All Countries"
    
    print("\n" + "="*80)
    print(f"SPEARMAN RANK CORRELATION ANALYSIS: {income_label.upper()}")
    print("="*80)
    
    # Prepare correlation dataset
    correlation_df = prepare_correlation_data(data_dict, analysis_year, income_group_filter)
    
    print(f"\nFinal dataset shape: {correlation_df.shape}")
    print(f"Countries with complete data: {len(correlation_df)}")
    print(f"\nMissing values:")
    print(correlation_df.isnull().sum())
    
    # Get incidence column
    _, _, incidence_col = data_dict['measles']
    
    # Calculate correlations
    correlations, pvalues, sample_sizes = calculate_spearman_correlations(correlation_df, incidence_col)
    
    # Create visualizations
    corr_results = create_correlation_visualizations(
        correlation_df, incidence_col, correlations, pvalues, analysis_year, income_group_filter
    )
    
    # Print results
    print("\n" + "="*80)
    print(f"SPEARMAN RANK CORRELATIONS: METRICS vs MEASLES INCIDENCE RATE")
    print(f"({income_label})")
    print("="*80)
    print(f"\nYear: {analysis_year}")
    print(f"Number of countries analyzed: {len(correlation_df)}")
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1, ns = not significant")
    print(f"\nCorrelation Results:")
    print(corr_results.to_string(index=False))
    print("\n" + "="*80)
    
    # Save results to CSV
    ensure_results_dir()
    income_suffix = f"_{income_group_filter.lower().replace(' ', '_')}" if income_group_filter else "_all"
    results_file = RESULTS_DIR / f'spearman_correlation_results{income_suffix}.csv'
    corr_results.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return corr_results, len(correlation_df)

def main():
    """Main analysis function - runs analysis for Low income, High income, and All countries"""
    print("="*80)
    print("SPEARMAN RANK CORRELATION ANALYSIS")
    print("Measles Incidence Rate vs Various Factors")
    print("(Low Income, High Income, and All Countries)")
    print("="*80)
    
    # Load all data
    data_dict = load_all_data()
    
    # Find best year for analysis
    analysis_year = find_best_year_for_analysis(data_dict)
    print(f"\nUsing year {analysis_year} for correlation analysis")
    
    # Run analysis for each group
    all_results = {}
    country_counts = {}
    
    # 1. Low income countries
    all_results['Low income'], country_counts['Low income'] = run_single_analysis(data_dict, analysis_year, 'Low income')
    
    # 2. High income countries
    all_results['High income'], country_counts['High income'] = run_single_analysis(data_dict, analysis_year, 'High income')
    
    # 3. All countries
    all_results['All'], country_counts['All'] = run_single_analysis(data_dict, analysis_year, None)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON ACROSS INCOME GROUPS")
    print("="*80)
    print(f"\nYear: {analysis_year}")
    print(f"\nNumber of countries analyzed:")
    print(f"  Low income: {country_counts['Low income']} countries")
    print(f"  High income: {country_counts['High income']} countries")
    print(f"  All countries: {country_counts['All']} countries")
    print("\n" + "="*80)
    
    # Run multi-year analysis for all groups
    print("\n" + "="*80)
    print("STARTING MULTI-YEAR ANALYSIS FOR ALL INCOME GROUPS")
    print("="*80)
    
    # 1. Low income countries - multi-year
    analyze_multiple_years('Low income')
    
    # 2. High income countries - multi-year
    analyze_multiple_years('High income')
    
    # 3. All countries - multi-year
    analyze_multiple_years(None)
    
    print("\n" + "="*80)
    print("ALL ANALYSES COMPLETE!")
    print("="*80)

def analyze_multiple_years(income_group_filter=None):
    """Run Spearman correlation analysis for multiple years after 2010
    
    Args:
        income_group_filter: Optional income group to filter by (e.g., 'Low income', 'High income')
    """
    income_label = income_group_filter if income_group_filter else "All Countries"
    
    print("="*80)
    print(f"MULTI-YEAR SPEARMAN RANK CORRELATION ANALYSIS: {income_label.upper()}")
    print("Measles Incidence Rate vs Various Factors (2011-2024)")
    print("="*80)
    
    # Load all data
    data_dict = load_all_data()
    measles_df, _, incidence_col = data_dict['measles']
    
    # Get available years from measles data (after 2010)
    available_years = sorted([y for y in measles_df['Year'].unique() if y > 2010])
    print(f"\nAvailable years for analysis: {available_years}")
    
    # Store results for all years
    all_results = []
    
    # Metric display names
    metric_display_names = {
        'GDP per capita, PPP (constant 2021 international $)': 'GDP per Capita',
        'Public health expenditure as a share of GDP': 'Health Expenditure (% GDP)',
        'Share that disagrees vaccines are effective': 'Vaccine Hesitancy (%)',
        'Urban population (% of total population)': 'Urban Population (%)',
        'Birth Rate': 'Birth Rate',
        'Net Migration': 'Net Migration',
        'Population Density': 'Population Density',
        'Political Stability Index': 'Political Stability Index',
        'Average household size (number of members)': 'Household Size',
        'vaccine_coverage_1st_COVERAGE': 'Vaccine Coverage 1st Dose (%)',
        'vaccine_coverage_2nd_COVERAGE': 'Vaccine Coverage 2nd Dose (%)'
    }
    
    # Process each year
    for year in available_years:
        print(f"\n{'='*60}")
        print(f"Processing year {year} ({income_label})")
        print(f"{'='*60}")
        
        try:
            # Prepare correlation data for this year
            correlation_df = prepare_correlation_data(data_dict, year, income_group_filter)
            
            if len(correlation_df) < 3:
                print(f"  Skipping year {year}: insufficient data ({len(correlation_df)} countries)")
                continue
            
            # Calculate correlations
            correlations, pvalues, sample_sizes = calculate_spearman_correlations(
                correlation_df, incidence_col
            )
            
            # Store results
            for metric_name, metric_col in correlations.items():
                display_name = metric_display_names.get(metric_name, metric_name.replace('_', ' ').title())
                all_results.append({
                    'Year': year,
                    'Metric': display_name,
                    'Correlation': correlations[metric_name],
                    'P-value': pvalues[metric_name],
                    'N': sample_sizes[metric_name],
                    'Significance': format_significance(pvalues[metric_name])
                })
            
            print(f"  Processed {len(correlations)} metrics for {len(correlation_df)} countries")
            
        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nNo results to display!")
        return results_df
    
    # Save detailed results to spearman folder
    ensure_results_dir()
    spearman_dir = RESULTS_DIR / 'spearman'
    spearman_dir.mkdir(exist_ok=True)
    
    # Create income-specific subdirectory
    if income_group_filter:
        income_suffix = income_group_filter.lower().replace(' ', '_')
        income_dir = spearman_dir / income_suffix
        income_dir.mkdir(exist_ok=True)
    else:
        income_dir = spearman_dir / 'all'
        income_dir.mkdir(exist_ok=True)
    
    results_file = income_dir / 'spearman_correlation_multiyear_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to {results_file}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Correlation trends over time for each metric
    metrics = results_df['Metric'].unique()
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    fig.suptitle(f'Spearman Correlation Trends Over Time\n({income_label})', 
                 fontsize=16, fontweight='bold', y=1.02)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        if idx < len(axes):
            metric_data = results_df[results_df['Metric'] == metric].sort_values('Year')
            ax = axes[idx]
            
            # Plot correlation with confidence intervals (if we had them)
            ax.plot(metric_data['Year'], metric_data['Correlation'], 
                   marker='o', linewidth=2, markersize=6, label='Correlation')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Year', fontsize=10)
            ax.set_ylabel('Spearman ρ', fontsize=10)
            ax.set_title(f'{metric}\n(n={metric_data["N"].min()}-{metric_data["N"].max()})', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1, 1)
            
            # Add significance markers
            for _, row in metric_data.iterrows():
                if row['Significance'] and row['Significance'] != 'ns':
                    ax.scatter(row['Year'], row['Correlation'], 
                             s=100, marker='*', color='red', zorder=5)
    
    # Hide unused subplots
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    trends_file = income_dir / 'spearman_correlation_multiyear_trends.png'
    fig.savefig(trends_file, dpi=300, bbox_inches='tight')
    print(f"  Saved trends plot to {trends_file}")
    plt.close()
    
    # 2. Heatmap of correlations across years
    # Use pivot_table to handle missing values
    pivot_corr = results_df.pivot_table(index='Metric', columns='Year', values='Correlation', aggfunc='first')
    pivot_n = results_df.pivot_table(index='Metric', columns='Year', values='N', aggfunc='first')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, len(metrics) * 0.5)))
    
    # Correlation heatmap - no annotations, just color coding
    sns.heatmap(pivot_corr, annot=False, fmt='', cmap='coolwarm', center=0,
                square=False, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=axes[0],
                xticklabels=True, yticklabels=True)
    axes[0].set_title(f'Spearman Correlation Coefficients Across Years\n({income_label})', 
                     fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Year', fontsize=12)
    axes[0].set_ylabel('Metric', fontsize=12)
    
    
    sns.heatmap(pivot_n, annot=False, fmt='', cmap='YlOrRd', 
                square=False, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Sample Size (n)"}, 
                ax=axes[1], xticklabels=True, yticklabels=True)
    axes[1].set_title(f'Sample Sizes (n) Across Years\n({income_label})', 
                     fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Year', fontsize=12)
    axes[1].set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    heatmap_file = income_dir / 'spearman_correlation_multiyear_heatmap.png'
    fig.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"  Saved multiyear heatmap to {heatmap_file}")
    plt.close()
    
    # 3. Individual heatmaps for each year (saved in year-specific folders within income directory)
    years_to_plot = sorted(results_df['Year'].unique())
    
    for year in years_to_plot:
        # Create year-specific folder within income directory
        year_dir = income_dir / str(year)
        year_dir.mkdir(exist_ok=True)
        
        year_data = results_df[results_df['Year'] == year].copy()
        
        if len(year_data) == 0:
            continue
        
        # Create matrices for this year
        year_metrics = year_data['Metric'].values
        year_corr = year_data['Correlation'].values
        year_n = year_data['N'].values
        year_pval = year_data['P-value'].values
        
        # Create correlation matrix (just correlations with incidence)
        corr_matrix = pd.DataFrame({
            'Measles Incidence': year_corr
        }, index=year_metrics)
        
        # Create annotation with correlation, p-value, and sample size
        annot_matrix = corr_matrix.copy()
        for i, metric in enumerate(year_metrics):
            # Format p-value: show scientific notation if very small, otherwise 3 decimals
            if year_pval[i] < 0.001:
                pval_str = f'p<0.001'
            elif year_pval[i] < 0.01:
                pval_str = f'p={year_pval[i]:.3f}'
            else:
                pval_str = f'p={year_pval[i]:.3f}'
            
            annot_matrix.loc[metric, 'Measles Incidence'] = f'ρ={year_corr[i]:.2f}\n{pval_str}\n(n={int(year_n[i])})'
        
        # Create individual heatmap for this year
        # Increase height slightly to accommodate 3-line annotations
        fig, ax = plt.subplots(figsize=(10, max(7, len(year_metrics) * 0.5)))
        sns.heatmap(corr_matrix, annot=annot_matrix, fmt='', cmap='coolwarm', center=0,
                   square=False, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                   xticklabels=True, yticklabels=True)
        ax.set_title(f'Year {year}\nCorrelations with Measles Incidence Rate\n({income_label})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('Metric', fontsize=12)
        
        plt.tight_layout()
        year_file = year_dir / f'spearman_correlation_heatmap_{year}.png'
        fig.savefig(year_file, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap for year {year} to {year_file}")
        plt.close()
        
        # Also save CSV with detailed results for this year
        year_csv = year_dir / f'spearman_correlation_results_{year}.csv'
        year_data.to_csv(year_csv, index=False)
        print(f"  Saved detailed results for year {year} to {year_csv}")
        
        # Create full correlation matrix for this year
        try:
            # Prepare correlation data for this year
            year_correlation_df = prepare_correlation_data(data_dict, year, income_group_filter)
            if len(year_correlation_df) >= 3:
                matrix_file = year_dir / f'spearman_correlation_matrix_{year}.png'
                create_correlation_matrix_heatmap(
                    year_correlation_df, incidence_col, year, income_group_filter,
                    output_path=matrix_file
                )
                print(f"  Saved correlation matrix for year {year} to {matrix_file}")
        except Exception as e:
            print(f"  Warning: Could not create correlation matrix for year {year}: {e}")
    
    # 4. Summary table by year
    print("\n" + "="*80)
    print("SUMMARY BY YEAR")
    print("="*80)
    for year in sorted(results_df['Year'].unique()):
        year_data = results_df[results_df['Year'] == year].sort_values('Correlation', key=abs, ascending=False)
        print(f"\nYear {year}:")
        print("-" * 80)
        print(f"{'Metric':<35} {'Correlation':<12} {'P-value':<12} {'N':<8} {'Sig':<6}")
        print("-" * 80)
        for _, row in year_data.iterrows():
            print(f"{row['Metric']:<35} {row['Correlation']:>11.3f}  {row['P-value']:>11.4f}  {int(row['N']):>7}  {row['Significance']:<6}")
    
    # 5. Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary_stats = results_df.groupby('Metric').agg({
        'Correlation': ['mean', 'std', 'min', 'max'],
        'N': ['mean', 'min', 'max']
    }).round(3)
    print(summary_stats)
    
    print("\n" + "="*80)
    print("MULTI-YEAR ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--multiyear':
        # Run multi-year analysis for all income groups
        print("="*80)
        print("MULTI-YEAR ANALYSIS FOR ALL INCOME GROUPS")
        print("="*80)
        analyze_multiple_years('Low income')
        analyze_multiple_years('High income')
        analyze_multiple_years(None)
        print("\n" + "="*80)
        print("ALL MULTI-YEAR ANALYSES COMPLETE!")
        print("="*80)
    else:
        main()

