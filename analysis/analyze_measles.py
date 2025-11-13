"""
Analysis script for Measles Reporting Data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
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

def create_yearly_geopandas_maps():
    """Create yearly choropleth maps of measles incidence and total cases using GeoPandas"""
    print("\n" + "="*60)
    print("CREATING YEARLY GEOPANDAS MAPS - MEASLES DATA")
    print("="*60)
    
    # Load measles data
    df = load_data()
    incidence_col = 'Measles incidence rate per 1\'000\'000  total population'
    cases_col = 'Total confirmed  measles cases'
    iso_col = 'ISO country code'
    
    # Get unique years
    years = sorted(df['Year'].unique())
    print(f"\nCreating maps for {len(years)} years: {years[0]} - {years[-1]}")
    
    # Load world map data
    world = None
    world_iso_col = None
    
    # Try multiple methods to load world map with country codes
    print("Loading world map data...")
    
    # Method 1: Try naturalearth_lowres (has ISO codes and country names)
    try:
        dataset_path = gpd.datasets.get_path('naturalearth_lowres')
        print(f"Loading dataset from: {dataset_path}")
        world = gpd.read_file(dataset_path)
        print(f"Loaded dataset with {len(world)} features")
        print(f"Available columns: {world.columns.tolist()}")
        
        # Check if this is actually naturalearth.land (which only has geometry)
        if len(world.columns) <= 4 and 'geometry' in world.columns:
            print("Warning: Dataset appears to be naturalearth.land (no country data)")
            print("Trying alternative method...")
            raise ValueError("Dataset lacks country identifiers")
        
        # Check for ISO column (preferred)
        iso_cols = ['ISO_A3', 'iso_a3', 'ISO3', 'iso3']
        for col in iso_cols:
            if col in world.columns:
                world_iso_col = col
                print(f"Using ISO column: {world_iso_col}")
                break
        
        # If no ISO column, use country name
        if world_iso_col is None:
            name_cols = ['NAME', 'name', 'NAME_EN', 'name_en']
            for col in name_cols:
                if col in world.columns:
                    world_iso_col = col
                    print(f"Using country name column: {world_iso_col}")
                    break
    except Exception as e:
        print(f"Error loading naturalearth_lowres: {e}")
        # Method 2: Try downloading from a reliable GeoJSON source
        try:
            print("Attempting to download world map from alternative source...")
            world = gpd.read_file('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
            print(f"Loaded world map from URL with {len(world)} features")
            print(f"Available columns: {world.columns.tolist()}")
            
            # Check for ISO or name columns
            iso_cols = ['ISO_A3', 'iso_a3', 'ISO3', 'iso3', 'id', 'ID']
            name_cols = ['NAME', 'name', 'NAME_EN', 'name_en', 'country', 'Country']
            
            for col in iso_cols + name_cols:
                if col in world.columns:
                    world_iso_col = col
                    print(f"Using column: {world_iso_col}")
                    break
        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            print("Skipping GeoPandas maps - world map data not available")
            return
    
    if world is None or world_iso_col is None:
        print("Could not find suitable world map with country identifiers")
        print("Skipping GeoPandas maps")
        return
    
    # Ensure results directory exists
    ensure_results_dir()
    maps_dir = RESULTS_DIR / 'measles_yearly_maps'
    maps_dir.mkdir(exist_ok=True)
    
    # Helper function to create a map for a given column
    def create_map_for_column(year, column_name, title, label, filename_suffix, cmap='YlOrRd'):
        """Helper function to create a choropleth map for a specific column"""
        # Filter data for this year
        year_data = df[df['Year'] == year].copy()
        
        if year_data.empty:
            return False
        
        # Prepare data for merging
        year_data_clean = year_data[[iso_col, 'Member State', column_name]].copy()
        
        # If we're using country names, we need to match by country name instead of ISO code
        if world_iso_col in ['NAME', 'name', 'NAME_EN', 'name_en', 'country', 'Country']:
            # Match by country name
            year_data_clean = year_data_clean.rename(columns={'Member State': world_iso_col})
            year_data_clean = year_data_clean.dropna(subset=[world_iso_col, column_name])
            # Clean country names for better matching
            year_data_clean[world_iso_col] = year_data_clean[world_iso_col].str.strip()
            
            # Create a mapping for common name variations
            name_mapping = {
                'United States of America': 'United States',
                'United States': 'United States of America',
                'Russian Federation': 'Russia',
                'Russia': 'Russian Federation',
                'Czech Republic': 'Czechia',
                'Czechia': 'Czech Republic',
                'Republic of the Congo': 'Congo',
                'Democratic Republic of the Congo': 'Congo, Democratic Republic of',
                'Myanmar': 'Burma',
                'Burma': 'Myanmar',
            }
            
            # Apply name mapping if needed
            for old_name, new_name in name_mapping.items():
                if old_name in year_data_clean[world_iso_col].values:
                    year_data_clean[world_iso_col] = year_data_clean[world_iso_col].replace(old_name, new_name)
        else:
            # Match by ISO code
            year_data_clean = year_data_clean.rename(columns={iso_col: world_iso_col})
            year_data_clean = year_data_clean.dropna(subset=[world_iso_col, column_name])
        
        # Merge with world map
        world_year = world.merge(
            year_data_clean,
            on=world_iso_col,
            how='left'
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        
        # Plot world map
        world_year.plot(
            column=column_name,
            ax=ax,
            legend=True,
            cmap=cmap,
            missing_kwds={'color': 'lightgrey', 'label': 'No data'},
            legend_kwds={
                'label': label,
                'orientation': 'horizontal',
                'shrink': 0.8,
                'pad': 0.02
            },
            edgecolor='black',
            linewidth=0.3
        )
        
        ax.set_title(
            title,
            fontsize=18,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        
        # Add text with data info
        countries_with_data = year_data_clean[world_iso_col].nunique()
        total_countries = len(world)
        info_text = f'Countries with data: {countries_with_data}/{total_countries}'
        ax.text(
            0.02, 0.02,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # Save figure
        filename = maps_dir / f'measles_{filename_suffix}_{year}.png'
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {filename_suffix} map for {year}: {filename}")
        return True
    
    # Create maps for each year - both incidence rate and total cases
    for year in years:
        print(f"\nProcessing year {year}...")
        
        # Create incidence rate map
        create_map_for_column(
            year=year,
            column_name=incidence_col,
            title=f'Measles Incidence Rate by Country - {year}',
            label='Measles Incidence Rate (per 1M population)',
            filename_suffix='incidence',
            cmap='YlOrRd'
        )
        
        # Create total cases map
        create_map_for_column(
            year=year,
            column_name=cases_col,
            title=f'Total Confirmed Measles Cases by Country - {year}',
            label='Total Confirmed Cases',
            filename_suffix='cases',
            cmap='Reds'
        )
    
    print(f"\n{'='*60}")
    print(f"All yearly maps saved to: {maps_dir}")
    print(f"  - Incidence rate maps: measles_incidence_*.png")
    print(f"  - Total cases maps: measles_cases_*.png")
    print(f"{'='*60}")

if __name__ == '__main__':
    df, cases_col, incidence_col = analyze_measles()
    create_yearly_geopandas_maps()

