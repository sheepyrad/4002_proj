"""
Utility functions for data analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

# Set paths
DATA_DIR = Path(__file__).parent.parent / 'data'
RESULTS_DIR = Path(__file__).parent.parent / 'results'

def ensure_results_dir():
    """Ensure results directory exists"""
    RESULTS_DIR.mkdir(exist_ok=True)

def save_figure(fig, filename):
    """Save figure to results directory"""
    ensure_results_dir()
    # Handle subdirectories in filename (e.g., 'spearman/subfolder/file.png')
    filepath = RESULTS_DIR / filename
    # Create parent directories if they don't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {filepath}")

def convert_wide_to_long(df, id_vars, value_name):
    """Convert wide format data to long format"""
    df_long = df.melt(
        id_vars=id_vars,
        var_name='Year',
        value_name=value_name
    )
    # Convert Year to integer and filter out non-numeric years
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long[df_long['Year'].notna()].copy()
    df_long['Year'] = df_long['Year'].astype(int)
    # Convert value column to numeric
    df_long[value_name] = pd.to_numeric(df_long[value_name], errors='coerce')
    # Rename columns to match other datasets
    df_long = df_long.rename(columns={'Country Name': 'Entity', 'Country Code': 'Code'})
    return df_long

def extract_year_from_date(date_str):
    """Extract year from date string in format dd/mm/yyyy"""
    try:
        if pd.isna(date_str):
            return None
        parts = str(date_str).split('/')
        if len(parts) == 3:
            year = parts[2]
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            return int(year)
    except:
        return None
    return None

