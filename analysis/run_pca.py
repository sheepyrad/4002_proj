"""
Run PCA regression analysis
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from correlation_analysis import prepare_correlation_data, load_all_data, find_best_year_for_analysis
from pca_regression import run_pca_regression

if __name__ == '__main__':
    # Load data
    data_dict = load_all_data()
    analysis_year = find_best_year_for_analysis(data_dict)
    correlation_df = prepare_correlation_data(data_dict, analysis_year)
    _, _, incidence_col = data_dict['measles']

    # Run PCA regression
    run_pca_regression(correlation_df, incidence_col, n_components=3)
