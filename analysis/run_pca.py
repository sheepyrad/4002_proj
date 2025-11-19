"""
Run multi-year PCA regression analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from correlation_analysis import load_all_data
from pca_regression import analyze_pca_multiple_years

if __name__ == '__main__':
    print("Loading all datasets...")
    data_dict = load_all_data()
    _, _, incidence_col = data_dict['measles']

    results_df = analyze_pca_multiple_years(
        data_dict,
        incidence_col,
        start_year=2015,
        end_year=2024,
        n_components=3,
        use_both_coverage=True  # Include both 1st and 2nd coverage in formula
    )

    results_path = Path(__file__).parent / "results" / "pca_regression_multiyear.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSummary results saved to {results_path}")
