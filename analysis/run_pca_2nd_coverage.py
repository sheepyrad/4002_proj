"""
Run multi-year PCA regression analysis using both 1st and 2nd dose coverage in the same formula
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from correlation_analysis import load_all_data
from pca_regression import analyze_pca_multiple_years
from plot_top3_loadings_counts import (
    plot_top3_loadings_counts, 
    plot_top3_coefficients_counts
)

if __name__ == '__main__':
    print("="*60)
    print("PCA REGRESSION ANALYSIS - COMBINED 1ST AND 2ND DOSE COVERAGE")
    print("="*60)
    print("\nLoading all datasets...")
    data_dict = load_all_data()
    _, _, incidence_col = data_dict['measles']

    # Run analysis with both coverage columns in the formula
    print("\n" + "="*60)
    print("ANALYSIS: BOTH 1ST AND 2ND DOSE COVERAGE IN FORMULA")
    print("="*60)
    results_df = analyze_pca_multiple_years(
        data_dict,
        incidence_col,
        start_year=2015,
        end_year=2024,
        n_components=3,
        use_both_coverage=True
    )

    results_path = Path(__file__).parent / "results" / "pca_regression_multiyear_combined_coverage.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nSummary results saved to {results_path}")
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    print("\nPlotting Combined Coverage Analysis...")
    plot_top3_loadings_counts(suffix="_combined_coverage")
    plot_top3_coefficients_counts(suffix="_combined_coverage")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print(f"  - {results_path}")
    print(f"  - analysis/results/yearly_reports_combined_coverage/")
    print(f"  - results/pca_regression_combined_coverage/")
    print("\nRegression formula includes both vaccine_coverage_1st_COVERAGE and")
    print("vaccine_coverage_2nd_COVERAGE (when available) as separate predictors.")
    print("="*60)

