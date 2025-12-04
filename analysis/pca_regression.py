"""
PCA regression analysis: Measles incidence vs vaccine coverage + socioeconomic PCs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import statsmodels.formula.api as smf
from pathlib import Path


def _select_pca_covariates(correlation_df, incidence_col):
    """
    Select numeric socioeconomic covariates for PCA, explicitly excluding identifiers,
    incidence, and coverage columns.
    """
    exclude = {
        "Code",
        incidence_col,
        "vaccine_coverage_1st_COVERAGE",
        "vaccine_coverage_2nd_COVERAGE",
    }
    # Keep only numeric columns
    numeric_df = correlation_df.select_dtypes(include=["number"]).copy()
    # Drop excluded columns if present
    to_drop = [c for c in exclude if c in numeric_df.columns]
    X = numeric_df.drop(columns=to_drop, errors="ignore")

    # Defensive assertions
    assert "vaccine_coverage_1st_COVERAGE" not in X.columns, "Coverage 1st leaked into PCA inputs."
    assert "vaccine_coverage_2nd_COVERAGE" not in X.columns, "Coverage 2nd leaked into PCA inputs."
    assert incidence_col not in X.columns, "Incidence leaked into PCA inputs."

    # Also ensure we have at least 3 covariates to compute 3 PCs
    if X.shape[1] < 3:
        raise ValueError(f"Insufficient covariates for PCA (got {X.shape[1]}).")

    return X


def standardize_data(X, impute_strategy="mean"):
    """
    Standardize data for PCA analysis.
    
    This function performs two critical preprocessing steps:
    1. Imputation: Handles missing values (required before standardization)
    2. Standardization: Z-score normalization (mean=0, std=1)
    
    Why standardization is essential for PCA:
    - PCA is sensitive to the scale of variables
    - Variables with larger scales/variance will dominate the principal components
    - Standardization ensures all variables contribute equally to the analysis
    - Without standardization, PCA results would be biased toward variables with larger scales
    
    Parameters:
    -----------
    X : array-like or DataFrame
        Input data with potential missing values
    impute_strategy : str, default="mean"
        Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
    
    Returns:
    --------
    X_scaled : ndarray
        Standardized data ready for PCA (mean=0, std=1 for each variable)
    imputer : SimpleImputer
        Fitted imputer (for potential inverse transformation)
    scaler : StandardScaler
        Fitted scaler (for potential inverse transformation)
    """
    # Step 1: Handle missing values
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)
    
    # Step 2: Standardize (z-score normalization)
    # This transforms each variable to have mean=0 and std=1
    # Formula: z = (x - mean) / std
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, imputer, scaler


def run_pca_regression(correlation_df, incidence_col, n_components=3, use_both_coverage=True):
    """
    Run PCA on socioeconomic covariates with imputation and fit a regression model
    predicting measles incidence using vaccine coverage + PCs. Saves no files (used within the loop).
    
    Preprocessing pipeline:
    1. Select relevant covariates (exclude identifiers, incidence, coverage)
    2. Impute missing values
    3. Standardize data (z-score normalization) - CRITICAL for PCA
    4. Apply PCA to standardized data
    5. Fit regression model with coverage + PCs
    
    Args:
        correlation_df: DataFrame with covariates and incidence data
        incidence_col: Name of the incidence column
        n_components: Number of principal components to compute
        use_both_coverage: If True, include both 1st and 2nd coverage in the formula (if available)
    """
    # 1) Select covariates for PCA
    X = _select_pca_covariates(correlation_df, incidence_col)
    print(f"\nSelected {X.shape[1]} covariates for PCA: {list(X.columns)}")

    # 2) Preprocess data: impute missing values and standardize
    # Standardization is ESSENTIAL before PCA to ensure all variables contribute equally
    print("\nPreprocessing data for PCA:")
    print("  - Step 1: Imputing missing values...")
    print("  - Step 2: Standardizing data (z-score normalization)...")
    X_scaled, imputer, scaler = standardize_data(X, impute_strategy="mean")
    print(f"  - Data shape after preprocessing: {X_scaled.shape}")
    print(f"  - Standardized data statistics:")
    print(f"    Mean: {X_scaled.mean(axis=0).round(3)} (should be ~0)")
    print(f"    Std: {X_scaled.std(axis=0).round(3)} (should be ~1)")

    # 3) Apply PCA to standardized data
    print(f"\nApplying PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # Print explained variance
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    for i in range(n_components):
        correlation_df[f"PC{i+1}"] = components[:, i]

    # Ensure loadings DataFrame matches the number of features
    # Handle case where n_features might differ from X.columns length
    # This can happen if some columns are constant after imputation
    n_features = pca.components_.shape[1]
    n_cols = len(X.columns)
    
    if n_features != n_cols:
        print(f"Warning: Mismatch between PCA features ({n_features}) and X.columns ({n_cols})")
        print(f"  This may occur if some covariates are constant or have insufficient variation.")
        # Check for constant columns after imputation
        X_imputed_reconstructed = imputer.transform(X)
        X_imputed_df = pd.DataFrame(X_imputed_reconstructed, columns=X.columns, index=X.index)
        constant_cols = []
        for col in X.columns:
            if X_imputed_df[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"  Found constant columns: {constant_cols}")
            # Remove constant columns from index
            valid_cols = [col for col in X.columns if col not in constant_cols]
            if len(valid_cols) == n_features:
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(n_components)],
                    index=valid_cols
                )
            else:
                # Fallback: use first n_features columns
                loadings = pd.DataFrame(
                    pca.components_.T,
                    columns=[f"PC{i+1}" for i in range(n_components)],
                    index=X.columns[:n_features]
                )
        else:
            # Use first n_features columns as fallback
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f"PC{i+1}" for i in range(n_components)],
                index=X.columns[:n_features]
            )
    else:
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=X.columns
        )

    print("\nPCA Loadings (covariate contributions to PCs):")
    print(loadings)

    # 2) Choose coverage columns for regression
    coverage_cols = []
    if use_both_coverage:
        # Include both coverage columns if available
        if "vaccine_coverage_1st_COVERAGE" in correlation_df.columns:
            coverage_cols.append("vaccine_coverage_1st_COVERAGE")
        if "vaccine_coverage_2nd_COVERAGE" in correlation_df.columns:
            coverage_cols.append("vaccine_coverage_2nd_COVERAGE")
    else:
        # Use single coverage (prefer 1st, fallback to 2nd)
        if "vaccine_coverage_1st_COVERAGE" in correlation_df.columns:
            coverage_cols.append("vaccine_coverage_1st_COVERAGE")
        elif "vaccine_coverage_2nd_COVERAGE" in correlation_df.columns:
            coverage_cols.append("vaccine_coverage_2nd_COVERAGE")
    
    if len(coverage_cols) == 0:
        raise ValueError("No coverage column found in correlation_df.")

    # 3) Build a safe regression target without quoting issues
    #    Create a temporary clean column name and use that in the formula.
    target_alias = "__incidence__"
    correlation_df[target_alias] = correlation_df[incidence_col]

    predictors = coverage_cols + [f"PC{i+1}" for i in range(n_components)]
    formula = f"{target_alias} ~ " + " + ".join(predictors)

    # Debug visibility
    print(f"\nRegression formula: {formula}")

    # 4) Fit regression
    model = smf.ols(formula, data=correlation_df).fit()
    print("\nRegression Summary:")
    print(model.summary())

    return model, loadings


def analyze_pca_multiple_years(data_dict, incidence_col, start_year=2015, end_year=2024, n_components=3, use_both_coverage=True):
    """
    Run PCA regression for multiple years and collect results.
    Also saves separate regression reports and loadings per year.
    Analyzes PC1 mapped coefficients and identifies top actionable factors.
    
    Args:
        data_dict: Dictionary of dataframes
        incidence_col: Name of the incidence column
        start_year: Starting year for analysis
        end_year: Ending year for analysis
        n_components: Number of principal components
        use_both_coverage: If True, include both 1st and 2nd coverage in the formula (if available)
    """
    from correlation_analysis import prepare_correlation_data

    # Local results directory (analysis/results/yearly_reports)
    # Add suffix for combined coverage analysis
    suffix = "_combined_coverage" if use_both_coverage else ""
    reports_dir = Path(__file__).parent / "results" / f"yearly_reports{suffix}"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Project root results directory (project_root/results)
    project_root = Path(__file__).parent.parent
    root_results_dir = project_root / "results" / f"pca_regression{suffix}"
    root_results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    # Store mapped coefficients for PC1 across years
    pc1_mapped_coefs = {}  # {factor_name: {year: mapped_coef, ...}, ...}
    top3_per_year = []  # List of {year, top3_factors} dicts
    
    # Store mapped coefficients for all PCs across years
    pc_mapped_coefs = {
        "PC1": {},  # {factor_name: {year: mapped_coef, ...}, ...}
        "PC2": {},
        "PC3": {}
    }
    
    # Store top 3 coefficients per year for each PC
    top3_coefficients_per_pc = {
        "PC1": [],  # List of {year, top3_factors, top3_coefficients} dicts
        "PC2": [],
        "PC3": []
    }
    
    # Store top 3 loadings for each PC across years
    top3_loadings_per_pc = {
        "PC1": [],  # List of {year, top3_factors, top3_loadings} dicts
        "PC2": [],
        "PC3": []
    }

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}\nProcessing year {year}\n{'='*60}")
        try:
            correlation_df = prepare_correlation_data(data_dict, year)
            print(f"  correlation_df shape: {correlation_df.shape}")

            if len(correlation_df) == 0:
                print(f"  Skipping year {year}: no data")
                continue

            # Ensure required predictors exist
            if use_both_coverage:
                has_coverage = (
                    ("vaccine_coverage_1st_COVERAGE" in correlation_df.columns) or
                    ("vaccine_coverage_2nd_COVERAGE" in correlation_df.columns)
                )
            else:
                has_coverage = (
                    ("vaccine_coverage_1st_COVERAGE" in correlation_df.columns) or
                    ("vaccine_coverage_2nd_COVERAGE" in correlation_df.columns)
                )
            if not has_coverage:
                print(f"  Skipping year {year}: no coverage columns present")
                continue

            # Run PCA + regression
            model, loadings = run_pca_regression(correlation_df, incidence_col, n_components=n_components, use_both_coverage=use_both_coverage)

            # Save regression summary to text file
            report_path = reports_dir / f"pca_regression_{year}.txt"
            with open(report_path, "w") as f:
                f.write(model.summary().as_text())
            print(f"  Saved regression report to {report_path}")

            # Save loadings to CSV
            loadings_path = reports_dir / f"pca_loadings_{year}.csv"
            loadings.to_csv(loadings_path)
            print(f"  Saved PCA loadings to {loadings_path}")

            # Extract top 3 loadings for each PC (by absolute value)
            for pc_num in range(1, n_components + 1):
                pc_name = f"PC{pc_num}"
                if pc_name in loadings.columns:
                    pc_loadings = loadings[pc_name]
                    abs_loadings = pc_loadings.abs().sort_values(ascending=False)
                    top3_factors = abs_loadings.head(3).index.tolist()
                    top3_loadings = pc_loadings[top3_factors].to_dict()
                    
                    top3_loadings_per_pc[pc_name].append({
                        "Year": year,
                        "Top3_Factors": top3_factors,
                        "Top3_Loadings": top3_loadings
                    })
                    
                    print(f"\n  Top 3 loadings for {pc_name} (by absolute value):")
                    for i, factor in enumerate(top3_factors, 1):
                        print(f"    {i}. {factor}: {pc_loadings[factor]:.6f}")

            # Compute mapped coefficients for all PCs
            # mapped_coef = PC_regression_coef * PC_loading
            for pc_num in range(1, n_components + 1):
                pc_name = f"PC{pc_num}"
                pc_coef = model.params.get(pc_name, None)
                if pc_coef is not None and pc_name in loadings.columns:
                    pc_loadings = loadings[pc_name]
                    mapped_coefs = pc_coef * pc_loadings
                    
                    # Store mapped coefficients
                    for factor_name, mapped_coef in mapped_coefs.items():
                        if factor_name not in pc_mapped_coefs[pc_name]:
                            pc_mapped_coefs[pc_name][factor_name] = {}
                        pc_mapped_coefs[pc_name][factor_name][year] = mapped_coef
                    
                    # Identify top 3 factors by absolute value
                    abs_mapped = mapped_coefs.abs().sort_values(ascending=False)
                    top3_factors = abs_mapped.head(3).index.tolist()
                    top3_values = mapped_coefs[top3_factors].to_dict()
                    
                    top3_coefficients_per_pc[pc_name].append({
                        "Year": year,
                        "Top3_Factors": top3_factors,
                        "Top3_Coefficients": top3_values
                    })
            
            # Keep PC1-specific tracking for backward compatibility
            pc1_coef = model.params.get("PC1", None)
            if pc1_coef is not None and "PC1" in loadings.columns:
                pc1_loadings = loadings["PC1"]
                mapped_coefs = pc1_coef * pc1_loadings
                
                # Store mapped coefficients
                for factor_name, mapped_coef in mapped_coefs.items():
                    if factor_name not in pc1_mapped_coefs:
                        pc1_mapped_coefs[factor_name] = {}
                    pc1_mapped_coefs[factor_name][year] = mapped_coef
                
                # Identify top 3 factors by absolute value
                abs_mapped = mapped_coefs.abs().sort_values(ascending=False)
                top3_factors = abs_mapped.head(3).index.tolist()
                top3_values = mapped_coefs[top3_factors].to_dict()
                
                top3_per_year.append({
                    "Year": year,
                    "Top3_Factors": top3_factors,
                    "Top3_MappedCoefficients": top3_values
                })

            # Store summary stats
            # Coverage params could be both first and second
            coverage_1st_param = model.params.get("vaccine_coverage_1st_COVERAGE", None)
            coverage_2nd_param = model.params.get("vaccine_coverage_2nd_COVERAGE", None)

            row = {
                "Year": year,
                "N": len(correlation_df),
                "R2": model.rsquared,
                "Adj_R2": model.rsquared_adj,
                "Coverage_1st_coef": coverage_1st_param,
                "Coverage_2nd_coef": coverage_2nd_param,
            }
            # Add PC coefs if present
            for i in range(n_components):
                pc_name = f"PC{i+1}"
                row[f"{pc_name}_coef"] = model.params.get(pc_name, None)

            all_results.append(row)

        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            # Also save an error note so the summary isn't empty silently
            error_path = reports_dir / f"ERROR_{year}.txt"
            with open(error_path, "w") as f:
                f.write(str(e))
            print(f"  Saved error report to {error_path}")
            continue

    results_df = pd.DataFrame(all_results)
    print("\nSummary of PCA regression across years:")
    print(results_df)
    
    # Analyze top loadings for all PCs
    if top3_loadings_per_pc:
        analyze_pc_loadings(top3_loadings_per_pc, reports_dir, root_results_dir, n_components=n_components)
    
    # Analyze top coefficients for all PCs
    if top3_coefficients_per_pc:
        analyze_pc_coefficients(top3_coefficients_per_pc, reports_dir, root_results_dir, n_components=n_components)
    
    # Analyze top factors and create visualizations for PC1 mapped coefficients
    if pc1_mapped_coefs and top3_per_year:
        analyze_pc1_trends(pc1_mapped_coefs, top3_per_year, reports_dir, root_results_dir)
    
    return results_df


def analyze_pc1_trends(pc1_mapped_coefs, top3_per_year, reports_dir, root_results_dir=None):
    """
    Analyze PC1 mapped coefficient trends and identify actionable targets.
    
    Args:
        pc1_mapped_coefs: Dict mapping factor names to dicts of {year: mapped_coef}
        top3_per_year: List of dicts with top 3 factors per year
        reports_dir: Directory to save results (local)
        root_results_dir: Directory to save results (project root), optional
    """
    print("\n" + "="*60)
    print("PC1 MAPPED COEFFICIENT ANALYSIS")
    print("="*60)
    
    # Convert to DataFrame for easier manipulation
    all_years = sorted(set(year_dict["Year"] for year_dict in top3_per_year))
    
    # Build DataFrame of mapped coefficients over time
    factors = list(pc1_mapped_coefs.keys())
    trend_data = []
    for year in all_years:
        row = {"Year": year}
        for factor in factors:
            row[factor] = pc1_mapped_coefs[factor].get(year, np.nan)
        trend_data.append(row)
    
    trend_df = pd.DataFrame(trend_data)
    
    # Count frequency of top 3 appearances
    factor_counts = {}
    for entry in top3_per_year:
        for factor in entry["Top3_Factors"]:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    
    # Sort by frequency
    sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 3 Factors Frequency Count (across all years):")
    print("-" * 60)
    for factor, count in sorted_factors:
        print(f"  {factor}: appears in top 3 for {count} year(s)")
    
    # Identify top 3 actionable targets
    top3_actionable = [factor for factor, _ in sorted_factors[:3]]
    print("\n" + "="*60)
    print("TOP 3 ACTIONABLE TARGETS:")
    print("="*60)
    for i, factor in enumerate(top3_actionable, 1):
        print(f"{i}. {factor} (appeared in top 3 for {factor_counts[factor]} year(s))")
    
    # Create visualization: Trend of PC1 mapped coefficients
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All factors trend
    ax1 = axes[0]
    for factor in factors:
        years = trend_df["Year"]
        values = trend_df[factor]
        # Only plot if we have at least 2 data points
        if values.notna().sum() >= 2:
            ax1.plot(years, values, marker='o', label=factor, alpha=0.7, linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("PC1 Mapped Coefficient\n(PC1_coef × PC1_loading)", fontsize=12)
    ax1.set_title("PC1 Mapped Coefficients Trend Over Time\n(All Factors)", fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top 3 actionable targets only
    ax2 = axes[1]
    for factor in top3_actionable:
        years = trend_df["Year"]
        values = trend_df[factor]
        if values.notna().sum() >= 2:
            ax2.plot(years, values, marker='o', label=f"{factor}\n({factor_counts[factor]} years)", 
                    linewidth=2.5, markersize=8)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax2.set_xlabel("Year", fontsize=12)
    ax2.set_ylabel("PC1 Mapped Coefficient\n(PC1_coef × PC1_loading)", fontsize=12)
    ax2.set_title("Top 3 Actionable Targets: PC1 Mapped Coefficients Trend", fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = reports_dir / "pc1_mapped_coefficients_trends.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved trend plot to {plot_path}")
    # Also save to project root
    if root_results_dir:
        root_plot_path = root_results_dir / "pc1_mapped_coefficients_trends.png"
        plt.savefig(root_plot_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_plot_path}")
    plt.close()
    
    # Create bar chart of frequency counts
    fig, ax = plt.subplots(figsize=(12, 8))
    factors_plot = [f for f, _ in sorted_factors[:10]]  # Top 10
    counts_plot = [c for _, c in sorted_factors[:10]]
    colors = ['#1f77b4' if f in top3_actionable else '#888888' for f in factors_plot]
    
    bars = ax.barh(factors_plot, counts_plot, color=colors)
    ax.set_xlabel("Number of Years in Top 3", fontsize=12)
    ax.set_ylabel("Factor", fontsize=12)
    ax.set_title("Frequency of Top 3 PC1 Factors Across All Years", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts_plot)):
        ax.text(count + 0.1, i, str(count), va='center', fontsize=10)
    
    plt.tight_layout()
    freq_path = reports_dir / "pc1_top3_frequency.png"
    plt.savefig(freq_path, dpi=300, bbox_inches='tight')
    print(f"Saved frequency plot to {freq_path}")
    # Also save to project root
    if root_results_dir:
        root_freq_path = root_results_dir / "pc1_top3_frequency.png"
        plt.savefig(root_freq_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_freq_path}")
    plt.close()
    
    # Save detailed results to CSV
    trend_csv_path = reports_dir / "pc1_mapped_coefficients_trends.csv"
    trend_df.to_csv(trend_csv_path, index=False)
    if root_results_dir:
        root_trend_csv = root_results_dir / "pc1_mapped_coefficients_trends.csv"
        trend_df.to_csv(root_trend_csv, index=False)
        print(f"Saved trend CSV to {root_trend_csv}")
    
    # Save top 3 per year summary
    top3_summary = []
    for entry in top3_per_year:
        row = {"Year": entry["Year"]}
        for i, factor in enumerate(entry["Top3_Factors"], 1):
            row[f"Rank_{i}_Factor"] = factor
            row[f"Rank_{i}_MappedCoeff"] = entry["Top3_MappedCoefficients"][factor]
        top3_summary.append(row)
    
    top3_df = pd.DataFrame(top3_summary)
    top3_csv_path = reports_dir / "pc1_top3_per_year.csv"
    top3_df.to_csv(top3_csv_path, index=False)
    print(f"Saved top 3 per year summary to {top3_csv_path}")
    if root_results_dir:
        root_top3_csv = root_results_dir / "pc1_top3_per_year.csv"
        top3_df.to_csv(root_top3_csv, index=False)
        print(f"  Also saved to {root_top3_csv}")
    
    # Save actionable targets summary
    summary_text = f"""
PC1 MAPPED COEFFICIENT ANALYSIS SUMMARY
=======================================

Top 3 Actionable Targets (by frequency in top 3 across all years):
1. {top3_actionable[0]} - appeared in top 3 for {factor_counts[top3_actionable[0]]} year(s)
2. {top3_actionable[1]} - appeared in top 3 for {factor_counts[top3_actionable[1]]} year(s)
3. {top3_actionable[2]} - appeared in top 3 for {factor_counts[top3_actionable[2]]} year(s)

Note: Mapped coefficient = PC1_regression_coefficient × PC1_loading
This represents the contribution of each original factor to measles incidence through PC1.

Files generated:
- pc1_mapped_coefficients_trends.png: Trend plots of all factors and top 3 targets
- pc1_mapped_coefficients_trends.csv: Full data of mapped coefficients over time
- pc1_top3_frequency.png: Bar chart of frequency counts
- pc1_top3_per_year.csv: Detailed top 3 factors for each year
"""
    
    summary_path = reports_dir / "pc1_actionable_targets_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved summary to {summary_path}")
    if root_results_dir:
        root_summary_path = root_results_dir / "pc1_actionable_targets_summary.txt"
        with open(root_summary_path, "w") as f:
            f.write(summary_text)
        print(f"  Also saved to {root_summary_path}")
    
    print("\n" + "="*60)


def analyze_pc_loadings(top3_loadings_per_pc, reports_dir, root_results_dir=None, n_components=3):
    """
    Analyze top 3 loadings for each PC across all years and identify most important factors.
    
    Args:
        top3_loadings_per_pc: Dict mapping PC names to lists of {year, top3_factors, top3_loadings}
        reports_dir: Directory to save results (local)
        root_results_dir: Directory to save results (project root), optional
        n_components: Number of principal components
    """
    print("\n" + "="*60)
    print("PCA LOADINGS ANALYSIS - TOP 3 FACTORS PER PC")
    print("="*60)
    
    pc_summaries = {}
    
    for pc_num in range(1, n_components + 1):
        pc_name = f"PC{pc_num}"
        if pc_name not in top3_loadings_per_pc or not top3_loadings_per_pc[pc_name]:
            continue
        
        print(f"\n{pc_name} Analysis:")
        print("-" * 60)
        
        # Count frequency of top 3 appearances
        factor_counts = {}
        for entry in top3_loadings_per_pc[pc_name]:
            for factor in entry["Top3_Factors"]:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 3 Loadings Frequency Count for {pc_name} (across all years):")
        for factor, count in sorted_factors:
            print(f"  {factor}: appears in top 3 for {count} year(s)")
        
        # Identify top 3 factors
        top3_factors = [factor for factor, _ in sorted_factors[:3]]
        
        print(f"\nTop 3 Most Important Factors for {pc_name}:")
        for i, factor in enumerate(top3_factors, 1):
            print(f"  {i}. {factor} (appeared in top 3 for {factor_counts[factor]} year(s))")
        
        pc_summaries[pc_name] = {
            "top3_factors": top3_factors,
            "factor_counts": factor_counts,
            "sorted_factors": sorted_factors
        }
    
    # Create visualizations
    create_pc_loadings_visualizations(top3_loadings_per_pc, pc_summaries, reports_dir, root_results_dir, n_components)
    
    # Save summary report
    save_pc_loadings_summary(top3_loadings_per_pc, pc_summaries, reports_dir, root_results_dir)
    
    print("\n" + "="*60)


def create_pc_loadings_visualizations(top3_loadings_per_pc, pc_summaries, reports_dir, root_results_dir=None, n_components=3):
    """Create visualizations for PC loadings analysis."""
    
    # Create frequency bar charts for each PC
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 8))
    if n_components == 1:
        axes = [axes]
    
    for idx, pc_num in enumerate(range(1, n_components + 1)):
        pc_name = f"PC{pc_num}"
        if pc_name not in pc_summaries:
            continue
        
        ax = axes[idx]
        summary = pc_summaries[pc_name]
        sorted_factors = summary["sorted_factors"]
        top3_factors = summary["top3_factors"]
        
        # Plot top 10 factors
        factors_plot = [f for f, _ in sorted_factors[:10]]
        counts_plot = [c for _, c in sorted_factors[:10]]
        colors = ['#1f77b4' if f in top3_factors else '#888888' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=colors)
        ax.set_xlabel("Number of Years in Top 3", fontsize=11)
        ax.set_ylabel("Factor", fontsize=11)
        ax.set_title(f"{pc_name} - Top 3 Loadings Frequency", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    freq_path = reports_dir / "pc_loadings_top3_frequency.png"
    plt.savefig(freq_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved frequency plots to {freq_path}")
    # Also save to project root
    if root_results_dir:
        root_freq_path = root_results_dir / "pc_loadings_top3_frequency.png"
        plt.savefig(root_freq_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_freq_path}")
    plt.close()
    
    # Create combined summary plot showing top 3 for each PC side by side
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_positions = {}
    y_pos = 0
    colors_pc = {'PC1': '#1f77b4', 'PC2': '#ff7f0e', 'PC3': '#2ca02c'}
    labels_added = set()
    
    for pc_num in range(1, n_components + 1):
        pc_name = f"PC{pc_num}"
        if pc_name not in pc_summaries:
            continue
        
        summary = pc_summaries[pc_name]
        top3_factors = summary["top3_factors"]
        factor_counts = summary["factor_counts"]
        
        for i, factor in enumerate(top3_factors):
            if factor not in y_positions:
                y_positions[factor] = y_pos
                y_pos += 1
        
        # Plot bars for this PC's top 3
        for i, factor in enumerate(top3_factors):
            y_pos_factor = y_positions[factor]
            count = factor_counts[factor]
            offset = (i - 1) * 0.25  # Offset bars for same factor across PCs
            label = pc_name if pc_name not in labels_added else ""
            if pc_name not in labels_added:
                labels_added.add(pc_name)
            ax.barh(y_pos_factor + offset, count, height=0.2, 
                   color=colors_pc[pc_name], alpha=0.7, label=label)
    
    # Set y-axis labels
    factor_list = sorted(y_positions.items(), key=lambda x: x[1])
    ax.set_yticks([pos for _, pos in factor_list])
    ax.set_yticklabels([factor for factor, _ in factor_list], fontsize=9)
    ax.set_xlabel("Number of Years in Top 3", fontsize=12)
    ax.set_ylabel("Factor", fontsize=12)
    ax.set_title("Top 3 Factors by PC - Frequency Across All Years", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    combined_path = reports_dir / "pc_loadings_top3_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined summary plot to {combined_path}")
    # Also save to project root
    if root_results_dir:
        root_combined_path = root_results_dir / "pc_loadings_top3_combined.png"
        plt.savefig(root_combined_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_combined_path}")
    plt.close()


def save_pc_loadings_summary(top3_loadings_per_pc, pc_summaries, reports_dir, root_results_dir=None):
    """Save detailed summary of PC loadings analysis."""
    
    # Save top 3 per year for each PC to CSV
    for pc_name in pc_summaries.keys():
        if pc_name not in top3_loadings_per_pc:
            continue
        
        top3_summary = []
        for entry in top3_loadings_per_pc[pc_name]:
            row = {"Year": entry["Year"]}
            for i, factor in enumerate(entry["Top3_Factors"], 1):
                row[f"Rank_{i}_Factor"] = factor
                row[f"Rank_{i}_Loading"] = entry["Top3_Loadings"][factor]
            top3_summary.append(row)
        
        top3_df = pd.DataFrame(top3_summary)
        csv_path = reports_dir / f"{pc_name.lower()}_top3_loadings_per_year.csv"
        top3_df.to_csv(csv_path, index=False)
        print(f"Saved {pc_name} top 3 per year to {csv_path}")
        # Also save to project root
        if root_results_dir:
            root_csv_path = root_results_dir / f"{pc_name.lower()}_top3_loadings_per_year.csv"
            top3_df.to_csv(root_csv_path, index=False)
            print(f"  Also saved to {root_csv_path}")
    
    # Create summary text
    summary_lines = [
        "PCA LOADINGS ANALYSIS SUMMARY",
        "=" * 60,
        "",
        "Top 3 Factors by Principal Component (based on frequency in top 3 across all years):",
        ""
    ]
    
    for pc_name, summary in pc_summaries.items():
        summary_lines.append(f"{pc_name}:")
        summary_lines.append("-" * 60)
        top3_factors = summary["top3_factors"]
        factor_counts = summary["factor_counts"]
        for i, factor in enumerate(top3_factors, 1):
            summary_lines.append(
                f"  {i}. {factor} - appeared in top 3 for {factor_counts[factor]} year(s)"
            )
        summary_lines.append("")
    
    summary_lines.extend([
        "Note: Loadings represent the contribution of each original factor to the principal component.",
        "Top 3 factors are identified by absolute loading value for each year, then counted across all years.",
        "",
        "Files generated:",
        "- pc_loadings_top3_frequency.png: Frequency bar charts for each PC",
        "- pc_loadings_top3_combined.png: Combined summary showing top 3 factors for all PCs",
        "- pc1_top3_loadings_per_year.csv: Detailed top 3 factors for PC1 each year",
        "- pc2_top3_loadings_per_year.csv: Detailed top 3 factors for PC2 each year",
        "- pc3_top3_loadings_per_year.csv: Detailed top 3 factors for PC3 each year"
    ])
    
    summary_text = "\n".join(summary_lines)
    
    summary_path = reports_dir / "pc_loadings_top3_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved PC loadings summary to {summary_path}")
    # Also save to project root
    if root_results_dir:
        root_summary_path = root_results_dir / "pc_loadings_top3_summary.txt"
        with open(root_summary_path, "w") as f:
            f.write(summary_text)
        print(f"  Also saved to {root_summary_path}")


def analyze_pc_coefficients(top3_coefficients_per_pc, reports_dir, root_results_dir=None, n_components=3):
    """
    Analyze top 3 coefficients for each PC across all years and identify most important factors.
    
    Args:
        top3_coefficients_per_pc: Dict mapping PC names to lists of {year, top3_factors, top3_coefficients}
        reports_dir: Directory to save results (local)
        root_results_dir: Directory to save results (project root), optional
        n_components: Number of principal components
    """
    print("\n" + "="*60)
    print("PCA COEFFICIENTS ANALYSIS - TOP 3 FACTORS PER PC")
    print("="*60)
    
    pc_summaries = {}
    
    for pc_num in range(1, n_components + 1):
        pc_name = f"PC{pc_num}"
        if pc_name not in top3_coefficients_per_pc or not top3_coefficients_per_pc[pc_name]:
            continue
        
        print(f"\n{pc_name} Coefficients Analysis:")
        print("-" * 60)
        
        # Count frequency of top 3 appearances
        factor_counts = {}
        for entry in top3_coefficients_per_pc[pc_name]:
            for factor in entry["Top3_Factors"]:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 3 Coefficients Frequency Count for {pc_name} (across all years):")
        for factor, count in sorted_factors:
            print(f"  {factor}: appears in top 3 for {count} year(s)")
        
        # Identify top 3 factors
        top3_factors = [factor for factor, _ in sorted_factors[:3]]
        
        print(f"\nTop 3 Most Important Factors for {pc_name} (by coefficient frequency):")
        for i, factor in enumerate(top3_factors, 1):
            print(f"  {i}. {factor} (appeared in top 3 for {factor_counts[factor]} year(s))")
        
        pc_summaries[pc_name] = {
            "top3_factors": top3_factors,
            "factor_counts": factor_counts,
            "sorted_factors": sorted_factors
        }
    
    # Create visualizations
    create_pc_coefficients_visualizations(top3_coefficients_per_pc, pc_summaries, reports_dir, root_results_dir, n_components)
    
    # Save summary report
    save_pc_coefficients_summary(top3_coefficients_per_pc, pc_summaries, reports_dir, root_results_dir)
    
    print("\n" + "="*60)


def create_pc_coefficients_visualizations(top3_coefficients_per_pc, pc_summaries, reports_dir, root_results_dir=None, n_components=3):
    """Create visualizations for PC coefficients analysis."""
    
    # Create frequency bar charts for each PC
    fig, axes = plt.subplots(1, n_components, figsize=(6*n_components, 8))
    if n_components == 1:
        axes = [axes]
    
    for idx, pc_num in enumerate(range(1, n_components + 1)):
        pc_name = f"PC{pc_num}"
        if pc_name not in pc_summaries:
            continue
        
        ax = axes[idx]
        summary = pc_summaries[pc_name]
        sorted_factors = summary["sorted_factors"]
        top3_factors = summary["top3_factors"]
        
        # Plot top 10 factors
        factors_plot = [f for f, _ in sorted_factors[:10]]
        counts_plot = [c for _, c in sorted_factors[:10]]
        colors = ['#1f77b4' if f in top3_factors else '#888888' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=colors)
        ax.set_xlabel("Number of Years in Top 3", fontsize=11)
        ax.set_ylabel("Factor", fontsize=11)
        ax.set_title(f"{pc_name} - Top 3 Coefficients Frequency", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    freq_path = reports_dir / "pc_coefficients_top3_frequency.png"
    plt.savefig(freq_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved coefficients frequency plots to {freq_path}")
    # Also save to project root
    if root_results_dir:
        root_freq_path = root_results_dir / "pc_coefficients_top3_frequency.png"
        plt.savefig(root_freq_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_freq_path}")
    plt.close()
    
    # Create combined summary plot showing top 3 for each PC side by side
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_positions = {}
    y_pos = 0
    colors_pc = {'PC1': '#1f77b4', 'PC2': '#ff7f0e', 'PC3': '#2ca02c'}
    labels_added = set()
    
    for pc_num in range(1, n_components + 1):
        pc_name = f"PC{pc_num}"
        if pc_name not in pc_summaries:
            continue
        
        summary = pc_summaries[pc_name]
        top3_factors = summary["top3_factors"]
        factor_counts = summary["factor_counts"]
        
        for i, factor in enumerate(top3_factors):
            if factor not in y_positions:
                y_positions[factor] = y_pos
                y_pos += 1
        
        # Plot bars for this PC's top 3
        for i, factor in enumerate(top3_factors):
            y_pos_factor = y_positions[factor]
            count = factor_counts[factor]
            offset = (i - 1) * 0.25  # Offset bars for same factor across PCs
            label = pc_name if pc_name not in labels_added else ""
            if pc_name not in labels_added:
                labels_added.add(pc_name)
            ax.barh(y_pos_factor + offset, count, height=0.2, 
                   color=colors_pc[pc_name], alpha=0.7, label=label)
    
    # Set y-axis labels
    factor_list = sorted(y_positions.items(), key=lambda x: x[1])
    ax.set_yticks([pos for _, pos in factor_list])
    ax.set_yticklabels([factor for factor, _ in factor_list], fontsize=9)
    ax.set_xlabel("Number of Years in Top 3", fontsize=12)
    ax.set_ylabel("Factor", fontsize=12)
    ax.set_title("Top 3 Coefficients by PC - Frequency Across All Years", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    combined_path = reports_dir / "pc_coefficients_top3_combined.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined coefficients summary plot to {combined_path}")
    # Also save to project root
    if root_results_dir:
        root_combined_path = root_results_dir / "pc_coefficients_top3_combined.png"
        plt.savefig(root_combined_path, dpi=300, bbox_inches='tight')
        print(f"  Also saved to {root_combined_path}")
    plt.close()


def save_pc_coefficients_summary(top3_coefficients_per_pc, pc_summaries, reports_dir, root_results_dir=None):
    """Save detailed summary of PC coefficients analysis."""
    
    # Save top 3 per year for each PC to CSV
    for pc_name in pc_summaries.keys():
        if pc_name not in top3_coefficients_per_pc:
            continue
        
        top3_summary = []
        for entry in top3_coefficients_per_pc[pc_name]:
            row = {"Year": entry["Year"]}
            for i, factor in enumerate(entry["Top3_Factors"], 1):
                row[f"Rank_{i}_Factor"] = factor
                row[f"Rank_{i}_Coefficient"] = entry["Top3_Coefficients"][factor]
            top3_summary.append(row)
        
        top3_df = pd.DataFrame(top3_summary)
        csv_path = reports_dir / f"{pc_name.lower()}_top3_coefficients_per_year.csv"
        top3_df.to_csv(csv_path, index=False)
        print(f"Saved {pc_name} top 3 coefficients per year to {csv_path}")
        # Also save to project root
        if root_results_dir:
            root_csv_path = root_results_dir / f"{pc_name.lower()}_top3_coefficients_per_year.csv"
            top3_df.to_csv(root_csv_path, index=False)
            print(f"  Also saved to {root_csv_path}")
    
    # Create summary text
    summary_lines = [
        "PCA COEFFICIENTS ANALYSIS SUMMARY",
        "=" * 60,
        "",
        "Top 3 Factors by Principal Component (based on frequency in top 3 coefficients across all years):",
        ""
    ]
    
    for pc_name, summary in pc_summaries.items():
        summary_lines.append(f"{pc_name}:")
        summary_lines.append("-" * 60)
        top3_factors = summary["top3_factors"]
        factor_counts = summary["factor_counts"]
        for i, factor in enumerate(top3_factors, 1):
            summary_lines.append(
                f"  {i}. {factor} - appeared in top 3 for {factor_counts[factor]} year(s)"
            )
        summary_lines.append("")
    
    summary_lines.extend([
        "Note: Mapped coefficients = PC_regression_coefficient × PC_loading",
        "This represents the contribution of each original factor to measles incidence through the PC.",
        "Top 3 factors are identified by absolute mapped coefficient value for each year, then counted across all years.",
        "",
        "Files generated:",
        "- pc_coefficients_top3_frequency.png: Frequency bar charts for each PC",
        "- pc_coefficients_top3_combined.png: Combined summary showing top 3 factors for all PCs",
        "- pc1_top3_coefficients_per_year.csv: Detailed top 3 factors for PC1 each year",
        "- pc2_top3_coefficients_per_year.csv: Detailed top 3 factors for PC2 each year",
        "- pc3_top3_coefficients_per_year.csv: Detailed top 3 factors for PC3 each year"
    ])
    
    summary_text = "\n".join(summary_lines)
    
    summary_path = reports_dir / "pc_coefficients_top3_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved PC coefficients summary to {summary_path}")
    # Also save to project root
    if root_results_dir:
        root_summary_path = root_results_dir / "pc_coefficients_top3_summary.txt"
        with open(root_summary_path, "w") as f:
            f.write(summary_text)
        print(f"  Also saved to {root_summary_path}")
