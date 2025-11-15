"""
PCA regression analysis: Measles incidence vs vaccine coverage + socioeconomic PCs
"""

import pandas as pd
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


def run_pca_regression(correlation_df, incidence_col, n_components=3):
    """
    Run PCA on socioeconomic covariates with imputation and fit a regression model
    predicting measles incidence using vaccine coverage + PCs. Saves no files (used within the loop).
    """
    # 1) PCA on covariates only
    X = _select_pca_covariates(correlation_df, incidence_col)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    for i in range(n_components):
        correlation_df[f"PC{i+1}"] = components[:, i]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X.columns
    )

    print("\nPCA Loadings (covariate contributions to PCs):")
    print(loadings)

    # 2) Choose coverage column
    coverage_col = "vaccine_coverage_1st_COVERAGE" if "vaccine_coverage_1st_COVERAGE" in correlation_df.columns else None
    if coverage_col is None and "vaccine_coverage_2nd_COVERAGE" in correlation_df.columns:
        coverage_col = "vaccine_coverage_2nd_COVERAGE"
    if coverage_col is None:
        raise ValueError("No coverage column found in correlation_df.")

    # 3) Build a safe regression target without quoting issues
    #    Create a temporary clean column name and use that in the formula.
    target_alias = "__incidence__"
    correlation_df[target_alias] = correlation_df[incidence_col]

    predictors = [coverage_col] + [f"PC{i+1}" for i in range(n_components)]
    formula = f"{target_alias} ~ " + " + ".join(predictors)

    # Debug visibility
    print(f"\nRegression formula: {formula}")

    # 4) Fit regression
    model = smf.ols(formula, data=correlation_df).fit()
    print("\nRegression Summary:")
    print(model.summary())

    return model, loadings


def analyze_pca_multiple_years(data_dict, incidence_col, start_year=2010, end_year=2024, n_components=3):
    """
    Run PCA regression for multiple years and collect results.
    Also saves separate regression reports and loadings per year.
    """
    from correlation_analysis import prepare_correlation_data

    reports_dir = Path(__file__).parent / "results" / "yearly_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for year in range(start_year, end_year + 1):
        print(f"\n{'='*60}\nProcessing year {year}\n{'='*60}")
        try:
            correlation_df = prepare_correlation_data(data_dict, year)
            print(f"  correlation_df shape: {correlation_df.shape}")

            if len(correlation_df) == 0:
                print(f"  Skipping year {year}: no data")
                continue

            # Ensure required predictors exist
            has_coverage = (
                ("vaccine_coverage_1st_COVERAGE" in correlation_df.columns) or
                ("vaccine_coverage_2nd_COVERAGE" in correlation_df.columns)
            )
            if not has_coverage:
                print(f"  Skipping year {year}: no coverage columns present")
                continue

            # Run PCA + regression
            model, loadings = run_pca_regression(correlation_df, incidence_col, n_components=n_components)

            # Save regression summary to text file
            report_path = reports_dir / f"pca_regression_{year}.txt"
            with open(report_path, "w") as f:
                f.write(model.summary().as_text())
            print(f"  Saved regression report to {report_path}")

            # Save loadings to CSV
            loadings_path = reports_dir / f"pca_loadings_{year}.csv"
            loadings.to_csv(loadings_path)
            print(f"  Saved PCA loadings to {loadings_path}")

            # Store summary stats
            # Coverage param could be either first or second
            coverage_param = None
            for name in ["vaccine_coverage_1st_COVERAGE", "vaccine_coverage_2nd_COVERAGE"]:
                if name in model.params.index:
                    coverage_param = model.params[name]
                    break

            row = {
                "Year": year,
                "N": len(correlation_df),
                "R2": model.rsquared,
                "Adj_R2": model.rsquared_adj,
                "Coverage_coef": coverage_param,
            }
            # Add PC coefs if present
            for i in range(n_components):
                pc_name = f"PC{i+1}"
                row[f"{pc_name}_coef"] = model.params.get(pc_name, None)

            all_results.append(row)

        except Exception as e:
            print(f"  Error processing year {year}: {e}")
            # Also save an error note so the summary isn’t empty silently
            error_path = reports_dir / f"ERROR_{year}.txt"
            with open(error_path, "w") as f:
                f.write(str(e))
            print(f"  Saved error report to {error_path}")
            continue

    results_df = pd.DataFrame(all_results)
    print("\nSummary of PCA regression across years:")
    print(results_df)
    return results_df
