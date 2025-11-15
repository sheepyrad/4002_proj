"""
Run PCA regression analysis: Measles Incidence vs Vaccine Coverage + PCs
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf

def run_pca_regression(correlation_df, incidence_col, n_components=3):
    """Run PCA on covariates and regression with vaccine coverage + PCs."""
    exclude_cols = ["Code", incidence_col]
    X = correlation_df.drop(columns=[c for c in exclude_cols if c in correlation_df.columns]).dropna()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)

    # Add PCs back
    for i in range(n_components):
        correlation_df[f"PC{i+1}"] = components[:, i]

    # Loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X.columns
    )

    print("\nPCA Loadings:")
    print(loadings)

    # Choose coverage column
    coverage_col = "vaccine_coverage_1st_COVERAGE"
    if coverage_col not in correlation_df.columns:
        coverage_col = "vaccine_coverage_2nd_COVERAGE"

    formula = f"Q('{incidence_col}') ~ Q('{coverage_col}')"
    for i in range(n_components):
        formula += f" + PC{i+1}"

    model = smf.ols(formula, data=correlation_df).fit()
    print("\nRegression Summary:")
    print(model.summary())

    return model, loadings
