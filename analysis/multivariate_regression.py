from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import pandas as pd

def run_pca_regression(correlation_df, incidence_col, n_components=3):
    """
    Run PCA on socioeconomic covariates and fit a regression model
    predicting measles incidence using vaccine coverage + PCs.
    
    Parameters
    ----------
    correlation_df : pd.DataFrame
        Merged dataset with measles incidence and covariates.
    incidence_col : str
        Column name for measles incidence.
    n_components : int
        Number of principal components to keep.
    
    Returns
    -------
    model : statsmodels regression results
        Fitted regression model.
    loadings : pd.DataFrame
        PCA loadings showing how each covariate contributes to PCs.
    """
    # Select covariates (exclude identifiers and incidence)
    exclude_cols = ["Code", incidence_col]
    X = correlation_df.drop(columns=[c for c in exclude_cols if c in correlation_df.columns]).dropna()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(X_scaled)
    
    # Add PCs back to dataframe
    for i in range(n_components):
        correlation_df[f"PC{i+1}"] = components[:, i]
    
    # Loadings (interpretation of PCs)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=X.columns
    )
    
    print("\nPCA Loadings (covariate contributions to PCs):")
    print(loadings)
    
    # Build regression formula: incidence ~ vaccine coverage + PCs
    # Adjust coverage column name if needed
    coverage_col = "vaccine_coverage_1st_COVERAGE"
    if coverage_col not in correlation_df.columns:
        coverage_col = "vaccine_coverage_2nd_COVERAGE"
    
    formula = f"Q('{incidence_col}') ~ Q('{coverage_col}')"
    for i in range(n_components):
        formula += f" + PC{i+1}"
    
    # Fit regression
    model = smf.ols(formula, data=correlation_df).fit()
    print("\nRegression Summary:")
    print(model.summary())
    
    return model, loadings
