"""
Fit mixed-effects models for measles incidence analysis
- Linear mixed model with log-transformed incidence
- Negative binomial mixed-effects model (using GLM with country fixed effects as approximation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')


def fit_linear_mixed_model(panel_df, use_random_slope=True):
    """
    Fit linear mixed-effects model with log-transformed incidence
    
    Model specification:
    log(MeaslesIncidence_it) = β₀ + β₁MCV1_it + β₂MCV2_it + β₃GDPpc_it + 
                               β₄HealthExpPC_it + β₅BirthRate_it + 
                               β₆PopDensity_it + β₇Migration_it + 
                               β₈PolStability_it + β₉HIC_i + 
                               u₀ᵢ + u₁ᵢMCV1_it + ε_it
    
    Where:
    - u₀ᵢ = country-level random intercept
    - u₁ᵢ = country-level random slope for MCV1 (optional)
    
    Parameters:
    -----------
    panel_df : DataFrame
        Panel dataset prepared by prepare_data.py
    use_random_slope : bool
        If True, include random slope for MCV1
        
    Returns:
    --------
    model : MixedLMResults
        Fitted mixed-effects model
    """
    print("\n" + "="*60)
    print("FITTING LINEAR MIXED-EFFECTS MODEL")
    print("="*60)
    
    # Prepare data: drop rows with missing key variables
    df = panel_df.copy()
    
    # Required variables
    required_vars = ['Code', 'Year', 'LogIncidence', 'MCV1', 'MCV2', 
                     'GDPpc', 'HealthExpPC', 'BirthRate', 'PopDensity',
                     'NetMigration', 'PolStability', 'HIC']
    
    # Check availability
    missing_vars = [v for v in required_vars if v not in df.columns]
    if missing_vars:
        print(f"Warning: Missing variables: {missing_vars}")
        print("Proceeding with available variables...")
        required_vars = [v for v in required_vars if v in df.columns]
    
    # Drop rows with missing outcome or key predictors
    df = df.dropna(subset=['LogIncidence', 'Code'])
    
    # For predictors, use mean imputation by country (conservative approach)
    predictor_vars = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                      'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    predictor_vars = [v for v in predictor_vars if v in df.columns]
    
    for var in predictor_vars:
        if df[var].isnull().any():
            # Impute by country mean, then overall mean
            country_means = df.groupby('Code')[var].transform('mean')
            df[var] = df[var].fillna(country_means)
            overall_mean = df[var].mean()
            df[var] = df[var].fillna(overall_mean)
            print(f"  Imputed {var}: {df[var].isnull().sum()} missing values")
    
    print(f"\nData for modeling:")
    print(f"  Observations: {len(df)}")
    print(f"  Countries: {df['Code'].nunique()}")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    
    # Build formula
    # Fixed effects
    fixed_effects = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                     'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    fixed_effects = [v for v in fixed_effects if v in df.columns]
    
    formula = f"LogIncidence ~ {' + '.join(fixed_effects)}"
    
    # Random effects
    if use_random_slope and 'MCV1' in df.columns:
        groups = df['Code']
        re_formula = "1 + MCV1"  # Random intercept + random slope for MCV1
    else:
        groups = df['Code']
        re_formula = "1"  # Random intercept only
    
    print(f"\nModel formula:")
    print(f"  Fixed: {formula}")
    print(f"  Random: {re_formula} (grouped by country)")
    
    # Fit model
    try:
        model = smf.mixedlm(formula, df, groups=groups, re_formula=re_formula)
        result = model.fit(reml=False)  # Use ML instead of REML for comparison
        
        print(f"\nModel converged successfully!")
        print(f"  Log-likelihood: {result.llf:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"\nFixed effects:")
        print(result.summary().tables[1])
        
        if len(result.summary().tables) > 2:
            print(f"\nRandom effects:")
            print(result.summary().tables[2])
        
        return result
        
    except Exception as e:
        print(f"\nError fitting model: {e}")
        print("Trying simpler model without random slope...")
        if use_random_slope:
            return fit_linear_mixed_model(panel_df, use_random_slope=False)
        else:
            raise


def fit_negative_binomial_model(panel_df):
    """
    Fit negative binomial model for count data (measles cases)
    Uses country fixed effects to approximate random effects
    
    Model specification:
    MeaslesCases_it ~ NegativeBinomial(μ_it, α)
    log(μ_it) = log(Population_it) + β₀ + β₁MCV1_it + β₂MCV2_it + 
                β₃GDPpc_it + β₄HealthExpPC_it + β₅BirthRate_it + 
                β₆PopDensity_it + β₇Migration_it + β₈PolStability_it + 
                β₉HIC_i + Country_i
    
    Where Population is used as offset
    
    Parameters:
    -----------
    panel_df : DataFrame
        Panel dataset prepared by prepare_data.py
        
    Returns:
    --------
    model : NegativeBinomialResults
        Fitted negative binomial model
    """
    print("\n" + "="*60)
    print("FITTING NEGATIVE BINOMIAL MODEL")
    print("="*60)
    
    # Prepare data
    df = panel_df.copy()
    
    # Drop rows with missing key variables
    required_vars = ['Code', 'Year', 'MeaslesCases', 'Population', 'MCV1', 'MCV2',
                     'GDPpc', 'HealthExpPC', 'BirthRate', 'PopDensity',
                     'NetMigration', 'PolStability', 'HIC']
    
    df = df.dropna(subset=['MeaslesCases', 'Population', 'Code'])
    
    # Ensure non-negative cases
    df = df[df['MeaslesCases'] >= 0].copy()
    df = df[df['Population'] > 0].copy()
    
    # Impute missing predictors
    predictor_vars = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                      'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    predictor_vars = [v for v in predictor_vars if v in df.columns]
    
    for var in predictor_vars:
        if df[var].isnull().any():
            country_means = df.groupby('Code')[var].transform('mean')
            df[var] = df[var].fillna(country_means)
            overall_mean = df[var].mean()
            df[var] = df[var].fillna(overall_mean)
    
    print(f"\nData for modeling:")
    print(f"  Observations: {len(df)}")
    print(f"  Countries: {df['Code'].nunique()}")
    print(f"  Years: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  Mean cases: {df['MeaslesCases'].mean():.1f}")
    print(f"  Zero cases: {(df['MeaslesCases'] == 0).sum()} ({(df['MeaslesCases'] == 0).mean()*100:.1f}%)")
    
    # Determine if we should use country fixed effects
    n_countries = df['Code'].nunique()
    use_country_fe = n_countries <= 50  # Only use country FE if <= 50 countries
    
    # Prepare design matrix using helper function with standardization
    X, offset, scaler = prepare_nb_data_for_prediction(df, include_country_fe=use_country_fe, standardize=True)
    
    # Count predictors
    X_vars = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
              'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    X_vars = [v for v in X_vars if v in df.columns]
    
    if use_country_fe:
        n_country_fe = n_countries - 1  # -1 because drop_first=True
        print(f"\nModel specification:")
        print(f"  Outcome: MeaslesCases (count)")
        print(f"  Offset: log(Population/1M) [scaled to prevent overflow]")
        print(f"  Predictors: {len(X_vars)} fixed effects + {n_country_fe} country effects")
        print(f"  Note: Predictors are standardized (mean=0, std=1) to prevent numerical overflow")
    else:
        print(f"\nModel specification:")
        print(f"  Outcome: MeaslesCases (count)")
        print(f"  Offset: log(Population/1M) [scaled to prevent overflow]")
        print(f"  Predictors: {len(X_vars)} fixed effects")
        print(f"  Note: Skipping country fixed effects ({n_countries} countries)")
        print(f"  Note: Predictors are standardized (mean=0, std=1) to prevent numerical overflow")
    
    # Outcome - ensure float type
    y = np.asarray(df['MeaslesCases'].values, dtype=np.float64)
    
    # Check for extreme values that might cause issues
    if np.any(y > 1e6):
        print(f"  Warning: Very large case counts detected (max: {y.max():.0f})")
        print(f"  Consider using incidence rate instead of counts")
    
    # Fit negative binomial model with better initialization
    try:
        # Use NegativeBinomial with log link
        model = NegativeBinomial(y, X, loglike_method='nb2', offset=offset)
        
        # Set better starting values to prevent overflow
        # Start with small coefficients (near zero) since predictors are standardized
        n_params = X.shape[1] + 1  # +1 for alpha parameter
        start_params = np.zeros(n_params)
        start_params[-1] = 1.0  # Start alpha at 1.0
        
        # Try fitting with different methods and starting values
        try:
            result = model.fit(start_params=start_params, method='lbfgs', 
                             maxiter=1000, disp=0, warn_convergence=False)
        except:
            try:
                # Try with even smaller starting values
                start_params = start_params * 0.1
                result = model.fit(start_params=start_params, method='newton', 
                                 maxiter=500, disp=0, warn_convergence=False)
            except:
                # Last resort: use basic fit with default starting values
                result = model.fit(disp=0, warn_convergence=False)
        
        # Check if model converged and has valid results
        if np.any(np.isnan(result.params)):
            raise ValueError("Model fitting produced NaN coefficients - numerical instability detected")
        
        if hasattr(result, 'mle_retvals') and result.mle_retvals.get('converged', False):
            print(f"\nModel converged successfully!")
        else:
            print(f"\nModel fitting completed (convergence status unclear)")
        
        if np.isnan(result.llf) or np.isinf(result.llf):
            raise ValueError("Model log-likelihood is invalid - numerical instability detected")
        
        print(f"  Log-likelihood: {result.llf:.2f}")
        print(f"  AIC: {result.aic:.2f}")
        
        # Store scaler in model for later predictions
        if scaler is not None:
            result.scaler = scaler
            result.standardized = True
        
        # Show coefficients (limit to avoid too much output)
        print(f"\nFixed effects (first 15 coefficients):")
        try:
            coef_summary = result.summary().tables[1]
            # Print first few rows
            for i, row in enumerate(coef_summary.data[:16]):  # First 15 + header
                print('  '.join(str(cell) for cell in row))
        except:
            # Fallback: print params directly
            params_df = pd.DataFrame({
                'Coefficient': result.params[:15],
                'Std Err': result.bse[:15] if hasattr(result, 'bse') else np.nan
            })
            print(params_df)
        
        return result
        
    except Exception as e:
        print(f"\nError fitting negative binomial model: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTrying simplest specification with standardized predictors...")
        # Use only fixed effects with standardized variables
        try:
            from sklearn.preprocessing import StandardScaler
            
            X_simple = df[X_vars].copy()
            for col in X_simple.columns:
                X_simple[col] = pd.to_numeric(X_simple[col], errors='coerce')
                X_simple[col] = X_simple[col].fillna(X_simple[col].median())
            
            # Log-transform large variables
            if 'GDPpc' in X_simple.columns:
                X_simple['GDPpc'] = np.log1p(X_simple['GDPpc'])
            if 'PopDensity' in X_simple.columns:
                X_simple['PopDensity'] = np.log1p(X_simple['PopDensity'])
            
            # Standardize predictors
            scaler_simple = StandardScaler()
            X_simple_scaled = scaler_simple.fit_transform(X_simple)
            X_simple = pd.DataFrame(X_simple_scaled, columns=X_vars, index=X_simple.index)
            
            X_simple = X_simple.astype(float)
            X_simple = sm.add_constant(X_simple)
            X_simple = np.asarray(X_simple, dtype=np.float64)
            
            # Scale offset
            offset_simple = np.log(df['Population'].values / 1e6)
            y_simple = np.asarray(df['MeaslesCases'].values, dtype=np.float64)
            
            # Use small starting values
            n_params_simple = X_simple.shape[1] + 1
            start_params_simple = np.zeros(n_params_simple)
            start_params_simple[-1] = 1.0
            
            model_simple = NegativeBinomial(y_simple, X_simple, loglike_method='nb2', offset=offset_simple)
            result_simple = model_simple.fit(start_params=start_params_simple, method='lbfgs', 
                                           maxiter=1000, disp=0, warn_convergence=False)
            
            # Check for valid results
            if np.any(np.isnan(result_simple.params)) or np.isnan(result_simple.llf):
                raise ValueError("Simplified model also produced invalid results")
            
            # Store scaler
            result_simple.scaler = scaler_simple
            result_simple.standardized = True
            
            print("Simplified model fitted successfully!")
            return result_simple
        except Exception as e2:
            print(f"Error with simplified model: {e2}")
            print("\nNegative binomial model failed. Consider using linear mixed model instead.")
            raise


def prepare_nb_data_for_prediction(panel_df, include_country_fe=False, standardize=True, scaler=None):
    """
    Helper function to prepare data for negative binomial model prediction
    Applies same transformations as in model fitting
    
    Parameters:
    -----------
    panel_df : DataFrame
        Panel dataset
    include_country_fe : bool
        Whether to include country fixed effects
    standardize : bool
        Whether to standardize predictors (mean=0, std=1)
    scaler : StandardScaler, optional
        Pre-fitted scaler to use. If None and standardize=True, will fit new scaler
        
    Returns:
    --------
    X : ndarray
        Design matrix
    offset : ndarray
        Offset values (log population scaled)
    scaler : StandardScaler, optional
        Fitted scaler (if standardize=True)
    """
    from sklearn.preprocessing import StandardScaler
    
    df = panel_df.copy()
    
    # Same predictor variables
    X_vars = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
              'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    X_vars = [v for v in X_vars if v in df.columns]
    
    X = df[X_vars].copy()
    
    # Apply same transformations
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].median())
    
    # Log-transform large variables (same as in fitting)
    if 'GDPpc' in X.columns:
        X['GDPpc'] = np.log1p(X['GDPpc'])
    if 'PopDensity' in X.columns:
        X['PopDensity'] = np.log1p(X['PopDensity'])
    
    # Store column names before standardization
    col_names = X.columns.tolist()
    
    # Standardize predictors to prevent overflow
    if standardize:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=col_names, index=X.index)
    
    # Add country effects if requested
    if include_country_fe and 'Code' in df.columns:
        country_dummies = pd.get_dummies(df['Code'], prefix='Country', drop_first=True)
        country_dummies = country_dummies.astype(float)
        X = pd.concat([X, country_dummies], axis=1)
    
    X = X.astype(float)
    X = sm.add_constant(X)
    X = np.asarray(X, dtype=np.float64)
    
    # Offset (same scaling as in fitting)
    # Use log(Population/1M) to keep offset values reasonable
    offset = np.log(df['Population'].values / 1e6)
    
    if standardize:
        return X, offset, scaler
    else:
        return X, offset


def save_model(model, filepath):
    """Save fitted model to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {filepath}")


def load_model(filepath):
    """Load fitted model from pickle file"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == '__main__':
    # Load prepared data
    from prepare_data import prepare_panel_data
    
    print("Loading panel data...")
    panel_df = prepare_panel_data()
    
    # Fit linear mixed model
    linear_model = fit_linear_mixed_model(panel_df, use_random_slope=True)
    
    # Save model
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis'
    save_model(linear_model, results_dir / 'linear_mixed_model.pkl')
    
    # Fit negative binomial model
    nb_model = fit_negative_binomial_model(panel_df)
    
    # Save model
    save_model(nb_model, results_dir / 'negative_binomial_model.pkl')

