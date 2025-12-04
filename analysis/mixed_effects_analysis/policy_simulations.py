"""
Policy simulation scenarios for mixed-effects models
Simulates intervention outcomes and produces policy recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def simulate_intervention(model, panel_df, intervention_type, intervention_value=None, 
                         target_countries=None, baseline_year=None):
    """
    Simulate intervention scenarios using fitted model
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model from fit_models.py
    panel_df : DataFrame
        Panel dataset
    intervention_type : str
        Type of intervention:
        - 'MCV1_95': Increase MCV1 to 95% in LMICs
        - 'MCV1_increase': Increase MCV1 by specified amount
        - 'HealthExp_100': Increase health expenditure by $100 per capita
        - 'PolStability_1sd': Improve political stability by 1 SD
        - 'MCV2_95': Increase MCV2 to 95% in LMICs
    intervention_value : float, optional
        Specific intervention value (for custom interventions)
    target_countries : list, optional
        List of country codes to target. If None, targets LMICs (HIC=0)
    baseline_year : int, optional
        Year to use as baseline. If None, uses most recent year
        
    Returns:
    --------
    results : DataFrame
        Comparison of baseline vs intervention outcomes
    """
    print("\n" + "="*60)
    print(f"SIMULATING INTERVENTION: {intervention_type}")
    print("="*60)
    
    df = panel_df.copy()
    
    # Determine baseline year (default to 2024, not max year which might be incomplete)
    if baseline_year is None:
        available_years = sorted(df['Year'].unique())
        # Prefer 2024, fall back to most recent year if 2024 not available
        if 2024 in available_years:
            baseline_year = 2024
        else:
            baseline_year = max(y for y in available_years if y <= 2024) if any(y <= 2024 for y in available_years) else available_years[-1]
    
    print(f"\nUsing baseline year: {baseline_year}")
    
    # Filter to baseline year
    baseline_df = df[df['Year'] == baseline_year].copy()
    
    # Ensure Code is string type for consistent matching
    if 'Code' in baseline_df.columns:
        baseline_df['Code'] = baseline_df['Code'].astype(str)
    else:
        raise ValueError("Code column not found in panel data")
    
    # Determine target countries
    if target_countries is None:
        # Default: target LMICs
        target_countries = baseline_df[baseline_df['HIC'] == 0]['Code'].unique().tolist()
        # Ensure target_countries are strings
        target_countries = [str(c) for c in target_countries]
        print(f"\nTargeting LMICs: {len(target_countries)} countries")
    else:
        # Ensure target_countries are strings
        target_countries = [str(c) for c in target_countries]
        print(f"\nTargeting specified countries: {len(target_countries)} countries")
    
    # Create intervention dataset
    intervention_df = baseline_df.copy()
    
    # Apply intervention
    if intervention_type == 'MCV1_95':
        mask = intervention_df['Code'].isin(target_countries)
        intervention_df.loc[mask, 'MCV1'] = 95.0
        print(f"  Setting MCV1 to 95% for {mask.sum()} country-year observations")
        
    elif intervention_type == 'MCV1_increase':
        if intervention_value is None:
            intervention_value = 10.0  # Default: increase by 10 percentage points
        mask = intervention_df['Code'].isin(target_countries)
        intervention_df.loc[mask, 'MCV1'] = np.minimum(
            intervention_df.loc[mask, 'MCV1'] + intervention_value, 100.0
        )
        print(f"  Increasing MCV1 by {intervention_value} percentage points for {mask.sum()} observations")
        
    elif intervention_type == 'MCV2_95':
        mask = intervention_df['Code'].isin(target_countries)
        intervention_df.loc[mask, 'MCV2'] = 95.0
        print(f"  Setting MCV2 to 95% for {mask.sum()} country-year observations")
        
    elif intervention_type == 'HealthExp_1pct':
        if intervention_value is None:
            intervention_value = 1.0  # Default: increase by 1 percentage point
        mask = intervention_df['Code'].isin(target_countries)
        intervention_df.loc[mask, 'HealthExpPC'] = (
            intervention_df.loc[mask, 'HealthExpPC'] + intervention_value
        )
        print(f"  Increasing health expenditure by ${intervention_value} per capita for {mask.sum()} observations")
        
    elif intervention_type == 'PolStability_1sd':
        pol_stab_std = baseline_df['PolStability'].std()
        if intervention_value is None:
            intervention_value = pol_stab_std  # Default: 1 SD
        mask = intervention_df['Code'].isin(target_countries)
        intervention_df.loc[mask, 'PolStability'] = (
            intervention_df.loc[mask, 'PolStability'] + intervention_value
        )
        print(f"  Improving political stability by {intervention_value:.2f} (1 SD) for {mask.sum()} observations")
        
    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")
    
    # Predict outcomes
    print("\nPredicting outcomes...")
    
    # CRITICAL: Preserve Code column BEFORE any processing!
    # Store original Code values to prevent corruption
    if 'Code' not in baseline_df.columns:
        raise ValueError("Code column missing from baseline_df - cannot proceed")
    
    original_codes_baseline = baseline_df['Code'].copy()
    original_codes_intervention = intervention_df['Code'].copy()
    
    # Prepare dataframes for prediction - ensure all required columns exist
    # Get required columns from model (excluding Code - it's handled separately)
    if hasattr(model, 'model') and hasattr(model.model, 'formula'):
        # Linear mixed model - need to ensure all formula variables are present
        required_cols = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                        'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    else:
        required_cols = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                        'PopDensity', 'NetMigration', 'PolStability', 'HIC']
    
    # Ensure all required columns exist and are numeric
    # Code is handled separately - DO NOT convert it to numeric!
    for col in required_cols:
        if col not in baseline_df.columns:
            print(f"  Warning: Missing column {col}, using median value")
            baseline_df[col] = panel_df[col].median() if col in panel_df.columns else 0
            intervention_df[col] = baseline_df[col]
        
        # Ensure numeric (but NOT Code!)
        baseline_df[col] = pd.to_numeric(baseline_df[col], errors='coerce')
        intervention_df[col] = pd.to_numeric(intervention_df[col], errors='coerce')
        
        # Fill any remaining NaN
        if baseline_df[col].isnull().any():
            fill_val = baseline_df[col].median() if not baseline_df[col].isnull().all() else 0
            baseline_df[col] = baseline_df[col].fillna(fill_val)
            intervention_df[col] = intervention_df[col].fillna(fill_val)
    
    # Restore Code column (preserve as string!)
    baseline_df['Code'] = original_codes_baseline.astype(str)
    intervention_df['Code'] = original_codes_intervention.astype(str)
    
    # Debug: Verify Code values are preserved
    if len(baseline_df) > 0:
        sample_code = baseline_df['Code'].iloc[0]
        if sample_code == '0.0' or sample_code == '0' or sample_code == 'nan':
            print(f"  ERROR: Code column corrupted! First value: {sample_code}")
            print(f"  Original first code was: {original_codes_baseline.iloc[0]}")
            # Force restore
            baseline_df['Code'] = original_codes_baseline.astype(str)
            intervention_df['Code'] = original_codes_intervention.astype(str)
    
    # Use ACTUAL OBSERVED incidence as baseline (country-specific)
    # Then calculate intervention effect using model coefficients
    
    # Get actual observed baseline incidence
    baseline_incidence = baseline_df['MeaslesIncidence'].values.copy()
    baseline_incidence = np.where(pd.isna(baseline_incidence), 0, baseline_incidence)
    
    print(f"  Using actual observed incidence as baseline")
    print(f"    Mean baseline: {np.mean(baseline_incidence):.2f}, Range: [{np.min(baseline_incidence):.2f}, {np.max(baseline_incidence):.2f}]")
    
    # Check model type and calculate intervention effect
    if hasattr(model, 'params') and hasattr(model, 'model'):
        # Linear mixed model - use coefficients to calculate effect
        try:
            params = model.params
            
            # Calculate the change in log-incidence based on intervention
            # The model is: log(1 + Incidence) = X * beta
            # So change in log-incidence = delta_X * beta
            
            log_effect = 0.0
            
            # Calculate effect based on which variables changed
            if intervention_type == 'MCV1_95':
                if 'MCV1' in params.index:
                    # Effect = coefficient * (new_value - old_value)
                    old_mcv1 = baseline_df['MCV1'].values
                    new_mcv1 = 95.0
                    delta_mcv1 = new_mcv1 - old_mcv1
                    log_effect = params['MCV1'] * delta_mcv1
                    print(f"    MCV1 coefficient: {params['MCV1']:.4f}")
                    print(f"    Mean MCV1 change: {np.mean(delta_mcv1):.2f} percentage points")
                    
            elif intervention_type == 'MCV1_increase':
                if 'MCV1' in params.index:
                    delta_mcv1 = 10.0  # 10 percentage point increase
                    log_effect = params['MCV1'] * delta_mcv1
                    print(f"    MCV1 coefficient: {params['MCV1']:.4f}")
                    
            elif intervention_type == 'MCV2_95':
                if 'MCV2' in params.index:
                    old_mcv2 = baseline_df['MCV2'].values
                    new_mcv2 = 95.0
                    delta_mcv2 = new_mcv2 - old_mcv2
                    log_effect = params['MCV2'] * delta_mcv2
                    print(f"    MCV2 coefficient: {params['MCV2']:.4f}")
                    print(f"    Mean MCV2 change: {np.nanmean(delta_mcv2):.2f} percentage points")
                    
            elif intervention_type == 'HealthExp_1pct':
                if 'HealthExpPC' in params.index:
                    delta_health = 1.0  # 1 percentage point increase
                    log_effect = params['HealthExpPC'] * delta_health
                    print(f"    HealthExpPC coefficient: {params['HealthExpPC']:.4f}")
                    
            elif intervention_type == 'PolStability_1sd':
                if 'PolStability' in params.index:
                    pol_std = baseline_df['PolStability'].std()
                    if pd.notna(pol_std) and pol_std > 0:
                        delta_pol = pol_std
                    else:
                        delta_pol = 1.0  # Default if std is invalid
                    log_effect = params['PolStability'] * delta_pol
                    print(f"    PolStability coefficient: {params['PolStability']:.4f}")
                    print(f"    PolStability SD: {delta_pol:.4f}")
            
            # Handle array or scalar log_effect
            if isinstance(log_effect, (int, float)):
                log_effect = np.full(len(baseline_df), log_effect)
            log_effect = np.where(pd.isna(log_effect), 0, log_effect)
            
            # Convert baseline to log scale, apply effect, convert back
            # log(1 + baseline) + log_effect = log(1 + intervention)
            log_baseline = np.log1p(baseline_incidence)
            log_intervention = log_baseline + log_effect
            intervention_incidence = np.expm1(log_intervention)
            
            # Ensure non-negative
            intervention_incidence = np.maximum(intervention_incidence, 0)
            
            print(f"    Mean intervention effect: {np.mean(log_effect):.4f} (log scale)")
            print(f"    Mean intervention incidence: {np.mean(intervention_incidence):.2f}")
            
        except Exception as e:
            print(f"  Warning: Error calculating intervention effect: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: no change
            intervention_incidence = baseline_incidence.copy()
        
    else:
        # For other model types (GLM, NB, etc.) - use observed baseline
        print("  Using observed incidence (model type doesn't support coefficient extraction)")
        intervention_incidence = baseline_incidence.copy()
    
    # Ensure Code column is properly formatted (string, not numeric)
    if 'Code' in baseline_df.columns:
        baseline_codes = baseline_df['Code'].astype(str).values
    else:
        raise ValueError("Code column missing from baseline_df")
    
    # Check which countries had intervention applied
    if target_countries is not None:
        # Convert target_countries to strings for comparison
        target_countries_str = [str(c) for c in target_countries]
        intervention_applied = pd.Series(baseline_codes).isin(target_countries_str).astype(int).values
    else:
        intervention_applied = np.zeros(len(baseline_df), dtype=int)
    
    # Get country names
    if 'Country' in baseline_df.columns:
        country_names = baseline_df['Country'].values
    elif 'Member State' in baseline_df.columns:
        country_names = baseline_df['Member State'].values
    else:
        country_names = ['Unknown'] * len(baseline_df)
    
    # Calculate changes
    results = pd.DataFrame({
        'Code': baseline_codes,
        'Country': country_names,
        'Year': baseline_df['Year'].values,
        'HIC': baseline_df['HIC'].values,
        'BaselineIncidence': baseline_incidence,
        'InterventionIncidence': intervention_incidence,
        'Change': intervention_incidence - baseline_incidence,
        'PercentChange': 100 * (intervention_incidence - baseline_incidence) / (baseline_incidence + 1e-10),
        'InterventionApplied': intervention_applied
    })
    
    # Debug: Print summary of intervention application
    n_applied = intervention_applied.sum()
    print(f"  Intervention applied to {n_applied} observations")
    if n_applied == 0:
        print(f"  Warning: No interventions applied! Checking target_countries...")
        print(f"    Target countries sample: {target_countries[:5] if len(target_countries) > 0 else 'None'}")
        print(f"    Baseline codes sample: {baseline_codes[:5] if len(baseline_codes) > 0 else 'None'}")
        print(f"    Code match test: {baseline_codes[0] in target_countries_str if len(baseline_codes) > 0 and len(target_countries_str) > 0 else 'N/A'}")
    
    # Summary statistics
    print("\nIntervention Results Summary:")
    print(f"  Countries affected: {results['InterventionApplied'].sum()}")
    
    affected = results[results['InterventionApplied'] == 1]
    if len(affected) > 0:
        print(f"\n  Among affected countries:")
        print(f"    Mean baseline incidence: {affected['BaselineIncidence'].mean():.2f}")
        print(f"    Mean intervention incidence: {affected['InterventionIncidence'].mean():.2f}")
        print(f"    Mean reduction: {affected['Change'].mean():.2f}")
        print(f"    Mean percent reduction: {affected['PercentChange'].mean():.2f}%")
        print(f"    Total cases prevented: {affected['Change'].sum():.0f}")
    
    return results


def run_all_interventions(model, panel_df, baseline_year=None):
    """
    Run all standard intervention scenarios
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model
    panel_df : DataFrame
        Panel dataset
    baseline_year : int, optional
        Baseline year
        
    Returns:
    --------
    all_results : dict
        Dictionary of intervention results
    """
    print("\n" + "="*60)
    print("RUNNING ALL INTERVENTION SCENARIOS")
    print("="*60)
    
    interventions = [
        'MCV1_95',
        'MCV2_95',
        'HealthExp_1pct',
        'PolStability_1sd',
        'MCV1_increase'  # Increase MCV1 by 10 percentage points
    ]
    
    all_results = {}
    
    for intervention in interventions:
        try:
            results = simulate_intervention(
                model, panel_df, intervention, 
                baseline_year=baseline_year
            )
            all_results[intervention] = results
        except Exception as e:
            print(f"\nError running {intervention}: {e}")
            continue
    
    return all_results


def calculate_marginal_effects(model, panel_df, variable, 
                               values=None, hic_split=True):
    """
    Calculate marginal effects of a variable across its range
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model
    panel_df : DataFrame
        Panel dataset
    variable : str
        Variable name to analyze
    values : array-like, optional
        Specific values to evaluate. If None, uses range from data
    hic_split : bool
        If True, calculate separately for HIC and LMIC
        
    Returns:
    --------
    marginal_effects : DataFrame
        Marginal effects at different values
    """
    print(f"\nCalculating marginal effects for {variable}...")
    
    df = panel_df.copy()
    
    # Determine evaluation values
    if values is None:
        valid_values = df[variable].dropna()
        values = np.linspace(valid_values.min(), valid_values.max(), 50)
    
    # Create prediction dataset
    # Use median/mode for other variables
    prediction_df = df.copy()
    
    # Set all rows to median values for other predictors
    numeric_cols = ['MCV1', 'MCV2', 'GDPpc', 'HealthExpPC', 'BirthRate',
                    'PopDensity', 'NetMigration', 'PolStability']
    numeric_cols = [c for c in numeric_cols if c in prediction_df.columns and c != variable]
    
    for col in numeric_cols:
        prediction_df[col] = prediction_df[col].median()
    
    # Set HIC to 0 or 1 depending on split
    if hic_split:
        results_list = []
        
        for hic_value in [0, 1]:
            pred_df = prediction_df.copy()
            pred_df['HIC'] = hic_value
            pred_df[variable] = np.nan  # Will be set per row
            
            effects = []
            for val in values:
                pred_df[variable] = val
                
                # Predict
                try:
                    if hasattr(model, 'fittedvalues') and hasattr(model, 'model'):
                        # Linear mixed model
                        pred_log = model.predict(pred_df.iloc[:1])
                        if isinstance(pred_log, pd.Series):
                            pred_log = pred_log.iloc[0]
                        elif isinstance(pred_log, np.ndarray):
                            pred_log = pred_log[0]
                        pred_incidence = np.expm1(pred_log)
                    else:
                        # GLM model
                        pred = model.predict(pred_df.iloc[:1])
                        if isinstance(pred, pd.Series):
                            pred_incidence = pred.iloc[0]
                        elif isinstance(pred, np.ndarray):
                            pred_incidence = pred[0]
                        else:
                            pred_incidence = float(pred)
                except Exception as e:
                    # Fallback: use median incidence
                    pred_incidence = df['MeaslesIncidence'].median()
                
                effects.append({
                    'Variable': variable,
                    'Value': val,
                    'HIC': hic_value,
                    'PredictedIncidence': pred_incidence
                })
            
            results_list.extend(effects)
        
        marginal_effects = pd.DataFrame(results_list)
        
    else:
        # Single calculation
        effects = []
        for val in values:
            pred_df = prediction_df.copy()
            pred_df[variable] = val
            
            try:
                if hasattr(model, 'fittedvalues') and hasattr(model, 'model'):
                    # Linear mixed model
                    pred_log = model.predict(pred_df.iloc[:1])
                    if isinstance(pred_log, pd.Series):
                        pred_log = pred_log.iloc[0]
                    elif isinstance(pred_log, np.ndarray):
                        pred_log = pred_log[0]
                    pred_incidence = np.expm1(pred_log)
                else:
                    # GLM model
                    pred = model.predict(pred_df.iloc[:1])
                    if isinstance(pred, pd.Series):
                        pred_incidence = pred.iloc[0]
                    elif isinstance(pred, np.ndarray):
                        pred_incidence = pred[0]
                    else:
                        pred_incidence = float(pred)
            except Exception as e:
                # Fallback: use median incidence
                pred_incidence = df['MeaslesIncidence'].median()
            
            effects.append({
                'Variable': variable,
                'Value': val,
                'PredictedIncidence': pred_incidence
            })
        
        marginal_effects = pd.DataFrame(effects)
    
    return marginal_effects


def save_intervention_results(all_results, output_dir=None):
    """Save intervention results to CSV files"""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'interventions'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for intervention, results in all_results.items():
        filepath = output_dir / f'{intervention}_results.csv'
        results.to_csv(filepath, index=False)
        print(f"Saved {intervention} results to {filepath}")
    
    # Create summary
    summary_data = []
    for intervention, results in all_results.items():
        affected = results[results['InterventionApplied'] == 1]
        if len(affected) > 0:
            summary_data.append({
                'Intervention': intervention,
                'CountriesAffected': affected['Code'].nunique(),
                'MeanBaselineIncidence': affected['BaselineIncidence'].mean(),
                'MeanInterventionIncidence': affected['InterventionIncidence'].mean(),
                'MeanReduction': affected['Change'].mean(),
                'MeanPercentReduction': affected['PercentChange'].mean(),
                'TotalCasesPrevented': affected['Change'].sum()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'intervention_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved intervention summary to {summary_path}")


if __name__ == '__main__':
    from prepare_data import prepare_panel_data
    from fit_models import load_model
    
    # Load data and model
    print("Loading data and model...")
    panel_df = prepare_panel_data()
    
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis'
    model = load_model(results_dir / 'linear_mixed_model.pkl')
    
    # Run interventions
    all_results = run_all_interventions(model, panel_df)
    
    # Save results
    save_intervention_results(all_results)

