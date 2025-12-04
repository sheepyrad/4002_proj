"""
DoWhy Causal Analysis for Measles Vaccine Coverage Study

This module implements causal inference using the DoWhy framework to estimate
the causal effect of vaccine coverage on measles incidence, accounting for
confounding factors.

Key analyses:
1. Effect of MCV1 (first dose) on measles incidence
2. Effect of MCV2 (second dose) on measles incidence
3. Robustness checks using multiple refutation methods
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from dowhy import CausalModel
    import networkx as nx
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("Warning: DoWhy not installed. Install with: pip install dowhy")

from causal_dag import (
    create_causal_dag, 
    create_simple_dag_for_mcv1,
    create_simple_dag_for_mcv2,
    dag_to_gml,
    visualize_dag
)
from prepare_imputed_data import get_analysis_ready_data


# Results directory
RESULTS_DIR = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis'


def load_panel_data(use_imputation=True):
    """
    Load the prepared panel data, optionally with imputation.
    
    Parameters:
    -----------
    use_imputation : bool
        If True, use income-group based imputation for missing values
        This significantly increases the number of countries (especially LIC/LMIC)
    """
    if use_imputation:
        print("Loading data with income-group based imputation...")
        df_full, df_complete, imputation_stats = get_analysis_ready_data(verbose=True)
        print(f"\nLoaded imputed panel data: {len(df_full)} observations, {df_full['Code'].nunique()} countries")
        return df_full
    else:
        data_path = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'panel_data.csv'
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"Panel data not found at {data_path}. "
                "Run the mixed_effects_analysis/prepare_data.py script first."
            )
        
        df = pd.read_csv(data_path)
        print(f"Loaded panel data (no imputation): {len(df)} observations, {df['Code'].nunique()} countries")
        return df


def prepare_data_for_causal_analysis(df, treatment='MCV1'):
    """
    Prepare data for causal analysis by handling missing values
    and selecting relevant variables.
    
    Parameters:
    -----------
    df : DataFrame
        Panel data
    treatment : str
        Treatment variable ('MCV1' or 'MCV2')
    
    Returns:
    --------
    DataFrame with complete cases for the analysis
    """
    # Variables needed for analysis
    outcome = 'LogIncidence'  # Use log-transformed incidence
    
    # Full set of confounders (based on epidemiological theory):
    # - LogGDPpc: economic capacity affects healthcare and disease burden
    # - HealthExpPC: health spending per capita determines vaccine program funding
    # - PolStability: governance affects vaccine program implementation
    # - HIC: structural differences in health systems
    # - UrbanPop: urbanization affects healthcare access and disease spread
    # - HouseholdSize: larger households = more transmission, affects vaccine uptake
    # - VaccineHesitancy: directly affects vaccine uptake and care-seeking
    # - NetMigration: affects herd immunity and disease importation
    confounders = [
        'LogGDPpc', 'LogHealthExpPC', 'PolStability', 'HIC',
        'UrbanPop', 'HouseholdSize', 'VaccineHesitancy', 'NetMigration'
    ]
    
    # Effect modifiers (affect outcome, may interact with treatment)
    effect_modifiers = ['LogPopDensity', 'BirthRate']
    
    # For MCV2, also control for MCV1 (sequential vaccination)
    if treatment == 'MCV2':
        confounders = confounders + ['MCV1']
    
    all_vars = [treatment, outcome] + confounders + effect_modifiers
    
    # Remove duplicates
    all_vars = list(dict.fromkeys(all_vars))
    
    # Get complete cases
    analysis_df = df[all_vars + ['Code', 'Year', 'Country', 'IncomeGroup']].dropna()
    
    print(f"\nData preparation for {treatment} analysis:")
    print(f"  Original observations: {len(df)}")
    print(f"  Complete cases: {len(analysis_df)}")
    print(f"  Countries with complete data: {analysis_df['Code'].nunique()}")
    print(f"  Confounders: {confounders}")
    
    return analysis_df, outcome, confounders, effect_modifiers


def estimate_causal_effect_mcv1(df, verbose=True):
    """
    Estimate the causal effect of MCV1 (first dose coverage) on measles incidence.
    
    Uses DoWhy's four-step approach:
    1. Model: Define causal model with DAG
    2. Identify: Find causal estimand using backdoor criterion
    3. Estimate: Compute causal effect using various methods
    4. Refute: Test robustness of the estimate
    
    Returns:
    --------
    dict with causal estimates and refutation results
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    # Prepare data
    analysis_df, outcome, confounders, effect_modifiers = prepare_data_for_causal_analysis(
        df, treatment='MCV1'
    )
    
    # Create the causal graph
    dag = create_simple_dag_for_mcv1()
    gml_graph = dag_to_gml(dag)
    
    if verbose:
        print("\n" + "="*60)
        print("CAUSAL ANALYSIS: Effect of MCV1 on Measles Incidence")
        print("="*60)
        print(f"\nTreatment: MCV1 (First dose vaccine coverage %)")
        print(f"Outcome: Log(Measles Incidence + 1)")
        print(f"Confounders: {confounders}")
    
    # Step 1: Create causal model
    model = CausalModel(
        data=analysis_df,
        treatment='MCV1',
        outcome='LogIncidence',
        graph=gml_graph,
        common_causes=confounders,
        effect_modifiers=effect_modifiers
    )
    
    if verbose:
        print("\n--- Step 1: Causal Model Created ---")
    
    # Step 2: Identify causal effect (backdoor criterion)
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    if verbose:
        print("\n--- Step 2: Causal Effect Identified ---")
        print(identified_estimand)
    
    # Step 3: Estimate effect using multiple methods
    results = {'estimates': {}, 'refutations': {}}
    
    # Method 1: Linear Regression
    estimate_lr = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
        method_params={'need_conditional_estimates': False}
    )
    results['estimates']['linear_regression'] = {
        'value': estimate_lr.value,
        'estimate_object': estimate_lr
    }
    
    if verbose:
        print("\n--- Step 3: Causal Effect Estimates ---")
        print(f"\nLinear Regression Estimate: {estimate_lr.value:.4f}")
    
    # Method 2: Propensity Score Stratification
    # First, discretize treatment for propensity score methods
    analysis_df_disc = analysis_df.copy()
    analysis_df_disc['MCV1_high'] = (analysis_df_disc['MCV1'] >= analysis_df_disc['MCV1'].median()).astype(int)
    
    model_disc = CausalModel(
        data=analysis_df_disc,
        treatment='MCV1_high',
        outcome='LogIncidence',
        common_causes=confounders,
        effect_modifiers=effect_modifiers
    )
    identified_disc = model_disc.identify_effect(proceed_when_unidentifiable=True)
    
    estimate_ps = model_disc.estimate_effect(
        identified_disc,
        method_name="backdoor.propensity_score_stratification",
        target_units="ate"
    )
    results['estimates']['propensity_stratification'] = {
        'value': estimate_ps.value,
        'estimate_object': estimate_ps
    }
    
    if verbose:
        print(f"Propensity Score Stratification (binary): {estimate_ps.value:.4f}")
    
    # Method 3: Propensity Score Matching (binary treatment)
    try:
        estimate_match = model_disc.estimate_effect(
            identified_disc,
            method_name="backdoor.propensity_score_matching",
            target_units="ate"
        )
        results['estimates']['propensity_matching'] = {
            'value': estimate_match.value,
            'estimate_object': estimate_match
        }
        if verbose:
            print(f"Propensity Score Matching (binary): {estimate_match.value:.4f}")
    except Exception as e:
        if verbose:
            print(f"Propensity Score Matching failed: {e}")
    
    # Step 4: Refutation tests
    if verbose:
        print("\n--- Step 4: Robustness Checks ---")
    
    # Refutation 1: Random Common Cause
    try:
        refute_random = model.refute_estimate(
            identified_estimand, 
            estimate_lr,
            method_name="random_common_cause"
        )
        results['refutations']['random_common_cause'] = {
            'new_estimate': refute_random.new_effect,
            'p_value': getattr(refute_random, 'refutation_result', None)
        }
        if verbose:
            print(f"\nRandom Common Cause Test:")
            print(f"  Original estimate: {estimate_lr.value:.4f}")
            print(f"  New estimate: {refute_random.new_effect:.4f}")
    except Exception as e:
        if verbose:
            print(f"Random Common Cause refutation failed: {e}")
    
    # Refutation 2: Placebo Treatment
    try:
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        results['refutations']['placebo_treatment'] = {
            'new_estimate': refute_placebo.new_effect,
            'p_value': getattr(refute_placebo, 'refutation_result', None)
        }
        if verbose:
            print(f"\nPlacebo Treatment Test:")
            print(f"  Original estimate: {estimate_lr.value:.4f}")
            print(f"  Placebo estimate: {refute_placebo.new_effect:.4f}")
            print("  (Placebo estimate should be ~0 if effect is causal)")
    except Exception as e:
        if verbose:
            print(f"Placebo Treatment refutation failed: {e}")
    
    # Refutation 3: Data Subset
    try:
        refute_subset = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="data_subset_refuter",
            subset_fraction=0.8
        )
        results['refutations']['data_subset'] = {
            'new_estimate': refute_subset.new_effect,
            'p_value': getattr(refute_subset, 'refutation_result', None)
        }
        if verbose:
            print(f"\nData Subset Test (80%):")
            print(f"  Original estimate: {estimate_lr.value:.4f}")
            print(f"  Subset estimate: {refute_subset.new_effect:.4f}")
            print("  (Estimates should be similar if effect is robust)")
    except Exception as e:
        if verbose:
            print(f"Data Subset refutation failed: {e}")
    
    # Summary
    results['summary'] = {
        'treatment': 'MCV1',
        'outcome': 'LogIncidence',
        'n_observations': len(analysis_df),
        'n_countries': analysis_df['Code'].nunique(),
        'primary_estimate': estimate_lr.value,
        'interpretation': interpret_effect(estimate_lr.value)
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nPrimary Causal Effect Estimate: {estimate_lr.value:.4f}")
        print(f"\nInterpretation: {results['summary']['interpretation']}")
    
    return results


def estimate_causal_effect_mcv2(df, verbose=True):
    """
    Estimate the causal effect of MCV2 (second dose coverage) on measles incidence.
    
    This analysis accounts for MCV1 as a confounder/mediator.
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    # Prepare data
    analysis_df, outcome, confounders, effect_modifiers = prepare_data_for_causal_analysis(
        df, treatment='MCV2'
    )
    
    # Create the causal graph
    dag = create_simple_dag_for_mcv2()
    gml_graph = dag_to_gml(dag)
    
    if verbose:
        print("\n" + "="*60)
        print("CAUSAL ANALYSIS: Effect of MCV2 on Measles Incidence")
        print("="*60)
        print(f"\nTreatment: MCV2 (Second dose vaccine coverage %)")
        print(f"Outcome: Log(Measles Incidence + 1)")
        print(f"Confounders (including MCV1): {confounders}")
    
    # Create causal model
    model = CausalModel(
        data=analysis_df,
        treatment='MCV2',
        outcome='LogIncidence',
        graph=gml_graph,
        common_causes=confounders,
        effect_modifiers=effect_modifiers
    )
    
    if verbose:
        print("\n--- Step 1: Causal Model Created ---")
    
    # Identify causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    if verbose:
        print("\n--- Step 2: Causal Effect Identified ---")
        print(identified_estimand)
    
    results = {'estimates': {}, 'refutations': {}}
    
    # Estimate using linear regression
    estimate_lr = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
        method_params={'need_conditional_estimates': False}
    )
    results['estimates']['linear_regression'] = {
        'value': estimate_lr.value,
        'estimate_object': estimate_lr
    }
    
    if verbose:
        print("\n--- Step 3: Causal Effect Estimates ---")
        print(f"\nLinear Regression Estimate: {estimate_lr.value:.4f}")
    
    # Refutations
    if verbose:
        print("\n--- Step 4: Robustness Checks ---")
    
    try:
        refute_random = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="random_common_cause"
        )
        results['refutations']['random_common_cause'] = {
            'new_estimate': refute_random.new_effect
        }
        if verbose:
            print(f"\nRandom Common Cause: {refute_random.new_effect:.4f}")
    except Exception as e:
        if verbose:
            print(f"Random Common Cause failed: {e}")
    
    try:
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        results['refutations']['placebo_treatment'] = {
            'new_estimate': refute_placebo.new_effect
        }
        if verbose:
            print(f"Placebo Treatment: {refute_placebo.new_effect:.4f}")
    except Exception as e:
        if verbose:
            print(f"Placebo Treatment failed: {e}")
    
    results['summary'] = {
        'treatment': 'MCV2',
        'outcome': 'LogIncidence',
        'n_observations': len(analysis_df),
        'n_countries': analysis_df['Code'].nunique(),
        'primary_estimate': estimate_lr.value,
        'interpretation': interpret_effect(estimate_lr.value, treatment='MCV2')
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nPrimary Causal Effect Estimate: {estimate_lr.value:.4f}")
        print(f"\nInterpretation: {results['summary']['interpretation']}")
    
    return results


def interpret_effect(effect_value, treatment='MCV1'):
    """
    Interpret the causal effect estimate.
    
    Effect is on log-transformed incidence, so:
    - A 1-unit increase in vaccine coverage (1%)
    - Changes log(incidence+1) by effect_value
    - Multiplicative effect on incidence: exp(effect_value)
    """
    # Convert to percentage change in incidence
    pct_change = (np.exp(effect_value) - 1) * 100
    
    direction = "decreases" if effect_value < 0 else "increases"
    
    interpretation = (
        f"A 1 percentage point increase in {treatment} coverage "
        f"{direction} measles incidence by approximately {abs(pct_change):.2f}%.\n"
        f"A 10 percentage point increase in coverage would {direction} "
        f"incidence by approximately {abs((np.exp(effect_value * 10) - 1) * 100):.1f}%."
    )
    
    return interpretation


def sensitivity_analysis(df, treatment='MCV1', verbose=True):
    """
    Perform sensitivity analysis for unobserved confounding.
    
    Uses E-value method to assess how strong an unmeasured confounder
    would need to be to explain away the observed effect.
    
    Note: Uses simplified model without effect modifiers (DoWhy limitation).
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    analysis_df, outcome, confounders, effect_modifiers = prepare_data_for_causal_analysis(
        df, treatment=treatment
    )
    
    # E-value analysis requires model WITHOUT effect modifiers (DoWhy limitation)
    # Use simpler model specification for sensitivity analysis
    model = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome='LogIncidence',
        common_causes=confounders
        # Note: effect_modifiers excluded for E-value compatibility
    )
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    
    if verbose:
        print("\n" + "="*60)
        print(f"SENSITIVITY ANALYSIS: {treatment}")
        print("="*60)
    
    results = {'original_estimate': estimate.value}
    
    # E-value analysis - quantifies robustness to unmeasured confounding
    try:
        refute_eval = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            simulation_method="e-value"
        )
        
        results['e_value_refutation'] = refute_eval
        
        if verbose:
            print("\nE-value Analysis:")
            print(refute_eval)
        
    except Exception as e:
        results['e_value_error'] = str(e)
        if verbose:
            print(f"E-value analysis failed: {e}")
    
    # Additional sensitivity: effect of varying confounder strength
    try:
        refute_strength = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="add_unobserved_common_cause",
            confounders_effect_on_treatment="linear",
            confounders_effect_on_outcome="linear",
            effect_strength_on_treatment=0.1,
            effect_strength_on_outcome=0.1
        )
        
        results['confounder_strength_refutation'] = {
            'new_estimate': refute_strength.new_effect,
            'effect_strength': 0.1
        }
        
        if verbose:
            print(f"\nConfounder Strength Test (effect=0.1):")
            print(f"  Original estimate: {estimate.value:.4f}")
            print(f"  New estimate: {refute_strength.new_effect:.4f}")
            change_pct = abs(refute_strength.new_effect - estimate.value) / abs(estimate.value) * 100
            print(f"  Change: {change_pct:.1f}%")
            
    except Exception as e:
        results['confounder_strength_error'] = str(e)
        if verbose:
            print(f"Confounder strength test failed: {e}")
    
    return results


def estimate_causal_effect_hesitancy(df, verbose=True):
    """
    Estimate the causal effect of Vaccine Hesitancy on measles incidence.
    
    Vaccine hesitancy as treatment allows us to estimate:
    1. Direct effect: Hesitancy → Incidence (e.g., delayed care-seeking)
    2. Total effect including mediation through vaccine coverage
    
    This analysis is policy-relevant: what would happen if we reduced
    vaccine hesitancy through public health campaigns?
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    # Prepare data - VaccineHesitancy as treatment
    outcome = 'LogIncidence'
    
    # Confounders that affect both hesitancy and incidence
    # Note: MCV1/MCV2 are MEDIATORS, not confounders (hesitancy → coverage → incidence)
    confounders = ['LogGDPpc', 'LogHealthExpPC', 'PolStability', 'HIC', 
                   'UrbanPop', 'HouseholdSize', 'NetMigration']
    effect_modifiers = ['LogPopDensity', 'BirthRate']
    
    all_vars = ['VaccineHesitancy', outcome] + confounders + effect_modifiers
    all_vars = list(dict.fromkeys(all_vars))
    
    analysis_df = df[all_vars + ['Code', 'Year', 'Country', 'IncomeGroup']].dropna()
    
    if verbose:
        print("\n" + "="*60)
        print("CAUSAL ANALYSIS: Effect of Vaccine Hesitancy on Measles Incidence")
        print("="*60)
        print(f"\nTreatment: VaccineHesitancy (% disagreeing vaccines are effective)")
        print(f"Outcome: Log(Measles Incidence + 1)")
        print(f"Confounders: {confounders}")
        print(f"\nData: {len(analysis_df)} observations, {analysis_df['Code'].nunique()} countries")
    
    # Create causal model
    model = CausalModel(
        data=analysis_df,
        treatment='VaccineHesitancy',
        outcome='LogIncidence',
        common_causes=confounders,
        effect_modifiers=effect_modifiers
    )
    
    if verbose:
        print("\n--- Step 1: Causal Model Created ---")
    
    # Identify effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    if verbose:
        print("\n--- Step 2: Causal Effect Identified ---")
        print(identified_estimand)
    
    results = {'estimates': {}, 'refutations': {}}
    
    # Estimate using linear regression
    estimate_lr = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        target_units="ate",
        method_params={'need_conditional_estimates': False}
    )
    results['estimates']['linear_regression'] = {
        'value': estimate_lr.value,
        'estimate_object': estimate_lr
    }
    
    if verbose:
        print("\n--- Step 3: Causal Effect Estimates ---")
        print(f"\nLinear Regression Estimate: {estimate_lr.value:.4f}")
    
    # Robustness checks
    if verbose:
        print("\n--- Step 4: Robustness Checks ---")
    
    try:
        refute_random = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="random_common_cause"
        )
        results['refutations']['random_common_cause'] = {
            'new_estimate': refute_random.new_effect
        }
        if verbose:
            print(f"\nRandom Common Cause: {refute_random.new_effect:.4f}")
    except Exception as e:
        if verbose:
            print(f"Random Common Cause failed: {e}")
    
    try:
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_lr,
            method_name="placebo_treatment_refuter",
            placebo_type="permute"
        )
        results['refutations']['placebo_treatment'] = {
            'new_estimate': refute_placebo.new_effect
        }
        if verbose:
            print(f"Placebo Treatment: {refute_placebo.new_effect:.4f}")
    except Exception as e:
        if verbose:
            print(f"Placebo Treatment failed: {e}")
    
    # Interpretation
    pct_change = (np.exp(estimate_lr.value) - 1) * 100
    direction = "increases" if estimate_lr.value > 0 else "decreases"
    
    interpretation = (
        f"A 1 percentage point increase in vaccine hesitancy "
        f"{direction} measles incidence by approximately {abs(pct_change):.2f}%.\n"
        f"POLICY IMPLICATION: Reducing hesitancy by 10 percentage points "
        f"would {('decrease' if estimate_lr.value > 0 else 'increase')} "
        f"incidence by approximately {abs((np.exp(-estimate_lr.value * 10) - 1) * 100):.1f}%."
    )
    
    results['summary'] = {
        'treatment': 'VaccineHesitancy',
        'outcome': 'LogIncidence',
        'n_observations': len(analysis_df),
        'n_countries': analysis_df['Code'].nunique(),
        'primary_estimate': estimate_lr.value,
        'interpretation': interpretation
    }
    
    if verbose:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"\nPrimary Causal Effect Estimate: {estimate_lr.value:.4f}")
        print(f"\nInterpretation: {interpretation}")
    
    return results


def estimate_mediation_hesitancy_coverage(df, verbose=True):
    """
    Estimate mediation analysis: How much of hesitancy's effect on incidence
    is mediated through vaccine coverage (MCV1 and MCV2)?
    
    Causal paths:
    - VaccineHesitancy → MCV1 → MeaslesIncidence
    - VaccineHesitancy → MCV2 → MeaslesIncidence
    - VaccineHesitancy → MCV1 → MCV2 → MeaslesIncidence (sequential)
    
    This decomposition helps understand:
    - Direct effect: Hesitancy affects incidence through other channels
    - Indirect effect: Hesitancy reduces coverage, which increases incidence
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    outcome = 'LogIncidence'
    treatment = 'VaccineHesitancy'
    
    confounders = ['LogGDPpc', 'LogHealthExpPC', 'PolStability', 'HIC', 
                   'UrbanPop', 'HouseholdSize', 'NetMigration']
    
    all_vars = [treatment, 'MCV1', 'MCV2', outcome] + confounders
    analysis_df = df[all_vars + ['Code', 'Year', 'IncomeGroup']].dropna()
    
    if verbose:
        print("\n" + "="*60)
        print("MEDIATION ANALYSIS: Hesitancy → Coverage → Incidence")
        print("="*60)
        print(f"\nTreatment: {treatment}")
        print(f"Mediators: MCV1, MCV2 (both doses)")
        print(f"Outcome: {outcome}")
        print(f"\nData: {len(analysis_df)} observations")
    
    results = {}
    
    # ============================================
    # Step 1: Total effect (Hesitancy → Incidence)
    # Not controlling for any mediators
    # ============================================
    model_total = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders
    )
    estimand_total = model_total.identify_effect(proceed_when_unidentifiable=True)
    estimate_total = model_total.estimate_effect(
        estimand_total,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    results['total_effect'] = estimate_total.value
    
    # ============================================
    # Step 2: Direct effect controlling for BOTH MCV1 and MCV2
    # ============================================
    model_direct = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders + ['MCV1', 'MCV2']  # Control for both mediators
    )
    estimand_direct = model_direct.identify_effect(proceed_when_unidentifiable=True)
    estimate_direct = model_direct.estimate_effect(
        estimand_direct,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    results['direct_effect'] = estimate_direct.value
    
    # ============================================
    # Step 3: Effect of Hesitancy on MCV1
    # ============================================
    model_mcv1 = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome='MCV1',
        common_causes=confounders
    )
    estimand_mcv1 = model_mcv1.identify_effect(proceed_when_unidentifiable=True)
    estimate_mcv1 = model_mcv1.estimate_effect(
        estimand_mcv1,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    results['hesitancy_to_mcv1'] = estimate_mcv1.value
    
    # ============================================
    # Step 4: Effect of Hesitancy on MCV2 (controlling for MCV1)
    # This captures the ADDITIONAL reluctance for second dose
    # ============================================
    model_mcv2 = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome='MCV2',
        common_causes=confounders + ['MCV1']  # Control for MCV1 (sequential)
    )
    estimand_mcv2 = model_mcv2.identify_effect(proceed_when_unidentifiable=True)
    estimate_mcv2 = model_mcv2.estimate_effect(
        estimand_mcv2,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    results['hesitancy_to_mcv2_direct'] = estimate_mcv2.value
    
    # Also get total effect on MCV2 (not controlling for MCV1)
    model_mcv2_total = CausalModel(
        data=analysis_df,
        treatment=treatment,
        outcome='MCV2',
        common_causes=confounders
    )
    estimand_mcv2_total = model_mcv2_total.identify_effect(proceed_when_unidentifiable=True)
    estimate_mcv2_total = model_mcv2_total.estimate_effect(
        estimand_mcv2_total,
        method_name="backdoor.linear_regression",
        target_units="ate"
    )
    results['hesitancy_to_mcv2_total'] = estimate_mcv2_total.value
    
    # ============================================
    # Calculate indirect effects
    # ============================================
    results['indirect_effect_total'] = results['total_effect'] - results['direct_effect']
    
    # Proportion mediated
    if abs(results['total_effect']) > 0.0001:
        results['proportion_mediated'] = results['indirect_effect_total'] / results['total_effect']
    else:
        results['proportion_mediated'] = 0
    
    if verbose:
        print("\n" + "-"*60)
        print("EFFECT OF HESITANCY ON VACCINE COVERAGE")
        print("-"*60)
        print(f"\nHesitancy → MCV1: {results['hesitancy_to_mcv1']:.4f}")
        print(f"  (1% ↑ hesitancy → {results['hesitancy_to_mcv1']:.2f}% change in 1st dose coverage)")
        
        print(f"\nHesitancy → MCV2 (total): {results['hesitancy_to_mcv2_total']:.4f}")
        print(f"  (1% ↑ hesitancy → {results['hesitancy_to_mcv2_total']:.2f}% change in 2nd dose coverage)")
        
        print(f"\nHesitancy → MCV2 (direct, controlling for MCV1): {results['hesitancy_to_mcv2_direct']:.4f}")
        print(f"  (ADDITIONAL reluctance for 2nd shot beyond 1st)")
        
        # Check if people are more reluctant for 2nd dose
        if abs(results['hesitancy_to_mcv2_direct']) > 0.01:
            if results['hesitancy_to_mcv2_direct'] < 0:
                print("\n  ⚠️  YES: Hesitant people are EXTRA reluctant to get 2nd dose!")
            else:
                print("\n  Note: Hesitant people who get 1st dose are more likely to complete 2nd")
        
        print("\n" + "-"*60)
        print("MEDIATION DECOMPOSITION")
        print("-"*60)
        print(f"\nTotal Effect (Hesitancy → Incidence): {results['total_effect']:.4f}")
        print(f"Direct Effect (controlling for MCV1 & MCV2): {results['direct_effect']:.4f}")
        print(f"Indirect Effect (via coverage): {results['indirect_effect_total']:.4f}")
        print(f"\nProportion Mediated through Coverage: {abs(results['proportion_mediated'])*100:.1f}%")
        
        print("\n" + "-"*60)
        print("INTERPRETATION")
        print("-"*60)
        if results['total_effect'] > 0:
            print("\n✓ Vaccine hesitancy INCREASES measles incidence.")
        else:
            print("\n⚠ Unexpected: Hesitancy appears to decrease incidence.")
        
        if results['hesitancy_to_mcv1'] < 0:
            print(f"✓ Hesitancy reduces MCV1 uptake by {abs(results['hesitancy_to_mcv1']):.2f}% per 1% hesitancy")
        
        if results['hesitancy_to_mcv2_direct'] < 0:
            print(f"✓ Hesitancy has ADDITIONAL effect on MCV2: {results['hesitancy_to_mcv2_direct']:.2f}% per 1% hesitancy")
            print("  (People don't want extra shots!)")
    
    return results


def estimate_by_income_group(df, treatment='MCV1', verbose=True):
    """
    Estimate causal effects stratified by income group.
    
    This allows us to see if the effect of vaccination differs
    between high-income and low/middle-income countries.
    
    Note: Uses reduced confounder set for stratified analysis to maintain
    adequate sample sizes within each stratum.
    
    Parameters:
    -----------
    df : DataFrame
        Panel data with imputed values
    treatment : str
        'MCV1' or 'MCV2'
    verbose : bool
        Print progress
    """
    if not DOWHY_AVAILABLE:
        raise ImportError("DoWhy is required for this analysis")
    
    results = {}
    
    # Define income group categories
    income_groups = {
        'High Income': ['High income'],
        'Upper Middle Income': ['Upper middle income'],
        'Lower Middle Income': ['Lower middle income'],
        'Low Income': ['Low income'],
        'Low & Middle Income (Combined)': ['Upper middle income', 'Lower middle income', 'Low income']
    }
    
    if verbose:
        print("\n" + "="*60)
        print(f"STRATIFIED ANALYSIS: {treatment} by Income Group")
        print("="*60)
    
    for group_name, income_levels in income_groups.items():
        # Filter data
        df_subset = df[df['IncomeGroup'].isin(income_levels)].copy()
        
        # Use core confounders for stratified analysis (maintains sample size)
        # HIC is excluded as it's constant within income strata
        outcome = 'LogIncidence'
        confounders = ['LogGDPpc', 'LogHealthExpPC', 'PolStability']
        effect_modifiers = ['LogPopDensity', 'BirthRate']
        
        if treatment == 'MCV2':
            confounders = confounders + ['MCV1']
        
        all_vars = [treatment, outcome] + confounders + effect_modifiers
        all_vars = list(dict.fromkeys(all_vars))  # Remove duplicates
        analysis_df = df_subset[all_vars + ['Code', 'Year']].dropna()
        
        n_obs = len(analysis_df)
        n_countries = analysis_df['Code'].nunique()
        
        if n_obs < 30 or n_countries < 5:
            if verbose:
                print(f"\n{group_name}: Insufficient data (n={n_obs}, countries={n_countries})")
            continue
        
        try:
            # Create model without graph (simpler)
            model = CausalModel(
                data=analysis_df,
                treatment=treatment,
                outcome=outcome,
                common_causes=confounders
            )
            
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                target_units="ate",
                method_params={'need_conditional_estimates': False}
            )
            
            results[group_name] = {
                'estimate': estimate.value,
                'n_observations': n_obs,
                'n_countries': n_countries,
                'pct_change_per_1pct': (np.exp(estimate.value) - 1) * 100
            }
            
            if verbose:
                print(f"\n{group_name}:")
                print(f"  Countries: {n_countries}, Observations: {n_obs}")
                print(f"  Effect: {estimate.value:.4f}")
                print(f"  Interpretation: 1% increase in {treatment} -> {results[group_name]['pct_change_per_1pct']:.2f}% change in incidence")
        
        except Exception as e:
            if verbose:
                print(f"\n{group_name}: Analysis failed - {e}")
    
    return results


def run_full_analysis(save_results=True, verbose=True, use_imputation=True):
    """
    Run the complete causal analysis pipeline.
    
    Parameters:
    -----------
    save_results : bool
        Save results to files
    verbose : bool
        Print progress
    use_imputation : bool
        Use income-group based imputation (recommended for more countries)
    """
    print("\n" + "="*70)
    print("DOWHY CAUSAL ANALYSIS: VACCINE COVERAGE AND MEASLES INCIDENCE")
    if use_imputation:
        print("(Using income-group based imputation for broader country coverage)")
    print("="*70)
    
    # Load data
    df = load_panel_data(use_imputation=use_imputation)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Analysis 1: MCV1 Effect
    print("\n\n" + "#"*70)
    print("ANALYSIS 1: FIRST DOSE VACCINE COVERAGE (MCV1)")
    print("#"*70)
    
    mcv1_results = estimate_causal_effect_mcv1(df, verbose=verbose)
    all_results['MCV1'] = mcv1_results
    
    # Analysis 2: MCV2 Effect
    print("\n\n" + "#"*70)
    print("ANALYSIS 2: SECOND DOSE VACCINE COVERAGE (MCV2)")
    print("#"*70)
    
    mcv2_results = estimate_causal_effect_mcv2(df, verbose=verbose)
    all_results['MCV2'] = mcv2_results
    
    # Analysis 3: Vaccine Hesitancy as Treatment
    print("\n\n" + "#"*70)
    print("ANALYSIS 3: VACCINE HESITANCY AS INTERVENTION TARGET")
    print("#"*70)
    
    hesitancy_results = estimate_causal_effect_hesitancy(df, verbose=verbose)
    all_results['VaccineHesitancy'] = hesitancy_results
    
    # Analysis 4: Mediation Analysis
    print("\n\n" + "#"*70)
    print("ANALYSIS 4: MEDIATION ANALYSIS")
    print("#"*70)
    
    mediation_results = estimate_mediation_hesitancy_coverage(df, verbose=verbose)
    all_results['mediation'] = mediation_results
    
    # Stratified Analysis by Income Group
    print("\n\n" + "#"*70)
    print("STRATIFIED ANALYSIS BY INCOME GROUP")
    print("#"*70)
    
    stratified_mcv1 = estimate_by_income_group(df, treatment='MCV1', verbose=verbose)
    stratified_mcv2 = estimate_by_income_group(df, treatment='MCV2', verbose=verbose)
    all_results['stratified'] = {'MCV1': stratified_mcv1, 'MCV2': stratified_mcv2}
    
    # Sensitivity Analysis
    print("\n\n" + "#"*70)
    print("SENSITIVITY ANALYSIS")
    print("#"*70)
    
    sens_mcv1 = sensitivity_analysis(df, treatment='MCV1', verbose=verbose)
    sens_mcv2 = sensitivity_analysis(df, treatment='MCV2', verbose=verbose)
    all_results['sensitivity'] = {'MCV1': sens_mcv1, 'MCV2': sens_mcv2}
    
    # Generate report
    if save_results:
        save_analysis_report(all_results, RESULTS_DIR / 'causal_analysis_report.txt')
        save_estimates_csv(all_results, RESULTS_DIR / 'causal_estimates.csv')
        save_stratified_results(all_results, RESULTS_DIR / 'causal_estimates_by_income.csv')
        save_mediation_results(all_results, RESULTS_DIR / 'causal_mediation_results.csv')
    
    # Visualize DAGs
    from causal_dag import create_dag_for_hesitancy
    
    # Main DAG (vaccine coverage)
    dag = create_causal_dag()
    visualize_dag(dag, RESULTS_DIR / 'causal_dag_coverage.png')
    
    # Hesitancy DAG
    dag_hesitancy = create_dag_for_hesitancy()
    visualize_hesitancy_dag(dag_hesitancy, RESULTS_DIR / 'causal_dag_hesitancy.png')
    
    print("\n\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    
    return all_results


def save_analysis_report(results, output_path):
    """Save analysis results as a text report"""
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CAUSAL ANALYSIS REPORT: VACCINE COVERAGE AND MEASLES INCIDENCE\n")
        f.write("Using DoWhy Framework\n")
        f.write("="*70 + "\n\n")
        
        # MCV1 and MCV2 Analysis
        for treatment in ['MCV1', 'MCV2']:
            if treatment not in results:
                continue
            f.write(f"\n{'='*50}\n")
            f.write(f"{treatment} ANALYSIS\n")
            f.write(f"{'='*50}\n\n")
            
            summary = results[treatment]['summary']
            
            f.write(f"Treatment: {summary['treatment']}\n")
            f.write(f"Outcome: {summary['outcome']}\n")
            f.write(f"Observations: {summary['n_observations']}\n")
            f.write(f"Countries: {summary['n_countries']}\n\n")
            
            f.write("ESTIMATES:\n")
            f.write("-"*40 + "\n")
            for method, est in results[treatment]['estimates'].items():
                f.write(f"  {method}: {est['value']:.4f}\n")
            
            f.write(f"\nPRIMARY ESTIMATE: {summary['primary_estimate']:.4f}\n\n")
            f.write("INTERPRETATION:\n")
            f.write(summary['interpretation'] + "\n\n")
            
            f.write("ROBUSTNESS CHECKS:\n")
            f.write("-"*40 + "\n")
            for method, ref in results[treatment]['refutations'].items():
                f.write(f"  {method}: {ref['new_estimate']:.4f}\n")
        
        # Vaccine Hesitancy Analysis
        if 'VaccineHesitancy' in results:
            f.write(f"\n\n{'='*50}\n")
            f.write("VACCINE HESITANCY AS TREATMENT\n")
            f.write(f"{'='*50}\n\n")
            
            summary = results['VaccineHesitancy']['summary']
            
            f.write(f"Treatment: {summary['treatment']}\n")
            f.write(f"Outcome: {summary['outcome']}\n")
            f.write(f"Observations: {summary['n_observations']}\n")
            f.write(f"Countries: {summary['n_countries']}\n\n")
            
            f.write("ESTIMATES:\n")
            f.write("-"*40 + "\n")
            for method, est in results['VaccineHesitancy']['estimates'].items():
                f.write(f"  {method}: {est['value']:.4f}\n")
            
            f.write(f"\nPRIMARY ESTIMATE: {summary['primary_estimate']:.4f}\n\n")
            f.write("INTERPRETATION:\n")
            f.write(summary['interpretation'] + "\n\n")
            
            f.write("ROBUSTNESS CHECKS:\n")
            f.write("-"*40 + "\n")
            for method, ref in results['VaccineHesitancy']['refutations'].items():
                f.write(f"  {method}: {ref['new_estimate']:.4f}\n")
        
        # Mediation Analysis
        if 'mediation' in results:
            f.write(f"\n\n{'='*50}\n")
            f.write("MEDIATION ANALYSIS\n")
            f.write("Hesitancy → Coverage (MCV1 & MCV2) → Incidence\n")
            f.write(f"{'='*50}\n\n")
            
            med = results['mediation']
            f.write(f"Total Effect (Hesitancy → Incidence): {med['total_effect']:.4f}\n")
            f.write(f"Direct Effect (controlling for MCV1 & MCV2): {med['direct_effect']:.4f}\n")
            f.write(f"Indirect Effect (via coverage): {med['indirect_effect_total']:.4f}\n\n")
            
            f.write("Effect of Hesitancy on Coverage:\n")
            f.write(f"  Hesitancy → MCV1: {med['hesitancy_to_mcv1']:.4f}\n")
            f.write(f"  Hesitancy → MCV2 (total): {med['hesitancy_to_mcv2_total']:.4f}\n")
            f.write(f"  Hesitancy → MCV2 (direct, controlling for MCV1): {med['hesitancy_to_mcv2_direct']:.4f}\n\n")
            f.write(f"Proportion Mediated: {abs(med['proportion_mediated'])*100:.1f}%\n\n")
            
            f.write("INTERPRETATION:\n")
            if med['total_effect'] > 0:
                f.write("Vaccine hesitancy INCREASES measles incidence.\n")
            if med['hesitancy_to_mcv1'] < 0:
                f.write(f"Hesitancy reduces MCV1 uptake by {abs(med['hesitancy_to_mcv1']):.2f}% per 1% hesitancy.\n")
            if med['hesitancy_to_mcv2_direct'] > 0:
                f.write("Hesitant people who DO get 1st dose are MORE likely to complete 2nd.\n")
                f.write("(Selection effect: those who overcome hesitancy for 1st dose are committed)\n")
        
        # Stratified Results
        if 'stratified' in results:
            f.write(f"\n\n{'='*50}\n")
            f.write("STRATIFIED ANALYSIS BY INCOME GROUP\n")
            f.write(f"{'='*50}\n\n")
            
            for treatment in ['MCV1', 'MCV2']:
                if treatment in results['stratified']:
                    f.write(f"\n{treatment}:\n")
                    f.write("-"*40 + "\n")
                    for group, est in results['stratified'][treatment].items():
                        f.write(f"  {group}: {est['estimate']:.4f} ({est['n_countries']} countries)\n")
        
        f.write("\n\n" + "="*70 + "\n")
        f.write("METHODOLOGY NOTES\n")
        f.write("="*70 + "\n\n")
        f.write("""
This analysis uses the DoWhy causal inference framework to estimate the
causal effect of vaccine coverage and vaccine hesitancy on measles incidence.

CONFOUNDERS CONTROLLED FOR:
- LogGDPpc (GDP per capita, log-transformed)
- LogHealthExpPC (Health expenditure per capita, log-transformed)
- PolStability (Political stability index)
- HIC (High-income country indicator)
- UrbanPop (Urban population percentage)
- HouseholdSize (Average household size)
- VaccineHesitancy (% disagreeing vaccines are effective)
- NetMigration (Net migration)

KEY ASSUMPTIONS:
1. The causal DAG correctly represents the data generating process
2. No unmeasured confounders (conditional ignorability)
3. Positivity (all covariate strata have treated and untreated units)
4. SUTVA (no interference between units)

ROBUSTNESS CHECKS:
- Random Common Cause: Adds a random confounder to test sensitivity
- Placebo Treatment: Permutes treatment to test if effect disappears
- Data Subset: Re-estimates on 80% of data to test stability

LIMITATIONS:
- Panel data may have temporal dependencies not fully captured
- Country-level aggregation may mask within-country heterogeneity
- Causal graph assumptions may not perfectly reflect reality
- Vaccine hesitancy data was heavily imputed (85% missing)
""")
    
    print(f"\nReport saved to: {output_path}")


def save_estimates_csv(results, output_path):
    """Save estimates to CSV"""
    rows = []
    
    # MCV1 and MCV2 results
    for treatment in ['MCV1', 'MCV2']:
        if treatment in results:
            summary = results[treatment]['summary']
            for method, est in results[treatment]['estimates'].items():
                rows.append({
                    'Treatment': treatment,
                    'Method': method,
                    'Estimate': est['value'],
                    'N_Observations': summary['n_observations'],
                    'N_Countries': summary['n_countries']
                })
            
            for method, ref in results[treatment]['refutations'].items():
                rows.append({
                    'Treatment': treatment,
                    'Method': f'refutation_{method}',
                    'Estimate': ref['new_estimate'],
                    'N_Observations': summary['n_observations'],
                    'N_Countries': summary['n_countries']
                })
    
    # Vaccine Hesitancy results
    if 'VaccineHesitancy' in results:
        summary = results['VaccineHesitancy']['summary']
        for method, est in results['VaccineHesitancy']['estimates'].items():
            rows.append({
                'Treatment': 'VaccineHesitancy',
                'Method': method,
                'Estimate': est['value'],
                'N_Observations': summary['n_observations'],
                'N_Countries': summary['n_countries']
            })
        
        for method, ref in results['VaccineHesitancy']['refutations'].items():
            rows.append({
                'Treatment': 'VaccineHesitancy',
                'Method': f'refutation_{method}',
                'Estimate': ref['new_estimate'],
                'N_Observations': summary['n_observations'],
                'N_Countries': summary['n_countries']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Estimates saved to: {output_path}")


def save_stratified_results(results, output_path):
    """Save income-group stratified results to CSV"""
    rows = []
    
    if 'stratified' in results:
        for treatment in ['MCV1', 'MCV2']:
            if treatment in results['stratified']:
                for income_group, est in results['stratified'][treatment].items():
                    rows.append({
                        'Treatment': treatment,
                        'Income_Group': income_group,
                        'Estimate': est['estimate'],
                        'Pct_Change_Per_1Pct': est['pct_change_per_1pct'],
                        'N_Observations': est['n_observations'],
                        'N_Countries': est['n_countries']
                    })
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Stratified estimates saved to: {output_path}")


def save_mediation_results(results, output_path):
    """Save mediation analysis results to CSV"""
    if 'mediation' not in results:
        return
    
    med = results['mediation']
    rows = [{
        'Analysis': 'Mediation: VaccineHesitancy → MCV1/MCV2 → LogIncidence',
        'Total_Effect': med.get('total_effect', None),
        'Direct_Effect': med.get('direct_effect', None),
        'Indirect_Effect_Total': med.get('indirect_effect_total', None),
        'Hesitancy_to_MCV1': med.get('hesitancy_to_mcv1', None),
        'Hesitancy_to_MCV2_Total': med.get('hesitancy_to_mcv2_total', None),
        'Hesitancy_to_MCV2_Direct': med.get('hesitancy_to_mcv2_direct', None),
        'Proportion_Mediated': med.get('proportion_mediated', None)
    }]
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Mediation results saved to: {output_path}")


def visualize_hesitancy_dag(dag, output_path=None):
    """Visualize the vaccine hesitancy causal DAG"""
    import matplotlib.pyplot as plt
    
    # Define node colors by role
    node_colors = []
    treatment_nodes = ['VaccineHesitancy']
    outcome_nodes = ['LogIncidence']
    mediator_nodes = ['MCV1', 'MCV2']
    confounder_nodes = ['LogGDPpc', 'LogHealthExpPC', 'PolStability', 'HIC', 
                        'UrbanPop', 'HouseholdSize', 'NetMigration']
    
    for node in dag.nodes():
        if node in treatment_nodes:
            node_colors.append('#E91E63')  # Pink for treatment
        elif node in outcome_nodes:
            node_colors.append('#F44336')  # Red for outcome
        elif node in mediator_nodes:
            node_colors.append('#4CAF50')  # Green for mediators
        elif node in confounder_nodes:
            node_colors.append('#2196F3')  # Blue for confounders
        else:
            node_colors.append('#9E9E9E')  # Gray for others
    
    # Use spring layout
    try:
        pos = nx.nx_agraph.graphviz_layout(dag, prog='dot')
    except:
        pos = nx.spring_layout(dag, k=2, iterations=50, seed=42)
    
    plt.figure(figsize=(16, 12))
    
    nx.draw(dag, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=2500,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='#666666',
            alpha=0.9)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#E91E63', s=200, label='Treatment (Hesitancy)'),
        plt.scatter([], [], c='#F44336', s=200, label='Outcome'),
        plt.scatter([], [], c='#4CAF50', s=200, label='Mediators (Coverage)'),
        plt.scatter([], [], c='#2196F3', s=200, label='Confounders'),
        plt.scatter([], [], c='#9E9E9E', s=200, label='Other'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.title('Causal DAG: Vaccine Hesitancy → Coverage → Measles Incidence\n' +
              '(Mediation Analysis)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Hesitancy DAG visualization saved to: {output_path}")
    
    plt.close()
    
    return pos


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    results = run_full_analysis(save_results=True, verbose=True)

