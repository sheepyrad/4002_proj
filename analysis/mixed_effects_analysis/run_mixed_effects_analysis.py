"""
Main script to run complete mixed-effects negative binomial analysis
Executes data preparation, model fitting, policy simulations, and visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from prepare_data import prepare_panel_data, save_panel_data
from fit_models import fit_linear_mixed_model, fit_negative_binomial_model, save_model
from policy_simulations import run_all_interventions, save_intervention_results, calculate_marginal_effects
from visualize_results import create_comprehensive_visualizations


def run_complete_analysis():
    """
    Run complete mixed-effects analysis pipeline
    
    Steps:
    1. Prepare panel data
    2. Fit linear mixed-effects model
    3. Fit negative binomial model
    4. Run policy simulations
    5. Create visualizations
    """
    print("="*70)
    print("MIXED-EFFECTS NEGATIVE BINOMIAL ANALYSIS")
    print("="*70)
    
    # Set up output directory
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare panel data
    print("\n" + "="*70)
    print("STEP 1: PREPARING PANEL DATA")
    print("="*70)
    panel_df = prepare_panel_data()
    save_panel_data(panel_df, results_dir / 'panel_data.csv')
    
    # Step 2: Fit models
    print("\n" + "="*70)
    print("STEP 2: FITTING MIXED-EFFECTS MODELS")
    print("="*70)
    
    # Fit linear mixed model
    print("\n--- Fitting Linear Mixed-Effects Model ---")
    try:
        linear_model = fit_linear_mixed_model(panel_df, use_random_slope=True)
        save_model(linear_model, results_dir / 'linear_mixed_model.pkl')
        
        # Save model summary
        with open(results_dir / 'linear_mixed_model_summary.txt', 'w') as f:
            f.write(str(linear_model.summary()))
        print(f"\nModel summary saved to: {results_dir / 'linear_mixed_model_summary.txt'}")
        
    except Exception as e:
        print(f"\nError fitting linear mixed model: {e}")
        print("Continuing with available models...")
        linear_model = None
    
    # Fit negative binomial model
    print("\n--- Fitting Negative Binomial Model ---")
    try:
        nb_model = fit_negative_binomial_model(panel_df)
        save_model(nb_model, results_dir / 'negative_binomial_model.pkl')
        
        # Save model summary
        with open(results_dir / 'negative_binomial_model_summary.txt', 'w') as f:
            f.write(str(nb_model.summary()))
        print(f"\nModel summary saved to: {results_dir / 'negative_binomial_model_summary.txt'}")
        
    except Exception as e:
        print(f"\nError fitting negative binomial model: {e}")
        print("Continuing with available models...")
        nb_model = None
    
    # Select primary model for policy analysis
    if linear_model is not None:
        primary_model = linear_model
        print("\nUsing linear mixed-effects model for policy analysis")
    elif nb_model is not None:
        primary_model = nb_model
        print("\nUsing negative binomial model for policy analysis")
    else:
        print("\nERROR: No models successfully fitted. Cannot proceed with policy analysis.")
        return
    
    # Step 3: Policy simulations
    print("\n" + "="*70)
    print("STEP 3: RUNNING POLICY SIMULATIONS")
    print("="*70)
    
    try:
        intervention_results = run_all_interventions(primary_model, panel_df)
        save_intervention_results(intervention_results, results_dir / 'interventions')
        
        # Print summary
        print("\n" + "-"*70)
        print("INTERVENTION SUMMARY")
        print("-"*70)
        for intervention, results in intervention_results.items():
            affected = results[results['InterventionApplied'] == 1]
            if len(affected) > 0:
                print(f"\n{intervention}:")
                print(f"  Countries affected: {affected['Code'].nunique()}")
                print(f"  Mean reduction: {affected['Change'].mean():.2f} cases per 1M")
                print(f"  Mean percent reduction: {affected['PercentChange'].mean():.2f}%")
                print(f"  Total cases prevented: {affected['Change'].sum():.0f}")
        
    except Exception as e:
        print(f"\nError running policy simulations: {e}")
        intervention_results = None
    
    # Step 4: Visualizations
    print("\n" + "="*70)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("="*70)
    
    try:
        create_comprehensive_visualizations(
            primary_model, 
            panel_df, 
            intervention_results_dict=intervention_results,
            output_dir=results_dir / 'figures'
        )
    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 5: Generate policy recommendations report
    print("\n" + "="*70)
    print("STEP 5: GENERATING POLICY RECOMMENDATIONS REPORT")
    print("="*70)
    
    try:
        generate_policy_report(primary_model, panel_df, intervention_results, results_dir)
    except Exception as e:
        print(f"\nError generating policy report: {e}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {results_dir}")
    print("\nGenerated files:")
    print("  - panel_data.csv: Prepared panel dataset")
    print("  - linear_mixed_model.pkl: Fitted linear mixed model")
    print("  - negative_binomial_model.pkl: Fitted negative binomial model")
    print("  - interventions/: Intervention scenario results")
    print("  - figures/: All visualizations")
    print("  - policy_recommendations.txt: Policy recommendations report")


def generate_policy_report(model, panel_df, intervention_results, output_dir):
    """
    Generate a text report with policy recommendations
    
    Parameters:
    -----------
    model : MixedLMResults or NegativeBinomialResults
        Fitted model
    panel_df : DataFrame
        Panel dataset
    intervention_results : dict
        Intervention results
    output_dir : Path
        Output directory
    """
    report_path = output_dir / 'policy_recommendations.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("POLICY RECOMMENDATIONS REPORT\n")
        f.write("Mixed-Effects Analysis of Measles Incidence\n")
        f.write("="*70 + "\n\n")
        
        # Model summary
        f.write("MODEL SUMMARY\n")
        f.write("-"*70 + "\n")
        if hasattr(model, 'summary'):
            f.write(str(model.summary()) + "\n\n")
        
        # Key findings
        f.write("KEY FINDINGS\n")
        f.write("-"*70 + "\n")
        
        # Extract coefficients if available
        if hasattr(model, 'params'):
            params = model.params
            if 'MCV1' in params.index:
                mcv1_coef = params['MCV1']
                f.write(f"MCV1 Coverage Effect: {mcv1_coef:.4f}\n")
                f.write(f"  Interpretation: A 1 percentage point increase in MCV1 coverage\n")
                f.write(f"  is associated with a {mcv1_coef:.4f} change in log(incidence).\n\n")
            
            if 'MCV2' in params.index:
                mcv2_coef = params['MCV2']
                f.write(f"MCV2 Coverage Effect: {mcv2_coef:.4f}\n")
                f.write(f"  Interpretation: A 1 percentage point increase in MCV2 coverage\n")
                f.write(f"  is associated with a {mcv2_coef:.4f} change in log(incidence).\n\n")
        
        # Intervention results
        if intervention_results:
            f.write("INTERVENTION SCENARIOS\n")
            f.write("-"*70 + "\n")
            
            for intervention, results in intervention_results.items():
                affected = results[results['InterventionApplied'] == 1]
                if len(affected) > 0:
                    f.write(f"\n{intervention.replace('_', ' ').title()}:\n")
                    f.write(f"  Countries affected: {affected['Code'].nunique()}\n")
                    f.write(f"  Mean reduction: {affected['Change'].mean():.2f} cases per 1M population\n")
                    f.write(f"  Mean percent reduction: {affected['PercentChange'].mean():.2f}%\n")
                    f.write(f"  Total cases prevented: {affected['Change'].sum():.0f}\n")
        
        # Policy recommendations
        f.write("\n\nPOLICY RECOMMENDATIONS\n")
        f.write("-"*70 + "\n")
        
        if intervention_results:
            # Find most effective intervention
            best_intervention = None
            best_reduction = -np.inf
            
            for intervention, results in intervention_results.items():
                affected = results[results['InterventionApplied'] == 1]
                if len(affected) > 0:
                    mean_reduction = affected['Change'].mean()
                    if mean_reduction > best_reduction:
                        best_reduction = mean_reduction
                        best_intervention = intervention
            
            if best_intervention:
                f.write(f"\n1. PRIORITY INTERVENTION: {best_intervention.replace('_', ' ').title()}\n")
                f.write(f"   This intervention shows the largest mean reduction in measles incidence.\n")
            
            f.write("\n2. RECOMMENDED ACTIONS:\n")
            f.write("   - Increase MCV1 coverage to 95% in low and middle-income countries\n")
            f.write("   - Increase MCV2 coverage to 95% in low and middle-income countries\n")
            f.write("   - Increase health expenditure by at least 1% of GDP\n")
            f.write("   - Improve political stability through governance reforms\n")
            
            f.write("\n3. TARGET COUNTRIES:\n")
            f.write("   Focus interventions on low and middle-income countries (LMICs)\n")
            f.write("   where baseline coverage is below 95%.\n")
        
        f.write("\n4. UNCERTAINTY AND LIMITATIONS:\n")
        f.write("   - Model predictions include uncertainty intervals\n")
        f.write("   - Results are based on historical data and may not capture\n")
        f.write("     all factors affecting measles incidence\n")
        f.write("   - Policy implementation should consider local context\n")
        f.write("     and feasibility constraints\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Policy recommendations report saved to: {report_path}")


if __name__ == '__main__':
    run_complete_analysis()

