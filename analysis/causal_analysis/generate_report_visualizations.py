"""
Generate visualizations for the causal inference report.
Focus on MCV1 administration and vaccine hesitancy.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_dag import create_simple_dag_for_mcv1, create_dag_for_hesitancy, visualize_dag
from prepare_imputed_data import prepare_imputed_data

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                          'results', 'causal_analysis')

def plot_effect_by_income_group():
    """Plot MCV1 causal effects by income group."""
    # Data from analysis results
    income_groups = ['High\nIncome', 'Upper Middle\nIncome', 'Lower Middle\nIncome', 'Low\nIncome']
    effects = [-4.60, -0.73, -1.54, -3.12]  # % change per 1% MCV1 coverage
    countries = [63, 52, 49, 25]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax.barh(income_groups, effects, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, effect, n in zip(bars, effects, countries):
        width = bar.get_width()
        ax.text(width - 0.3, bar.get_y() + bar.get_height()/2, 
                f'{effect:.2f}%\n(n={n})', 
                ha='right', va='center', fontsize=11, fontweight='bold', color='white')
    
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('% Change in Measles Incidence per 1% Increase in MCV1 Coverage', fontsize=12)
    ax.set_title('Causal Effect of MCV1 on Measles Incidence by Income Group', fontsize=14, fontweight='bold')
    ax.set_xlim(-5.5, 0.5)
    
    # Add interpretation
    ax.text(-5.3, -0.8, 'Negative values indicate protective effect (reduced incidence)', 
            fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mcv1_effect_by_income.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: mcv1_effect_by_income.png")

def plot_evalue_visualization():
    """Visualize E-value sensitivity analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # E-value data
    categories = ['E-value\n(Point Est.)', 'E-value\n(95% CI)', 'Largest Observed\nConfounder']
    values = [1.12, 1.11, 1.04]
    colors = ['#E63946', '#457B9D', '#A8DADC']
    
    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Null (no effect)')
    ax.set_ylabel('Risk Ratio Required to Explain Away Effect', fontsize=12)
    ax.set_title('E-Value Sensitivity Analysis for MCV1 → Measles Incidence', fontsize=14, fontweight='bold')
    ax.set_ylim(0.95, 1.20)
    
    # Add interpretation box
    textstr = 'Interpretation:\n• E-value > Largest confounder → Robust\n• An unmeasured confounder would need\n  RR ≥ 1.12 to nullify the effect'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'evalue_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: evalue_analysis.png")

def plot_hesitancy_pathway():
    """Visualize the mediation pathway for vaccine hesitancy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create pathway diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Boxes
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=2)
    
    # Vaccine Hesitancy
    ax.text(1.5, 3, 'Vaccine\nHesitancy', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F4A261', edgecolor='black', linewidth=2))
    
    # MCV1
    ax.text(5, 4.5, 'MCV1\nCoverage', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2A9D8F', edgecolor='black', linewidth=2))
    
    # Incidence
    ax.text(8.5, 3, 'Measles\nIncidence', fontsize=14, ha='center', va='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E76F51', edgecolor='black', linewidth=2))
    
    # Arrows
    ax.annotate('', xy=(4.0, 4.2), xytext=(2.5, 3.3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7.5, 3.3), xytext=(6.0, 4.2),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(7.5, 3), xytext=(2.5, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=2, linestyle='dashed'))
    
    # Labels
    ax.text(3.5, 4.5, '-0.23%', fontsize=12, fontweight='bold', color='#264653')
    ax.text(6.7, 4.3, '-2.30%', fontsize=12, fontweight='bold', color='#264653')
    ax.text(5, 2.5, '+0.18%\n(direct)', fontsize=11, fontweight='bold', color='#E76F51', ha='center')
    
    # Title
    ax.text(5, 5.8, 'Mediation Pathway: Vaccine Hesitancy → Measles Incidence', 
            fontsize=16, fontweight='bold', ha='center')
    
    ax.text(5, 0.5, 'Note: Per 1 percentage point increase. Dashed line = direct effect.', 
            fontsize=10, ha='center', style='italic', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hesitancy_pathway.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: hesitancy_pathway.png")

def plot_policy_impact_projections():
    """Project expected impact of policy interventions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Coverage scenarios
    ax1 = axes[0]
    scenarios = ['Current\n(84%)', '+5%\n(89%)', '+10%\n(94%)', '+15%\n(99%)']
    coverage = [84, 89, 94, 99]
    # Calculate incidence reduction: exp(-0.023 * coverage_increase) - 1
    incidence_reduction = [0, 10.9, 20.6, 29.2]  # approximate % reduction from baseline
    
    bars = ax1.bar(scenarios, incidence_reduction, color=['#264653', '#2A9D8F', '#E9C46A', '#F4A261'], 
                   edgecolor='black', linewidth=1.5)
    
    for bar, pct in zip(bars, incidence_reduction):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1, f'{pct:.1f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('Expected % Reduction in Measles Incidence', fontsize=12)
    ax1.set_xlabel('MCV1 Coverage Scenario', fontsize=12)
    ax1.set_title('Projected Impact of MCV1 Coverage Increases', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 35)
    
    # Right plot: Income group targets
    ax2 = axes[1]
    groups = ['Low\nIncome', 'Lower Middle\nIncome', 'Upper Middle\nIncome', 'High\nIncome']
    current = [74, 82, 88, 94]
    target = [90, 93, 95, 98]
    
    x = np.arange(len(groups))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, current, width, label='Current Coverage', 
                    color='#457B9D', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x + width/2, target, width, label='Target Coverage', 
                    color='#E63946', edgecolor='black', linewidth=1.5)
    
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Herd Immunity Threshold')
    
    ax2.set_ylabel('MCV1 Coverage (%)', fontsize=12)
    ax2.set_xlabel('Income Group', fontsize=12)
    ax2.set_title('Current vs. Target MCV1 Coverage', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylim(65, 100)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'policy_impact_projections.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: policy_impact_projections.png")

def plot_refutation_tests():
    """Visualize refutation test results."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    tests = ['Original\nEstimate', 'Random Common\nCause Added', 'Placebo\nTreatment', 'Data Subset\n(80%)']
    estimates = [-0.0230, -0.0230, -0.0004, -0.0228]
    colors = ['#2A9D8F', '#2A9D8F', '#E76F51', '#2A9D8F']
    
    bars = ax.bar(tests, estimates, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.001, f'{height:.4f}', 
                ha='center', va='top', fontsize=11, fontweight='bold', color='white')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Estimated Causal Effect', fontsize=12)
    ax.set_title('Robustness Checks: MCV1 → Measles Incidence', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.03, 0.005)
    
    # Add pass/fail indicators
    ax.text(0, 0.003, '✓ Robust', fontsize=12, ha='center', color='green', fontweight='bold')
    ax.text(1, 0.003, '✓ Robust', fontsize=12, ha='center', color='green', fontweight='bold')
    ax.text(2, 0.003, '✓ Effect disappears', fontsize=10, ha='center', color='green', fontweight='bold')
    ax.text(3, 0.003, '✓ Stable', fontsize=12, ha='center', color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'refutation_tests.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: refutation_tests.png")

def plot_hesitancy_data_coverage():
    """Show vaccine hesitancy data coverage by income group."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    income_groups = ['High Income', 'Upper Middle', 'Lower Middle', 'Low Income']
    missing_pct = [79.7, 84.6, 87.6, 86.4]  # From the analysis output
    observed_pct = [100 - m for m in missing_pct]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax.barh(income_groups, observed_pct, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, obs, miss in zip(bars, observed_pct, missing_pct):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{obs:.1f}% observed\n({miss:.1f}% imputed)', 
                ha='left', va='center', fontsize=10)
    
    ax.set_xlabel('% of Observations with Original (Non-Imputed) Data', fontsize=12)
    ax.set_title('Vaccine Hesitancy Data Availability by Income Group', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 50)
    
    # Add warning
    ax.text(25, -0.8, '⚠ Hesitancy estimates limited by high imputation rates (84% overall)', 
            fontsize=10, ha='center', style='italic', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hesitancy_data_coverage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: hesitancy_data_coverage.png")

def main():
    """Generate all report visualizations."""
    print("=" * 60)
    print("Generating Report Visualizations")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate visualizations
    plot_effect_by_income_group()
    plot_evalue_visualization()
    plot_hesitancy_pathway()
    plot_policy_impact_projections()
    plot_refutation_tests()
    plot_hesitancy_data_coverage()
    
    # Generate DAGs using existing functions
    print("\nGenerating causal DAGs...")
    try:
        dag = create_simple_dag_for_mcv1(outcome='LogIncidence', use_log_vars=True)
        visualize_dag(dag, output_path=os.path.join(OUTPUT_DIR, 'causal_dag_MCV1.png'))
        print(f"Saved: causal_dag_MCV1.png")
    except Exception as e:
        print(f"Warning: Could not generate MCV1 DAG: {e}")
    
    try:
        dag = create_dag_for_hesitancy(outcome='LogIncidence', use_log_vars=True)
        visualize_dag(dag, output_path=os.path.join(OUTPUT_DIR, 'causal_dag_hesitancy.png'))
        print(f"Saved: causal_dag_hesitancy.png")
    except Exception as e:
        print(f"Warning: Could not generate hesitancy DAG: {e}")
    
    print("\n" + "=" * 60)
    print("All visualizations saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()

