"""
Generate per-country intervention reports
Focuses on Low-Income Countries (LICs) to identify which countries 
benefit most from each intervention
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils import save_figure


def load_intervention_results(results_dir=None):
    """Load all intervention results"""
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'interventions'
    else:
        results_dir = Path(results_dir)
    
    interventions = {}
    for filepath in results_dir.glob('*_results.csv'):
        intervention_name = filepath.stem.replace('_results', '')
        df = pd.read_csv(filepath)
        interventions[intervention_name] = df
    
    return interventions


def load_income_groups():
    """Load income group classifications"""
    data_dir = Path(__file__).parent.parent.parent / 'data'
    income_df = pd.read_csv(data_dir / 'incomegroups.csv')
    income_map = dict(zip(income_df['Country.Code'], income_df['IncomeGroup']))
    return income_map


def generate_country_reports(intervention_results, panel_df, income_filter='Low income', 
                            output_dir=None):
    """
    Generate detailed per-country reports for specified income group
    
    Parameters:
    -----------
    intervention_results : dict
        Dictionary of intervention results from run_all_interventions
    panel_df : DataFrame
        Panel dataset with country information
    income_filter : str or list
        Income group(s) to filter: 'Low income', 'Lower middle income', etc.
        Use 'LMIC' for Low and Lower-middle income combined
    output_dir : Path
        Output directory for reports
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis' / 'country_reports'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING PER-COUNTRY INTERVENTION REPORTS")
    print("="*70)
    
    # Load income groups
    income_map = load_income_groups()
    
    # Determine which income groups to include
    if income_filter == 'LMIC':
        target_groups = ['Low income', 'Lower middle income']
    elif income_filter == 'LIC':
        target_groups = ['Low income']
    elif isinstance(income_filter, list):
        target_groups = income_filter
    else:
        target_groups = [income_filter]
    
    print(f"\nFiltering for income groups: {target_groups}")
    
    # Get baseline country info
    baseline_year = panel_df['Year'].max()
    baseline_df = panel_df[panel_df['Year'] == baseline_year].copy()
    baseline_df['IncomeGroup'] = baseline_df['Code'].map(income_map)
    
    # Filter to target income groups
    target_countries = baseline_df[baseline_df['IncomeGroup'].isin(target_groups)]['Code'].unique().tolist()
    print(f"Found {len(target_countries)} countries in target income groups")
    
    # Collect all country-level results
    all_country_results = []
    
    for intervention_name, results in intervention_results.items():
        if results is None or len(results) == 0:
            continue
        
        # Filter to target countries
        country_results = results[results['Code'].isin(target_countries)].copy()
        
        if len(country_results) == 0:
            continue
        
        # Add income group
        country_results['IncomeGroup'] = country_results['Code'].map(income_map)
        country_results['Intervention'] = intervention_name
        
        all_country_results.append(country_results)
    
    if len(all_country_results) == 0:
        print("No intervention results found for target countries!")
        return None
    
    # Combine all results
    combined_df = pd.concat(all_country_results, ignore_index=True)
    
    # Generate reports for each intervention
    for intervention_name in combined_df['Intervention'].unique():
        intervention_df = combined_df[combined_df['Intervention'] == intervention_name].copy()
        
        print(f"\n--- {intervention_name} ---")
        
        # Sort by reduction (most benefit first)
        intervention_df = intervention_df.sort_values('Change', ascending=True)
        
        # Save detailed CSV
        csv_path = output_dir / f'{intervention_name}_country_details.csv'
        intervention_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        
        # Create summary statistics by country
        summary_cols = ['Code', 'Country', 'IncomeGroup', 'BaselineIncidence', 
                       'InterventionIncidence', 'Change', 'PercentChange']
        summary_cols = [c for c in summary_cols if c in intervention_df.columns]
        
        # Print top 10 benefiting countries
        print(f"\n  Top 10 countries benefiting from {intervention_name}:")
        top_10 = intervention_df.head(10)[summary_cols]
        for _, row in top_10.iterrows():
            change = row.get('Change', 0)
            pct_change = row.get('PercentChange', 0)
            if pd.notna(change) and change != 0:
                print(f"    {row['Code']}: {row.get('Country', 'Unknown')[:25]:<25} "
                      f"Change: {change:>8.2f}  ({pct_change:>6.2f}%)")
    
    # Generate combined report
    generate_combined_country_report(combined_df, output_dir)
    
    # Generate visualizations
    generate_country_visualizations(combined_df, output_dir)
    
    return combined_df


def generate_combined_country_report(combined_df, output_dir):
    """Generate a combined report showing all interventions per country"""
    
    # Pivot to show all interventions per country
    pivot_df = combined_df.pivot_table(
        index=['Code', 'Country', 'IncomeGroup'],
        columns='Intervention',
        values='Change',
        aggfunc='first'
    ).reset_index()
    
    # Calculate average effect across all interventions
    intervention_cols = [c for c in pivot_df.columns if c not in ['Code', 'Country', 'IncomeGroup']]
    pivot_df['AvgChange'] = pivot_df[intervention_cols].mean(axis=1)
    
    # Sort by average change
    pivot_df = pivot_df.sort_values('AvgChange', ascending=True)
    
    # Save
    csv_path = output_dir / 'all_interventions_by_country.csv'
    pivot_df.to_csv(csv_path, index=False)
    print(f"\nSaved combined report: {csv_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("COUNTRIES WITH HIGHEST POTENTIAL BENEFIT (ALL INTERVENTIONS)")
    print("="*70)
    
    for _, row in pivot_df.head(15).iterrows():
        print(f"{row['Code']}: {row.get('Country', 'Unknown')[:30]:<30} "
              f"({row.get('IncomeGroup', 'Unknown')[:15]:<15}) "
              f"Avg Change: {row['AvgChange']:.2f}")
    
    return pivot_df


def generate_country_visualizations(combined_df, output_dir):
    """Generate visualizations for country-level results"""
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("\nGenerating country-level visualizations...")
    
    # 1. Distribution of intervention effects by country
    fig, ax = plt.subplots(figsize=(12, 6))
    
    interventions = combined_df['Intervention'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(interventions)))
    
    for i, intervention in enumerate(interventions):
        subset = combined_df[combined_df['Intervention'] == intervention]
        changes = subset['Change'].dropna()
        if len(changes) > 0:
            ax.hist(changes, bins=20, alpha=0.5, label=intervention.replace('_', ' ').title(),
                   color=colors[i], edgecolor='black', linewidth=0.5)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax.set_xlabel('Change in Incidence (per 1M population)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Countries', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Intervention Effects Across Countries', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'mixed_effects_analysis/country_reports/figures/effect_distribution.png')
    plt.close()
    
    # 2. Top benefiting countries per intervention
    n_top = 15
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, intervention in enumerate(interventions[:6]):  # Max 6 interventions
        ax = axes[idx]
        subset = combined_df[combined_df['Intervention'] == intervention].copy()
        subset = subset.sort_values('Change', ascending=True).head(n_top)
        
        if len(subset) > 0:
            bars = ax.barh(range(len(subset)), subset['Change'], color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels([f"{c} ({n[:15]})" for c, n in zip(subset['Code'], subset['Country'].fillna('Unknown'))],
                             fontsize=9)
            ax.set_xlabel('Change in Incidence', fontsize=10)
            ax.set_title(intervention.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='x')
    
    # Hide unused axes
    for idx in range(len(interventions), 6):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Top {n_top} Countries Benefiting from Each Intervention', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'mixed_effects_analysis/country_reports/figures/top_countries_by_intervention.png')
    plt.close()
    
    # 3. Heatmap of intervention effects by country
    pivot_df = combined_df.pivot_table(
        index=['Code', 'Country'],
        columns='Intervention',
        values='Change',
        aggfunc='first'
    ).reset_index()
    
    # Get top 30 countries by average effect
    intervention_cols = [c for c in pivot_df.columns if c not in ['Code', 'Country']]
    pivot_df['AvgChange'] = pivot_df[intervention_cols].mean(axis=1)
    top_30 = pivot_df.nsmallest(30, 'AvgChange')
    
    if len(top_30) > 0 and len(intervention_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, 12))
        
        heatmap_data = top_30.set_index('Code')[intervention_cols]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                   center=0, ax=ax, cbar_kws={'label': 'Change in Incidence'},
                   annot_kws={'fontsize': 8})
        
        ax.set_xlabel('Intervention', fontsize=12, fontweight='bold')
        ax.set_ylabel('Country', fontsize=12, fontweight='bold')
        ax.set_title('Intervention Effects by Country (Top 30 Benefiting)', fontsize=14, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        save_figure(fig, 'mixed_effects_analysis/country_reports/figures/intervention_heatmap.png')
        plt.close()
    
    # 4. Box plot comparing interventions
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for box plot
    plot_data = []
    labels = []
    for intervention in interventions:
        subset = combined_df[combined_df['Intervention'] == intervention]['Change'].dropna()
        if len(subset) > 0:
            plot_data.append(subset.values)
            labels.append(intervention.replace('_', '\n').title())
    
    if len(plot_data) > 0:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_ylabel('Change in Incidence (per 1M)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Intervention', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Intervention Effects Across Countries', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_figure(fig, 'mixed_effects_analysis/country_reports/figures/intervention_boxplot.png')
        plt.close()
    
    print(f"Visualizations saved to: {figures_dir}")


def generate_text_report(combined_df, output_dir):
    """Generate a detailed text report"""
    
    report_path = output_dir / 'detailed_country_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED PER-COUNTRY INTERVENTION ANALYSIS\n")
        f.write("Focus: Low-Income Countries (LICs)\n")
        f.write("="*80 + "\n\n")
        
        # Get unique interventions
        interventions = combined_df['Intervention'].unique()
        
        for intervention in interventions:
            f.write("\n" + "="*80 + "\n")
            f.write(f"INTERVENTION: {intervention.replace('_', ' ').upper()}\n")
            f.write("="*80 + "\n\n")
            
            subset = combined_df[combined_df['Intervention'] == intervention].copy()
            subset = subset.sort_values('Change', ascending=True)
            
            # Statistics
            changes = subset['Change'].dropna()
            if len(changes) > 0:
                f.write("Summary Statistics:\n")
                f.write(f"  Countries analyzed: {len(subset)}\n")
                f.write(f"  Mean change: {changes.mean():.2f}\n")
                f.write(f"  Median change: {changes.median():.2f}\n")
                f.write(f"  Std deviation: {changes.std():.2f}\n")
                f.write(f"  Min change: {changes.min():.2f}\n")
                f.write(f"  Max change: {changes.max():.2f}\n")
                f.write(f"  Countries with reduction: {(changes < 0).sum()}\n")
                f.write(f"  Countries with increase: {(changes > 0).sum()}\n")
                f.write(f"  Countries with no change: {(changes == 0).sum()}\n\n")
            
            # Top benefiting countries
            f.write("Countries with Largest Reduction (Top 20):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Code':<6} {'Country':<30} {'Income Group':<20} {'Change':>10} {'% Change':>10}\n")
            f.write("-"*80 + "\n")
            
            for _, row in subset.head(20).iterrows():
                code = str(row.get('Code', 'N/A'))[:5]
                country = str(row.get('Country', 'Unknown'))[:29]
                income = str(row.get('IncomeGroup', 'Unknown'))[:19]
                change = row.get('Change', 0)
                pct = row.get('PercentChange', 0)
                
                if pd.notna(change):
                    f.write(f"{code:<6} {country:<30} {income:<20} {change:>10.2f} {pct:>9.2f}%\n")
            
            f.write("\n")
        
        # Overall recommendations
        f.write("\n" + "="*80 + "\n")
        f.write("POLICY RECOMMENDATIONS BY COUNTRY\n")
        f.write("="*80 + "\n\n")
        
        # Calculate best intervention per country
        pivot = combined_df.pivot_table(
            index=['Code', 'Country', 'IncomeGroup'],
            columns='Intervention',
            values='Change',
            aggfunc='first'
        ).reset_index()
        
        intervention_cols = [c for c in pivot.columns if c not in ['Code', 'Country', 'IncomeGroup']]
        
        if len(intervention_cols) > 0:
            # Find best intervention for each country
            pivot['BestIntervention'] = pivot[intervention_cols].idxmin(axis=1)
            pivot['BestReduction'] = pivot[intervention_cols].min(axis=1)
            
            # Sort by best reduction
            pivot = pivot.sort_values('BestReduction', ascending=True)
            
            f.write(f"{'Code':<6} {'Country':<30} {'Best Intervention':<25} {'Expected Reduction':>15}\n")
            f.write("-"*80 + "\n")
            
            for _, row in pivot.head(30).iterrows():
                code = str(row.get('Code', 'N/A'))[:5]
                country = str(row.get('Country', 'Unknown'))[:29]
                best = str(row.get('BestIntervention', 'N/A')).replace('_', ' ')[:24]
                reduction = row.get('BestReduction', 0)
                
                if pd.notna(reduction):
                    f.write(f"{code:<6} {country:<30} {best:<25} {reduction:>15.2f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Text report saved to: {report_path}")


def run_country_analysis(panel_df=None, intervention_results=None):
    """Run complete country-level analysis"""
    
    from pathlib import Path
    
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'mixed_effects_analysis'
    output_dir = results_dir / 'country_reports'
    
    # Load panel data if not provided
    if panel_df is None:
        panel_path = results_dir / 'panel_data.csv'
        if panel_path.exists():
            panel_df = pd.read_csv(panel_path)
        else:
            from prepare_data import prepare_panel_data
            panel_df = prepare_panel_data()
    
    # Load intervention results if not provided
    if intervention_results is None:
        intervention_results = load_intervention_results()
    
    if not intervention_results:
        print("No intervention results found! Run the main analysis first.")
        return
    
    # Generate reports for LICs (Low Income Countries)
    print("\n" + "="*70)
    print("ANALYSIS FOR LOW-INCOME COUNTRIES (LICs)")
    print("="*70)
    lic_results = generate_country_reports(
        intervention_results, panel_df, 
        income_filter='Low income',
        output_dir=output_dir / 'LIC'
    )
    
    if lic_results is not None and len(lic_results) > 0:
        generate_text_report(lic_results, output_dir / 'LIC')
    
    # Generate reports for LMICs (Low and Lower-Middle Income)
    print("\n" + "="*70)
    print("ANALYSIS FOR LOW AND LOWER-MIDDLE INCOME COUNTRIES (LMICs)")
    print("="*70)
    lmic_results = generate_country_reports(
        intervention_results, panel_df,
        income_filter='LMIC',
        output_dir=output_dir / 'LMIC'
    )
    
    if lmic_results is not None and len(lmic_results) > 0:
        generate_text_report(lmic_results, output_dir / 'LMIC')
    
    print("\n" + "="*70)
    print("COUNTRY ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nReports saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - LIC/: Low-income country reports")
    print("  - LMIC/: Low and lower-middle income country reports")
    print("  - */figures/: Country-level visualizations")
    print("  - */detailed_country_report.txt: Detailed text report")
    print("  - */*_country_details.csv: Per-intervention country data")


if __name__ == '__main__':
    run_country_analysis()


