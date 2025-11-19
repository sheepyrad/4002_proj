"""
Plot top 3 loadings frequency counts separately for each PC
Also plot top 3 coefficients frequency counts separately for each PC
Supports both 1st and 2nd coverage analyses
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_top3_loadings_counts(suffix=""):
    """Plot top 3 loadings frequency counts for each PC separately
    
    Args:
        suffix: Suffix for results directory (e.g., "_2nd_coverage" or "")
    """
    
    # Set up paths
    results_dir = Path(__file__).parent / "results" / f"yearly_reports{suffix}"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    coverage_label = "2nd Coverage" if suffix == "_2nd_coverage" else "1st Coverage"
    
    # Load data for each PC
    pc_files = {
        'PC1': results_dir / 'pc1_top3_loadings_per_year.csv',
        'PC2': results_dir / 'pc2_top3_loadings_per_year.csv',
        'PC3': results_dir / 'pc3_top3_loadings_per_year.csv'
    }
    
    pc_data = {}
    for pc_name, file_path in pc_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            pc_data[pc_name] = df
            print(f"Loaded {pc_name} data: {len(df)} years")
        else:
            print(f"Warning: {file_path} not found")
    
    if not pc_data:
        print("No data files found!")
        return
    
    # Count frequencies for each PC
    pc_counts = {}
    for pc_name, df in pc_data.items():
        factor_counts = {}
        
        # Count occurrences in Rank_1, Rank_2, Rank_3
        for rank in [1, 2, 3]:
            factor_col = f'Rank_{rank}_Factor'
            if factor_col in df.columns:
                factors = df[factor_col].dropna()
                for factor in factors:
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        pc_counts[pc_name] = {
            'counts': factor_counts,
            'sorted': sorted_factors
        }
        
        print(f"\n{pc_name} - Top factors:")
        for factor, count in sorted_factors[:10]:  # Show top 10
            print(f"  {factor}: {count} year(s)")
    
    # Create separate plots for each PC
    colors = {'PC1': '#1f77b4', 'PC2': '#ff7f0e', 'PC3': '#2ca02c'}
    
    for pc_name, counts_data in pc_counts.items():
        sorted_factors = counts_data['sorted']
        factor_counts = counts_data['counts']
        
        # Get top factors (at least top 10, or all if fewer)
        n_factors = min(15, len(sorted_factors))
        factors_plot = [f for f, _ in sorted_factors[:n_factors]]
        counts_plot = [c for _, c in sorted_factors[:n_factors]]
        
        # Identify top 3
        top3_factors = [f for f, _ in sorted_factors[:3]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color bars: highlight top 3
        bar_colors = [colors[pc_name] if f in top3_factors else '#cccccc' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Number of Years in Top 3", fontsize=12, fontweight='bold')
        ax.set_ylabel("Factor", fontsize=12, fontweight='bold')
        ax.set_title(f"{pc_name} - Top 3 Loadings Frequency ({coverage_label})", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[pc_name], alpha=0.8, label='Top 3 Most Frequent'),
            Patch(facecolor='#cccccc', alpha=0.8, label='Other Factors')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        if suffix:
            output_path = output_dir / f"{pc_name.lower()}_top3_loadings_frequency{suffix}.png"
        else:
            output_path = output_dir / f"{pc_name.lower()}_top3_loadings_frequency.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved {pc_name} plot to {output_path}")
        plt.close()
    
    # Also create a combined figure with subplots
    fig, axes = plt.subplots(1, len(pc_data), figsize=(7*len(pc_data), 8))
    if len(pc_data) == 1:
        axes = [axes]
    
    for idx, (pc_name, counts_data) in enumerate(pc_counts.items()):
        ax = axes[idx]
        sorted_factors = counts_data['sorted']
        factor_counts = counts_data['counts']
        
        # Get top factors
        n_factors = min(10, len(sorted_factors))
        factors_plot = [f for f, _ in sorted_factors[:n_factors]]
        counts_plot = [c for _, c in sorted_factors[:n_factors]]
        
        # Identify top 3
        top3_factors = [f for f, _ in sorted_factors[:3]]
        
        # Color bars
        bar_colors = [colors[pc_name] if f in top3_factors else '#cccccc' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
        
        ax.set_xlabel("Years in Top 3", fontsize=11, fontweight='bold')
        ax.set_ylabel("Factor", fontsize=11, fontweight='bold')
        ax.set_title(f"{pc_name}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f"Top 3 Loadings Frequency by Principal Component ({coverage_label})", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save combined figure
    if suffix:
        combined_path = output_dir / f"all_pcs_top3_loadings_frequency{suffix}.png"
    else:
        combined_path = output_dir / "all_pcs_top3_loadings_frequency.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot to {combined_path}")
    plt.close()
    
    print("\n" + "="*60)
    print(f"All loadings plots generated successfully for {coverage_label}!")
    print("="*60)


def plot_top3_coefficients_counts(suffix=""):
    """Plot top 3 coefficients frequency counts for each PC separately
    
    Args:
        suffix: Suffix for results directory (e.g., "_2nd_coverage" or "")
    """
    
    # Set up paths
    results_dir = Path(__file__).parent / "results" / f"yearly_reports{suffix}"
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    coverage_label = "2nd Coverage" if suffix == "_2nd_coverage" else "1st Coverage"
    
    # Load data for each PC
    pc_files = {
        'PC1': results_dir / 'pc1_top3_coefficients_per_year.csv',
        'PC2': results_dir / 'pc2_top3_coefficients_per_year.csv',
        'PC3': results_dir / 'pc3_top3_coefficients_per_year.csv'
    }
    
    pc_data = {}
    for pc_name, file_path in pc_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            pc_data[pc_name] = df
            print(f"Loaded {pc_name} coefficients data: {len(df)} years")
        else:
            print(f"Warning: {file_path} not found")
    
    if not pc_data:
        print("No coefficients data files found!")
        return
    
    # Count frequencies for each PC
    pc_counts = {}
    for pc_name, df in pc_data.items():
        factor_counts = {}
        
        # Count occurrences in Rank_1, Rank_2, Rank_3
        for rank in [1, 2, 3]:
            factor_col = f'Rank_{rank}_Factor'
            if factor_col in df.columns:
                factors = df[factor_col].dropna()
                for factor in factors:
                    factor_counts[factor] = factor_counts.get(factor, 0) + 1
        
        # Sort by frequency
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        pc_counts[pc_name] = {
            'counts': factor_counts,
            'sorted': sorted_factors
        }
        
        print(f"\n{pc_name} Coefficients - Top factors:")
        for factor, count in sorted_factors[:10]:  # Show top 10
            print(f"  {factor}: {count} year(s)")
    
    # Create separate plots for each PC
    colors = {'PC1': '#1f77b4', 'PC2': '#ff7f0e', 'PC3': '#2ca02c'}
    
    for pc_name, counts_data in pc_counts.items():
        sorted_factors = counts_data['sorted']
        factor_counts = counts_data['counts']
        
        # Get top factors (at least top 10, or all if fewer)
        n_factors = min(15, len(sorted_factors))
        factors_plot = [f for f, _ in sorted_factors[:n_factors]]
        counts_plot = [c for _, c in sorted_factors[:n_factors]]
        
        # Identify top 3
        top3_factors = [f for f, _ in sorted_factors[:3]]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color bars: highlight top 3
        bar_colors = [colors[pc_name] if f in top3_factors else '#cccccc' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel("Number of Years in Top 3", fontsize=12, fontweight='bold')
        ax.set_ylabel("Factor", fontsize=12, fontweight='bold')
        ax.set_title(f"{pc_name} - Top 3 Coefficients Frequency ({coverage_label})", 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[pc_name], alpha=0.8, label='Top 3 Most Frequent'),
            Patch(facecolor='#cccccc', alpha=0.8, label='Other Factors')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        if suffix:
            output_path = output_dir / f"{pc_name.lower()}_top3_coefficients_frequency{suffix}.png"
        else:
            output_path = output_dir / f"{pc_name.lower()}_top3_coefficients_frequency.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved {pc_name} coefficients plot to {output_path}")
        plt.close()
    
    # Also create a combined figure with subplots
    fig, axes = plt.subplots(1, len(pc_data), figsize=(7*len(pc_data), 8))
    if len(pc_data) == 1:
        axes = [axes]
    
    for idx, (pc_name, counts_data) in enumerate(pc_counts.items()):
        ax = axes[idx]
        sorted_factors = counts_data['sorted']
        factor_counts = counts_data['counts']
        
        # Get top factors
        n_factors = min(10, len(sorted_factors))
        factors_plot = [f for f, _ in sorted_factors[:n_factors]]
        counts_plot = [c for _, c in sorted_factors[:n_factors]]
        
        # Identify top 3
        top3_factors = [f for f, _ in sorted_factors[:3]]
        
        # Color bars
        bar_colors = [colors[pc_name] if f in top3_factors else '#cccccc' for f in factors_plot]
        
        bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts_plot)):
            ax.text(count + 0.1, i, str(count), va='center', fontsize=9)
        
        ax.set_xlabel("Years in Top 3", fontsize=11, fontweight='bold')
        ax.set_ylabel("Factor", fontsize=11, fontweight='bold')
        ax.set_title(f"{pc_name}", fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle(f"Top 3 Coefficients Frequency by Principal Component ({coverage_label})", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save combined figure
    if suffix:
        combined_path = output_dir / f"all_pcs_top3_coefficients_frequency{suffix}.png"
    else:
        combined_path = output_dir / "all_pcs_top3_coefficients_frequency.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined coefficients plot to {combined_path}")
    plt.close()
    
    print("\n" + "="*60)
    print(f"All coefficients plots generated successfully for {coverage_label}!")
    print("="*60)


def plot_comparison_loadings():
    """Create comparison plots showing both 1st and 2nd coverage loadings side by side"""
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data for both analyses
    results_dirs = {
        "1st Coverage": Path(__file__).parent / "results" / "yearly_reports",
        "2nd Coverage": Path(__file__).parent / "results" / "yearly_reports_2nd_coverage"
    }
    
    all_pc_data = {}
    for coverage_type, results_dir in results_dirs.items():
        if not results_dir.exists():
            print(f"Warning: {results_dir} not found, skipping {coverage_type}")
            continue
            
        pc_files = {
            'PC1': results_dir / 'pc1_top3_loadings_per_year.csv',
            'PC2': results_dir / 'pc2_top3_loadings_per_year.csv',
            'PC3': results_dir / 'pc3_top3_loadings_per_year.csv'
        }
        
        pc_data = {}
        for pc_name, file_path in pc_files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                pc_data[pc_name] = df
            else:
                print(f"Warning: {file_path} not found")
        
        if pc_data:
            all_pc_data[coverage_type] = pc_data
    
    if len(all_pc_data) < 2:
        print("Need both 1st and 2nd coverage data for comparison plots")
        return
    
    # Create comparison plots for each PC
    colors_coverage = {"1st Coverage": '#1f77b4', "2nd Coverage": '#ff7f0e'}
    
    for pc_name in ['PC1', 'PC2', 'PC3']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, (coverage_type, pc_data) in enumerate(all_pc_data.items()):
            if pc_name not in pc_data:
                continue
                
            ax = axes[idx]
            df = pc_data[pc_name]
            
            # Count frequencies
            factor_counts = {}
            for rank in [1, 2, 3]:
                factor_col = f'Rank_{rank}_Factor'
                if factor_col in df.columns:
                    factors = df[factor_col].dropna()
                    for factor in factors:
                        factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
            n_factors = min(15, len(sorted_factors))
            factors_plot = [f for f, _ in sorted_factors[:n_factors]]
            counts_plot = [c for _, c in sorted_factors[:n_factors]]
            top3_factors = [f for f, _ in sorted_factors[:3]]
            
            bar_colors = [colors_coverage[coverage_type] if f in top3_factors else '#cccccc' for f in factors_plot]
            bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
            
            for i, (bar, count) in enumerate(zip(bars, counts_plot)):
                ax.text(count + 0.1, i, str(count), va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel("Number of Years in Top 3", fontsize=12, fontweight='bold')
            ax.set_ylabel("Factor", fontsize=12, fontweight='bold')
            ax.set_title(f"{pc_name} - {coverage_type}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f"{pc_name} - Top 3 Loadings Frequency Comparison", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        comparison_path = output_dir / f"{pc_name.lower()}_loadings_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved {pc_name} comparison plot to {comparison_path}")
        plt.close()


def plot_comparison_coefficients():
    """Create comparison plots showing both 1st and 2nd coverage coefficients side by side"""
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Load data for both analyses
    results_dirs = {
        "1st Coverage": Path(__file__).parent / "results" / "yearly_reports",
        "2nd Coverage": Path(__file__).parent / "results" / "yearly_reports_2nd_coverage"
    }
    
    all_pc_data = {}
    for coverage_type, results_dir in results_dirs.items():
        if not results_dir.exists():
            print(f"Warning: {results_dir} not found, skipping {coverage_type}")
            continue
            
        pc_files = {
            'PC1': results_dir / 'pc1_top3_coefficients_per_year.csv',
            'PC2': results_dir / 'pc2_top3_coefficients_per_year.csv',
            'PC3': results_dir / 'pc3_top3_coefficients_per_year.csv'
        }
        
        pc_data = {}
        for pc_name, file_path in pc_files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                pc_data[pc_name] = df
            else:
                print(f"Warning: {file_path} not found")
        
        if pc_data:
            all_pc_data[coverage_type] = pc_data
    
    if len(all_pc_data) < 2:
        print("Need both 1st and 2nd coverage data for comparison plots")
        return
    
    # Create comparison plots for each PC
    colors_coverage = {"1st Coverage": '#1f77b4', "2nd Coverage": '#ff7f0e'}
    
    for pc_name in ['PC1', 'PC2', 'PC3']:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, (coverage_type, pc_data) in enumerate(all_pc_data.items()):
            if pc_name not in pc_data:
                continue
                
            ax = axes[idx]
            df = pc_data[pc_name]
            
            # Count frequencies
            factor_counts = {}
            for rank in [1, 2, 3]:
                factor_col = f'Rank_{rank}_Factor'
                if factor_col in df.columns:
                    factors = df[factor_col].dropna()
                    for factor in factors:
                        factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
            sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
            n_factors = min(15, len(sorted_factors))
            factors_plot = [f for f, _ in sorted_factors[:n_factors]]
            counts_plot = [c for _, c in sorted_factors[:n_factors]]
            top3_factors = [f for f, _ in sorted_factors[:3]]
            
            bar_colors = [colors_coverage[coverage_type] if f in top3_factors else '#cccccc' for f in factors_plot]
            bars = ax.barh(factors_plot, counts_plot, color=bar_colors, alpha=0.8)
            
            for i, (bar, count) in enumerate(zip(bars, counts_plot)):
                ax.text(count + 0.1, i, str(count), va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel("Number of Years in Top 3", fontsize=12, fontweight='bold')
            ax.set_ylabel("Factor", fontsize=12, fontweight='bold')
            ax.set_title(f"{pc_name} - {coverage_type}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle(f"{pc_name} - Top 3 Coefficients Frequency Comparison", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        comparison_path = output_dir / f"{pc_name.lower()}_coefficients_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved {pc_name} coefficients comparison plot to {comparison_path}")
        plt.close()


if __name__ == '__main__':
    print("="*60)
    print("PLOTTING TOP 3 LOADINGS AND COEFFICIENTS")
    print("="*60)
    
    # Plot 1st coverage
    print("\n1. Plotting 1st Coverage Analysis...")
    plot_top3_loadings_counts(suffix="")
    plot_top3_coefficients_counts(suffix="")
    
    # Plot 2nd coverage
    print("\n2. Plotting 2nd Coverage Analysis...")
    plot_top3_loadings_counts(suffix="_2nd_coverage")
    plot_top3_coefficients_counts(suffix="_2nd_coverage")
    
    # Create comparison plots
    print("\n3. Creating Comparison Plots...")
    plot_comparison_loadings()
    plot_comparison_coefficients()
    
    print("\n" + "="*60)
    print("ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*60)


