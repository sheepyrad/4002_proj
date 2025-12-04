"""
Generate properly color-coded DAG visualizations for the causal analysis.
"""

import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                          'results', 'causal_analysis')


def create_mcv1_dag_with_types():
    """
    Create DAG for MCV1 analysis with proper node type annotations.
    """
    dag = nx.DiGraph()
    
    # Define nodes with their types
    node_types = {
        # Treatment
        'MCV1': 'treatment',
        
        # Outcome
        'LogIncidence': 'outcome',
        
        # Confounders
        'LogGDPpc': 'confounder',
        'LogHealthExpPC': 'confounder',
        'PolStability': 'confounder',
        'HIC': 'confounder',
        'UrbanPop': 'confounder',
        'HouseholdSize': 'confounder',
        'VaccineHesitancy': 'confounder',
        'NetMigration': 'confounder',
        
        # Effect modifiers
        'LogPopDensity': 'effect_modifier',
        'BirthRate': 'effect_modifier',
        
        # Mediator
        'MCV2': 'mediator',
    }
    
    # Add nodes with types
    for node, ntype in node_types.items():
        dag.add_node(node, type=ntype)
    
    # Define edges
    edges = [
        # Socioeconomic confounders
        ('LogGDPpc', 'LogHealthExpPC'),
        ('LogGDPpc', 'MCV1'),
        ('LogGDPpc', 'LogIncidence'),
        ('LogHealthExpPC', 'MCV1'),
        ('LogHealthExpPC', 'LogIncidence'),
        ('PolStability', 'MCV1'),
        ('PolStability', 'LogIncidence'),
        ('HIC', 'MCV1'),
        ('HIC', 'LogIncidence'),
        
        # Social/demographic confounders
        ('UrbanPop', 'MCV1'),
        ('UrbanPop', 'LogIncidence'),
        ('HouseholdSize', 'MCV1'),
        ('HouseholdSize', 'LogIncidence'),
        ('VaccineHesitancy', 'MCV1'),
        ('VaccineHesitancy', 'LogIncidence'),
        ('NetMigration', 'MCV1'),
        ('NetMigration', 'LogIncidence'),
        
        # Effect modifiers (affect outcome only)
        ('LogPopDensity', 'LogIncidence'),
        ('BirthRate', 'LogIncidence'),
        
        # CAUSAL EFFECT OF INTEREST
        ('MCV1', 'LogIncidence'),
        
        # Mediation through MCV2
        ('MCV1', 'MCV2'),
        ('MCV2', 'LogIncidence'),
    ]
    
    dag.add_edges_from(edges)
    return dag


def create_hesitancy_dag_with_types():
    """
    Create DAG for Vaccine Hesitancy analysis with proper node type annotations.
    """
    dag = nx.DiGraph()
    
    node_types = {
        # Treatment
        'VaccineHesitancy': 'treatment',
        
        # Outcome
        'LogIncidence': 'outcome',
        
        # Mediators (coverage)
        'MCV1': 'mediator',
        'MCV2': 'mediator',
        
        # Confounders
        'LogGDPpc': 'confounder',
        'LogHealthExpPC': 'confounder',
        'PolStability': 'confounder',
        'HIC': 'confounder',
        'UrbanPop': 'confounder',
        'HouseholdSize': 'confounder',
        'NetMigration': 'confounder',
        
        # Effect modifiers
        'LogPopDensity': 'effect_modifier',
        'BirthRate': 'effect_modifier',
    }
    
    for node, ntype in node_types.items():
        dag.add_node(node, type=ntype)
    
    edges = [
        # Confounders → Treatment (Hesitancy)
        ('LogGDPpc', 'VaccineHesitancy'),
        ('UrbanPop', 'VaccineHesitancy'),
        ('HIC', 'VaccineHesitancy'),
        
        # Confounders → Outcome
        ('LogGDPpc', 'LogIncidence'),
        ('LogHealthExpPC', 'LogIncidence'),
        ('PolStability', 'LogIncidence'),
        ('HIC', 'LogIncidence'),
        ('LogGDPpc', 'LogHealthExpPC'),
        
        # Confounders → Mediators
        ('LogGDPpc', 'MCV1'),
        ('LogHealthExpPC', 'MCV1'),
        ('PolStability', 'MCV1'),
        
        # TREATMENT → MEDIATORS (indirect pathway)
        ('VaccineHesitancy', 'MCV1'),
        ('VaccineHesitancy', 'MCV2'),
        
        # TREATMENT → OUTCOME (direct effect)
        ('VaccineHesitancy', 'LogIncidence'),
        
        # MEDIATORS → OUTCOME
        ('MCV1', 'MCV2'),
        ('MCV1', 'LogIncidence'),
        ('MCV2', 'LogIncidence'),
        
        # Effect modifiers
        ('LogPopDensity', 'LogIncidence'),
        ('BirthRate', 'LogIncidence'),
    ]
    
    dag.add_edges_from(edges)
    return dag


def visualize_dag_colored(dag, title, output_path, highlight_causal_path=None):
    """
    Visualize DAG with proper color coding and legend.
    
    Colors:
    - Treatment: Green
    - Outcome: Red
    - Confounder: Blue
    - Mediator: Orange
    - Effect Modifier: Purple
    """
    
    # Color mapping
    color_map = {
        'treatment': '#2ECC71',      # Green
        'outcome': '#E74C3C',        # Red
        'confounder': '#3498DB',     # Blue
        'mediator': '#F39C12',       # Orange
        'effect_modifier': '#9B59B6', # Purple
        'unknown': '#95A5A6',        # Gray
    }
    
    # Assign colors to nodes
    node_colors = []
    for node in dag.nodes():
        node_type = dag.nodes[node].get('type', 'unknown')
        node_colors.append(color_map.get(node_type, color_map['unknown']))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use manual positioning for better layout
    if 'MCV1' in dag.nodes() and 'LogIncidence' in dag.nodes():
        # Create hierarchical layout
        pos = {}
        
        # Get node types
        treatments = [n for n in dag.nodes() if dag.nodes[n].get('type') == 'treatment']
        outcomes = [n for n in dag.nodes() if dag.nodes[n].get('type') == 'outcome']
        confounders = [n for n in dag.nodes() if dag.nodes[n].get('type') == 'confounder']
        mediators = [n for n in dag.nodes() if dag.nodes[n].get('type') == 'mediator']
        effect_mods = [n for n in dag.nodes() if dag.nodes[n].get('type') == 'effect_modifier']
        
        # Position confounders at top
        for i, node in enumerate(confounders):
            pos[node] = (i * 1.5 - len(confounders) * 0.75, 2)
        
        # Position treatments in middle-left
        for i, node in enumerate(treatments):
            pos[node] = (-2, 0)
        
        # Position mediators in middle
        for i, node in enumerate(mediators):
            pos[node] = (0, -0.5 * i)
        
        # Position outcomes in middle-right
        for i, node in enumerate(outcomes):
            pos[node] = (3, 0)
        
        # Position effect modifiers at bottom
        for i, node in enumerate(effect_mods):
            pos[node] = (i * 2, -2)
    else:
        try:
            pos = nx.nx_agraph.graphviz_layout(dag, prog='dot')
        except:
            pos = nx.spring_layout(dag, k=3, iterations=100, seed=42)
    
    # Draw edges
    # Highlight causal path if specified
    edge_colors = []
    edge_widths = []
    for u, v in dag.edges():
        if highlight_causal_path and (u, v) in highlight_causal_path:
            edge_colors.append('#E74C3C')  # Red for causal path
            edge_widths.append(3.0)
        else:
            edge_colors.append('#7F8C8D')  # Gray for other edges
            edge_widths.append(1.5)
    
    # Draw the graph
    nx.draw_networkx_edges(dag, pos, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=25, arrowstyle='-|>',
                           connectionstyle='arc3,rad=0.1', alpha=0.7, ax=ax)
    
    nx.draw_networkx_nodes(dag, pos, node_color=node_colors, 
                           node_size=3500, alpha=0.9, ax=ax)
    
    nx.draw_networkx_labels(dag, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Create legend
    legend_handles = [
        mpatches.Patch(color=color_map['treatment'], label='Treatment'),
        mpatches.Patch(color=color_map['outcome'], label='Outcome'),
        mpatches.Patch(color=color_map['confounder'], label='Confounder'),
        mpatches.Patch(color=color_map['mediator'], label='Mediator'),
        mpatches.Patch(color=color_map['effect_modifier'], label='Effect Modifier'),
    ]
    
    ax.legend(handles=legend_handles, loc='upper left', fontsize=11, 
              framealpha=0.9, title='Node Types', title_fontsize=12)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    """Generate all colored DAG visualizations."""
    print("="*60)
    print("Generating Color-Coded DAG Visualizations")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # MCV1 DAG (Main Analysis)
    print("\n1. MCV1 Analysis DAG (Primary)")
    dag_mcv1 = create_mcv1_dag_with_types()
    visualize_dag_colored(
        dag_mcv1,
        title='Causal DAG: Effect of MCV1 on Measles Incidence\n(Primary Analysis)',
        output_path=os.path.join(OUTPUT_DIR, 'causal_dag_MCV1_colored.png'),
        highlight_causal_path=[('MCV1', 'LogIncidence')]
    )
    
    # Hesitancy DAG
    print("\n2. Vaccine Hesitancy DAG (Mediation Analysis)")
    dag_hes = create_hesitancy_dag_with_types()
    visualize_dag_colored(
        dag_hes,
        title='Causal DAG: Vaccine Hesitancy Pathways\n(Mediation Analysis)',
        output_path=os.path.join(OUTPUT_DIR, 'causal_dag_hesitancy_colored.png'),
        highlight_causal_path=[('VaccineHesitancy', 'MCV1'), ('MCV1', 'LogIncidence'),
                               ('VaccineHesitancy', 'LogIncidence')]
    )
    
    # Create a simplified summary DAG
    print("\n3. Simplified Summary DAG")
    create_simplified_dag()
    
    # Create stratified DAGs
    print("\n4. Stratified DAG (HICs vs LICs)")
    create_stratified_dag()
    
    # Create income group comparison chart
    print("\n5. Income Group Comparison Chart")
    create_all_income_comparison()
    
    print("\n" + "="*60)
    print("All DAG visualizations saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*60)


def create_simplified_dag():
    """Create a simplified DAG showing only the key relationships."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Manual drawing of simplified DAG
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Colors
    treatment_color = '#2ECC71'
    outcome_color = '#E74C3C'
    confounder_color = '#3498DB'
    mediator_color = '#F39C12'
    
    # Draw boxes
    def draw_box(x, y, text, color, width=1.8, height=0.8):
        rect = mpatches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Confounders (top)
    draw_box(5, 6, 'Confounders\n(GDP, Health Exp,\nPol. Stability, etc.)', confounder_color, width=3, height=1.2)
    
    # Treatment (left)
    draw_box(2, 3.5, 'MCV1\n(Treatment)', treatment_color, width=2, height=1)
    
    # Mediator (middle)
    draw_box(5, 2, 'MCV2\n(Mediator)', mediator_color, width=2, height=1)
    
    # Hesitancy (middle-left)
    draw_box(2, 1, 'Vaccine\nHesitancy', '#9B59B6', width=2, height=1)
    
    # Outcome (right)
    draw_box(8, 3.5, 'Measles\nIncidence\n(Outcome)', outcome_color, width=2, height=1.2)
    
    # Draw arrows
    def draw_arrow(x1, y1, x2, y2, color='#7F8C8D', lw=2, style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, 
                                  linestyle=style, shrinkA=10, shrinkB=10))
    
    # Confounder arrows
    draw_arrow(4, 5.4, 2.5, 4.1, confounder_color)  # Conf -> MCV1
    draw_arrow(6, 5.4, 7.5, 4.1, confounder_color)  # Conf -> Outcome
    
    # Causal arrows (highlighted in red)
    draw_arrow(3, 3.5, 7, 3.5, '#E74C3C', lw=3)  # MCV1 -> Outcome (MAIN EFFECT)
    draw_arrow(2.5, 3, 4, 2.3, '#F39C12', lw=2)  # MCV1 -> MCV2
    draw_arrow(6, 2, 7.5, 3, '#F39C12', lw=2)  # MCV2 -> Outcome
    
    # Hesitancy arrows
    draw_arrow(2, 1.5, 2, 3, '#9B59B6', lw=2)  # Hesitancy -> MCV1
    draw_arrow(3, 1, 7, 3, '#9B59B6', lw=2, style='--')  # Hesitancy -> Outcome (direct)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=treatment_color, label='Treatment (MCV1)'),
        mpatches.Patch(color=outcome_color, label='Outcome (Incidence)'),
        mpatches.Patch(color=confounder_color, label='Confounders'),
        mpatches.Patch(color=mediator_color, label='Mediator (MCV2)'),
        mpatches.Patch(color='#9B59B6', label='Vaccine Hesitancy'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add effect size annotation
    ax.text(5, 3.8, 'Causal Effect: -2.3%\nper 1% coverage\n(E-value: 1.12)', 
            ha='center', va='bottom', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_title('Simplified Causal DAG: MCV1 → Measles Incidence', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'causal_dag_simplified.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: causal_dag_simplified.png")


def create_stratified_dag():
    """Create a DAG showing stratified effects for HICs vs LICs."""
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # Colors
    treatment_color = '#2ECC71'
    outcome_color = '#E74C3C'
    confounder_color = '#3498DB'
    mediator_color = '#F39C12'
    hesitancy_color = '#9B59B6'
    
    def draw_dag_panel(ax, title, effect_size, effect_pct, n_countries, color_theme):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')
        
        # Draw boxes
        def draw_box(x, y, text, color, width=1.8, height=0.8):
            rect = mpatches.FancyBboxPatch((x-width/2, y-height/2), width, height,
                                            boxstyle="round,pad=0.05",
                                            facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Confounders (top)
        draw_box(5, 6, 'Confounders\n(GDP, Health Exp,\nPol. Stability, etc.)', confounder_color, width=3, height=1.2)
        
        # Treatment (left)
        draw_box(2, 3.5, 'MCV1\n(Treatment)', treatment_color, width=2, height=1)
        
        # Mediator (middle)
        draw_box(5, 2, 'MCV2\n(Mediator)', mediator_color, width=2, height=1)
        
        # Hesitancy (bottom-left)
        draw_box(2, 0.8, 'Vaccine\nHesitancy', hesitancy_color, width=2, height=0.9)
        
        # Outcome (right)
        draw_box(8, 3.5, 'Measles\nIncidence\n(Outcome)', outcome_color, width=2, height=1.2)
        
        # Draw arrows
        def draw_arrow(x1, y1, x2, y2, color='#7F8C8D', lw=2, style='-'):
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, 
                                      linestyle=style, shrinkA=10, shrinkB=10))
        
        # Confounder arrows
        draw_arrow(4, 5.4, 2.5, 4.1, confounder_color)
        draw_arrow(6, 5.4, 7.5, 4.1, confounder_color)
        
        # Causal arrows - use theme color for main effect
        draw_arrow(3, 3.5, 7, 3.5, color_theme, lw=4)  # MCV1 -> Outcome
        draw_arrow(2.5, 3, 4, 2.3, mediator_color, lw=2)  # MCV1 -> MCV2
        draw_arrow(6, 2, 7.5, 3, mediator_color, lw=2)  # MCV2 -> Outcome
        
        # Hesitancy arrows
        draw_arrow(2, 1.3, 2, 3, hesitancy_color, lw=2)
        draw_arrow(3, 0.8, 7, 3, hesitancy_color, lw=1.5, style='--')
        
        # Add effect size annotation - PROMINENT
        effect_box = ax.text(5, 4.2, f'Causal Effect:\n{effect_pct}%\nper 1% coverage', 
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color_theme, 
                         edgecolor='black', linewidth=2, alpha=0.9),
                color='white')
        
        # Add sample size
        ax.text(5, 0.2, f'n = {n_countries} countries', 
                ha='center', fontsize=11, style='italic', alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Left panel: High Income Countries (2023 data)
    draw_dag_panel(axes[0], 
                   'High Income Countries (HICs)', 
                   -0.0392, '-3.92', 63,
                   '#1E88E5')  # Blue theme
    
    # Right panel: Low Income Countries (2023 data)
    draw_dag_panel(axes[1], 
                   'Low Income Countries (LICs)', 
                   -0.0301, '-3.01', 25,
                   '#D32F2F')  # Red theme
    
    # Add overall legend at bottom
    legend_elements = [
        mpatches.Patch(color=treatment_color, label='Treatment (MCV1)'),
        mpatches.Patch(color=outcome_color, label='Outcome (Incidence)'),
        mpatches.Patch(color=confounder_color, label='Confounders'),
        mpatches.Patch(color=mediator_color, label='Mediator (MCV2)'),
        mpatches.Patch(color=hesitancy_color, label='Vaccine Hesitancy'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    
    # Add comparison annotation
    fig.text(0.5, 0.95, 
             'Stratified Causal Effects: MCV1 → Measles Incidence by Income Group',
             ha='center', fontsize=16, fontweight='bold')
    
    fig.text(0.5, 0.06, 
             'Key Finding: HICs show stronger effect (-4.60%) than LICs (-3.12%), '
             'possibly due to better surveillance or different baseline coverage levels.',
             ha='center', fontsize=11, style='italic', alpha=0.8)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(os.path.join(OUTPUT_DIR, 'causal_dag_stratified_income.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: causal_dag_stratified_income.png")


def create_all_income_comparison():
    """Create a bar chart comparing effects across all income groups."""
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Data from 2023 analysis
    income_groups = ['High\nIncome', 'Upper Middle\nIncome', 'Lower Middle\nIncome', 'Low\nIncome']
    effects = [-3.92, -0.39, -1.54, -3.01]  # Updated for 2023 cutoff
    n_countries = [63, 52, 49, 25]
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#D32F2F']
    
    bars = ax.bar(income_groups, effects, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels and sample sizes
    for bar, effect, n in zip(bars, effects, n_countries):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height - 0.25, 
                f'{effect:.2f}%', ha='center', va='top', 
                fontsize=14, fontweight='bold', color='white')
        ax.text(bar.get_x() + bar.get_width()/2, 0.15, 
                f'n={n}', ha='center', va='bottom', 
                fontsize=11, color='black')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=-2.33, color='red', linewidth=2, linestyle='--', label='Overall effect (-2.33%)')
    
    ax.set_ylabel('% Change in Measles Incidence\nper 1% Increase in MCV1', fontsize=12)
    ax.set_xlabel('Income Group', fontsize=12)
    ax.set_title('Causal Effect of MCV1 on Measles Incidence\nStratified by Income Group (2012-2023)', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(-5.0, 0.8)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add interpretation box
    textstr = ('Key Findings:\n'
               '• HICs: -3.92% (p=0.020)\n'
               '• LICs: -3.01% (p<0.001)\n'
               '• Overall: -2.33% (p<0.001)\n\n'
               'E-value: 1.12')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'causal_effect_by_income_group.png'), 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: causal_effect_by_income_group.png")


if __name__ == "__main__":
    main()

