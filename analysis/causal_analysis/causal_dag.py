"""
Causal DAG Definition for Measles Vaccine Coverage Analysis

This module defines the causal graph structure based on epidemiological
domain knowledge, specifying the assumed causal relationships between:
- Vaccine coverage (treatment)
- Measles incidence (outcome)
- Socioeconomic confounders
"""

import networkx as nx


def create_causal_dag():
    """
    Create the causal DAG for measles vaccine analysis.
    
    Causal Structure (based on epidemiological domain knowledge):
    ==============================================================
    
    TREATMENTS (Interventions):
    - MCV1: First dose measles vaccine coverage
    - MCV2: Second dose measles vaccine coverage
    
    OUTCOME:
    - MeaslesIncidence: Measles cases per million population
    
    CONFOUNDERS (Common causes of both treatment and outcome):
    - GDPpc: GDP per capita affects both healthcare access (vaccines) and disease burden
    - LogHealthExpPC: Health expenditure per capita determines vaccine program funding and healthcare
    - PolStability: Political stability affects vaccine program implementation and disease control
    - IncomeGroup: Income classification influences both vaccine access and disease outcomes
    
    EFFECT MODIFIERS (Affect outcome through treatment or directly):
    - PopDensity: Population density affects disease transmission but not vaccine coverage directly
    - BirthRate: Birth rate affects population susceptibility to measles
    - NetMigration: Migration patterns affect herd immunity
    """
    
    # Create directed acyclic graph
    dag = nx.DiGraph()
    
    # Add nodes with types
    nodes = {
        # Treatment variables
        'MCV1': {'type': 'treatment', 'description': 'First dose coverage (%)'},
        'MCV2': {'type': 'treatment', 'description': 'Second dose coverage (%)'},
        
        # Outcome variable
        'MeaslesIncidence': {'type': 'outcome', 'description': 'Cases per million'},
        
        # Confounders (common causes)
        'GDPpc': {'type': 'confounder', 'description': 'GDP per capita'},
        'LogHealthExpPC': {'type': 'confounder', 'description': 'Health expenditure per capita (PPP $)'},
        'PolStability': {'type': 'confounder', 'description': 'Political stability index'},
        'HIC': {'type': 'confounder', 'description': 'High-income country indicator'},
        
        # Effect modifiers / risk factors
        'PopDensity': {'type': 'effect_modifier', 'description': 'Population density'},
        'BirthRate': {'type': 'effect_modifier', 'description': 'Birth rate'},
        'NetMigration': {'type': 'effect_modifier', 'description': 'Net migration'},
    }
    
    for node, attrs in nodes.items():
        dag.add_node(node, **attrs)
    
    # Define causal edges based on domain knowledge
    edges = [
        # GDP affects health expenditure, vaccine coverage, and directly affects disease
        ('GDPpc', 'LogHealthExpPC'),  # Richer countries spend more on health
        ('GDPpc', 'MCV1'),           # Economic resources enable vaccination programs
        ('GDPpc', 'MCV2'),
        ('GDPpc', 'MeaslesIncidence'),  # Direct effect through healthcare quality
        
        # Health expenditure affects vaccine programs and disease outcomes
        ('LogHealthExpPC', 'MCV1'),    # More health spending = better vaccine programs
        ('LogHealthExpPC', 'MCV2'),
        ('LogHealthExpPC', 'MeaslesIncidence'),  # Healthcare quality affects outcomes
        
        # Political stability affects vaccine programs and disease control
        ('PolStability', 'MCV1'),   # Stable governments maintain vaccine programs
        ('PolStability', 'MCV2'),
        ('PolStability', 'MeaslesIncidence'),  # Instability disrupts disease control
        
        # Income group classification (proxy for development)
        ('HIC', 'MCV1'),
        ('HIC', 'MCV2'),
        ('HIC', 'MeaslesIncidence'),
        ('HIC', 'GDPpc'),  # Income group determines GDP range
        
        # First dose coverage affects second dose (sequential vaccination)
        ('MCV1', 'MCV2'),
        
        # Treatment effects on outcome
        ('MCV1', 'MeaslesIncidence'),  # Primary causal effect of interest
        ('MCV2', 'MeaslesIncidence'),  # Secondary causal effect
        
        # Effect modifiers - affect outcome directly
        ('PopDensity', 'MeaslesIncidence'),   # Dense populations transmit disease faster
        ('BirthRate', 'MeaslesIncidence'),     # More births = more susceptibles
        ('NetMigration', 'MeaslesIncidence'),  # Migration affects herd immunity
    ]
    
    dag.add_edges_from(edges)
    
    return dag


def dag_to_gml(dag):
    """Convert NetworkX DAG to GML string format for DoWhy"""
    return "\n".join(nx.generate_gml(dag))


def create_dag_for_hesitancy(outcome='LogIncidence', use_log_vars=True):
    """
    Create DAG for analyzing Vaccine Hesitancy as treatment.
    
    This DAG captures:
    - Direct effect: Hesitancy → Incidence (care-seeking behavior, etc.)
    - Indirect effect: Hesitancy → Coverage → Incidence (mediation)
    
    Note: MCV1/MCV2 are MEDIATORS, not confounders when hesitancy is treatment.
    """
    gdp_var = 'LogGDPpc' if use_log_vars else 'GDPpc'
    pop_var = 'LogPopDensity' if use_log_vars else 'PopDensity'
    
    edges = [
        # Socioeconomic confounders (affect hesitancy and incidence)
        (gdp_var, 'VaccineHesitancy'),
        (gdp_var, outcome),
        ('LogHealthExpPC', 'VaccineHesitancy'),
        ('LogHealthExpPC', outcome),
        ('PolStability', 'VaccineHesitancy'),
        ('PolStability', outcome),
        ('HIC', 'VaccineHesitancy'),
        ('HIC', outcome),
        
        # Socioeconomic relationships
        (gdp_var, 'LogHealthExpPC'),
        
        # Urban/household confounders
        ('UrbanPop', 'VaccineHesitancy'),
        ('UrbanPop', outcome),
        ('HouseholdSize', 'VaccineHesitancy'),
        ('HouseholdSize', outcome),
        ('NetMigration', 'VaccineHesitancy'),
        ('NetMigration', outcome),
        
        # Treatment direct effect
        ('VaccineHesitancy', outcome),
        
        # Mediation pathway: Hesitancy → Coverage → Incidence
        ('VaccineHesitancy', 'MCV1'),
        ('VaccineHesitancy', 'MCV2'),
        ('MCV1', 'MCV2'),
        ('MCV1', outcome),
        ('MCV2', outcome),
        
        # Confounders also affect coverage
        (gdp_var, 'MCV1'),
        (gdp_var, 'MCV2'),
        ('LogHealthExpPC', 'MCV1'),
        ('LogHealthExpPC', 'MCV2'),
        ('PolStability', 'MCV1'),
        ('PolStability', 'MCV2'),
        
        # Other factors
        (pop_var, outcome),
        ('BirthRate', outcome),
    ]
    
    dag = nx.DiGraph(edges)
    return dag


def create_simple_dag_for_mcv1(outcome='LogIncidence', use_log_vars=True):
    """
    Create DAG for estimating MCV1 -> Measles Incidence effect.
    
    Includes full set of confounders based on epidemiological theory.
    
    Parameters:
    -----------
    outcome : str
        Name of the outcome variable (default: 'LogIncidence')
    use_log_vars : bool
        If True, use log-transformed variable names (LogGDPpc, LogPopDensity)
    """
    gdp_var = 'LogGDPpc' if use_log_vars else 'GDPpc'
    pop_var = 'LogPopDensity' if use_log_vars else 'PopDensity'
    
    edges = [
        # === CONFOUNDERS (affect both treatment and outcome) ===
        
        # Economic factors
        (gdp_var, 'MCV1'),              # Economic resources enable vaccination
        (gdp_var, outcome),              # Healthcare quality affects outcomes
        (gdp_var, 'LogHealthExpPC'),       # Richer countries spend more on health
        
        # Health system
        ('LogHealthExpPC', 'MCV1'),        # Health spending funds vaccine programs
        ('LogHealthExpPC', outcome),        # Healthcare capacity affects outcomes
        
        # Governance
        ('PolStability', 'MCV1'),        # Stable governments maintain programs
        ('PolStability', outcome),        # Instability disrupts disease control
        
        # Income classification
        ('HIC', 'MCV1'),
        ('HIC', outcome),
        
        # Urbanization
        ('UrbanPop', 'MCV1'),            # Urban areas have better vaccine access
        ('UrbanPop', outcome),            # Urban density affects transmission
        
        # Household structure
        ('HouseholdSize', 'MCV1'),        # Family size affects vaccine decisions
        ('HouseholdSize', outcome),        # Larger households = more transmission
        
        # Vaccine attitudes
        ('VaccineHesitancy', 'MCV1'),     # Hesitancy reduces coverage
        ('VaccineHesitancy', outcome),     # May affect care-seeking behavior
        
        # Population movement
        ('NetMigration', 'MCV1'),         # Migration affects coverage estimates
        ('NetMigration', outcome),         # Migration affects disease importation
        
        # === TREATMENT EFFECT ===
        ('MCV1', outcome),
        
        # === EFFECT MODIFIERS (affect outcome directly) ===
        (pop_var, outcome),               # Dense populations transmit faster
        ('BirthRate', outcome),            # More births = more susceptibles
        
        # === STRUCTURAL RELATIONSHIPS ===
        (gdp_var, 'UrbanPop'),            # Development drives urbanization
        (gdp_var, 'HouseholdSize'),       # Development affects family size
    ]
    
    dag = nx.DiGraph(edges)
    return dag


def create_simple_dag_for_mcv2(outcome='LogIncidence', use_log_vars=True):
    """
    Create DAG for estimating MCV2 -> Measles Incidence effect,
    accounting for MCV1 as a prior treatment in the causal chain.
    
    Parameters:
    -----------
    outcome : str
        Name of the outcome variable (default: 'LogIncidence')
    use_log_vars : bool
        If True, use log-transformed variable names (LogGDPpc, LogPopDensity)
    """
    gdp_var = 'LogGDPpc' if use_log_vars else 'GDPpc'
    pop_var = 'LogPopDensity' if use_log_vars else 'PopDensity'
    
    edges = [
        # === CONFOUNDERS (affect both MCV1, MCV2, and outcome) ===
        
        # Economic factors
        (gdp_var, 'MCV1'),
        (gdp_var, 'MCV2'),
        (gdp_var, outcome),
        (gdp_var, 'LogHealthExpPC'),
        
        # Health system
        ('LogHealthExpPC', 'MCV1'),
        ('LogHealthExpPC', 'MCV2'),
        ('LogHealthExpPC', outcome),
        
        # Governance
        ('PolStability', 'MCV1'),
        ('PolStability', 'MCV2'),
        ('PolStability', outcome),
        
        # Income classification
        ('HIC', 'MCV1'),
        ('HIC', 'MCV2'),
        ('HIC', outcome),
        
        # Urbanization
        ('UrbanPop', 'MCV1'),
        ('UrbanPop', 'MCV2'),
        ('UrbanPop', outcome),
        
        # Household structure
        ('HouseholdSize', 'MCV1'),
        ('HouseholdSize', 'MCV2'),
        ('HouseholdSize', outcome),
        
        # Vaccine attitudes
        ('VaccineHesitancy', 'MCV1'),
        ('VaccineHesitancy', 'MCV2'),
        ('VaccineHesitancy', outcome),
        
        # Population movement
        ('NetMigration', 'MCV1'),
        ('NetMigration', 'MCV2'),
        ('NetMigration', outcome),
        
        # === SEQUENTIAL VACCINATION ===
        ('MCV1', 'MCV2'),              # First dose enables second dose
        
        # === TREATMENT EFFECTS ===
        ('MCV1', outcome),
        ('MCV2', outcome),
        
        # === EFFECT MODIFIERS ===
        (pop_var, outcome),
        ('BirthRate', outcome),
        
        # === STRUCTURAL RELATIONSHIPS ===
        (gdp_var, 'UrbanPop'),
        (gdp_var, 'HouseholdSize'),
    ]
    
    dag = nx.DiGraph(edges)
    return dag


def visualize_dag(dag, output_path=None):
    """Visualize the causal DAG"""
    import matplotlib.pyplot as plt
    
    # Define node colors by type
    node_colors = []
    for node in dag.nodes():
        node_type = dag.nodes[node].get('type', 'unknown')
        if node_type == 'treatment':
            node_colors.append('#4CAF50')  # Green for treatment
        elif node_type == 'outcome':
            node_colors.append('#F44336')  # Red for outcome
        elif node_type == 'confounder':
            node_colors.append('#2196F3')  # Blue for confounders
        else:
            node_colors.append('#9E9E9E')  # Gray for others
    
    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(dag, prog='dot')
    except:
        pos = nx.spring_layout(dag, k=2, iterations=50)
    
    plt.figure(figsize=(14, 10))
    
    nx.draw(dag, pos, 
            with_labels=True, 
            node_color=node_colors,
            node_size=2500,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='#666666',
            alpha=0.9)
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#4CAF50', s=200, label='Treatment'),
        plt.scatter([], [], c='#F44336', s=200, label='Outcome'),
        plt.scatter([], [], c='#2196F3', s=200, label='Confounder'),
        plt.scatter([], [], c='#9E9E9E', s=200, label='Effect Modifier'),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    plt.title('Causal DAG: Vaccine Coverage and Measles Incidence', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"DAG visualization saved to: {output_path}")
    
    plt.close()
    
    return pos


if __name__ == '__main__':
    from pathlib import Path
    
    # Create and visualize the full DAG
    dag = create_causal_dag()
    print("Full Causal DAG:")
    print(f"  Nodes: {list(dag.nodes())}")
    print(f"  Edges: {list(dag.edges())}")
    
    # Visualize
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'causal_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    visualize_dag(dag, output_dir / 'causal_dag_full.png')
    
    # Simple DAG for MCV1
    dag_mcv1 = create_simple_dag_for_mcv1()
    print("\nSimplified DAG for MCV1 analysis:")
    print(f"  Nodes: {list(dag_mcv1.nodes())}")
    print(f"  Edges: {list(dag_mcv1.edges())}")

