"""
Main script to run all individual analyses and correlation analysis
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("RUNNING ALL ANALYSES")
print("="*80)

# Run individual analyses
print("\n1. Running GDP per capita analysis...")
from analyze_gdp import analyze_gdp
analyze_gdp()

print("\n2. Running Health expenditure analysis...")
from analyze_health_expenditure import analyze_health_expenditure
analyze_health_expenditure()

print("\n3. Running Vaccine hesitancy analysis...")
from analyze_vaccine_hesitancy import analyze_vaccine_hesitancy
analyze_vaccine_hesitancy()

print("\n4. Running Urban population analysis...")
from analyze_urban_pop import analyze_urban_pop
analyze_urban_pop()

print("\n5. Running Measles reporting data analysis...")
from analyze_measles import analyze_measles
analyze_measles()

print("\n6. Running Birth rate analysis...")
from analyze_birth_rate import analyze_birth_rate
analyze_birth_rate()

print("\n7. Running Net migration analysis...")
from analyze_net_migration import analyze_net_migration
analyze_net_migration()

print("\n8. Running Population density analysis...")
from analyze_pop_density import analyze_pop_density
analyze_pop_density()

print("\n9. Running Political stability analysis...")
from analyze_political_stability import analyze_political_stability
analyze_political_stability()

print("\n10. Running Household size analysis...")
from analyze_household_size import analyze_household_size
analyze_household_size()

print("\n11. Running Vaccine coverage analysis...")
from analyze_vaccine_coverage import analyze_vaccine_coverage
analyze_vaccine_coverage()

print("\n12. Running Spearman correlation analysis (single year)...")
from correlation_analysis import main as run_correlation
run_correlation()

print("\n13. Running Multi-year Spearman correlation analysis (2011-2024)...")
from correlation_analysis import analyze_multiple_years
analyze_multiple_years()

print("\n" + "="*80)
print("ALL ANALYSES COMPLETED!")
print("="*80)
print("\nAll figures have been saved to the /results folder")
print("Individual analyses: results/[dataset]_analysis.png")
print("Spearman correlation: results/spearman/")
print("  - Multi-year results: results/spearman/spearman_correlation_multiyear_*.png")
print("  - Year-specific results: results/spearman/[year]/")

