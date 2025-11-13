"""
Main script to run all individual analyses and correlation analysis
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging to file
project_root = Path(__file__).parent.parent
log_file = project_root / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

class Tee:
    """Class to write to both file and stdout"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Open log file and set up Tee to write to both console and file
log_file_handle = open(log_file, 'w', encoding='utf-8')
original_stdout = sys.stdout
original_stderr = sys.stderr

# Redirect stdout and stderr to both console and file
sys.stdout = Tee(original_stdout, log_file_handle)
sys.stderr = Tee(original_stderr, log_file_handle)

try:
    print("="*80)
    print("RUNNING ALL ANALYSES")
    print(f"Log file: {log_file}")
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

    print("\n11b. Creating yearly GeoPandas maps for measles incidence...")
    from analyze_measles import create_yearly_geopandas_maps
    create_yearly_geopandas_maps()

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
    print("Yearly measles maps: results/measles_yearly_maps/measles_incidence_*.png")
    print("Spearman correlation: results/spearman/")
    print("  - Multi-year results: results/spearman/spearman_correlation_multiyear_*.png")
    print("  - Year-specific results: results/spearman/[year]/")

except Exception as e:
    print(f"\nERROR: Analysis failed with error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    raise
finally:
    # Restore stdout and stderr, close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file_handle.close()
    print(f"\nAnalysis log saved to: {log_file}")

