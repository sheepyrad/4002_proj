# Analysis Scripts

This folder contains modular analysis scripts for each dataset in the project.

## Structure

Each script corresponds to one dataset and performs comprehensive exploratory data analysis:

- `analyze_gdp.py` - GDP per capita analysis
- `analyze_health_expenditure.py` - Public health expenditure analysis
- `analyze_vaccine_hesitancy.py` - Vaccine hesitancy analysis
- `analyze_urban_pop.py` - Urban population analysis
- `analyze_measles.py` - Measles reporting data analysis
- `analyze_birth_rate.py` - Birth rate analysis
- `analyze_net_migration.py` - Net migration analysis
- `analyze_pop_density.py` - Population density analysis
- `analyze_political_stability.py` - Political stability index analysis
- `analyze_household_size.py` - Household size analysis
- `analyze_vaccine_coverage.py` - Vaccine coverage (1st and 2nd dose) analysis
- `correlation_analysis.py` - Spearman rank correlation analysis between measles incidence and all factors
- `run_all_analyses.py` - Main script to run all analyses

## Utilities

- `utils.py` - Common utility functions used across all scripts

## Usage

### Run individual analysis:
```bash
python analysis/analyze_gdp.py
```

### Run all analyses:
```bash
python analysis/run_all_analyses.py
```

### Run only correlation analysis:
```bash
python analysis/correlation_analysis.py
```

## Output

All figures are saved to the `/results` folder with descriptive filenames:
- `gdp_analysis.png`
- `health_expenditure_analysis.png`
- `vaccine_hesitancy_analysis.png`
- `urban_pop_analysis.png`
- `measles_analysis.png`
- `measles_detailed_analysis.png`
- `birth_rate_analysis.png`
- `net_migration_analysis.png`
- `pop_density_analysis.png`
- `political_stability_analysis.png`
- `household_size_analysis.png`
- `vaccine_coverage_analysis.png`
- `spearman_correlation_barplot.png`
- `spearman_correlation_scatterplots.png`
- `spearman_correlation_heatmap.png`

Correlation results are also saved as CSV:
- `spearman_correlation_results.csv`

