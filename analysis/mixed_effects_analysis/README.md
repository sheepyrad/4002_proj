# Mixed-Effects Negative Binomial Analysis

This folder contains scripts for conducting mixed-effects negative binomial analysis of measles incidence data.

## Overview

The analysis uses mixed-effects models to account for country-level heterogeneity while examining the relationship between measles incidence and various predictors including vaccine coverage, socioeconomic factors, and health system indicators.

## Model Specification

### Linear Mixed Model (Log-Transformed Incidence)

```
log(MeaslesIncidence_it) = β₀ + β₁MCV1_it + β₂MCV2_it + β₃GDPpc_it + 
                           β₄HealthExpGDP_it + β₅BirthRate_it + 
                           β₆PopDensity_it + β₇Migration_it + 
                           β₈PolStability_it + β₉HIC_i + 
                           u₀ᵢ + u₁ᵢMCV1_it + ε_it
```

Where:
- `i` indexes countries, `t` indexes years
- `u₀ᵢ` = country-level random intercept
- `u₁ᵢ` = country-level random slope for MCV1 (optional)
- `HIC_i` = high-income country indicator (time-invariant)

### Negative Binomial Model (Count Data)

For count data (measles cases), uses negative binomial regression with:
- Population as offset
- Country fixed effects to approximate random effects
- Log link function

## Scripts

### 1. `prepare_data.py`
Prepares panel dataset by merging all available data sources:
- Measles incidence and cases
- Vaccine coverage (MCV1, MCV2)
- GDP per capita
- Health expenditure (% of GDP)
- Birth rate
- Population density
- Net migration
- Political stability index
- Income group classifications

**Usage:**
```python
from prepare_data import prepare_panel_data
panel_df = prepare_panel_data()
```

### 2. `fit_models.py`
Fits mixed-effects models:
- Linear mixed model with log-transformed incidence
- Negative binomial model for count data

**Note on Negative Binomial Model:**
- Uses log-transformed GDPpc and PopDensity to prevent numerical overflow
- Scales population offset by 1M to improve numerical stability
- Automatically falls back to simpler specification if country fixed effects cause issues
- Handles data type conversions to ensure numeric arrays

**Usage:**
```python
from fit_models import fit_linear_mixed_model, fit_negative_binomial_model
linear_model = fit_linear_mixed_model(panel_df)
nb_model = fit_negative_binomial_model(panel_df)
```

### 3. `policy_simulations.py`
Simulates intervention scenarios:
- Increase MCV1 to 95% in LMICs
- Increase MCV2 to 95% in LMICs
- Increase health expenditure by 1% of GDP
- Improve political stability by 1 SD
- Custom interventions

Also calculates marginal effects for key variables.

**Usage:**
```python
from policy_simulations import run_all_interventions, calculate_marginal_effects
interventions = run_all_interventions(model, panel_df)
marginal_effects = calculate_marginal_effects(model, panel_df, 'MCV1')
```

### 4. `visualize_results.py`
Creates visualizations:
- Marginal effects plots (by HIC/LMIC status)
- Uncertainty intervals (confidence intervals)
- Country clusters based on random effects
- Intervention comparison charts

**Usage:**
```python
from visualize_results import create_comprehensive_visualizations
create_comprehensive_visualizations(model, panel_df, intervention_results)
```

### 5. `run_mixed_effects_analysis.py`
Main script that runs the complete analysis pipeline:
1. Data preparation
2. Model fitting
3. Policy simulations
4. Visualizations
5. Policy recommendations report

**Usage:**
```bash
python run_mixed_effects_analysis.py
```

Or:
```python
from run_mixed_effects_analysis import run_complete_analysis
run_complete_analysis()
```

## Output Files

Results are saved to `results/mixed_effects_analysis/`:

- `panel_data.csv`: Prepared panel dataset
- `linear_mixed_model.pkl`: Fitted linear mixed model (pickle)
- `negative_binomial_model.pkl`: Fitted negative binomial model (pickle)
- `linear_mixed_model_summary.txt`: Model summary
- `negative_binomial_model_summary.txt`: Model summary
- `interventions/`: Intervention scenario results (CSV files)
- `figures/`: All visualizations (PNG files)
- `policy_recommendations.txt`: Policy recommendations report
- `country_clusters.csv`: Country cluster assignments

## Policy Recommendations

The analysis produces actionable policy recommendations including:

1. **Simulated intervention scenarios**: Predicted reductions in measles incidence for various interventions
2. **Marginal effects**: Impact magnitude of each variable across HICs and LMICs
3. **Uncertainty intervals**: Confidence intervals for model predictions
4. **Country clusters**: Identification of high-risk vs low-risk country profiles

## Dependencies

- pandas
- numpy
- statsmodels
- scikit-learn
- matplotlib
- seaborn

## Notes

- Models handle missing data through imputation (country means, then overall means)
- Random effects capture country-level heterogeneity
- Policy simulations focus on LMICs by default
- Visualizations compare HICs vs LMICs where applicable

