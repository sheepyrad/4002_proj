# Causal Analysis: Vaccine Coverage and Measles Incidence

This module implements causal inference using the **DoWhy** framework to estimate the causal effect of measles vaccine coverage on measles incidence rates.

## Overview

The analysis uses Directed Acyclic Graphs (DAGs) to encode domain knowledge about causal relationships between variables, then employs various statistical methods to estimate causal effects while accounting for confounding.

## Causal Structure

### Research Question
**What is the causal effect of increasing vaccine coverage (MCV1/MCV2) on measles incidence rates?**

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| MCV1 | Treatment | First dose measles vaccine coverage (%) |
| MCV2 | Treatment | Second dose measles vaccine coverage (%) |
| MeaslesIncidence | Outcome | Measles cases per million population |
| GDPpc | Confounder | GDP per capita (PPP) |
| HealthExpGDP | Confounder | Public health expenditure (% of GDP) |
| PolStability | Confounder | Political stability index |
| HIC | Confounder | High-income country indicator |
| PopDensity | Effect Modifier | Population density |
| BirthRate | Effect Modifier | Birth rate |

### Causal DAG

```
                      ┌─────────────┐
                      │   GDPpc     │
                      └──────┬──────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌────────────────┐    ┌─────────────┐    ┌─────────────────┐
│ HealthExpGDP   │───▶│    MCV1     │───▶│ MeaslesIncidence│
└────────────────┘    └──────┬──────┘    └─────────────────┘
         │                   │                   ▲
         │                   ▼                   │
         │            ┌─────────────┐            │
         └───────────▶│    MCV2     │────────────┘
                      └─────────────┘
                             ▲
                             │
                      ┌─────────────┐
                      │ PolStability│
                      └─────────────┘
```

## Methodology

### DoWhy Four-Step Approach

1. **Model**: Define causal model with DAG encoding domain knowledge
2. **Identify**: Find causal estimand using backdoor criterion
3. **Estimate**: Compute causal effect using multiple methods:
   - Linear regression with adjustment
   - Propensity score stratification
   - Propensity score matching
4. **Refute**: Test robustness with:
   - Random common cause test
   - Placebo treatment test
   - Data subset test
   - Sensitivity analysis (E-value)

### Key Assumptions

1. **Conditional Ignorability**: Given observed confounders, treatment assignment is independent of potential outcomes
2. **Positivity**: All covariate strata have both treated and untreated units
3. **SUTVA**: Stable Unit Treatment Value Assumption - no interference between countries
4. **Correct Graph Specification**: The DAG correctly represents the data generating process

## Files

| File | Description |
|------|-------------|
| `causal_dag.py` | Defines causal graphs using NetworkX |
| `dowhy_analysis.py` | Main analysis script using DoWhy |
| `README.md` | This documentation |

## Usage

### Prerequisites

```bash
pip install dowhy networkx pandas numpy matplotlib
```

### Running the Analysis

```python
# Full analysis pipeline
python dowhy_analysis.py

# Or import and run specific analyses
from dowhy_analysis import run_full_analysis, estimate_causal_effect_mcv1

# Run full analysis
results = run_full_analysis(save_results=True, verbose=True)

# Or analyze specific treatment
from dowhy_analysis import load_panel_data, estimate_causal_effect_mcv1
df = load_panel_data()
mcv1_results = estimate_causal_effect_mcv1(df)
```

### Output

Results are saved to `results/causal_analysis/`:
- `causal_analysis_report.txt` - Detailed text report
- `causal_estimates.csv` - Numerical estimates in CSV format
- `causal_dag.png` - Visualization of the causal DAG

## Interpretation

### Effect Estimates

The effect is estimated on log-transformed incidence:
- **Coefficient**: Change in log(incidence + 1) per 1% increase in coverage
- **Percentage change**: `(exp(coefficient) - 1) × 100%`

Example interpretation:
> A coefficient of -0.05 means a 1 percentage point increase in vaccine coverage decreases measles incidence by approximately 5%.

### Robustness Checks

- **Random Common Cause**: If estimate changes significantly, effect may be confounded
- **Placebo Treatment**: Estimate should be ~0 with randomized treatment (indicates no spurious correlation)
- **Data Subset**: Estimate should be stable across subsamples (indicates reliability)

## Limitations

1. **Ecological Fallacy**: Country-level analysis may not reflect individual-level effects
2. **Temporal Dynamics**: Panel data has temporal dependencies not fully captured
3. **Unmeasured Confounding**: Despite adjustment, unobserved confounders may exist
4. **DAG Assumptions**: Causal structure is based on domain knowledge, may be misspecified

## References

- Sharma, A., & Kiciman, E. (2020). DoWhy: An End-to-End Library for Causal Inference
- Pearl, J. (2009). Causality: Models, Reasoning, and Inference
- Hernán, M. A., & Robins, J. M. (2020). Causal Inference: What If

