# Measles Incidence Analysis Project

This project analyzes measles incidence rates and their relationships with various socioeconomic and health factors using exploratory data analysis and Spearman rank correlation analysis.

## Datasets

All datasets are located in the `data/` folder:

1. **GDP_per_cap.csv** - GDP per capita (PPP, constant 2021 international $)
2. **Public_health_expenditure_share_gdp.csv** - Public health expenditure as share of GDP
3. **Vaccine_hesitancy.csv** - Share of population that disagrees vaccines are effective
4. **Urban_pop.csv** - Urban population as percentage of total population
5. **Measles_reporting_data.csv** - Measles reporting data from WHO (cases, incidence rates)
6. **Birth_rate.csv** - Crude birth rate (per 1,000 people)
7. **Net_migration.csv** - Net migration data
8. **Pop_density.csv** - Population density (people per sq. km)
9. **Political_Stability_idx.csv** - Political Stability and Absence of Violence/Terrorism Index
10. **Household_size.csv** - Average household size (number of members)
11. **1st_dose_WUENIC.csv** - Measles vaccine coverage (1st dose)
12. **2nd_dose_WUENIC.csv** - Measles vaccine coverage (2nd dose)

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate 4002
```

## Running Analyses

### Individual Dataset Analyses

Each dataset has its own analysis script that generates visualizations:

```bash
# Run analysis for a specific dataset
python analysis/analyze_gdp.py
python analysis/analyze_health_expenditure.py
python analysis/analyze_vaccine_hesitancy.py
# ... etc for each dataset
```

### Run All Individual Analyses

```bash
python analysis/run_all_analyses.py
```

This will generate analysis figures for all datasets and save them to the `results/` folder.
```


