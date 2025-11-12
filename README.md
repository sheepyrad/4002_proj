# Exploratory Data Analysis

This directory contains datasets and a Jupyter notebook for conducting exploratory data analysis.

## Datasets

1. **gdp-per-capita-worldbank.csv** - GDP per capita data from World Bank
2. **public-health-expenditure-share-gdp.csv** - Public health expenditure as share of GDP
3. **share-disagrees-vaccines-are-effective.csv** - Share of population that disagrees vaccines are effective
4. **share-of-population-urban.csv** - Share of population that is urban
5. **403-table-web-reporting-data.xlsx** - Additional Excel data
6. **undesa_pd_2022_hh-size-composition.xlsx** - UNDESA household size composition data

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate 4002
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `EDA.ipynb` and run all cells

**Note:** For more detailed information about the analysis, please refer to the `EDA.ipynb` notebook.

## Notebook Structure

The EDA notebook includes:

1. **Data Loading** - Loads all CSV and Excel files
2. **Dataset Overview** - Basic information about each dataset
3. **Data Quality Assessment** - Missing values and duplicates
4. **Temporal Analysis** - Year ranges and coverage
5. **Geographic Coverage** - Countries and regions
6. **Visualizations** - Individual analysis for each dataset:
   - GDP per capita distributions and trends
   - Health expenditure analysis
   - Vaccine disagreement patterns
   - Urban population trends
7. **Relationship Analysis** - Correlations and scatter plots between variables
8. **Summary Statistics** - Comprehensive statistical summaries
9. **Key Findings** - Insights and conclusions

## Analysis Features

- Distribution analysis
- Time series trends
- Geographic comparisons
- Correlation analysis
- Relationship visualization
- Statistical summaries



