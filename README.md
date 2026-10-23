# Predicting Life Expectancy with Machine Learning

**Python | Scikit-learn | Random Forest | Neural Networks | PCA | K-Means**

An end-to-end machine learning project examining how economic conditions, healthcare resources, sanitation, and nutrition relate to life expectancy across countries and territories.

**Key result:** A random forest achieved a **3.12-year mean absolute error (MAE)** on countries held out from training—a **56.8% reduction in MAE** compared with predicting the training-set mean.

## Project overview

**Research question:** Can economic, healthcare, sanitation, and dietary indicators predict historical life expectancy in countries excluded from model training?

This project investigates relationships between global socioeconomic and health indicators and life expectancy through data cleaning, exploratory analysis, dimensionality reduction, clustering, and supervised regression. Unlike a random row-based split, the evaluation holds entire countries out of training to assess generalization to previously unseen geographic units.

## Dataset

- **Source:** Miguel Roca (2025), [*Global Health, Nutrition, Mortality, Economic Data — Version 1*](https://www.kaggle.com/datasets/miguelroca/global-health-nutrition-mortality-economic-data) (Kaggle).
- **Original file:** 22,050 rows and 150 columns, covering 1990–2019.
- **Analysis sample:** **3,016 country-year observations from 195 countries and territories**, covering **2000–2017** after filtering.
- **Target:** Life expectancy at birth, in years.
- **Inputs:** Twelve indicators relating to GDP, health expenditure, drinking water, sanitation, clean fuels, medical staffing, and diet, plus calendar year.

The dataset publisher lists **CC BY-NC-SA 4.0** as its license. Check the license conditions before redistributing the source CSV.

## Approach

### 1. Data preparation and exploration

- Selected combined-sex country observations, excluding geographic aggregates.
- Required at least six nonmissing indicators per observation.
- Used a **country-grouped 60/20/20 approximate train/validation/test split**, keeping each country's years in a single partition.
- Used training data only to determine which predictors to retain and to fit median imputation and scaling; log-transformed GDP per capita.
- Explored missing-data patterns and associations between life expectancy, GDP, and sanitation.

| Partition | Observations | Countries/territories |
|---|---:|---:|
| Training | 1,845 | 117 |
| Validation | 568 | 39 |
| Test | 603 | 39 |

### 2. Unsupervised learning

On a consistent **2010 snapshot of 107 training countries/territories**, I used PCA for visualization and k-means for grouping profiles using standardized indicators. Life expectancy was **not used** to form the clusters.

- The first two principal components captured **72.9%** of the standardized predictor variance.
- Silhouette-based comparison of 2–6 clusters selected **two groups** (silhouette score **0.342**).
- The groups differed in median life expectancy (**62.5 vs. 75.9 years**) and measures of GDP and sanitation access. These differences are descriptive, not causal.

### 3. Supervised learning

I compared the following approaches:

- **Training-mean baseline:** Predict the training-set average life expectancy for every observation.
- **Random forest regression:** 200 trees, comparing maximum depths of 8 and unrestricted depth.
- **Feedforward neural networks:** One hidden layer of 32 units versus two hidden layers of 64 and 32 units, using standardized inputs, Adam optimization, and validation-based early stopping.

I used validation MAE to compare configurations and reserved the country-held-out test partition for final evaluation.

## Results

### Performance on held-out countries

| Model | MAE (years) ↓ | RMSE (years) ↓ | R² ↑ |
|---|---:|---:|---:|
| Training-mean baseline | 7.23 | 9.04 | -0.001 |
| **Random forest** | **3.12** | **4.69** | **0.731** |
| Neural network | 3.72 | 6.08 | 0.548 |

**Interpretation:** The two-layer neural network had the lowest **validation** MAE (**3.40 years**), but the selected random forest configuration performed better on the **test** countries (**3.12 vs. 3.72 years MAE**). The validation-selected forest itself recorded **3.58 years** validation MAE. These results illustrate that relative model performance can change on unseen countries; the test results were not used to retune the models.

Random forest permutation importance on the validation data highlighted **sanitation access, GDP per capita, and drinking water access** as influential predictors. Importance indicates predictive reliance, **not causation**.

### Investigating prediction errors

The neural network made unusually large predictions for some **Belize** observations. An inspection found reported nurse/midwife values of **1,853 per 10,000 residents**, far above the training-data maximum of **201.6**. The cause of those records was not verified. I retained those observations in the reported test metrics rather than removing difficult cases after evaluation.

This case highlights the difficulty of generalizing beyond the input ranges seen during training, particularly for neural networks.

## Technologies

- **Language / environment:** Python, Jupyter Notebook
- **Libraries:** NumPy, pandas, Matplotlib, scikit-learn, threadpoolctl
- **Methods:** Missing-data treatment, feature engineering, PCA, k-means clustering, random forest regression, multilayer perceptrons, permutation importance
- **Evaluation:** Country-grouped splitting, validation-based model selection, MAE, RMSE, R², residual analysis

## Run the analysis

1. Download this repository and obtain **`UnifiedDataset.csv`** from the [original dataset page](https://www.kaggle.com/datasets/miguelroca/global-health-nutrition-mortality-economic-data), unless it is already included in the repository.
2. Put `UnifiedDataset.csv` **in the same working directory from which you run Jupyter**. The notebook currently loads it with `Path("UnifiedDataset.csv")`; it does **not** mount Google Drive.
3. Install the Python dependencies listed in `requirements.txt` (including Jupyter Notebook):

   ```bash
   pip install -r requirements.txt
   ```

4. Open **`DAT402_Project2_Life_Expectancy.ipynb`** in Jupyter and run all cells in order.

The notebook saves intermediate analysis tables, validation and test results, test predictions, and cluster summaries as CSV files. It also generates **nine PNG visualizations** in a `figures/` directory.

**Main files:**

- `DAT402_Project2_Life_Expectancy.ipynb` — full analysis, methods, plots, and conclusions.
- `requirements.txt` — Python packages needed to run the notebook.
- `UnifiedDataset.csv` — input dataset; obtain separately if not redistributed.
- `Project2_Test_Results.csv` — generated model test metrics.
- `Project2_Test_Predictions.csv` — generated country-year predictions.
- `Project2_Validation_Results.csv` — generated configuration comparisons.
- `Project2_Cluster_Summary.csv` — generated unsupervised cluster summaries.
- `figures/` — generated plots, including model comparison and error analysis.

The CSV outputs and `figures/` directory are **created when the notebook runs**; they are not required to be pre-existing files.

## Limitations

- The study makes **retrospective country-level estimates**, not forecasts of future life expectancy or predictions of individual lifespan.
- Observational associations and feature importance do not establish causality.
- Missingness, uneven country coverage, and potentially inconsistent upstream measurements affect data quality.
- Reported accuracy comes from **one country-held-out split**; different splits may produce different results.
- Large out-of-range feature values can create unreliable predictions, as illustrated by the Belize case.

## Attribution

Dataset: Roca, M. (2025). *Global Health, Nutrition, Mortality, Economic Data (Version 1).* [Kaggle](https://www.kaggle.com/datasets/miguelroca/global-health-nutrition-mortality-economic-data).

The project notebook also acknowledges ChatGPT assistance with Python syntax and debugging.
