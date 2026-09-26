# WHO Tuberculosis Surveillance — Machine Learning Analysis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Data](https://img.shields.io/badge/Data-WHO%20TB%20Surveillance-2E8B57)](https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data)

A machine-learning project based on **WHO country-level tuberculosis surveillance data**, developed for the 5BM753 course.

The project combines **temporal supervised learning** and **unsupervised trajectory analysis** to study tuberculosis burden across countries while respecting the longitudinal structure of the data.

## Project at a glance

**Data source:** WHO Global Tuberculosis Programme  
**Unit of analysis:** country-year observations  
**Time span used in the main pipeline:** 2000–2023  
**Main language:** Python  
**Supervised task:** predict next-year TB incidence from current-year indicators  
**Main supervised models:** Ridge Regression and Random Forest Regression  
**Unsupervised task:** group countries according to multi-year TB trajectories  
**Unsupervised methods:** K-means, Ward hierarchical clustering, spectral clustering and PCA

## Biological and epidemiological questions

### Supervised learning

**Can the tuberculosis incidence rate in year t+1 be predicted from epidemiological indicators observed in year t?**

This is treated as a temporal regression problem rather than a random cross-sectional prediction task.

### Unsupervised learning

**Do countries form reproducible groups with similar tuberculosis trajectories over time?**

The unsupervised component analyzes longitudinal patterns in incidence, mortality, case detection and related TB indicators.

## Data

The raw data come from the **WHO Global Tuberculosis Programme**.

Official source:  
https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data

The repository does not version the large raw WHO files. Instead, it documents how to download and rebuild the analysis datasets.

See:

[02_data/README.md](02_data/README.md)

### Temporal split

The main preprocessing pipeline uses a chronological split:

- **training:** 2000–2018
- **held-out test:** 2019–2023

A random row split is deliberately avoided because consecutive observations from the same country are correlated and could create optimistic leakage across the train/test boundary.

## Preprocessing principles

The preprocessing pipeline includes several decisions designed for epidemiological time-series data:

- retain normalized TB rates rather than raw burden counts where appropriate;
- preserve extreme values when they represent plausible observations;
- avoid globally fitting preprocessing statistics before the temporal split;
- interpolate missing values within countries first;
- use WHO-region information for remaining missingness where needed;
- add indicators identifying imputed values;
- one-hot encode WHO regions;
- keep separate core and TB/HIV variants for sensitivity analysis;
- avoid arbitrary exclusion of countries solely because their series are shorter.

The main preprocessing implementation is available in:

[01_scripts/data_preprocessing.py](01_scripts/data_preprocessing.py)

## Supervised-learning workflow

The supervised component focuses on a simple and interpretable comparison between two model families.

### 1. Ridge Regression

Ridge Regression provides a regularized linear baseline.

It is useful here because it:

- handles correlated predictors better than unregularized linear regression;
- produces a stable linear reference model;
- makes the value of additional nonlinear complexity easier to assess.

### 2. Random Forest Regression

Random Forest Regression is used to model nonlinear relationships and interactions between epidemiological indicators.

Its performance is compared with the linear model using the same temporal validation logic.

### Evaluation

The main regression metrics are:

- MAE
- RMSE
- R²

The supervised workflow also uses temporal validation rather than random cross-validation so that model selection more closely reflects the real forecasting setting.

The objective is not to maximize a single score, but to determine whether the models improve meaningfully over simple temporal baselines.

## Unsupervised trajectory analysis

The unsupervised component represents each country through its longitudinal TB trajectory and compares several clustering strategies:

- **K-means**
- **Ward hierarchical clustering**
- **Spectral clustering**

The analysis includes:

- country-level trajectory reshaping;
- standardization;
- evaluation across multiple values of k;
- comparison of cluster assignments with Adjusted Rand Index;
- PCA visualization;
- threshold-sensitivity analyses;
- comparison of TB-only and TB/HIV representations.

The implementation is available in:

[01_scripts/unsupervised_analysis.py](01_scripts/unsupervised_analysis.py)

## Core trajectory variables

The primary trajectory representation includes:

- estimated TB incidence per 100,000
- estimated TB mortality per 100,000
- case detection rate
- case fatality ratio

A second analysis adds an HIV-related TB indicator to evaluate how much the country grouping changes when the TB/HIV dimension is included.

## Repository structure

| Path | Purpose |
|---|---|
| [01_scripts](01_scripts) | Preprocessing and analysis code |
| [02_data](02_data) | Data documentation and locally generated datasets |
| [03_notebooks](03_notebooks) | Exploratory notebooks |
| [04_results](04_results) | Generated numerical results |
| [05_figures](05_figures) | Generated figures |
| [06_report](06_report) | Project report |
| [07_orange](07_orange) | Orange3 workflows used in the course project |
| [08_slides](08_slides) | Presentation material |

## Reproduce the environment

Using Conda:

    conda env create -f environment.yml
    conda activate aida-project

or with pip:

    python -m pip install -r requirements.txt

The raw WHO files should then be placed in the location described in:

[02_data/README.md](02_data/README.md)

## Methodological strengths

This project emphasizes several points that are particularly important for longitudinal biomedical data:

- chronological train/test separation;
- explicit prevention of temporal leakage;
- reproducible preprocessing decisions;
- comparison with simple baselines;
- limited model complexity rather than unnecessary algorithm proliferation;
- sensitivity analyses for missing-data thresholds;
- separate interpretation of supervised and unsupervised results.

## Limitations

- WHO surveillance indicators can contain measurement uncertainty and country-specific reporting differences;
- imputation cannot recover unobserved information with certainty;
- country-level associations should not be interpreted as individual-level effects;
- clustering does not create validated epidemiological disease categories;
- predictive performance on historical WHO data does not automatically imply transportability to future surveillance periods.

## Data source

World Health Organization — Global Tuberculosis Programme  
https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data

## Author

**Nahi El Akoum**  
Master 2 — Artificial Intelligence and Data Analysis in Biology (AIDA)  
Sorbonne Université
