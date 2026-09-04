# 5BM753 Project — <title>

Team: <name 1>, <name 2>, <name 3>

## Biological question

<Which question? On which data? Why machine learning is relevant here.>

## Data

- Source: <link or reference to the paper>
- Size: <n samples x p features>
- Target variable: <name, number of classes, class balance>
- Missing values: <how many, where>

Raw data is not versioned. Each member downloads it into `02_data/raw/`.

## How to reproduce the analysis

```bash
conda env create -f environment.yml
conda activate aida-project
python 01_scripts/main.py        # <update once the script exists>
```

## Repository layout

```
01_scripts/     analysis code
02_data/        raw/ (never modified) and processed/ (regenerable) - not versioned
03_notebooks/   throwaway exploration, outputs cleared before commit
04_results/     tables produced by the code - not versioned
05_figures/     figures produced by the code - not versioned
06_report/      8-10 page report, exported PDF only
07_orange/      Orange3 workflows (.ows), one file per person
08_slides/      defence slides, exported PDF only
```

## Task allocation

| Member | Scope |
|---|---|
| <name 1> | <e.g. preprocessing + unsupervised> |
| <name 2> | <e.g. supervised + evaluation> |
| <name 3> | <e.g. Orange3 + report> |

## Logbook

| Date | What was done | By whom |
|---|---|---|
|  |  |  |
