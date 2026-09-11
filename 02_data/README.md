# 02_data/

## Raw data

**File:** `raw/TB_burden_countries_2025-09-06.csv`
**Source:** WHO Global Tuberculosis Programme — https://www.who.int/teams/global-programme-on-tuberculosis-and-lung-health/data
**Extraction date:** 2025-09-06 (per filename)

To rebuild this file: go to the URL above, download the "TB burden estimates" country-level
CSV, and place it at `raw/TB_burden_countries_2025-09-06.csv`. The accompanying data
dictionary (`TB_data_dictionary_2025-09-06.csv`) is available from the same page.

Per `.gitignore`, this raw file is not committed, only this README, so anyone can rebuild it.

## Processed data

**Produced by:** `01_scripts/data_preprocessing.py` (run it, or open `03_notebooks/02_preprocessing.ipynb`)

| File | Rows | Description |
|---|---|---|
| `processed/tb_core_train.csv` | 2000-2018 | No HIV variable, max country coverage |
| `processed/tb_core_test.csv` | 2019-2023 | Same, held out for evaluation |
| `processed/tb_extended_train.csv` | 2000-2018 | + `e_tbhiv_prct` |
| `processed/tb_extended_test.csv` | 2019-2023 | Same, held out for evaluation |

Split is temporal (train <= 2018, test >= 2019), not random, see `data_preprocessing.py`
docstring for the full list of preprocessing decisions (D1-D9, A1-A2).

Also not committed (regenerable), run the script to produce them locally.
