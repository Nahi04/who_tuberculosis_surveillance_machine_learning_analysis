"""
WHO TB Burden : preprocessing pipeline.

Builds the clean, leak-safe analysis table from the raw data file, implementing
decisions D1-D9 validated in `01_EDA.ipynb`, and A1/A2 as resolved below.

D1  Column selection      -> drop _lo/_hi, _num, cfr_pct, iso2/iso_numeric.
D2  Analysis scale         -> normalized rates (_100k); e_pop_num kept as a
                              plain explanatory variable but not a burden signal.
D3  Extreme values         -> kept as is (checked as real in the EDA), no clipping.
D4  Log transform          -> NOT applied here. It only helps distance-based
                              methods (PCA, k-means, hierarchical clustering, KNN)
                              and would be wasted/wrong on tree-based models.
                              Provided as a separate, optional function
                              (`apply_log1p`) to be called downstream by whichever
                              method needs it.
D5  No data leakage        -> every fitted statistic (regional median, scaler)
                              is fit on the training split only.
D6  Train/test split       -> temporal, not random (train: <=TRAIN_END_YEAR,
                              test: after). Two consecutive years of the same
                              country are correlated, a random row split would
                              leak information across the boundary.
D7  Imputation rule        -> never a global mean/median. Temporal interpolation
                              within each country first, then the WHO-region
                              median (fit on train only) for what remains.
                              A binary "<col>_is_imputed" flag is added per column.
D8  Territories            -> all countries kept, including the six with a
                              shorter series (no arbitrary exclusion).
D9  Region encoding        -> one-hot, no assumed order between regions.

A1  HIV variables (resolved for this pipeline): produced as two variants
    instead of picking one option -- "core" (no HIV variable, max country
    coverage) and "extended" (+ e_tbhiv_prct). Downstream analyses run both
    and compare, rather than deciding this upfront.
A2  Correlated blocks (resolved for this pipeline): one representative
    variable kept per correlated block identified in the EDA (finding 8)
    instead of all of them -- e_inc_100k over c_newinc_100k (r=0.938),
    e_tbhiv_prct over e_inc_tbhiv_100k (r=0.956), e_mort_100k over
    e_mort_exc_tbhiv_100k and e_mort_tbhiv_100k. A PCA on the full block is
    left as a comparison point for Milestone 2, not implemented here.


Usage:
    python data_preprocessing.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Paths
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "02_data" / "raw" / "TB_burden_countries_2025-09-06.csv"
OUT_DIR = ROOT / "02_data" / "processed"

# ============================================================================
# Variable groups (D1 selection + A2 resolution: one representative per block)
# ============================================================================
ID_COLS = ["country", "iso3", "year", "g_whoregion"]

CORE_NUMERIC = [
    "e_pop_num",     # D2: explanatory, not a burden signal
    "e_inc_100k",    # kept over c_newinc_100k, r=0.938 (A2)
    "e_mort_100k",   # kept over e_mort_exc_tbhiv_100k + e_mort_tbhiv_100k (A2)
    "c_cdr",
    "cfr",           # kept over cfr_pct, r=1.0 (D1)
]

HIV_VARIABLE = "e_tbhiv_prct"  # kept over e_inc_tbhiv_100k, r=0.956 (A2) --> = e_inc_100k x e_tbhiv_prct

# D4 refinement, checked empirically in 02_preprocessing.ipynb: log1p reduces
# skew for every candidate EXCEPT c_cdr, whose skew gets worse after log1p
# (-0.98 -> -4.69). c_cdr was already the only non-skewed candidate in the EDA
# (finding 7) -- exclude it whenever apply_log1p is used downstream.
NO_LOG_COLS = ["c_cdr"]

# D6
TRAIN_END_YEAR = 2018  # train: <= 2018, test: >= 2019


# ============================================================================
# Step 1 -- load and select columns (D1, A1, A2)
# ============================================================================
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw shape: {df.shape}")
    return df


#variant: 'core' (no HIV variable) or 'extended' (+ e_tbhiv_prct).
#Also carries along each kept variable's _lo/_hi uncertainty bounds (D1:
#excluded from modelling features, but kept for visualization/interpretation). 
#They ride through untouched: not imputed, not
#log-transformed, not scaled --> just for plots that need them
def select_variant(df: pd.DataFrame, variant: str):
    assert variant in {"core", "extended"}
    numeric_cols = CORE_NUMERIC + ([HIV_VARIABLE] if variant == "extended" else [])
    bound_cols = [c for base in numeric_cols for c in (f"{base}_lo", f"{base}_hi") if c in df.columns]
    out = df[ID_COLS + numeric_cols + bound_cols].copy()
    print(f"[{variant}] numeric columns: {numeric_cols}")
    print(f"[{variant}] uncertainty bounds kept (not modelled): {bound_cols}")
    return out, numeric_cols


# ============================================================================
# Step 2 -- temporal split (D6), before any imputation
# ============================================================================
def temporal_split(df: pd.DataFrame, train_end_year: int = TRAIN_END_YEAR):
    train = df[df["year"] <= train_end_year].copy()
    test = df[df["year"] > train_end_year].copy()
    print(f"Train: {train['year'].min()}-{train['year'].max()} ({len(train)} rows)")
    print(f"Test:  {test['year'].min()}-{test['year'].max()} ({len(test)} rows)")
    return train, test


# ============================================================================
# Step 3 -- imputation (D7), fit on train only (D5)
# ============================================================================

#Interpolate within each country, using only the training years, then
#fill what remains with the WHO-region median computed on train only.
#Returns the imputed train set and the fitted regional medians (for test).
def impute_train(train: pd.DataFrame, numeric_cols: list[str]):
    train = train.copy()

    for col in numeric_cols:
        train[f"{col}_is_imputed"] = train[col].isna().astype(int)

    train[numeric_cols] = train.groupby("country")[numeric_cols].transform(
        lambda s: s.interpolate(method="linear", limit_direction="both")
    )

    regional_median = train.groupby("g_whoregion")[numeric_cols].median()

    train[numeric_cols] = train.groupby("g_whoregion")[numeric_cols].transform(
        lambda s: s.fillna(s.median())
    )

    remaining = train[numeric_cols].isna().sum().sum()
    print(f"Train -- remaining NaN after imputation: {remaining}")
    return train, regional_median


def impute_test(test: pd.DataFrame, numeric_cols: list[str], regional_median: pd.DataFrame):
#Forward-fill within each country (chronological, no look-ahead), then
#fill anything left with the TRAIN-derived regional median (not test's own).
    test = test.copy()

    for col in numeric_cols:
        test[f"{col}_is_imputed"] = test[col].isna().astype(int)

    test[numeric_cols] = test.groupby("country")[numeric_cols].transform(lambda s: s.ffill())

    for region, row in regional_median.iterrows():
        mask = test["g_whoregion"] == region
        test.loc[mask, numeric_cols] = test.loc[mask, numeric_cols].fillna(row)

    remaining = test[numeric_cols].isna().sum().sum()
    print(f"Test  -- remaining NaN after imputation: {remaining}")
    return test


# ============================================================================
# Step 4 -- region encoding (D9)
# ============================================================================
def encode_region(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["g_whoregion"], prefix="region", dtype=int)


# ============================================================================
# Optional, downstream transforms (D4) -- NOT applied automatically.
# Call only for distance-based methods (PCA, k-means, hierarchical
# clustering, KNN). Skip them entirely for tree-based models.
# ============================================================================
def apply_log1p(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[numeric_cols] = np.log1p(df[numeric_cols])
    return df


def fit_scaler(train: pd.DataFrame, numeric_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train[numeric_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df


# ============================================================================
# Pipeline -- CLEAN table (no log, no scaling)
# ============================================================================
def build_variant(df_raw: pd.DataFrame, variant: str, out_dir: Path = OUT_DIR):
    print(f"\n=== Building '{variant}' variant ===")
    df, numeric_cols = select_variant(df_raw, variant)

    train_raw, test_raw = temporal_split(df)
    train_clean, regional_median = impute_train(train_raw, numeric_cols)
    test_clean = impute_test(test_raw, numeric_cols, regional_median)

    train_clean = encode_region(train_clean)
    test_clean = encode_region(test_clean)

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / f"tb_{variant}_train.csv"
    test_path = out_dir / f"tb_{variant}_test.csv"
    train_clean.to_csv(train_path, index=False)
    test_clean.to_csv(test_path, index=False)
    print(f"Saved: {train_path.name}, {test_path.name}")

    return train_clean, test_clean, numeric_cols


def main():
    df_raw = load_raw()
    for variant in ("core", "extended"):
        build_variant(df_raw, variant)


if __name__ == "__main__":
    main()
