"""Reusable functions for the tuberculosis trajectory analysis."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_RANDOM_SEED = 42


def fill_missing_values(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Fill missing annual values within each country."""

    filled = df.sort_values(["iso3", "year"]).copy()

    # I keep flags so I can later report which observations were imputed.
    for feature in features:
        filled[f"{feature}_is_imputed"] = (
            filled[feature].isna().astype(int)
        )

    # I interpolate because the data describe annual country trajectories.
    filled[features] = (
        filled.groupby("iso3")[features]
        .transform(
            lambda values: values.interpolate(
                method="linear",
                limit_direction="both",
            )
        )
    )

    if filled[features].isna().any().any():
        raise ValueError("Missing values remain after interpolation.")

    return filled


def reshape_country_trajectories(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Create one row per country and one feature per indicator-year pair."""

    # I keep every year as a feature because clustering compares full trajectories.
    wide = df.pivot(
        index="iso3",
        columns="year",
        values=features,
    ).sort_index(axis=1)

    wide.columns = [
        f"{feature}_{year}"
        for feature, year in wide.columns
    ]

    if wide.isna().any().any():
        raise ValueError(
            "The country-level table contains missing cells."
        )

    return wide


def standardize_features(
    df_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize each trajectory feature across countries."""

    # I standardize features so indicators with larger scales do not dominate.
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_wide)

    scaled = pd.DataFrame(
        scaled_values,
        index=df_wide.index,
        columns=df_wide.columns,
    )

    return scaled, scaler


def _cluster_summary(
    df_scaled: pd.DataFrame,
    labels,
) -> dict:
    """Calculate metrics shared by the clustering methods."""

    cluster_sizes = pd.Series(labels).value_counts()

    return {
        "silhouette": silhouette_score(
            df_scaled,
            labels,
        ),
        "smallest_cluster": int(cluster_sizes.min()),
        "largest_cluster": int(cluster_sizes.max()),
    }


def evaluate_kmeans(
    df_scaled: pd.DataFrame,
    k_values: list[int],
    random_state: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare K-means solutions for several cluster counts."""

    results = []
    labels_by_k = pd.DataFrame(index=df_scaled.index)

    for k in k_values:
        # I use multiple starts and a fixed seed for reproducible results.
        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state,
        )

        labels = model.fit_predict(df_scaled)
        summary = _cluster_summary(df_scaled, labels)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                **summary,
            }
        )

        labels_by_k[f"k_{k}"] = labels

    return pd.DataFrame(results), labels_by_k


def evaluate_hierarchical(
    df_scaled: pd.DataFrame,
    k_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare Ward hierarchical clustering solutions."""

    results = []
    labels_by_k = pd.DataFrame(index=df_scaled.index)

    for k in k_values:
        # I use Ward linkage because it minimizes within-cluster variance.
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage="ward",
        )

        labels = model.fit_predict(df_scaled)
        summary = _cluster_summary(df_scaled, labels)

        results.append(
            {
                "k": k,
                **summary,
            }
        )

        labels_by_k[f"k_{k}"] = labels

    return pd.DataFrame(results), labels_by_k


def evaluate_spectral(
    df_scaled: pd.DataFrame,
    affinity_graph,
    k_values: list[int],
    random_state: int = DEFAULT_RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare spectral clustering solutions on a neighbour graph."""

    results = []
    labels_by_k = pd.DataFrame(index=df_scaled.index)

    for k in k_values:
        # I use the country similarity graph to capture non-linear structure.
        model = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            n_init=20,
            random_state=random_state,
        )

        labels = model.fit_predict(affinity_graph)
        summary = _cluster_summary(df_scaled, labels)

        results.append(
            {
                "k": k,
                **summary,
            }
        )

        labels_by_k[f"k_{k}"] = labels

    return pd.DataFrame(results), labels_by_k

def keep_complete_countries(df):
    """Keep countries with one observation for every year from 2000 to 2023."""

    # Define the common period used to compare country trajectories.
    expected_years = set(range(2000, 2024))

    # Keep observations within the selected period.
    df_period = df.loc[df["year"].isin(expected_years)].copy()

    # Each country must have at most one observation per year.
    if df_period.duplicated(subset=["iso3", "year"]).any():
        raise ValueError("Duplicate country-year observations were found.")

    # Collect the available years for each country.
    years_by_country = df_period.groupby("iso3")["year"].agg(set)

    # Identify countries that have every year in the selected period.
    complete_mask = years_by_country.apply(
        lambda years: years == expected_years
    )
    complete_country_ids = years_by_country.index[complete_mask]

    # Retain all observations belonging to the eligible countries.
    df_complete = df_period.loc[
        df_period["iso3"].isin(complete_country_ids)
    ].copy()

    # Arrange observations chronologically within each country.
    df_complete = df_complete.sort_values(
        ["iso3", "year"]
    ).reset_index(drop=True)

    # Report how many countries were retained and excluded.
    print("Countries before filtering:", df_period["iso3"].nunique())
    print("Countries retained:", df_complete["iso3"].nunique())
    print("Countries excluded:", (~complete_mask).sum())

    return df_complete