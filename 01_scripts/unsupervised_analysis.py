"""Main analysis stages for the unsupervised TB trajectory project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
)
from sklearn.neighbors import kneighbors_graph

from unsupervised_utils import (
    evaluate_hierarchical,
    evaluate_kmeans,
    evaluate_spectral,
    fill_missing_values,
    keep_complete_countries,
    reshape_country_trajectories,
    standardize_features,
)

from unsupervised_plots import (
    plot_cluster_trajectories,
    plot_country_cfr,
    plot_hiv_trajectories,
    plot_pca_hiv_comparison,
    plot_threshold_comparison,
)


RANDOM_SEED = 42
K_VALUES = list(range(2, 9))
CORE_FEATURES = [
    "e_inc_100k",
    "e_mort_100k",
    "c_cdr",
    "cfr",
]
HIV_FEATURE = "e_tbhiv_prct"

def load_input_data(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw TB observations and the variable dictionary."""

    # I keep file locations explicit so the analysis is reproducible.
    data_path = raw_dir / "TB_burden_countries_2025-09-06.csv"
    dictionary_path = raw_dir / "TB_data_dictionary_2025-09-06.csv"

    df_raw = pd.read_csv(data_path)
    data_dictionary = pd.read_csv(dictionary_path)

    return df_raw, data_dictionary

def prepare_complete_data(
    df_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Keep countries with complete observations from 2000 to 2023."""

    # I restricted the analysis to the common period used for all trajectories.
    df_complete = keep_complete_countries(df_raw)

    print(
        "Filtered dataset shape:",
        df_complete.shape,
    )

    return df_complete 

def prepare_imputed_core_data(
    df_complete: pd.DataFrame,
    core_features: list[str],
) -> pd.DataFrame:
    """Fill missing values in the core TB indicators."""

    # I interpolated each country's trajectory before reshaping it.
    df_filled = fill_missing_values(
        df_complete,
        core_features,
    )

    remaining_missing = df_filled[core_features].isna().sum()

    print("\nMissing values after imputation:")
    print(remaining_missing)

    return df_filled

def prepare_all_input_data(
    raw_dir: Path,
) -> dict:
    """Load and prepare all input data before clustering."""

    # I loaded the raw observations and the variable definitions.
    df_raw, data_dictionary = load_input_data(
        raw_dir,
    )

    # I kept only countries with complete coverage from 2000 to 2023.
    df_complete = prepare_complete_data(
        df_raw,
    )

    # I identified countries with an entirely missing core indicator.
    missing_by_country = (
        df_complete.groupby("iso3")[CORE_FEATURES]
        .agg(lambda values: values.isna().sum())
    )

    years_per_country = df_complete.groupby("iso3").size()

    entirely_missing = missing_by_country.eq(
        years_per_country,
        axis="index",
    )

    countries_to_exclude = entirely_missing.any(axis=1)
    excluded_ids = countries_to_exclude[
        countries_to_exclude
    ].index

    print(
        "Excluded countries with an entirely missing core indicator:",
        excluded_ids.tolist(),
    )

    # I excluded these countries before interpolation because no trajectory exists.
    df_complete = df_complete[
        ~df_complete["iso3"].isin(excluded_ids)
    ].copy()

    # I filled missing values in the four core TB indicators.
    df_filled = prepare_imputed_core_data(
        df_complete,
        CORE_FEATURES,
    )

    return {
        "df_raw": df_raw,
        "data_dictionary": data_dictionary,
        "df_complete": df_complete,
        "df_filled": df_filled,
    }

def select_core_countries(
    df_complete: pd.DataFrame,
    core_features: list[str],
    threshold: float,
) -> pd.Index:
    """Select countries below the chosen missing-data threshold."""

    # I calculated missingness before imputation to preserve the original data pattern.
    missing_percent = (
        df_complete.groupby("iso3")[core_features]
        .apply(lambda values: values.isna().mean() * 100)
    )

    # I kept a country only when every core indicator is below the threshold.
    eligible_countries = missing_percent[
        missing_percent.le(threshold).all(axis=1)
    ].index

    print(
        f"Countries retained at the {threshold:.0f}% threshold:",
        len(eligible_countries),
    )

    return eligible_countries

def prepare_core_samples(
    df_complete: pd.DataFrame,
    df_filled: pd.DataFrame,
    core_features: list[str],
) -> dict[str, pd.DataFrame]:
    """Create the imputed core datasets for both thresholds."""

    core_ids_20 = select_core_countries(
        df_complete,
        core_features,
        threshold=20,
    )

    core_ids_30 = select_core_countries(
        df_complete,
        core_features,
        threshold=30,
    )

    core_20_filled = df_filled[
        df_filled["iso3"].isin(core_ids_20)
    ].copy()

    core_30_filled = df_filled[
        df_filled["iso3"].isin(core_ids_30)
    ].copy()

    return {
        "ids_20": core_ids_20,
        "ids_30": core_ids_30,
        "data_20": core_20_filled,
        "data_30": core_30_filled,
    }

def prepare_shared_tb_hiv_sample(
    df_complete: pd.DataFrame,
    df_filled: pd.DataFrame,
    core_features: list[str],
    hiv_feature: str,
    threshold: float,
) -> dict:
    """Prepare countries eligible for both TB-only and TB-HIV clustering."""

    all_features = core_features + [hiv_feature]

    # I calculated missingness before imputation for all five indicators.
    missing_percent = (
        df_complete.groupby("iso3")[all_features]
        .apply(lambda values: values.isna().mean() * 100)
    )

    # I retained countries below the same threshold for every indicator.
    shared_ids = missing_percent[
        missing_percent.le(threshold).all(axis=1)
    ].index

    shared_data = df_filled[
        df_filled["iso3"].isin(shared_ids)
    ].copy()

    # I filled the HIV indicator within the shared sample.
    shared_data = fill_missing_values(
        shared_data,
        all_features,
    )

    print(
        f"Shared TB/TB-HIV sample at {threshold:.0f}%:",
        len(shared_ids),
        "countries",
    )

    return {
        "ids": shared_ids,
        "data": shared_data,
        "features": all_features,
    }

def run_tb_hiv_comparison(
    shared_data: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    k_values: list[int],
) -> dict:
    """Compare country groupings before and after adding the HIV trajectory."""

    # I created TB-only country trajectories on the shared country sample.
    core_wide, core_scaled = build_scaled_country_table(
        shared_data,
        CORE_FEATURES,
    )

    # I created extended trajectories using the four TB indicators and HIV.
    extended_wide, extended_scaled = build_scaled_country_table(
        shared_data,
        CORE_FEATURES + [HIV_FEATURE],
    )

    # I fited K-means to both feature sets using exactly the same countries.
    tb_results, tb_labels = evaluate_kmeans(
        core_scaled,
        k_values,
        random_state=RANDOM_SEED,
    )

    tb_hiv_results, tb_hiv_labels = evaluate_kmeans(
        extended_scaled,
        k_values,
        random_state=RANDOM_SEED,
    )

    # I compared the selected three-cluster solutions.
    agreement = compare_tb_and_hiv_assignments(
        tb_labels,
        tb_hiv_labels,
        k=3,
    )

    pd.DataFrame([agreement]).to_csv(
        results_dir / "tb_vs_tb_hiv_agreement_k3.csv",
        index=False,
    )

    # I visualized the HIV trajectories of the extended clusters.
    plot_hiv_trajectories(
        shared_data,
        tb_hiv_labels["k_3"],
        save_path=figures_dir / "hiv_trajectories_k3_20_percent.png",
    )

    # I visualized both assignments in one shared PCA projection.
    pca_scores, pca_model = plot_pca_hiv_comparison(
        extended_scaled,
        tb_labels["k_3"],
        tb_hiv_labels["k_3"],
        save_path=figures_dir / "pca_tb_vs_tb_hiv_k3_20_percent.png",
    )

    return {
        "core_wide": core_wide,
        "core_scaled": core_scaled,
        "extended_wide": extended_wide,
        "extended_scaled": extended_scaled,
        "tb_results": tb_results,
        "tb_labels": tb_labels,
        "tb_hiv_results": tb_hiv_results,
        "tb_hiv_labels": tb_hiv_labels,
        "agreement": agreement,
        "pca_scores": pca_scores,
        "pca_model": pca_model,
    }

def run_tb_hiv_threshold_sensitivity(
    df_complete: pd.DataFrame,
    df_filled: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> dict:
    """Compare TB-HIV clustering at the 20% and 30% thresholds."""

    # I prepared the stricter and less strict shared TB-HIV samples.
    shared_20 = prepare_shared_tb_hiv_sample(
        df_complete,
        df_filled,
        CORE_FEATURES,
        HIV_FEATURE,
        threshold=20,
    )

    shared_30 = prepare_shared_tb_hiv_sample(
        df_complete,
        df_filled,
        CORE_FEATURES,
        HIV_FEATURE,
        threshold=30,
    )

    # I ran the same TB-HIV clustering workflow for both thresholds.
    analysis_20 = run_tb_hiv_comparison(
        shared_20["data"],
        results_dir,
        figures_dir,
        k_values=[2, 3, 4, 5, 6, 7, 8],
    )

    analysis_30 = run_tb_hiv_comparison(
        shared_30["data"],
        results_dir,
        figures_dir,
        k_values=[2, 3, 4, 5, 6, 7, 8],
    )

    common_ids = (
        analysis_20["tb_labels"][f"k_3"].index
        .intersection(
            analysis_30["tb_labels"][f"k_3"].index
        )
    )

    comparisons = []

    for label_type in ["tb_labels", "tb_hiv_labels"]:
        ari = adjusted_rand_score(
            analysis_20[label_type]["k_3"].loc[common_ids],
            analysis_30[label_type]["k_3"].loc[common_ids],
        )

        comparisons.append(
            {
                "partition": label_type,
                "k": 3,
                "ARI_20_vs_30": ari,
            }
        )

    sensitivity_results = pd.DataFrame(comparisons)

    sensitivity_results.to_csv(
        results_dir / "tb_hiv_threshold_sensitivity_k3.csv",
        index=False,
    )

    print("\nTB-HIV threshold sensitivity:")
    print(sensitivity_results.round(3).to_string(index=False))

    return {
        "analysis_20": analysis_20,
        "analysis_30": analysis_30,
        "comparisons": sensitivity_results,
    }

def build_scaled_country_table(
    df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create and standardize one-row-per-country trajectory features."""

    # I reshaped before scaling so every country is represented by one row.
    country_wide = reshape_country_trajectories(
        df,
        features,
    )

    # I standardized all indicator-year features before clustering.
    country_scaled, scaler = standardize_features(
        country_wide,
    )

    print(
        "\nCountry-level dataset shape:",
        country_wide.shape,
    )

    print(
        "Standardized dataset shape:",
        country_scaled.shape,
    )

    return country_wide, country_scaled 

def compare_clustering_methods(
    scaled_data: pd.DataFrame,
    k_values: list[int],
    random_seed: int = RANDOM_SEED,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Compare K-means, Ward and spectral clustering."""

    # I evaluated K-means and Ward in the same Euclidean feature space.
    kmeans_results, kmeans_labels = evaluate_kmeans(
        scaled_data,
        k_values,
        random_state=random_seed,
    )

    ward_results, ward_labels = evaluate_hierarchical(
        scaled_data,
        k_values,
    )

    # I built a symmetric neighbour graph for spectral clustering.
    n_neighbors = 10

    neighbour_graph = kneighbors_graph(
        scaled_data,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )

    affinity_graph = 0.5 * (
        neighbour_graph + neighbour_graph.T
    )

    spectral_results, spectral_labels = evaluate_spectral(
        scaled_data,
        affinity_graph,
        k_values,
        random_state=random_seed,
    )

    return {
        "kmeans": (kmeans_results, kmeans_labels),
        "ward": (ward_results, ward_labels),
        "spectral": (spectral_results, spectral_labels),
    }

def save_clustering_results(
    results: dict,
    results_dir: Path,
    prefix: str,
) -> None:
    """Save clustering evaluation tables as CSV files."""

    results_dir.mkdir(exist_ok=True)

    # I saved each method separately so the results are easier to inspect.
    for method_name, (metrics, labels) in results.items():
        metrics.to_csv(
            results_dir / f"{prefix}_{method_name}_metrics.csv",
            index=False,
        )

        labels.to_csv(
            results_dir / f"{prefix}_{method_name}_labels.csv",
            index=True,
        )

def run_clustering_stage(
    df_analysis: pd.DataFrame,
    features: list[str],
    k_values: list[int],
    results_dir: Path,
    prefix: str,
) -> dict:
    """Prepare features, compare methods and save their results."""

    # I built the one-row-per-country standardized feature table.
    country_wide, country_scaled = build_scaled_country_table(
        df_analysis,
        features,
    )

    # I compared all three clustering methods on the same input data.
    clustering_results = compare_clustering_methods(
        country_scaled,
        k_values,
    )

    # I saved metrics and country assignments for later reporting.
    save_clustering_results(
        clustering_results,
        results_dir,
        prefix,
    )

    return {
        "country_wide": country_wide,
        "country_scaled": country_scaled,
        "results": clustering_results,
    }

def compare_method_assignments(
    clustering_results: dict,
    k: int,
) -> pd.DataFrame:
    """Compare cluster assignments with pairwise adjusted Rand index."""

    method_labels = {
        method_name: result[1][f"k_{k}"]
        for method_name, result in clustering_results.items()
    }

    method_pairs = [
        ("kmeans", "ward"),
        ("kmeans", "spectral"),
        ("ward", "spectral"),
    ]

    comparisons = []

    for first_method, second_method in method_pairs:
        ari = adjusted_rand_score(
            method_labels[first_method],
            method_labels[second_method],
        )

        comparisons.append(
            {
                "k": k,
                "first_method": first_method,
                "second_method": second_method,
                "ARI": ari,
            }
        )

    return pd.DataFrame(comparisons)

def run_core_clustering_analysis(
    df_core: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
    threshold_label: str,
) -> dict:
    """Run the complete clustering comparison for core TB indicators."""

    # I compared all methods using the same country-level TB trajectories.
    analysis = run_clustering_stage(
        df_core,
        CORE_FEATURES,
        K_VALUES,
        results_dir,
        prefix=f"core_{threshold_label}",
    )

    # I compared the three methods at the selected three-cluster solution.
    agreement = compare_method_assignments(
        analysis["results"],
        k=3,
    )

    agreement.to_csv(
        results_dir / f"core_{threshold_label}_method_agreement_k3.csv",
        index=False,
    )

    # I saved the main K-means trajectory figure for interpretation.
    plot_cluster_trajectories(
        df_core,
        CORE_FEATURES,
        analysis["results"]["kmeans"][1]["k_3"],
        save_path=(
            figures_dir
            / f"kmeans_tb_trajectories_k3_{threshold_label}.png"
        ),
    )

    return analysis 

def run_threshold_sensitivity_analysis(
    core_20_analysis: dict,
    core_30_data: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> dict:
    """Compare the primary 20% analysis with the 30% sensitivity analysis."""

    # I repeated the same clustering workflow at the less strict threshold.
    core_30_analysis = run_core_clustering_analysis(
        core_30_data,
        results_dir,
        figures_dir,
        threshold_label="30_percent",
    )

    comparisons = []

    for k in [2, 3, 4]:
        labels_20 = core_20_analysis["results"]["kmeans"][1][f"k_{k}"]
        labels_30 = core_30_analysis["results"]["kmeans"][1][f"k_{k}"]

        common_ids = labels_20.index.intersection(
            labels_30.index,
        )

        ari = adjusted_rand_score(
            labels_20.loc[common_ids],
            labels_30.loc[common_ids],
        )

        comparisons.append(
            {
                "method": "K-means",
                "k": k,
                "ARI_20_vs_30": ari,
            }
        )

    sensitivity_results = pd.DataFrame(comparisons)

    sensitivity_results.to_csv(
        results_dir / "threshold_sensitivity_kmeans.csv",
        index=False,
    )

    print("\nThreshold sensitivity results:")
    print(sensitivity_results.round(3).to_string(index=False))

    return {
        "core_30_analysis": core_30_analysis,
        "comparisons": sensitivity_results,
    }

def compare_tb_and_hiv_assignments(
    tb_labels: pd.DataFrame,
    tb_hiv_labels: pd.DataFrame,
    k: int,
) -> dict:
    """Measure how country assignments change after adding HIV trajectories."""

    # I compared labels for the same countries and the same number of clusters.
    tb_partition = tb_labels[f"k_{k}"]
    tb_hiv_partition = tb_hiv_labels[f"k_{k}"]

    ari = adjusted_rand_score(
        tb_partition,
        tb_hiv_partition,
    )

    homogeneity, completeness, v_measure = (
        homogeneity_completeness_v_measure(
            tb_partition,
            tb_hiv_partition,
        )
    )

    return {
        "k": k,
        "ARI": ari,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "V_measure": v_measure,
    }

def run_pca(
    scaled_data: pd.DataFrame,
    n_components: int = 2,
) -> tuple[pd.DataFrame, PCA]:
    """Project standardized trajectories into a two-dimensional PCA space."""

    # I used two components so I can visualize the main structure of countries.
    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_SEED,
    )

    scores = pca.fit_transform(scaled_data)

    pca_scores = pd.DataFrame(
        scores,
        index=scaled_data.index,
        columns=[
            f"PC{i + 1}"
            for i in range(n_components)
        ],
    )

    explained_variance = pca.explained_variance_ratio_ * 100

    print("\nPCA explained variance:")
    for component, variance in enumerate(
        explained_variance,
        start=1,
    ):
        print(
            f"PC{component}: {variance:.2f}%"
        )

    print(
        "Total for the selected components:",
        f"{explained_variance.sum():.2f}%",
    )

    return pca_scores, pca