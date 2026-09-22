from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, homogeneity_completeness_v_measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components 
from sklearn.cluster import SpectralClustering 
from sklearn.decomposition import PCA 
from unsupervised_utils import (
    fill_missing_values,
    reshape_country_trajectories,
    standardize_features,
    evaluate_kmeans,
    evaluate_hierarchical,
    evaluate_spectral,
    keep_complete_countries,
)
from unsupervised_plots import (
    plot_cluster_trajectories,
    plot_country_cfr,
    plot_threshold_comparison,
    plot_hiv_trajectories,
    plot_pca_hiv_comparison,
)
from unsupervised_analysis import (
    load_input_data,
    prepare_complete_data,
    prepare_imputed_core_data,
    prepare_all_input_data,
    select_core_countries,
    prepare_core_samples,
    prepare_shared_tb_hiv_sample,
    run_tb_hiv_comparison,
    run_tb_hiv_threshold_sensitivity,
    build_scaled_country_table,
    compare_clustering_methods,
    save_clustering_results,
    run_clustering_stage,
    compare_method_assignments,
    run_core_clustering_analysis,
    run_threshold_sensitivity_analysis,
    compare_tb_and_hiv_assignments,
    run_pca,
)

# Locate the project root independently of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the folder containing the raw datasets.
RAW_DIR = PROJECT_ROOT / "02_data" / "raw"

# Default seed for reproducible K-means results.
RANDOM_SEED = 42

# Features used by the refactored clustering workflow.
CORE_FEATURES = [
    "e_inc_100k",
    "e_mort_100k",
    "c_cdr",
    "cfr",
]
HIV_FEATURE = "e_tbhiv_prct"


def main():
    """Run the streamlined unsupervised TB trajectory analysis."""

    results_dir = PROJECT_ROOT / "04_results"
    figures_dir = PROJECT_ROOT / "05_figures"

    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # I loaded and prepared all raw data before clustering.
    prepared = prepare_all_input_data(
        RAW_DIR,
    )

    df_complete = prepared["df_complete"]
    df_filled = prepared["df_filled"]

    # I prepared the 20% and 30% core TB analysis samples.
    core_samples = prepare_core_samples(
        df_complete,
        df_filled,
        CORE_FEATURES,
    )

    # I used the 20% threshold as the primary core analysis.
    core_analysis = run_core_clustering_analysis(
        core_samples["data_20"],
        results_dir,
        figures_dir,
        threshold_label="20_percent",
    )

    threshold_analysis = run_threshold_sensitivity_analysis(
    core_analysis,
    core_samples["data_30"],
    results_dir,
    figures_dir,
    )

    # I prepared the shared sample for the TB-only versus TB-HIV comparison.
    shared_20 = prepare_shared_tb_hiv_sample(
        df_complete,
        df_filled,
        CORE_FEATURES,
        HIV_FEATURE,
        threshold=20,
    )

    # I quantified how adding HIV changes the country groupings.
    tb_hiv_analysis = run_tb_hiv_comparison(
        shared_20["data"],
        results_dir,
        figures_dir,
        k_values=[2, 3, 4, 5, 6, 7, 8], 
    )

    hiv_threshold_analysis = run_tb_hiv_threshold_sensitivity(
    df_complete,
    df_filled,
    results_dir,
    figures_dir,
    )

    return {
    "core_analysis": core_analysis,
    "threshold_analysis": threshold_analysis,
    "tb_hiv_analysis": tb_hiv_analysis,
    "hiv_threshold_analysis": hiv_threshold_analysis,
    }

if __name__ == "__main__":
    main()
