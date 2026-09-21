"""Plotting functions for the tuberculosis clustering analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA

def plot_cluster_trajectories(
    df,
    features,
    labels,
    save_path=None,
    ):
    """Plot yearly cluster medians with the middle 50% of country values."""

    # Attach cluster labels to every country-year observation.
    plot_data = df.merge(
        labels.rename("cluster"),
        left_on="iso3",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    # Ensure that every country has a cluster assignment.
    if plot_data["cluster"].isna().any():
        raise ValueError("Some countries have no cluster assignment.")

    # Use readable titles and the original indicator units.
    titles = {
        "e_inc_100k": "TB incidence",
        "e_mort_100k": "TB mortality",
        "c_cdr": "TB treatment coverage",
        "cfr": "Estimated TB case fatality ratio",
    }

    units = {
        "e_inc_100k": "Per 100,000 population",
        "e_mort_100k": "Per 100,000 population",
        "c_cdr": "Percent",
        "cfr": "Proportion",
    }

    # Create one panel for each of the four core indicators.
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8), sharex=True
    )

    cluster_ids = sorted(plot_data["cluster"].unique())

    for ax, feature in zip(axes.flat, features):
        for cluster_id in cluster_ids:
            cluster_data = plot_data.loc[
                plot_data["cluster"] == cluster_id
            ]

            # Summarize the distribution across countries in each year.
            yearly_values = cluster_data.groupby("year")[feature]
            median = yearly_values.median()
            lower = yearly_values.quantile(0.25)
            upper = yearly_values.quantile(0.75)

            n_countries = cluster_data["iso3"].nunique()

            # Draw the median trajectory and use its color for the band.
            line, = ax.plot(
                median.index,
                median.values,
                label=f"Cluster {cluster_id} (n={n_countries})",
            )

            ax.fill_between(
                median.index,
                lower.values,
                upper.values,
                color=line.get_color(),
                alpha=0.15,
            )

        ax.set_title(titles[feature])
        ax.set_ylabel(units[feature])
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)

    # Place the title and legend at separate vertical positions.
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()

    fig.suptitle(
        "TB trajectories by K-means cluster (k=4)",
        y=0.98,
    )

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
    )

    # Reserve the upper part of the figure for the title and legend.
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig) 


def plot_country_cfr(df, country_ids, save_path=None):
    """Plot individual CFR trajectories and mark imputed values."""

    # Select the countries being inspected.
    selected = df.loc[df["iso3"].isin(country_ids)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    for iso3, country_data in selected.groupby("iso3"):
        # Arrange each country's observations chronologically.
        country_data = country_data.sort_values("year")
        country_name = country_data["country"].iloc[0]

        # Plot the trajectory, including any imputed values.
        line, = ax.plot(
            country_data["year"],
            country_data["cfr"],
            label=f"{country_name} ({iso3})",
        )

        # Identify values that were originally missing.
        imputed = country_data["cfr_is_imputed"].eq(1)

        # Mark imputed values with crosses in the country's line color.
        ax.scatter(
            country_data.loc[imputed, "year"],
            country_data.loc[imputed, "cfr"],
            color=line.get_color(),
            marker="x",
            s=65,
            zorder=3,
        )

    ax.set_title("Individual CFR trajectories in cluster 3\nCrosses mark imputed values")
    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated TB case fatality ratio")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close(fig)


def plot_threshold_comparison(
    df_20, df_30, features, labels_20, labels_30, method_name, k, save_path=None
):
    """Compare cluster trajectories at two missing-data thresholds."""

    titles = {
        "e_inc_100k": "TB incidence",
        "e_mort_100k": "TB mortality",
        "c_cdr": "TB treatment coverage",
        "cfr": "Estimated TB case fatality ratio",
    }

    units = {
        "e_inc_100k": "Per 100,000 population",
        "e_mort_100k": "Per 100,000 population",
        "c_cdr": "Percent",
        "cfr": "Proportion",
    }

    # Use one row per indicator and one column per threshold.
    fig, axes = plt.subplots(
        len(features),
        2,
        figsize=(14, 12),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    samples = [
        (20, df_20, labels_20),
        (30, df_30, labels_30),
    ]

    for column, (threshold, df, labels) in enumerate(samples):
        # Attach each country's cluster label to its annual observations.
        plot_data = df.merge(
            labels.rename("cluster"),
            left_on="iso3",
            right_index=True,
            how="left",
            validate="many_to_one",
        )

        if plot_data["cluster"].isna().any():
            raise ValueError("Some countries have no cluster assignment.")

        # Order colors by mean incidence to make the panels easier to read.
        # This is a display convention, not a formal matching of clusters.
        cluster_order = (
            plot_data.groupby("cluster")["e_inc_100k"]
            .mean()
            .sort_values()
            .index
        )

        for row, feature in enumerate(features):
            ax = axes[row, column]

            for color_number, cluster_id in enumerate(cluster_order):
                cluster_data = plot_data.loc[
                    plot_data["cluster"] == cluster_id
                ]

                yearly = cluster_data.groupby("year")[feature]
                median = yearly.median()
                lower = yearly.quantile(0.25)
                upper = yearly.quantile(0.75)

                n_countries = cluster_data["iso3"].nunique()
                color = f"C{color_number}"

                ax.plot(
                    median.index,
                    median.values,
                    color=color,
                    label=f"Cluster {cluster_id} (n={n_countries})",
                )

                ax.fill_between(
                    median.index,
                    lower.values,
                    upper.values,
                    color=color,
                    alpha=0.15,
                )

            ax.set_title(f"{titles[feature]} — threshold {threshold}%")
            ax.grid(alpha=0.25)

            if column == 0:
                ax.set_ylabel(units[feature])

            if row == 0:
                ax.legend(fontsize=8)

            if row == len(features) - 1:
                ax.set_xlabel("Year")

    fig.suptitle(
        f"{method_name}: {k} clusters — 20% versus 30% threshold",
        y=0.99,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def plot_hiv_trajectories(df, labels, save_path=None):
    """Plot yearly HIV proportions by cluster in the extended analysis."""

    # Attach cluster assignments to the annual country observations.
    plot_data = df.merge(
        labels.rename("cluster"),
        left_on="iso3",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    if plot_data["cluster"].isna().any():
        raise ValueError("Some countries have no cluster assignment.")

    fig, ax = plt.subplots(figsize=(10, 6))

    for cluster_id, cluster_data in plot_data.groupby("cluster"):
        # Summarize the HIV indicator across countries in each year.
        yearly = cluster_data.groupby("year")["e_tbhiv_prct"]

        median = yearly.median()
        lower = yearly.quantile(0.25)
        upper = yearly.quantile(0.75)

        n_countries = cluster_data["iso3"].nunique()

        line, = ax.plot(
            median.index,
            median.values,
            label=f"Cluster {cluster_id} (n={n_countries})",
        )

        # Show the middle 50% of country values, not a confidence interval.
        ax.fill_between(
            median.index,
            lower.values,
            upper.values,
            color=line.get_color(),
            alpha=0.15,
        )

    ax.set_title("HIV trajectories by TB-HIV cluster — K-means, k=3")
    ax.set_xlabel("Year")
    ax.set_ylabel("Estimated HIV among incident TB cases (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_pca_hiv_comparison(
    df_extended_scaled, baseline_labels, extended_labels, save_path=None
):
    """Display both partitions in the same PCA projection of extended data."""

    # Align both sets of cluster labels with the country order.
    baseline_labels = baseline_labels.reindex(df_extended_scaled.index)
    extended_labels = extended_labels.reindex(df_extended_scaled.index)

    if baseline_labels.isna().any() or extended_labels.isna().any():
        raise ValueError("Some countries have no cluster assignment.")

    # Calculate one shared two-dimensional projection.
    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(df_extended_scaled)

    # Keep country identifiers attached to their coordinates.
    pca_scores = pd.DataFrame(
        coordinates,
        index=df_extended_scaled.index,
        columns=["PC1", "PC2"],
    )

    explained_percent = pca.explained_variance_ratio_ * 100

    # Match baseline labels to extended labels for consistent display colors.
    # This mapping follows the current three-cluster membership table.
    baseline_to_extended = {0: 0, 1: 2, 2: 1}
    baseline_colors = baseline_labels.map(baseline_to_extended)

    if baseline_colors.isna().any():
        raise ValueError("The display mapping does not cover all clusters.")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 6),
        sharex=True,
        sharey=True,
    )

    panels = [
        (
            "TB-only cluster assignments",
            baseline_labels,
            baseline_colors,
        ),
        (
            "TB-HIV cluster assignments",
            extended_labels,
            extended_labels,
        ),
    ]

    for ax, (title, labels, color_labels) in zip(axes, panels):
        for cluster_id in sorted(labels.unique()):
            mask = labels == cluster_id
            color_id = int(color_labels.loc[mask].iloc[0])

            ax.scatter(
                pca_scores.loc[mask, "PC1"],
                pca_scores.loc[mask, "PC2"],
                color=f"C{color_id}",
                label=f"Cluster {cluster_id} (n={int(mask.sum())})",
                s=40,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
            )

        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({explained_percent[0]:.1f}% variance)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    axes[0].set_ylabel(
        f"PC2 ({explained_percent[1]:.1f}% variance)"
    )

    fig.suptitle(
        "Country groupings before and after adding HIV\n"
        "Shared PCA projection of TB-HIV features",
        y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    print("\nPCA explained variance:")
    print(f"PC1: {explained_percent[0]:.2f}%")
    print(f"PC2: {explained_percent[1]:.2f}%")
    print(f"Total for the two components: {explained_percent.sum():.2f}%")

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    
    return pca_scores, pca

