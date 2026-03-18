import matplotlib.pyplot as plt
import pandas as pd

import os

def plot_model_decomposition_overlay(datasets_dict, style_map):

    # create figures folder if it does not exist
    os.makedirs("figures", exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16,12))

    # -----------------------------
    # 1) Seasonal effect (full year)
    # -----------------------------
    for name, df in datasets_dict.items():

        style = style_map[name]

        axes[0].plot(
            df.index,
            df["sin_effect"],
            label=name,
            color=style["color"],
            linestyle=style["linestyle"]
        )

    axes[0].set_title("Seasonal effect comparison")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # -----------------------------
    # 2) Day-type effect (2 weeks)
    # -----------------------------
    for name, df in datasets_dict.items():

        style = style_map[name]

        start = df.index.min()
        end = start + pd.Timedelta(days=14)
        df_2w = df.loc[(df.index >= start) & (df.index < end)]

        axes[1].plot(
            df_2w.index,
            df_2w["daytype_effect"],
            label=name,
            color=style["color"],
            linestyle=style["linestyle"]
        )

    axes[1].set_title("Day-type effect comparison (first 2 weeks)")
    axes[1].grid(alpha=0.3)

    # -----------------------------
    # 3) SDH effect (1 day)
    # -----------------------------
    for name, df in datasets_dict.items():

        style = style_map[name]

        start = df.index.min()
        end = start + pd.Timedelta(days=1)
        df_1d = df.loc[(df.index >= start) & (df.index < end)]

        axes[2].plot(
            df_1d.index,
            df_1d["sdh_effect"],
            label=name,
            color=style["color"],
            linestyle=style["linestyle"]
        )

    axes[2].set_title("Intraday effect comparison (first day)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    # save figure
    fig.savefig("/content/figures/decomposition_full_year.png", dpi=300, bbox_inches="tight")

    plt.show()
    
def from_decompo_to_pred(datasets, style_map):

    start = 24*6
    end   = 24*13

    time = datasets["big_rfp"].index[start:end]

    fig, axes = plt.subplots(3, 1, figsize=(14,10), sharex=True)

    # -----------------------------
    # 1) Components comparison
    # -----------------------------
    for name, df in datasets.items():

        style = style_map[name]

        axes[0].plot(
            time,
            df["daytype_effect"][start:end],
            label=f"daytype ({name})",
            color=style["color"],
            linestyle=style["linestyle"]
        )

        axes[0].plot(
            time,
            df["sdh_effect"][start:end],
            label=f"sdh ({name})",
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.6
        )

        axes[0].plot(
            time,
            df["sin_effect"][start:end],
            label=f"sin ({name})",
            color=style["color"],
            linestyle=":"
        )

    axes[0].set_title("Model components comparison")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.3)


    # -----------------------------
    # 2) Prediction comparison
    # -----------------------------
    for name, df in datasets.items():

        style = style_map[name]

        axes[1].plot(
            time,
            df["y_pred"][start:end],
            label=f"prediction ({name})",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )

    axes[1].plot(
        time,
        datasets["big_rfp"]["benchmark_y_pred"][start:end],
        label="Benchmark",
        color="black",
        linestyle=":"
    )

    axes[1].set_title("Model predictions comparison")
    axes[1].legend()
    axes[1].grid(alpha=0.3)


    # -----------------------------
    # 3) Observed vs predictions
    # -----------------------------
    axes[2].plot(
        time,
        datasets["big_rfp"]["normalized_prices"][start:end],
        label="Observed",
        color="black"
    )

    for name, df in datasets.items():

        style = style_map[name]

        axes[2].plot(
            time,
            df["y_pred"][start:end],
            label=f"{name}",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )

    axes[2].set_title("Observed vs predictions")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle("Price decomposition comparison – Week of Jan 7, 2019", fontsize=14)

    plt.tight_layout()

    plt.savefig("/content/figures/decomposition_week.png", dpi=300)

    plt.show()

def plot_price_duration_curve(datasets_dict, observed_series, benchmark_series, style_map):


    plt.figure(figsize=(10,6))

    # --- observed
    prices = observed_series.sort_values(ascending=False).reset_index(drop=True)
    percent = prices.index / len(prices) * 100
    plt.plot(percent, prices, color="black", label="Observed")

    # --- models
    for name, df in datasets_dict.items():

        style = style_map[name]

        prices = df["y_pred"].sort_values(ascending=False).reset_index(drop=True)
        percent = prices.index / len(prices) * 100

        plt.plot(
            percent,
            prices,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"]
        )

    # --- benchmark
    prices = benchmark_series.sort_values(ascending=False).reset_index(drop=True)
    percent = prices.index / len(prices) * 100
    plt.plot(percent, prices, color="grey", linestyle=":", label="Benchmark")

    plt.xlabel("Percentage of time (%)")
    plt.ylabel("Normalized price")
    plt.title("Price Duration Curve")
    
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_test_weeks(datasets_dict, observed_series, benchmark_series, weeks_to_plot, style_map):

    # add column week
    for df in datasets_dict.values():
        df["week"] = df.index.isocalendar().week

    for w in weeks_to_plot:

        plt.figure(figsize=(20,7))

        # observed
        subset_obs = observed_series[observed_series.index.isocalendar().week == w]
        plt.plot(subset_obs.index, subset_obs, color="black", label="Observed")

        # models
        for name, df in datasets_dict.items():

            style = style_map[name]

            subset = df[df["week"] == w]

            plt.plot(
                subset.index,
                subset["y_pred"],
                label=name,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2
            )

        # benchmark
        subset_bench = benchmark_series[benchmark_series.index.isocalendar().week == w]

        plt.plot(
            subset_bench.index,
            subset_bench,
            color="grey",
            linestyle=":",
            label="Benchmark"
        )

        plt.title(f"Test Week {w}")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(f"/content/figures/Test_Week_{w}.png", dpi=300)

        plt.show()

def plot_hourly_profile(datasets, style_map, ):
    """
    Return a graph for the 24 hours average predicted price.
    This graph is run on the 4 methods to compare peaks and off-peaks.
    """

    plt.figure(figsize=(10,6))

    for name, df in datasets.items():

        style = style_map[name]

        profile = (
            df.groupby("hour")["y_pred"]
            .mean()
        )

        plt.plot(
            profile.index,
            profile.values,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2
        )


        # Add normalized_price from big_rfp
    ref_profile = datasets["big_rfp"].groupby("hour")["normalized_prices"].mean()

    plt.plot(
        ref_profile.index,
        ref_profile.values,
        label="Observed",
        color="black",
        linewidth=1
    )

    plt.xlabel("Hour")
    plt.ylabel("Average predicted price")
    plt.title("Average hourly profile")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig("/content/figures/hourly_profile.png")

    plt.show()

def plot_season_profile(datasets, style_map):

    plt.figure(figsize=(10,6))

    seasons_order = [1, 2, 3, 4]

    for name, df in datasets.items():

        style = style_map[name]

        profile = (
            df.groupby("season")["y_pred"]
            .mean()
            .reindex(seasons_order)
        )

        plt.plot(
            profile.index,
            profile.values,
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            linewidth=2
        )

    ref_profile = (
        datasets["big_rfp"]
        .groupby("season")["normalized_prices"]
        .mean()
        .reindex(seasons_order)
    )

    plt.plot(
        ref_profile.index,
        ref_profile.values,
        label="Observed",
        color="black",
        marker="o",
        linewidth=2
    )

    plt.xlabel("Season")
    plt.ylabel("Average predicted price")
    plt.title("Average seasonal profile")
    plt.xticks(seasons_order)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("/content/figures/season_profile.png")

    plt.show()