import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def standard_filter(df, price_col="residuals", col_output="filtered_std", col_tag = "is_outlier_std", threshold=3):
    df = df.copy()
    df["residuals_centered"] = (df[price_col] - df[price_col].mean()) / df[price_col].std()
    df[col_tag] = df["residuals_centered"].abs() >= threshold
    df[col_output] = df[price_col].where(~df[col_tag], np.nan)
    df.drop(columns = "residuals_centered", inplace = True)
    return df

def recursive_filter_prices( df, price_col="residuals", col_output="filtered_rfp", col_tag = "is_outlier_rfp", max_iter=20, threshold=3):
    """
    detect outliers with recursive filter.

    rules:
        |P_t - mean| > threshold * std

    at every iteration:
        - recompute mean/std on non outliers
        - add new outliers
        - stop when no more outliers

    """
    
    df = df.copy()
    is_outlier = pd.Series(False, index=df.index)

    prices = df[price_col]

    for i in range(max_iter):
        clean_prices = prices[~is_outlier]

        mu = clean_prices.mean()
        sigma = clean_prices.std()

        if pd.isna(sigma) or sigma == 0:
            break

        new_outliers = (np.abs(prices - mu) > threshold * sigma) & (~is_outlier)

        if not new_outliers.any():
            break

        is_outlier = is_outlier | new_outliers

    df[col_tag] = is_outlier
    df[col_output] = df[price_col].where(~df[col_tag], np.nan)

    return df
    
    
def plot_filter_comparisons(
    df,
    residual_col="residuals",
    std_flag_col="is_outlier_std",
    rfp_flag_col="is_outlier_rfp",
    std_filtered_col="filtered_std",
    rfp_filtered_col="filtered_rfp",
    year_col="year",
    day_type_col="day_type",
    sdh_col="sdh",
    top_n=5,
    show_tables=True,
    save_png=False,
    output_prefix="filter_comparison"
):
    """
    Compare 2 methods  filtering :
    - standard filter
    - recursive filter on prices

    display :
    1) résidus + outliers STD
    2) résidus + outliers RFP
    3) KDE of distributions (original / STD / RFP)
    4) synthetic table

    if save_png=True :
    - save the graph PNG
    - save summary in PNG
    """

    fig, axes = plt.subplots(3, 1, figsize=(20, 15))

    # -----------------------------
    # 1) Résidus + outliers STD
    # -----------------------------
    sns.lineplot(
        data=df,
        x=df.index,
        y=residual_col,
        ax=axes[0]
    )

    sns.scatterplot(
        data=df[df[std_flag_col]],
        x=df[df[std_flag_col]].index,
        y=residual_col,
        color="red",
        zorder=2,
        ax=axes[0]
    )

    axes[0].set_title("Residuals with STD outliers")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(residual_col)

    # -----------------------------
    # 2) Résidus + outliers RFP
    # -----------------------------
    sns.lineplot(
        data=df,
        x=df.index,
        y=residual_col,
        ax=axes[1]
    )

    sns.scatterplot(
        data=df[df[rfp_flag_col]],
        x=df[df[rfp_flag_col]].index,
        y=residual_col,
        color="red",
        zorder=2,
        ax=axes[1]
    )

    axes[1].set_title("Residuals with RFP outliers")
    axes[1].set_xlabel("")
    axes[1].set_ylabel(residual_col)

    # -----------------------------
    # 3) KDE distributions
    # -----------------------------
    sns.kdeplot(
        data=df,
        x=rfp_filtered_col,
        label="RFP filter",
        ax=axes[2]
    )

    sns.kdeplot(
        data=df,
        x=std_filtered_col,
        label="STD filter",
        ax=axes[2]
    )

    sns.kdeplot(
        data=df,
        x=residual_col,
        label="Original",
        ax=axes[2]
    )

    axes[2].set_title("Distribution comparison")
    axes[2].legend()

    plt.tight_layout()

    if save_png:
        fig.savefig(f"/content/data_exp/{output_prefix}_graphs.png", dpi=300, bbox_inches="tight")

    plt.show()

    # -----------------------------
    # 4) Summary table
    # -----------------------------
    summary = pd.DataFrame({
        "method": ["original", "STD filter", "RFP filter"],
        "count": [
            df[residual_col].count(),
            df[std_filtered_col].count(),
            df[rfp_filtered_col].count()
        ],
        "outliers": [
            0,
            df[std_flag_col].sum(),
            df[rfp_flag_col].sum()
        ],
        "max": [
            df[residual_col].max(),
            df[std_filtered_col].max(),
            df[rfp_filtered_col].max()
        ],
        "mean": [
            df[residual_col].mean(),
            df[std_filtered_col].mean(),
            df[rfp_filtered_col].mean()
        ],
        "std": [
            df[residual_col].std(),
            df[std_filtered_col].std(),
            df[rfp_filtered_col].std()
        ],
        "skew": [
            df[residual_col].skew(),
            df[std_filtered_col].skew(),
            df[rfp_filtered_col].skew()
        ],
        "kurt": [
            df[residual_col].kurt(),
            df[std_filtered_col].kurt(),
            df[rfp_filtered_col].kurt()
        ]
    })

    display(summary)

    if save_png:
        fig_table, ax_table = plt.subplots(figsize=(12, 2))
        ax_table.axis("off")

        summary_to_plot = summary.copy()
        for col in summary_to_plot.select_dtypes(include="number").columns:
            summary_to_plot[col] = summary_to_plot[col].round(4)

        table = ax_table.table(
            cellText=summary_to_plot.values,
            colLabels=summary_to_plot.columns,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        plt.tight_layout()
        fig_table.savefig(f"/content/data_exp/{output_prefix}_summary.png", dpi=300, bbox_inches="tight")
        plt.close(fig_table)

    if not show_tables:
        return summary

    outputs = {"summary": summary}

    # helper
    def add_percent_columns(table, std_col="std_outliers", rfp_col="rfp_outliers"):
        std_total = table[std_col].sum()
        rfp_total = table[rfp_col].sum()

        table["std_%"] = (
            100 * table[std_col] / std_total
        ).round(0) if std_total != 0 else 0

        table["rfp_%"] = (
            100 * table[rfp_col] / rfp_total
        ).round(0) if rfp_total != 0 else 0

        return table

    # -----------------------------
    # 5) Outliers by year
    # -----------------------------
    if year_col in df.columns:
        outliers_by_year = (
            df.groupby(year_col)
              .agg(
                  std_outliers=(std_flag_col, "sum"),
                  rfp_outliers=(rfp_flag_col, "sum")
              )
        )

        outliers_by_year = add_percent_columns(outliers_by_year)
        display(outliers_by_year)
        outputs["outliers_by_year"] = outliers_by_year

    # -----------------------------
    # 6) Outliers by day type
    # -----------------------------
    if day_type_col in df.columns:
        outliers_by_day_type = (
            df.groupby(day_type_col)
              .agg(
                  std_outliers=(std_flag_col, "sum"),
                  rfp_outliers=(rfp_flag_col, "sum"),
                  total_obs=(day_type_col, "size")
              )
        )

        outliers_by_day_type = add_percent_columns(outliers_by_day_type)
        display(outliers_by_day_type)
        outputs["outliers_by_day_type"] = outliers_by_day_type

    # -----------------------------
    # 7) Outliers by SDH
    # -----------------------------
    if sdh_col in df.columns:
        outliers_by_sdh = (
            df.groupby(sdh_col)
              .agg(
                  std_outliers=(std_flag_col, "sum"),
                  rfp_outliers=(rfp_flag_col, "sum"),
                  total_obs=(sdh_col, "size")
              )
        )

        outliers_by_sdh = add_percent_columns(outliers_by_sdh)
        outputs["outliers_by_sdh"] = outliers_by_sdh

        top_std = (
            outliers_by_sdh
            .sort_values("std_outliers", ascending=False)
            .head(top_n)
        )

        top_rfp = (
            outliers_by_sdh
            .sort_values("rfp_outliers", ascending=False)
            .head(top_n)
        )

        display(top_std)
        display(top_rfp)

        outputs["top_std"] = top_std
        outputs["top_rfp"] = top_rfp

    return outputs