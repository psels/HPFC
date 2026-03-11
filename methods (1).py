import statsmodels.api as sm
import pandas as pd
from datetime import date, timedelta
from dateutil.easter import easter
import holidays
import numpy as np

def get_quarter(dt):
    """Return quarter in {1,2,3,4}."""
    return ((dt.month - 1) // 3) + 1


def repentance_day(year):
    """
    Day of Prayer and Repentance = Wednesday before November 23.
    """
    nov23 = date(year, 11, 23)
    # Monday=0 ... Sunday=6, Wednesday=2
    offset = (nov23.weekday() - 2) % 7
    return nov23 - timedelta(days=offset if offset != 0 else 7)


def christmas_week_dates(year):
    """
    give the dates of the Christmas week for a specific year
    """
    start = date(year, 12, 24)
    end = date(year, 12, 31)
    return {start + timedelta(days=i) for i in range((end - start).days + 1)}

def build_german_day_sets(years):
    """
    Build three sets:
      - public_holidays
      - partial_holidays
      - bridge_days
    """
    if isinstance(years, int):
        years = [years]

    public_holidays = set()
    partial_holidays = set()

    for year in years:
        easter_sunday = easter(year)
        pentecost_sunday = easter_sunday + timedelta(days=49)

        public_holidays |= {
            date(year, 1, 1),
            date(year, 5, 1),
            date(year, 10, 3),
            date(year, 12, 25),
            date(year, 12, 26),
            easter_sunday - timedelta(days=2),
            easter_sunday,
            easter_sunday + timedelta(days=1),#eastern Monday
            easter_sunday + timedelta(days=39),#ascension
            easter_sunday + timedelta(days=50),#Pentecost Monday
        }

        partial_holidays |= {
            pentecost_sunday,
            repentance_day(year),
            *christmas_week_dates(year),
            date(year, 8, 15), # Asumption day
            date(year, 10, 31),# Reformation day
            date(year, 11, 1), # all saint's day
        }

    partial_holidays -= public_holidays

    bridge_days = set()
    all_public = sorted(public_holidays)

    # 1) weekdays between two nearby public holidays only
    for i in range(len(all_public) - 1):
        left = all_public[i]
        right = all_public[i + 1]

        gap = (right - left).days - 1  # number of days strictly between them

        # only keep "small gaps"
        if 1 <= gap <= 3:
            current = left + timedelta(days=1)
            while current < right:
                if current.weekday() < 5:  # Mon-Fri only
                    bridge_days.add(current)
                current += timedelta(days=1)

    # 2) Friday if Thursday is holiday
    for h in public_holidays:
        if h.weekday() == 3:
            friday = h + timedelta(days=1)
            if friday.weekday() == 4:
                bridge_days.add(friday)

    # 3) Monday if Tuesday is holiday
    for h in public_holidays:
        if h.weekday() == 1:
            monday = h - timedelta(days=1)
            if monday.weekday() == 0:
                bridge_days.add(monday)

    bridge_days -= public_holidays
    bridge_days -= partial_holidays

    return public_holidays, partial_holidays, bridge_days


def get_day_type(dt, public_holidays, partial_holidays, bridge_days):

    d = dt.date()
    wd = dt.weekday()   # Monday=0 ... Sunday=6

    # group 5
    if d in public_holidays or wd == 6:
        return "sun_holiday"

    # group 4
    elif wd == 5 or d in partial_holidays or d in bridge_days:
        return "sat_partial_bridge"

    # group 3
    elif wd == 4:
        return "friday"

    # group 2
    elif wd in [1,2,3]:
        return "tue_wed_thu"

    # group 1
    else:
        return "monday"

def add_time_features(df, datetime_col):
    data = df.copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col).reset_index(drop=True)

    dt = data[datetime_col]
    year_start = pd.to_datetime(dt.dt.year.astype(str) + "-01-01 00:00:00")
    next_year_start = pd.to_datetime((dt.dt.year + 1).astype(str) + "-01-01 00:00:00")

    hours_since_year_start = (dt - year_start).dt.total_seconds() / 3600
    hours_in_year = (next_year_start - year_start).dt.total_seconds() / 3600

    phase = 2 * np.pi * hours_since_year_start / hours_in_year

    data["sin_year"] = np.sin(phase)
    data["cos_year"] = np.cos(phase)

    data["quarter"] = data[datetime_col].apply(get_quarter)
    data["hour"] = data[datetime_col].dt.hour

    return data

def add_day_type(data, datetime_col, public_holidays, partial_holidays, bridge_days):
    out = data.copy()
    out["day_type"] = out[datetime_col].apply(
        lambda x: get_day_type(x, public_holidays, partial_holidays, bridge_days)
    )
    return out



def add_qdh_label(data):
    out = data.copy()
    out["qdh"] = (
        "Q" + out["quarter"].astype(str)
        + "_"
        + out["day_type"].astype(str)
        + "_H" + out["hour"].astype(str)
    )
    return out




def fit_median_dummy_sinusoidal_model(train_data, y_col="normalized_prices"):
    train = train_data.copy()

    # 1) day-type effects by median
    daytype_effects = train.groupby("day_type")[y_col].median()

    train["daytype_effect"] = train["day_type"].map(daytype_effects)
    train["resid_after_daytype"] = train[y_col] - train["daytype_effect"]

    # 2) qdh effects by median on residuals
    qdh_effects = train.groupby("qdh")["resid_after_daytype"].median()

    train["qdh_effect"] = train["qdh"].map(qdh_effects)
    train["resid_after_qdh"] = train["resid_after_daytype"] - train["qdh_effect"]

    # 3) sinusoidal part by OLS
    X_sin = sm.add_constant(train[["sin_year", "cos_year"]], has_constant="add")
    y_sin = train["resid_after_qdh"]

    sin_model = sm.OLS(y_sin, X_sin).fit()

    return {
        "daytype_effects": daytype_effects,
        "qdh_effects": qdh_effects,
        "sin_model": sin_model
    }


def predict_median_dummy_sinusoidal_model(model_dict, data):
    out = data.copy()

    daytype_effects = model_dict["daytype_effects"]
    qdh_effects = model_dict["qdh_effects"]
    sin_model = model_dict["sin_model"]

    # map medians from train
    out["daytype_effect"] = out["day_type"].map(daytype_effects)
    out["qdh_effect"] = out["qdh"].map(qdh_effects)

    # fallback if unseen category in test
    #global_daytype_default = 0.0
    #global_qdh_default = 0.0

    #out["daytype_effect"] = out["daytype_effect"].fillna(global_daytype_default)
    #out["qdh_effect"] = out["qdh_effect"].fillna(global_qdh_default)

    X_sin = sm.add_constant(out[["sin_year", "cos_year"]], has_constant="add")
    out["sin_effect"] = sin_model.predict(X_sin)

    out["y_pred"] = out["daytype_effect"] + out["qdh_effect"] + out["sin_effect"]

    return out


def benchmark(train_df):
    """
    Benchmark = median of normalized_prices per (month, weekday, hour)
    """
    return train_df.groupby(["month", "weekday", "hour"])["normalized_prices"].median()





def normalize_prices_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize electricity prices by yearly median.

    Steps
    -----
    1. Remove extreme negative prices (< -200).
    2. Convert 'Datum' to datetime.
    3. Extract the year.
    4. Remove years 2021 and 2022.
    5. Compute yearly median prices.
    6. Normalize prices by the yearly median.

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized prices.
    """

    df = df.copy()

    # Remove extreme negative prices
    df = df[df["Preis"] > -200]

    # Convert date column
    df["Datum"] = pd.to_datetime(df["Datum"])

    # Extract year
    df["year"] = df["Datum"].dt.year

    # Remove abnormal years
    df = df[~df["year"].isin([2021, 2022])]

    # Compute yearly median prices
    yearly_medians = df.groupby("year")["Preis"].median()

    # Normalize prices
    df["normalized_prices"] = df["Preis"] / df["year"].map(yearly_medians)

    # Drop unnecessary columns
    df = df.drop(columns=["Preis"])

    return df
    
    


def fit_probabilistic_model(
    train: pd.DataFrame,
    y_col: str,
    residual_group: str = "qdh",
    quantiles=(0.1, 0.5, 0.9)
):
    """
    Fit the hybrid deterministic + probabilistic model.

    Deterministic part:
        1) day_type effect
        2) qdh effect on residuals (median)
        3) sinusoidal OLS on remaining residuals

    Probabilistic part:
        empirical residual quantiles conditional on `residual_group`

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe. Must contain:
        - y_col
        - day_type
        - qdh
        - sin_year
        - cos_year
    y_col : str
        Target column name.
    residual_group : str
        Group used for conditional residual distributions.
        Typical choices: "qdh", "day_type".
    quantiles : tuple
        Quantiles to estimate on final residuals.

    Returns
    -------
    dict
        Fitted model components.
    """
    train = train.copy()

    # --------------------------------------------------
    # 1) day_type effect
    # --------------------------------------------------
    daytype_effects = train.groupby("day_type")[y_col].median()

    train["daytype_effect"] = train["day_type"].map(daytype_effects)
    train["resid_after_daytype"] = train[y_col] - train["daytype_effect"]

    # --------------------------------------------------
    # 2) qdh effect on residuals
    # --------------------------------------------------
    qdh_effects = train.groupby("qdh")["resid_after_daytype"].median()

    train["qdh_effect"] = train["qdh"].map(qdh_effects)
    train["resid_after_qdh"] = train["resid_after_daytype"] - train["qdh_effect"]

    # --------------------------------------------------
    # 3) sinusoidal part by OLS
    # --------------------------------------------------
    X_sin = sm.add_constant(train[["sin_year", "cos_year"]], has_constant="add")
    y_sin = train["resid_after_qdh"]

    sin_model = sm.OLS(y_sin, X_sin).fit()

    train["sin_effect"] = sin_model.predict(X_sin)

    # --------------------------------------------------
    # Final deterministic forecast on train
    # --------------------------------------------------
    train["y_hat_det"] = (
        train["daytype_effect"]
        + train["qdh_effect"]
        + train["sin_effect"]
    )

    # Final residuals
    train["final_residual"] = train[y_col] - train["y_hat_det"]

    # --------------------------------------------------
    # Residual quantiles by chosen group
    # --------------------------------------------------
    residual_quantiles = (
        train.groupby(residual_group)["final_residual"]
        .quantile(quantiles)
        .unstack()
    )

    # Global fallback quantiles if group unseen in test
    global_residual_quantiles = train["final_residual"].quantile(quantiles)

    return {
        "daytype_effects": daytype_effects,
        "qdh_effects": qdh_effects,
        "sin_model": sin_model,
        "residual_group": residual_group,
        "residual_quantiles": residual_quantiles,
        "global_residual_quantiles": global_residual_quantiles,
        "quantiles": quantiles,
        "train_fitted": train,
    }


def predict_probabilistic(model, df_future: pd.DataFrame) -> pd.DataFrame:
    """
    Produce deterministic and probabilistic forecasts on future data.

    Returns a dataframe containing:
    - deterministic forecast
    - predictive quantiles
    """
    df_future = df_future.copy()

    daytype_effects = model["daytype_effects"]
    qdh_effects = model["qdh_effects"]
    sin_model = model["sin_model"]
    residual_group = model["residual_group"]
    residual_quantiles = model["residual_quantiles"]
    global_residual_quantiles = model["global_residual_quantiles"]
    quantiles = model["quantiles"]

    # --------------------------------------------------
    # Deterministic components
    # --------------------------------------------------
    df_future["daytype_effect"] = df_future["day_type"].map(daytype_effects)
    df_future["qdh_effect"] = df_future["qdh"].map(qdh_effects)

    # Fallbacks for unseen categories
    df_future["daytype_effect"] = df_future["daytype_effect"].fillna(0.0)
    df_future["qdh_effect"] = df_future["qdh_effect"].fillna(0.0)

    X_sin_future = sm.add_constant(
        df_future[["sin_year", "cos_year"]],
        has_constant="add"
    )
    df_future["sin_effect"] = sin_model.predict(X_sin_future)

    df_future["y_hat_det"] = (
        df_future["daytype_effect"]
        + df_future["qdh_effect"]
        + df_future["sin_effect"]
    )

    # --------------------------------------------------
    # Add conditional residual quantiles
    # --------------------------------------------------
    for q in quantiles:
        q_name = f"q_{int(q*100)}"

        df_future[q_name] = df_future[residual_group].map(
            residual_quantiles[q]
        )

        # fallback if unseen group
        df_future[q_name] = df_future[q_name].fillna(global_residual_quantiles.loc[q])

        # predictive quantile = deterministic forecast + residual quantile
        df_future[f"y_hat_{q_name}"] = df_future["y_hat_det"] + df_future[q_name]

    return df_future