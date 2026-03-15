import statsmodels.api as sm
import pandas as pd
from datetime import date, timedelta
from dateutil.easter import easter
import holidays
import numpy as np

def get_season(dt, season_map_number):
    return season_map_number[dt.month]


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

def add_time_features(df, datetime_col, season_map_number):
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

    data["season"] = data[datetime_col].apply(lambda dt: get_season(dt, season_map_number))
    data["hour"] = data[datetime_col].dt.hour

    return data

def add_day_type(data, datetime_col, public_holidays, partial_holidays, bridge_days):
    out = data.copy()
    out["day_type"] = out[datetime_col].apply(
        lambda x: get_day_type(x, public_holidays, partial_holidays, bridge_days)
    )
    return out



def add_sdh_label(data):
    out = data.copy()
    out["sdh"] = (
        "S" + out["season"].astype(str)
        + "_"
        + out["day_type"].astype(str)
        + "_H" + out["hour"].astype(str)
    )
    return out


def build_local_time_effect(    df,    resid_col="resid_after_daytype",    date_col="Datum",    day_type_col="day_type",    hour_col="hour",    window_days=45,    year_length=366,):
    """
    Pour chaque combinaison (day_type, hour, day_year),
    calcule la médiane de resid_col sur les observations :
    - du même day_type
    - de la même heure
    - dans une fenêtre circulaire de +/- window_days autour du dayofyear

    Retourne un DataFrame avec :
    [day_type, hour, day_year, sdh_effect]
    """

    work = df[[date_col, day_type_col, hour_col, resid_col]].copy()
    work["day_year"] = work[date_col].dt.dayofyear

    effects = []

    for (day_type, hour), g in work.groupby([day_type_col, hour_col], observed=True):
        days = g["day_year"].to_numpy()
        vals = g[resid_col].to_numpy()

        for target_day in range(1, year_length + 1):
            dist = np.abs(days - target_day)
            circ_dist = np.minimum(dist, year_length - dist)
            mask = circ_dist <= window_days

            median_val = np.median(vals[mask]) if mask.any() else np.nan

            effects.append((day_type, hour, target_day, median_val))

    effects_df = pd.DataFrame(
        effects,
        columns=[day_type_col, hour_col, "day_year", "sdh_effect"]
    )

    return effects_df


def fit_median_dummy_sinusoidal_model(train_data, y_col="normalized_prices"):
    train = train_data.copy()

    # 1) day-type effects by median
    daytype_effects = train.groupby("day_type")[y_col].median()

    train["daytype_effect"] = train["day_type"].map(daytype_effects)
    train["resid_after_daytype"] = train[y_col] - train["daytype_effect"]

    # 2) sdh effects by median on residuals
        # 2) nouvel effet "sdh" = effet horaire local dans l'année
    sdh_effects = build_local_time_effect(
        df=train,
        resid_col="resid_after_daytype",
        date_col="Datum",
        day_type_col="day_type",
        hour_col="hour",        # ou quart_hour / slot si besoin
        window_days=45,
        year_length=366,
    )
    train["day_year"] = train["Datum"].dt.dayofyear
    train = train.merge(
        sdh_effects,
        on=["day_type", "hour", "day_year"],
        how="left"
    )
    
    train["resid_after_sdh"] = train["resid_after_daytype"] - train["sdh_effect"]

    # 3) sinusoidal part by OLS
    X_sin = sm.add_constant(train[["sin_year", "cos_year"]], has_constant="add")
    y_sin = train["resid_after_sdh"]

    sin_model = sm.OLS(y_sin, X_sin).fit()

    return {
        "daytype_effects": daytype_effects,
        "sdh_effects": sdh_effects,
        "sin_model": sin_model
    }


def predict_median_dummy_sinusoidal_model(model_dict, data):
    out = data.copy()

    daytype_effects = model_dict["daytype_effects"]
    sdh_effects = model_dict["sdh_effects"]
    sin_model = model_dict["sin_model"]

    out["day_year"] = out["Datum"].dt.dayofyear

    # map medians from train
    out["daytype_effect"] = out["day_type"].map(daytype_effects)
    
        # 2) effet sdh local via merge
    out = out.merge(
        sdh_effects,
        on=["day_type", "hour", "day_year"],
        how="left"
    )

    X_sin = sm.add_constant(out[["sin_year", "cos_year"]], has_constant="add")
    out["sin_effect"] = sin_model.predict(X_sin)

    out["y_pred"] = out["daytype_effect"] + out["sdh_effect"] + out["sin_effect"]

    return out


def benchmark(train_df):
    """
    Benchmark = median of normalized_prices per (month, weekday, hour)
    """
    return train_df.groupby(["month", "weekday", "hour"])["normalized_prices"].median()





def normalize_prices_by_season_year(df, season_map_number) -> pd.DataFrame:
    """
    Clean and normalize electricity prices by quarter of the year median.

    
    -------
    pd.DataFrame
        DataFrame with normalized prices.
    """

    df = df.copy()

    # Convert date column
    df["Datum"] = pd.to_datetime(df["Datum"])
    df["month"] = df["Datum"].dt.month
    df["year"] = df["Datum"].dt.year

    # Compute yearly_season median prices
    
    df["season"] = df["month"].apply(lambda x: season_map_number[x])
    norm_quarter_year = df.groupby(["season", "year"])["Preis"].median()

    df["normalized_median"] = df.set_index(["season", "year"]).index.map(norm_quarter_year)
    df["normalized_prices"] = df["Preis"] / df["normalized_median"]

    df.drop(columns=["season", "normalized_median"], inplace=True)
    
    return df
    
    


def fit_probabilistic_model(    train: pd.DataFrame,    y_col: str,    residual_group: str = "sdh",    quantiles=(0.1, 0.5, 0.9)
):
    """
    Fit the hybrid deterministic + probabilistic model.

    Deterministic part:
        1) day_type effect
        2) sdh effect on residuals (median)
        3) sinusoidal OLS on remaining residuals

    Probabilistic part:
        empirical residual quantiles conditional on `residual_group`

    Parameters
    ----------
    train : pd.DataFrame
        Training dataframe. Must contain:
        - y_col
        - day_type
        - sdh
        - sin_year
        - cos_year
    y_col : str
        Target column name.
    residual_group : str
        Group used for conditional residual distributions.
        Typical choices: "sdh", "day_type".
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
    # 2) sdh effect on residuals
    # --------------------------------------------------
    sdh_effects = train.groupby("sdh")["resid_after_daytype"].median()

    train["sdh_effect"] = train["sdh"].map(sdh_effects)
    train["resid_after_sdh"] = train["resid_after_daytype"] - train["sdh_effect"]

    # --------------------------------------------------
    # 3) sinusoidal part by OLS
    # --------------------------------------------------
    X_sin = sm.add_constant(train[["sin_year", "cos_year"]], has_constant="add")
    y_sin = train["resid_after_sdh"]

    sin_model = sm.OLS(y_sin, X_sin).fit()

    train["sin_effect"] = sin_model.predict(X_sin)

    # --------------------------------------------------
    # Final deterministic forecast on train
    # --------------------------------------------------
    train["y_hat_det"] = (
        train["daytype_effect"]
        + train["sdh_effect"]
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
        "sdh_effects": sdh_effects,
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
    sdh_effects = model["sdh_effects"]
    sin_model = model["sin_model"]
    residual_group = model["residual_group"]
    residual_quantiles = model["residual_quantiles"]
    global_residual_quantiles = model["global_residual_quantiles"]
    quantiles = model["quantiles"]

    # --------------------------------------------------
    # Deterministic components
    # --------------------------------------------------
    df_future["daytype_effect"] = df_future["day_type"].map(daytype_effects)
    df_future["sdh_effect"] = df_future["sdh"].map(sdh_effects)

    # Fallbacks for unseen categories
    #df_future["daytype_effect"] = df_future["daytype_effect"].fillna(0.0)
    #df_future["sdh_effect"] = df_future["sdh_effect"].fillna(0.0)

    X_sin_future = sm.add_constant(
        df_future[["sin_year", "cos_year"]],
        has_constant="add"
    )
    df_future["sin_effect"] = sin_model.predict(X_sin_future)

    df_future["y_hat_det"] = (
        df_future["daytype_effect"]
        + df_future["sdh_effect"]
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
    
    
    

def compute_season_shape(df):
    """
    this function is designed to reintroduce the seasonal component for the inference on year 2027
    take as input a df, compute the median of the cross year x quarter. compute the mean of this year.
    compute this ratio: median( quarter of year n) / mean year n
    take the average of these valeues on the df
    """
    season_median = (
        df.groupby(["year", "season"])["Preis"]
        .median()
        .rename("season_median")
        .reset_index()
    )

    annual_mean = (
        df.groupby("year")["Preis"]
        .mean()
        .rename("annual_mean")
        .reset_index()
    )

    ratios = season_median.merge(annual_mean, on="year")
    ratios["ratio_s_to_annual"] = ratios["season_median"] / ratios["annual_mean"]

    shape_ratios = (
        ratios.groupby("season")["ratio_s_to_annual"]
        .mean()
        .to_dict()
    )

    return shape_ratios