"""This is a boilerplate pipeline 'model_training' generated using Kedro 0.19.12"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def simple_moving_average(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Create simple moving averages.

    Args:
        df (pd.DataFrame): The input DataFrame.
        lags (list[int]): List of lag values to create lag features.

    Returns:
        (pd.DataFrame): DataFrame with lag features added.
    """
    df_prep = df.sort_values(by=["store_nbr", "family", "date"])
    for lag in lags:
        df_prep["SMA" + str(lag) + "_sales_lag16"] = (
            df
            .groupby(["store_nbr", "family"])
            .rolling(lag)
            .sales.mean().shift(16).to_numpy()
        )
        df_prep["SMA" + str(lag) + "_sales_lag30"] = (
            df
            .groupby(["store_nbr", "family"])
            .rolling(lag)
            .sales.mean().shift(30).to_numpy()
        )
        df_prep["SMA" + str(lag) + "_sales_lag60"] = (
            df
            .groupby(["store_nbr", "family"])
            .rolling(lag)
            .sales.mean().shift(60).to_numpy()
        )
    return df_prep
