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


def exponential_moving_average(df: pd.DataFrame, alphas: list[float], lags: list[int]) -> pd.DataFrame:
    """Create exponential moving averages.

    Args:
        df (pd.DataFrame): The input DataFrame.
        alphas (list[float]): List of alpha values for exponential moving averages.
        lags (list[int]): List of lag values to create lag features.

    Returns:
        (pd.DataFrame): DataFrame with lag features added.
    """
    df_prep = df.sort_values(by=["store_nbr", "family", "date"])
    for alpha in alphas:
        for lag in lags:
            df_prep['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = (
                df_prep
                .groupby(["store_nbr", "family"])['sales']
                .transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())  # noqa: B023
            )
    return df_prep


def merge_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """Merge train and test DataFrames.

    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_test (pd.DataFrame): The testing DataFrame.

    Returns:
        (pd.DataFrame): Merged DataFrame with train and test data.
    """
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    return pd.concat([df_train, df_test], ignore_index=True)


def create_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Create lag features for the sales data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        lags (list[int]): List of lag values to create lag features.

    Returns:
        (pd.DataFrame): DataFrame with lag features added.
    """
    df_prep = df.sort_values(by=["store_nbr", "family", "date"])
    for lag in lags:
        df_prep["sales_lag_" + str(lag)] = (
            df
            .groupby(["store_nbr", "family"])['sales']
            .shift(lag)
        )
    return df_prep


def create_rolling_mean_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Create rolling mean features for the sales data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        lags (list[int]): List of the lag features to create rolling means for.

    Returns:
        (pd.DataFrame): DataFrame with rolling mean features added.
    """
    df_prep = df.sort_values(by=["store_nbr", "family", "date"])
    for lag in lags:
        df_prep["rolling_mean_" + str(lag)] = (
            df
            .groupby(["store_nbr", "family"])['sales']
            .transform(lambda x: x.shift(1).rolling(window=lag).mean())  # noqa: B023
        )
    return df_prep

