"""This is a boilerplate pipeline 'feature_engineering' generated using Kedro 0.19.12"""
# =================
# ==== IMPORTS ====
# =================

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date features from the 'date' column in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with additional date features.
    """
    return df.assign(
        date = lambda x: pd.to_datetime(x['date']),
        year = lambda x: x["date"].dt.year,
        month = lambda x: x['date'].dt.month,
        day_of_month = lambda x: x["date"].dt.day,
        day_of_week = lambda x: x["date"].dt.dayofweek + 1,
        day_of_year = lambda x: x["date"].dt.dayofyear,
        week_of_year = lambda x: x["date"].dt.isocalendar().week,
        week_of_month = lambda x: (x["date"].dt.day - 1) // 7 + 1,
        quarter = lambda x: x["date"].dt.quarter,
        is_month_start = lambda x: x["date"].dt.is_month_start,
        is_month_end = lambda x: x["date"].dt.is_month_end,
        is_quarter_start = lambda x: x["date"].dt.is_quarter_start,
        is_quarter_end = lambda x: x["date"].dt.is_quarter_end,
        is_year_start = lambda x: x["date"].dt.is_year_start,
        is_year_end = lambda x: x["date"].dt.is_year_end,
    )


def create_season_info(df: pd.DataFrame) -> pd.DataFrame:
    """Create season information based on the month.

    The function maps each month to a corresponding season:
    - Winter: January, February, December (0)
    - Spring: March, April, May (1)
    - Summer: June, July, August (2)
    - Autumn: September, October, November (3)

    Args:
        df (pd.DataFrame): Input DataFrame containing a month column.

    Returns:
        (pd.DataFrame): DataFrame with additional season information.
    """
    return df.assign(
        season = lambda x: x['month'].map({
            1: 0, 2: 0,
            3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2,
            9: 3, 10: 3, 11: 3,
            12: 0
        })
    )


def create_workday_info(df: pd.DataFrame) -> pd.DataFrame:
    """Create workday information base on the day of the week and the holidays."""
    return df.assign(
        is_workday = lambda x: 0 if (
            x["day_of_week"].isin([6, 7])
            | x["holiday_national_binary"] == 1
            | x["holiday_local_binary"] == 1
            | x["holiday_regional_binary"] == 1
        ) else 1,
    )
