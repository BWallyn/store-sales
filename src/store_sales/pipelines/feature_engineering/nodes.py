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
