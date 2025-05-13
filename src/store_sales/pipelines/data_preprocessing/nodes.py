"""This is a boilerplate pipeline 'data_preprocessing' generated using Kedro 0.19.12"""
# =================
# ==== IMPORTS ====
# =================

from typing import Optional

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def merge_transactions(df: pd.DataFrame, df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Merge transactions to the datframe based on date and store number.

    Args:
        df (pd.DataFrame): Dataframe to merge transactions to.
        df_transactions (pd.DataFrame): Dataframe containing transactions.

    Returns:
        (pd.DataFrame): Merged DataFrame.
    """
    return df.merge(df_transactions, on=["date", "store_nbr"], how="left")


def merge_stores(df: pd.DataFrame, df_stores: pd.DataFrame) -> pd.DataFrame:
    """Merge stores to the datframe based on store number.

    Args:
        df (pd.DataFrame): Dataframe to merge stores to.
        df_stores (pd.DataFrame): Dataframe containing stores.

    Returns:
        (pd.DataFrame): Merged DataFrame.
    """
    return df.merge(df_stores, on="store_nbr", how="left")


def merge_on_date(df: pd.DataFrame, df_to_merge: pd.DataFrame, suffix_y: Optional[str]=None) -> pd.DataFrame:
    """Merge dataframe to the datframe based on date.

    Args:
        df (pd.DataFrame): Dataframe to merge using date column.
        df_to_merge (pd.DataFrame): Dataframe to merge.
        suffix_y (Optional[str]): Suffix to add to the columns of the merged dataframe.

    Returns:
        (pd.DataFrame): Merged DataFrame.
    """
    return df.merge(df_to_merge, on="date", how="left", suffixes=("", suffix_y if suffix_y else ""))

