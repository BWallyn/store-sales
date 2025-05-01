"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.12
"""
# =================
# ==== IMPORTS ====
# =================

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
