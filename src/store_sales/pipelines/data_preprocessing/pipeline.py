"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from store_sales.pipelines.data_preprocessing.nodes import (
    merge_transactions
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=merge_transactions,
                inputs=["df_train", "df_transactions"],
                outputs="df_train_w_transactions",
                name="Merge_transactions_to_train",
            ),
        ],
        inputs=["df_train", "df_transactions"],
    )
