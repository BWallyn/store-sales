"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from store_sales.pipelines.data_preprocessing.nodes import (
    merge_transactions,
    merge_stores,
    merge_on_date,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=merge_transactions,
                inputs=["df_train", "df_transactions"],
                outputs="df_train_w_transactions",
                name="Merge_transactions_info_to_train",
            ),
            node(
                func=merge_stores,
                inputs=["df_train_w_transactions", "df_stores"],
                outputs="df_train_w_stores",
                name="Merge_stores_info_to_train",
            ),
            node(
                func=merge_on_date,
                inputs=["df_train_w_stores", "df_oil"],
                outputs="df_train_w_oil",
                name="Merge_oil_info_to_train",
            ),
            node(
                func=merge_on_date,
                inputs=["df_train_w_oil", "df_holidays"],
                outputs="df_train_w_holidays",
                name="Merge_holidays_info_to_train",
            ),
            node(
                func=merge_stores,
                inputs=["df_test", "df_stores"],
                outputs="df_test_w_stores",
                name="Merge_stores_info_to_test",
            ),
            node(
                func=merge_on_date,
                inputs=["df_test_w_stores", "df_holidays", "params:suffix_holidays"],
                outputs="df_test_w_holidays",
                name="Merge_holidays_info_to_test",
            ),
        ],
        inputs=["df_train", "df_test", "df_transactions", "df_stores", "df_oil", "df_holidays"],
        outputs=["df_train_w_holidays", "df_test_w_holidays"],
        namespace="data_preprocessing",
    )
