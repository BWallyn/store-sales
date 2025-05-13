"""This is a boilerplate pipeline 'feature_engineering' generated using Kedro 0.19.12"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from store_sales.pipelines.feature_engineering.nodes import (
    create_date_features,
    create_holidays_info,
    create_season_info,
    create_workday_info,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create pipeline for feature engineering."""
    return pipeline(
        pipe=[
            node(
                func=create_date_features,
                inputs="df_train_w_holidays",
                outputs="df_train_w_date_feats",
                name="Add_date_features_to_train",
            ),
            node(
                func=create_season_info,
                inputs="df_train_w_date_feats",
                outputs="df_train_w_season",
                name="Add_season_info_to_train",
            ),
            node(
                func=create_holidays_info,
                inputs="df_train_w_season",
                outputs="df_train_w_holidays_ind",
                name="Create_holidays_indicators_for_train",
            ),
            node(
                func=create_workday_info,
                inputs="df_train_w_holidays_ind",
                outputs="df_train_w_workday",
                name="Add_workday_info_to_train",
            ),
            node(
                func=create_date_features,
                inputs="df_test_w_holidays",
                outputs="df_test_w_date_feats",
                name="Add_date_features_to_test",
            ),
            node(
                func=create_season_info,
                inputs="df_test_w_date_feats",
                outputs="df_test_w_season",
                name="Add_season_info_to_test",
            ),
            node(
                func=create_holidays_info,
                inputs="df_test_w_season",
                outputs="df_test_w_holidays_ind",
                name="Create_holidays_indicators_for_test",
            ),
            node(
                func=create_workday_info,
                inputs="df_test_w_holidays_ind",
                outputs="df_test_w_workday",
                name="Add_workday_info_to_test",
            ),
        ],
        namespace="feature_engineering",
        inputs=["df_train_w_holidays", "df_test_w_holidays"],
        outputs=["df_train_w_workday", "df_test_w_workday"],
    )
