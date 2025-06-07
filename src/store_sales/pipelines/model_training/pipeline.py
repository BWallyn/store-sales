"""This is a boilerplate pipeline 'model_training' generated using Kedro 0.19.12"""

from kedro.pipeline import Pipeline, node, pipeline

from store_sales.pipelines.model_training.nodes import (
    create_lag_features,
    create_or_get_mlflow_experiment,
    create_rolling_mean_features,
    merge_train_test,
    split_train_validation_test,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the model training pipeline."""
    return pipeline(
        pipe=[
            node(
                func=merge_train_test,
                inputs=["df_train_feature_engineered", "df_test_feature_engineered"],
                outputs="df_engineered",
                name="Merge_train_test_data",
            ),
            node(
                func=create_lag_features,
                inputs=["df_engineered", "params:lags"],
                outputs="df_lag_features",
                name="Create_lag_features"
            ),
            node(
                func=create_rolling_mean_features,
                inputs=["df_lag_features", "params:lags"],
                outputs="df_rolling_mean",
                name="Create_rolling_mean_features",
            ),
            node(
                func=create_or_get_mlflow_experiment,
                inputs=[
                    "params:mlflow_experiment_id_saved",
                    "params:mlflow_experiment_folder",
                    "params:mlflow_experiment_name",
                ],
                outputs="mlflow_experiment_id",
                name="Create_or_retrieve_mlflow_experiment",
            ),
            node(
                func=split_train_validation_test,
                inputs="df_rolling_mean",
                outputs=[
                    "df_train_model",
                    "df_valid_model",
                    "df_test_model",
                    "df_big_train_model",
                ],
                name="Split_train_validation_test_data",
            ),
            node(
                func=train_model,
                inputs=[
                    "df_train_model",
                    "df_valid_model",
                    "params:list_features",
                    "params:hgbm_params",
                    "mlflow_experiment_id",
                    "params:target_name",
                ],
                outputs="hgbm_model",
                name="Train_model",
            ),
        ],
        namespace="model_training",
        inputs=["df_train_feature_engineered", "df_test_feature_engineered"]
    )
