"""This is a boilerplate pipeline 'model_training' generated using Kedro 0.19.12"""
# =================
# ==== IMPORTS ====
# =================

import logging
from typing import Optional

import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

from store_sales.pipelines.model_training.mlflow import (
    _log_mlflow_metric,
    _log_model_mlflow,
    create_mlflow_experiment,
)

# Options
logger = logging.getLogger(__name__)

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
    df_train['is_big_train'] = 1
    df_test['is_big_train'] = 0
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


def create_or_get_mlflow_experiment(
    experiment_id: Optional[str]=None,
    experiment_folder: Optional[str]=None,
    experiment_name: Optional[str]=None,
) -> str:
    """Create an MLflow experiment or use an existing one.
    If experiment_id is not None, use the MLflow experiment of the experiment_id,
    otherwise, create a MLflow experiment.

    Args:
        experiment_id (Optional[str]): Experiment id if exists to reuse one
        experiment_folder (Optional[str]): Folder where to create the experiment
        experiment_name (Optional[str]): Name of the MLflow experiment
    Returns:
        experiment_id (str): Id of the MLflow experiment
    """
    if experiment_id is not None:
        logger.info("Using MLflow experiment id %s", experiment_id)
    else:
        experiment_id = create_mlflow_experiment(
            experiment_folder, experiment_name
        )
        logger.info("Creating MLflow experiment %s", experiment_id)
    return experiment_id


def _train_hgbm(
    df_train: pd.DataFrame,
    list_features: list[str],
    params: dict[str, any],
    target_name: str = "sales",
) -> HistGradientBoostingRegressor:
    """Train a HistGradientBoostingRegressor model.

    Args:
        df_train (pd.DataFrame): The training DataFrame.
        list_features (list[str]): List of feature names to use for training.
        params (dict[str, any]): Parameters for the HistGradientBoostingRegressor model.
        target_name (str): The name of the target variable.

    Returns:
        (HistGradientBoostingRegressor): The trained model.
    """
    model = HistGradientBoostingRegressor(**params)
    model.fit(df_train[list_features], df_train[target_name])
    return model


def _compute_metrics(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    list_features: list[str],
    model: HistGradientBoostingRegressor,
    target_name: str = "sales",
) -> dict[str, any]:
    """Compute metrics for the model on the training and evaluation DataFrames.

    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_eval (pd.DataFrame): The evaluation DataFrame.
        list_features (list[str]): List of feature names to use for training.
        model (HistGradientBoostingRegressor): The trained model.
        target_name (str): The name of the target variable.

    Returns:
        (dict[str, any]): Dictionary containing the metrics on training and evaluation DataFrames.
    """
    y_train_pred = model.predict(df_train[list_features])
    y_eval_pred = model.predict(df_eval[list_features])

    return {
        "rmse_train": root_mean_squared_error(df_train[target_name], y_train_pred),
        "rmse_eval": root_mean_squared_error(df_eval[target_name], y_eval_pred),
        "mape_train": mean_absolute_percentage_error(df_train[target_name], y_train_pred),
        "mape_eval": mean_absolute_percentage_error(df_eval[target_name], y_eval_pred),
    }


def train_model(  # noqa: PLR0913
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    list_features: list[str],
    params: dict[str, any],
    experiment_id: str,
    target_name: str = "sales",
):
    """Train a model using the training DataFrame and evaluate it on the evaluation DataFrame.

    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_eval (pd.DataFrame): The evaluation DataFrame.
        list_features (list[str]): List of feature names to use for training.
        params (dict[str, any]): Parameters for the HistGradientBoostingRegressor model.
        experiment_id (str): The ID of the MLflow experiment to log the model and metrics.
        target_name (str): The name of the target variable.
    """
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag("model_type", "HistGradientBoostingRegressor")
        mlflow.set_tag("target_name", target_name)
        mlflow.log_params(params)

        # Log the features used for training
        mlflow.log_param("features", list_features)

        # Log the number of training and evaluation samples
        mlflow.log_param("n_train_samples", len(df_train))
        mlflow.log_param("n_eval_samples", len(df_eval))

        # Train the model
        logger.info("Training the model...")
        model = _train_hgbm(df_train, list_features, params, target_name)
        logger.info("Model trained")

        # Compute metrics
        metrics = _compute_metrics(df_train, df_eval, list_features, model, target_name)
        _log_mlflow_metric(metrics)
        logger.info("Metrics logged to MLflow")

        # Log model
        _log_model_mlflow(model, df=df_train)
        logger.info("Model logged to MLflow")
    return model
