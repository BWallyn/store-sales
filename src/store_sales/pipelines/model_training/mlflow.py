# =================
# ==== IMPORTS ====
# =================

import mlflow
import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def create_mlflow_experiment(
    experiment_folder: str, experiment_name: str
) -> str:
    """Create an MLflow experiment.

    Args:
        experiment_folder (str): Path to the folder to store the experiment
        experiment_name (str): Name of the experiment

    Returns:
        experiment_id (str): Id of the experiment
    """
    experiment_path = f"{experiment_folder}{experiment_name}"
    experiment_id = mlflow.create_experiment(experiment_path)
    return experiment_id


def _log_model_mlflow(model, df: pd.DataFrame) -> None:
    """Log model to Mlflow and add input example

    Args:
        model: Model trained to log
        df (pd.DataFrame): DataFrame to use for example
    """
    mlflow.sklearn.log_model(
        model,
        "model",
        # input_example=df[model.feature_names_].sample(10, random_state=42)
    )


def _log_mlflow_metric(dict_metrics: dict[str, float]):
    """Log metrics to MLflow

    Args:
        dict_metrics (dict[str, float]): Dict of metrics to log to MLflow
    """
    mlflow.log_metrics(dict_metrics)
