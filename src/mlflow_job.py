import datetime
import os
from typing import Tuple, Union

import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model_fit_predict import evaluate_model, predict_model, train_model

load_dotenv()

Classifiers = Union[CatBoostClassifier, RandomForestClassifier, LogisticRegression]

def log_experiment_mlflow(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    checkpoint: datetime.date,
    model_params: dict,
    model_name: str
) -> Tuple[Classifiers, dict]:
    """Log a machine learning experiment using MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        The training dataset containing features.
    y_train : pd.Series
        The training dataset containing target labels.
    X_test : pd.DataFrame
        The testing dataset containing features.
    y_test : pd.Series
        The testing dataset containing target labels.
    checkpoint : datetime.date
        The date for which predictions are made, and the model's performance is evaluated.
    model_params : dict
        A dictionary of hyperparameters and settings for the selected classifier.
    model_name : str
        The name of the classifier to use for the experiment.
        Supported values: "CatBoost", "RandomForest", "LogisticRegression".

    Returns
    -------
    tuple
        A tuple containing the trained classifier and a dictionary of logged metrics.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    with mlflow.start_run() as run:
        print(f'run_id: {run.info.run_id}, status: {run.info.status}.')

        model = train_model(model_name=model_name,
                            X_train=X_train,
                            y_train=y_train,
                            model_params=model_params)

        y_pred_proba, y_pred = predict_model(model, X_test)

        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)

        params_to_log = {
            "checkpoint": checkpoint,
            "model_name": model.__class__.__name__,
            "features": X_train.columns.tolist(),
            "target": y_train.name,
        }

        mlflow.log_params(params_to_log)
        mlflow.log_params(model_params)
        print(f'run_id: {run.info.run_id}, status: DONE.')
        return model, metrics
