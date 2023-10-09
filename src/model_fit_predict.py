from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

Classifiers = Union[CatBoostClassifier, RandomForestClassifier, LogisticRegression]

def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict,
) -> Classifiers:
    """Train a CatBoostClassifier model.

    Parameters
    ----------
    model_name : str
        The name of the classifier model to train. Choose from "CatBoost", "LogisticRegression", or "RandomForest".

    X_train : pandas.DataFrame
        The training feature dataset.

    y_train : pandas.Series
        The target labels for training.

    model_params : dict
        A dictionary containing hyperparameters and settings for the selected classifier model.

    Returns
    -------
    model : CatBoostClassifier
        A trained CatBoostClassifier model.
    """
    if model_name == "CatBoost":
        model = CatBoostClassifier(**model_params)
    elif model_name == "LogisticRegression":
        model = LogisticRegression(**model_params)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    model.fit(X_train, y_train)
    return model


def predict_model(
    model: Classifiers,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make predictions using a trained model.

    Parameters
    ----------
    model : Classifiers
        A trained classifier model (e.g., CatBoostClassifier, LogisticRegression, or RandomForestClassifier).
    X_test : pandas.DataFrame
        The feature dataset for which predictions are to be made.

    Returns
    -------
    y_pred_proba : numpy.ndarray
        An array of predicted probabilities for the positive class (class 1) for each sample in X_test.

    y_pred : numpy.ndarray
        An array of predicted class labels (0 or 1) for each sample in X_test.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    return y_pred_proba, y_pred


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, Union[float, int]]:
    """Evaluate the performance of a binary classification model using various metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        True binary labels for the test data.

    y_pred : numpy.ndarray
        Predicted binary labels for the test data.

    y_pred_proba : numpy.ndarray
        Predicted probabilities for the positive class (class 1) for the test data.

    Returns
    -------
    metrics : dict
        A dictionary containing the following evaluation metrics:
        - 'roc_auc' (float): Receiver Operating Characteristic Area Under the Curve (ROC AUC) score.
        - 'true_positive' (int): Number of true positive predictions.
        - 'true_negative' (int): Number of true negative predictions.
        - 'false_positive' (int): Number of false positive predictions.
        - 'false_negative' (int): Number of false negative predictions.
    """
    roc_auc = roc_auc_score(y_true=y_true, y_score=y_pred_proba)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    metrics = {
        "roc_auc": roc_auc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    return metrics
