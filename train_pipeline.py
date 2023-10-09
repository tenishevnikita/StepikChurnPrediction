import argparse

from src.data_processing import DataProcessor
from src.mlflow_job import log_experiment_mlflow
from src.model_fit_predict import evaluate_model, predict_model, train_model
from src.train_pipeline_params import TrainingPipelineParams, read_training_pipeline_params


def train_pipeline(config_path: str):
    training_pipeline_params: TrainingPipelineParams = read_training_pipeline_params(config_path)
    data_processor = DataProcessor(
        submissions_path=training_pipeline_params.submissions_path,
        course_structure_path=training_pipeline_params.course_structure_path,
        first_course_day=training_pipeline_params.first_course_day,
        admin_ids=training_pipeline_params.admin_ids
    )

    checkpoint = training_pipeline_params.checkpoint

    data_processor.fit()

    df = data_processor.features_df

    features = training_pipeline_params.features
    target = training_pipeline_params.target

    train_inds = df['day'].dt.date < checkpoint
    test_inds = df['day'].dt.date == checkpoint

    X = df[features]
    y = df[target]

    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]

    if training_pipeline_params.use_mlflow:
        model, metrics = log_experiment_mlflow(
            model_name=training_pipeline_params.model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            checkpoint=checkpoint,
            model_params=training_pipeline_params.model_params
        )
    else:
        model = train_model(
            model_name=training_pipeline_params.model_name,
            X_train=X_train,
            y_train=y_train,
            model_params=training_pipeline_params.model_params
        )
        y_pred_proba, y_pred = predict_model(model=model, X_test=X_test)
        metrics = evaluate_model(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
        print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train_pipeline(args.config)
