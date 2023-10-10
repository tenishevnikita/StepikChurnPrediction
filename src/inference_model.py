from src.data_processing import DataProcessor
from src.model_fit_predict import train_model
from src.train_pipeline_params import TrainingPipelineParams


def inference_pipeline(pipeline_params: TrainingPipelineParams):
    data_processor = DataProcessor(
        submissions_path=pipeline_params.submissions_path,
        course_structure_path=pipeline_params.course_structure_path,
        first_course_day=pipeline_params.first_course_day,
        admin_ids=pipeline_params.admin_ids
    )

    checkpoint = pipeline_params.checkpoint

    data_processor.fit()

    df = data_processor.features_df

    features = pipeline_params.features
    target = pipeline_params.target

    train_inds = df['day'].dt.date < checkpoint
    test_inds = df['day'].dt.date == checkpoint

    X = df[features]
    y = df[target]

    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]


    model = train_model(
        model_name=pipeline_params.model_name,
        X_train=X_train,
        y_train=y_train,
        model_params=pipeline_params.model_params
    )

    predictions = df[['user_id']].merge(y_test, left_index=True, right_index=True)
    predictions['pred_proba'] = model.predict_proba(X_test)[:, 1]
    predictions['target_14d'] = model.predict(X_test)

    return predictions
