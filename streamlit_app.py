import pandas as pd
import streamlit as st

from src.inference_model import inference_pipeline
from src.train_pipeline_params import TrainingPipelineParams

st.title('Предсказание оттока студентов с образовательного курса на Stepik')

submissions_path = st.text_input('Отчёт "Решения учащихся"')
course_structure_path = st.text_input('Отчёт "Структура курса"')
gradebook_path = st.text_input('Отчёт "Табель успеваемости"')


if all([submissions_path, course_structure_path, gradebook_path]):
    gradebook = pd.read_csv(gradebook_path)
    gradebook_df = gradebook[['user_id', 'last_name', 'first_name', 'total']]
    gradebook_df['total'] = gradebook_df['total'].round(2)

    course_first_day = st.date_input('Дата начала курса', value=None)
    admin_ids = st.multiselect('Выберите админов', gradebook_df['user_id'].values.tolist())
    checkpoint = st.date_input('Дата, для которой сделать предсказания', value="today")
    certificate_threshold = st.number_input('Количество баллов для получения сертификата с отличием')

    if st.button('Сделать предсказания'):
        pipeline_params = TrainingPipelineParams(first_course_day=course_first_day,
                            admin_ids=admin_ids,
                            course_structure_path=course_structure_path,
                            submissions_path=submissions_path,
                            model_params={'random_state': 2023, 'verbose': False},
                            use_mlflow=False,
                            checkpoint=checkpoint,
                            features=["days_offline", "solved_total", "success_rate_14d", "avg_submits_14d"],
                            target="target_14d",
                            model_name="CatBoost"
        )

        predictions = inference_pipeline(pipeline_params)
        if predictions is not None:
            predictions = pd.merge(gradebook_df[['user_id', 'total']], predictions, on='user_id', how='left')
            predictions = predictions[~predictions['user_id'].isin(admin_ids)]
            predictions.sort_values(by=['total', 'pred_proba'], ascending=[False, False], inplace=True)
            predictions['user_id'] = predictions['user_id'].astype(str)
            predictions = predictions[predictions['total'] < certificate_threshold]
            predictions.fillna(value=0, inplace=True)

            predictions.rename(columns={
                'total': 'Количество баллов',
                'pred_proba': 'Вероятность оттока',
            }, inplace=True)

            st.write(predictions.drop(columns=['target_14d']).reset_index(drop=True))
