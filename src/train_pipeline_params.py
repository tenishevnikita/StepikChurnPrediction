from datetime import date
from dataclasses import dataclass
from typing import List

import yaml


@dataclass
class TrainingPipelineParams:
    first_course_day: date
    admin_ids: List[int]
    submissions_path: str
    course_structure_path: str
    model_params: dict
    use_mlflow: bool
    checkpoint: date
    features: List[str]
    target: str


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as file:
        config_dict = yaml.safe_load(file)
        return TrainingPipelineParams(**config_dict)


if __name__ == "__main__":
    path = "../configs/train_config.yaml"
    schema = read_training_pipeline_params(path)
    print(schema)
