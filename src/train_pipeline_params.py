import datetime
from dataclasses import dataclass, field
from typing import List

import yaml
from marshmallow_dataclass import class_schema

PATH = "../configs/train_config.yaml"


@dataclass
class TrainingPipelineParams:
    first_course_day: datetime.date
    admin_ids: List[int]
    submissions_path: str
    course_structure_path: str
    model_params: dict
    use_mlflow: bool
    checkpoint: datetime.date
    features: List[str]
    target: str

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        config_dict = yaml.safe_load(input_stream)
        schema = TrainingPipelineParamsSchema().load(config_dict)
        return schema


if __name__ == "__main__":
    schema = read_training_pipeline_params(PATH)
    print(schema)
