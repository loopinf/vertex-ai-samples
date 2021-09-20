from typing import NamedTuple
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def get_dataset(
    run_no : str, 
    ml_dataset_name : str,
    bros_dataset_name: str,
    ml_dataset: Output[Dataset],
    bros_dataset: Output[Dataset] 
    ) -> str:

    import pandas as pd
    ml_dataset_path = f'/gcs/pipeline-dots-stock/ml_dataset/{ml_dataset_name}'
    bros_dataset_path = f'/gcs/pipeline-dots-stock/ml_dataset/{bros_dataset_name}'

    df_ml_dataset = pd.read_pickle(ml_dataset_path)
    df_bros_dataset = pd.read_pickle(bros_dataset_path)

    df_ml_dataset.to_pickle(ml_dataset.path)
    df_bros_dataset.to_pickle(bros_dataset.path)

    return run_no