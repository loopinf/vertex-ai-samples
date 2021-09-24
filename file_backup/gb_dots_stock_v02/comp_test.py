from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath

def test(
    # market_info_dataset: Output[Dataset],
    market_info_dataset_path : InputPath('DataFrame'),
)-> str:

    import pandas as pd

    df = pd.read_pickle(market_info_dataset_path)

    size = str(df.shape[0])

    return size