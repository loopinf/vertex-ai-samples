from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def get_ml_dataset(
  # features_dataset_path : InputPath('DataFrame'),
  # target_dataset_path : InputPath('DataFrame'),
  # tech_indi_dataset_path : InputPath('DataFrame'),
  # ml_dataset_path : OutputPath('DataFrame')
  features_dataset: Input[Dataset],
  target_dataset: Input[Dataset],
  tech_indi_dataset: Input[Dataset],
  ml_dataset: Output[Dataset]
):

  import pandas as pd

  df_feats = pd.read_pickle(features_dataset.path)  

  df_target = pd.read_pickle(target_dataset.path)
  df_target['date'] = pd.to_datetime(df_target.date).dt.strftime('%Y%m%d')

  df_tech = pd.read_pickle(tech_indi_dataset.path)
  df_tech['date'] = pd.to_datetime(df_tech.date).dt.strftime('%Y%m%d')

  df_ml_dataset = (df_feats.merge(df_target,
                            left_on=['code', 'date'],
                            right_on=['code', 'date'],
                            how='left'))

  df_ml_dataset = (df_ml_dataset.merge(df_tech,
                              left_on=['code', 'date'],
                              right_on=['code', 'date'],
                              how='left'))

  df_ml_dataset.to_pickle(ml_dataset.path)