from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def get_full_tech_indi(
  # tech_indi_dataset01_path: InputPath('DataFrame'),
  # tech_indi_dataset02_path: InputPath('DataFrame'),
  # tech_indi_dataset03_path: InputPath('DataFrame'),
  # tech_indi_dataset04_path: InputPath('DataFrame'),
  # tech_indi_dataset05_path: InputPath('DataFrame'),
  # full_tech_indi_dataset_path: OutputPath('DataFrame')
  tech_indi_dataset01: Input[Dataset],
  tech_indi_dataset02: Input[Dataset],
  tech_indi_dataset03: Input[Dataset],
  tech_indi_dataset04: Input[Dataset],
  tech_indi_dataset05: Input[Dataset],
  full_tech_indi_dataset: Output[Dataset]
):

  import pandas as pd  

  df_01 = pd.read_pickle(tech_indi_dataset01.path)
  df_02 = pd.read_pickle(tech_indi_dataset02.path)
  df_03 = pd.read_pickle(tech_indi_dataset03.path)
  df_04 = pd.read_pickle(tech_indi_dataset04.path)
  df_05 = pd.read_pickle(tech_indi_dataset05.path)
  
  df_full = pd.concat([df_01, df_02, df_03,df_04, df_05])

  df_full.to_pickle(full_tech_indi_dataset.path)