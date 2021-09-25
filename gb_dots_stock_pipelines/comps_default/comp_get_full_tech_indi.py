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
  tech_indi_dataset06: Input[Dataset],
  tech_indi_dataset07: Input[Dataset],
  tech_indi_dataset08: Input[Dataset],
  tech_indi_dataset09: Input[Dataset],
  tech_indi_dataset10: Input[Dataset],
  tech_indi_dataset11: Input[Dataset],
  full_tech_indi_dataset: Output[Dataset]
):

  import pandas as pd  

  df_01 = pd.read_pickle(tech_indi_dataset01.path)
  df_02 = pd.read_pickle(tech_indi_dataset02.path)
  df_03 = pd.read_pickle(tech_indi_dataset03.path)
  df_04 = pd.read_pickle(tech_indi_dataset04.path)
  df_05 = pd.read_pickle(tech_indi_dataset05.path)
  df_06 = pd.read_pickle(tech_indi_dataset06.path)
  df_07 = pd.read_pickle(tech_indi_dataset07.path)
  df_08 = pd.read_pickle(tech_indi_dataset08.path)
  df_09 = pd.read_pickle(tech_indi_dataset09.path)
  df_10 = pd.read_pickle(tech_indi_dataset10.path)
  df_11 = pd.read_pickle(tech_indi_dataset11.path)
  
  df_full = pd.concat([df_01, df_02, df_03,df_04, df_05, df_06, df_07, df_08, df_09, df_10, df_11])

  df_full.to_pickle(full_tech_indi_dataset.path)