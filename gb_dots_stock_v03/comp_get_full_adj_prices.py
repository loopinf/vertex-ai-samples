from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

def get_full_adj_prices(
  # adj_price_dataset01_path: InputPath('DataFrame'),
  # adj_price_dataset02_path: InputPath('DataFrame'),
  # adj_price_dataset03_path: InputPath('DataFrame'),
  # adj_price_dataset04_path: InputPath('DataFrame'),
  # adj_price_dataset05_path: InputPath('DataFrame'),
  # full_adj_prices_dataset_path: OutputPath('DataFrame')
  adj_price_dataset01: Input[Dataset],
  adj_price_dataset02: Input[Dataset],
  adj_price_dataset03: Input[Dataset],
  adj_price_dataset04: Input[Dataset],
  adj_price_dataset05: Input[Dataset],
  adj_price_dataset06: Input[Dataset],
  adj_price_dataset07: Input[Dataset],
  adj_price_dataset08: Input[Dataset],
  adj_price_dataset09: Input[Dataset],
  adj_price_dataset10: Input[Dataset],
  adj_price_dataset11: Input[Dataset],
  full_adj_prices_dataset: Output[Dataset]
):

  import pandas as pd
  
  df_adj_price_01 = pd.read_pickle(adj_price_dataset01.path)
  df_adj_price_02 = pd.read_pickle(adj_price_dataset02.path)
  df_adj_price_03 = pd.read_pickle(adj_price_dataset03.path)
  df_adj_price_04 = pd.read_pickle(adj_price_dataset04.path)
  df_adj_price_05 = pd.read_pickle(adj_price_dataset05.path)
  df_adj_price_06 = pd.read_pickle(adj_price_dataset06.path)
  df_adj_price_07 = pd.read_pickle(adj_price_dataset07.path)
  df_adj_price_08 = pd.read_pickle(adj_price_dataset08.path)
  df_adj_price_09 = pd.read_pickle(adj_price_dataset09.path)
  df_adj_price_10 = pd.read_pickle(adj_price_dataset10.path)
  df_adj_price_11 = pd.read_pickle(adj_price_dataset11.path)

  
  df_full_adj_prices = pd.concat([df_adj_price_01, 
                                df_adj_price_02,
                                df_adj_price_03,
                                df_adj_price_04,
                                df_adj_price_05,
                                df_adj_price_06,
                                df_adj_price_07,
                                df_adj_price_08,
                                df_adj_price_09,
                                df_adj_price_10,
                                df_adj_price_11,
                                ])

  df_full_adj_prices.to_pickle(full_adj_prices_dataset.path)
