from kfp.components import InputPath, OutputPath

def get_full_adj_prices(
  adj_price_dataset01_path: InputPath('DataFrame'),
  adj_price_dataset02_path: InputPath('DataFrame'),
  adj_price_dataset03_path: InputPath('DataFrame'),
  adj_price_dataset04_path: InputPath('DataFrame'),
  adj_price_dataset05_path: InputPath('DataFrame'),
  full_adj_prices_dataset_path: OutputPath('DataFrame')
):

  import pandas as pd
  
  df_adj_price_01 = pd.read_pickle(adj_price_dataset01_path)
  df_adj_price_02 = pd.read_pickle(adj_price_dataset02_path)
  df_adj_price_03 = pd.read_pickle(adj_price_dataset03_path)
  df_adj_price_04 = pd.read_pickle(adj_price_dataset04_path)
  df_adj_price_05 = pd.read_pickle(adj_price_dataset05_path)

  
  df_full_adj_prices = pd.concat([df_adj_price_01, 
                                df_adj_price_02,
                                df_adj_price_03,
                                df_adj_price_04,
                                df_adj_price_05])

  df_full_adj_prices.to_pickle(full_adj_prices_dataset_path)
