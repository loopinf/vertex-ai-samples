from kfp.components import InputPath, OutputPath

def get_adj_prices(
  start_index :int,
  end_index : int,
  market_info_dataset_path: InputPath('DataFrame'),
  adj_price_dataset_path: OutputPath('DataFrame'),
  ):

  # import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_pickle(market_info_dataset_path)

  date_ref = df_market.날짜.max()
  date_start = '20210101'

  codes_stock = df_market[df_market.날짜 == date_ref].종목코드.to_list()

  def get_price_adj(code, start, end):
    return fdr.DataReader(code, start=start, end=end)

  def get_price(l_univ, date_start, date_end):
    df_price = pd.DataFrame()
    for code in l_univ :
      df_ = get_price_adj(code, date_start, date_end)
      print('size', df_.shape)
      df_['code'] = str(code)
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[ start_index : end_index ]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)
  
  df_adj_price = df_adj_price.reset_index()
  print('df_adj_cols =>', df_adj_price.columns)

  df_adj_price.to_pickle(adj_price_dataset_path)

  ae_log.debug(df_adj_price.shape)