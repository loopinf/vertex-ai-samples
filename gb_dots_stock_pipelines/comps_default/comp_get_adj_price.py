from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def get_adj_prices(
  start_index :int,
  end_index : int,
  # market_info_dataset_path: InputPath('DataFrame'),
  market_info_dataset: Input[Dataset] ,
  # adj_price_dataset_path: OutputPath('DataFrame'),
  adj_price_dataset: Output[Dataset],
  ):

  # import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd
  from multiprocessing import Pool

  df_market = pd.read_pickle(market_info_dataset.path)

  date_ref = df_market.날짜.max()
  date_start = df_market.날짜.min() #'20210101'

  codes_stock = df_market[df_market.날짜 == date_ref].종목코드.to_list()

  # def get_price_adj(code, start, end):
  #   return fdr.DataReader(code, start=start, end=end)

  # def get_price(l_univ, date_start, date_end):
  #   df_price = pd.DataFrame()
  #   for code in l_univ :
  #     df_ = get_price_adj(code, date_start, date_end)
  #     print('size', df_.shape)
  #     df_['code'] = str(code)
  #     df_price = df_price.append(df_)
  #   return df_price

  

  # ###########

  l_code = codes_stock[ start_index : end_index ]

  global get_price

  def get_price(code):
    return (
        fdr.DataReader(code, start=date_start, end=date_ref)
        .assign(code=code)
    )

  #####
  with Pool(15) as pool:
    result = pool.map(get_price, l_code)

  df_adj_price = pd.concat(result)
  df_adj_price = df_adj_price.reset_index()
  df_adj_price.columns = df_adj_price.columns.str.lower()
  df_adj_price['date'] = df_adj_price.date.dt.strftime('%Y%m%d')

  print('df_adj_cols =>', df_adj_price.columns)

  df_adj_price.to_pickle(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)