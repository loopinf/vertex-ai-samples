from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def get_adj_prices(
  market_info_dataset: Input[Dataset] ,
  adj_price_dataset: Output[Dataset],
  ):

  import FinanceDataReader as fdr
  import pandas as pd
  from multiprocessing import Pool

  df_market = pd.read_pickle(market_info_dataset.path)

  start_date = '20190101'

  global get_price

  l_code = df_market.code.unique().tolist()

  def get_price(code):
    return (
        fdr.DataReader(code, start=start_date)
        .assign(code=code)
    )
  with Pool(15) as pool:
    result = pool.map(get_price, l_code)

  df_adj_price = pd.concat(result)
  df_adj_price = df_adj_price.reset_index()
  df_adj_price.columns = df_adj_price.columns.str.lower()
  df_adj_price['date'] = df_adj_price.date.dt.strftime('%Y%m%d')


  df_adj_price.to_pickle(adj_price_dataset.path)