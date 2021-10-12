from datetime import date
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def get_adj_prices_daily(
  market_info_dataset: Input[Dataset],
  date_ref : str,
  adj_price_dataset: Output[Dataset],  
  ):

  import FinanceDataReader as fdr
  # from ae_module.ae_logger import ae_log
  import pandas as pd
  from multiprocessing import Pool
  from trading_calendars import get_calendar
  cal_KRX = get_calendar('XKRX')

  df_market = pd.read_pickle(market_info_dataset.path)

  start_date = '20210104'

  print(f'dates : {df_market.날짜.unique().tolist()}')

  l_code = df_market.종목코드.unique().tolist()

  global get_price

  def get_price(code):
    return (
        fdr.DataReader(code, start=start_date, end=date_ref)
        .assign(code=code)
    )

  with Pool(15) as pool:
    result = pool.map(get_price, l_code)

  df_adj_price = pd.concat(result)
  df_adj_price = df_adj_price.reset_index()
  df_adj_price.columns = df_adj_price.columns.str.lower()
  df_adj_price['date'] = df_adj_price.date.dt.strftime('%Y%m%d')


  df_adj_price.to_pickle(adj_price_dataset.path)

