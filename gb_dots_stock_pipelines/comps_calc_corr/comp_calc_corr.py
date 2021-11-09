from datetime import date
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def calc_corr_rolling5(
  adj_price_dataset: Input[Dataset],
  corr_rolling5_dataset : Output[Dataset] 
  ):

  # import FinanceDataReader as fdr
  # from ae_module.ae_logger import ae_log  
  # from multiprocessing import Pool
  # from trading_calendars import get_calendar
  # cal_KRX = get_calendar('XKRX')

  import pandas as pd
  import pandas_gbq

  df_adj_price = pd.read_pickle(adj_price_dataset.path)
  print()
  calc_period = 120 

  df_corr = (df_adj_price
              .pivot_table(values='close', index='date', columns='code')
              .iloc[-120:, :]
              .rolling(5)
              .mean()
              .dropna()
              .corr()
            )

  print(f'size of df_corr : {df_corr.shape}')
  print(f'first row if df_corr : {df_corr.head(2)}')

  sr_corr_ = (df_corr
              .where((df_corr) > 0.7)
              .where(df_corr != 1)
              .stack()
              .rename_axis(['source','target'])
              .dropna()
              .to_frame()
              .reset_index()
              .rename(columns={0:'corr'})
            )

  def send_to_gbq(df_corr):
    table_schema = [{'name':'date','type':'DATE'}]
    pandas_gbq.to_gbq(df_corr, 
                    'krx_dataset.corr_ohlc_roll_mean_5_120days', 
                    project_id='dots-stock', 
                    if_exists='append',
                    table_schema=table_schema)

  sr_corr_.to_pickle(corr_rolling5_dataset.path)

