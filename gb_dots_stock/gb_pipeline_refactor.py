# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd

PROJECT_ID = "dots-stock"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
USER = "shkim01"  # <---CHANGE THIS
BUCKET_NAME = "gs://pipeline-dots-stock"  # @param {type:"string"}
PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root/{USER}"

from typing import NamedTuple

from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)
from kfp.v2.google.client import AIPlatformClient


@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_market_info(
    # top30_univ_dataset: Output[Dataset], 
    market_info_dataset: Output[Dataset],
    today: str,
    n_days: int
) -> str:
  import pandas as pd
  from pandas.tseries.offsets import CustomBusinessDay
  from trading_calendars import get_calendar
  import functools

  import pickle
  import logging
  import networkx as nx
  import os
  from sqlalchemy import create_engine

  # today = pd.Timestamp.now('Asia/Seoul').strftime('%Y%m%d')
  # today = '20210809'
  cal_KRX = get_calendar('XKRX')
  custombd_KRX = CustomBusinessDay(holidays=cal_KRX.precomputed_holidays)

  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  # console handler
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # create formatter
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  # add ch to logger
  logger.addHandler(ch)

  # Preference
  #-----------------------------------------------------------------------------
  AWS_DB_ID = 'gb_master'
  AWS_DB_PWD = 'qwert12345'
  AWS_DB_ADDRESS = 'kwdb-daily.cf6e7v8fhede.ap-northeast-2.rds.amazonaws.com'
  AWS_DB_PORT = '3306'
  DB_DATABASE_NAME_daily_naver = 'daily_naver'
  PROJECT_ID = 'dots-stock'
  db_daily_naver_con = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'
                                      .format(AWS_DB_ID, AWS_DB_PWD, AWS_DB_ADDRESS, AWS_DB_PORT, DB_DATABASE_NAME_daily_naver),
                                      encoding='utf8',
                                      echo=False)

  # @functools.lru_cache()
  def get_market_from_naver_aws(date_ref):
      '''
      daily naver 에서 db값 그대로 parsing 내용 받아오기
      '''
      with db_daily_naver_con.connect() as conn:
          table_name = f'{date_ref}_daily_allstock_naver'
          str_sql = f'select * from {table_name} order by 등락률 DESC'
          df = pd.read_sql_query(str_sql, conn)  # self.get_db_daily_naver_con())
          df = df.reset_index().rename(columns={'index':'순위_상승률', 'N':'순위_시가총액'})
          df['순위_상승률'] = df.순위_상승률 + 1
      return df

  def get_krx_on_dates_n_days_ago(date_ref, n_days=20):
      return [date.strftime('%Y%m%d')
              for date in pd.bdate_range(
          end=date_ref, freq='C', periods=n_days,
          holidays=cal_KRX.precomputed_holidays)
      ]
  # 1. Market data
  #------------------------------------------------------------------------------
  def get_markets_aws(date_ref, n_days):
      '''
      장중일때는 해당날짜만 cache 안함
      '''
      dates_n_days_ago = get_krx_on_dates_n_days_ago(date_ref, n_days)
      df_market = pd.DataFrame()
      for date in dates_n_days_ago:
          df_ = get_market_from_naver_aws(date)
          # logger.debug(f'date : {date} and df_.shape {df_.shape}' )
          df_market  = df_market.append(df_)
      return df_market

  # def get_top30_list(df_market):
  #     cols_out = ['날짜','종목코드','종목명']
  #     return (df_market
  #             .sort_values(['날짜','등락률'], ascending=False)
  #             .groupby('날짜')
  #             .head(30)[cols_out]
  #     )
  
  df_market = get_markets_aws(date_ref=today, n_days=n_days)
  # df_top30s = get_top30_list(df_market)

  df_market.to_csv(market_info_dataset.path)
  # df_top30s.to_csv(top30_univ_dataset.path)

  return today

@component(
   base_image="gcr.io/dots-stock/python-img-v5.2"
)
def get_base_item(
  market_info_dataset: Input[Dataset],
  base_item_dataset: Output[Dataset]
):
  import pandas as pd

  # helper function
  def get_top30_list(df_market):
      cols_out = ['날짜','종목코드','종목명']
      return (df_market
              .sort_values(['날짜','등락률'], ascending=False)
              .groupby('날짜')
              .head(30)[cols_out])
  
  df_market = pd.read_csv(market_info_dataset.path)
  df_base_item = get_top30_list(df_market)
  df_base_item.to_csv(base_item_dataset.path)

@component(
   base_image="gcr.io/dots-stock/python-img-v5.2"
)
def get_bros(
    today: str,
    n_days: int, 
    bros_univ_dataset: Output[Dataset]
) -> str :
  '''
  
  Returns:
    list
  '''
  import pandas as pd
  import pandas_gbq
  import networkx as nx
  from trading_calendars import get_calendar 
  PROJECT_ID = 'dots-stock'
  cal_KRX = get_calendar('XKRX')

  # helper functions
  #-----------------------------------------------------------------------------
  def get_krx_on_dates_start_end(start, end):
    return [date.strftime('%Y%m%d')
            for date in pd.bdate_range(start=start, end=end, freq='C', 
        holidays=cal_KRX.precomputed_holidays) ]

  def get_krx_on_dates_n_days_ago(date_ref, n_days=20):
    return [date.strftime('%Y%m%d')
            for date in pd.bdate_range(
        end=date_ref, freq='C', periods=n_days,
        holidays=cal_KRX.precomputed_holidays) ]

  def get_corr_pairs_gbq(date_ref, period):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    sql = f'''
    SELECT
      DISTINCT source,
      target,
      corr_value,
      period,
      date
    FROM
      `dots-stock.krx_dataset.corr_ohlc_part1`
    WHERE
      date = "{date_ref_}"
      AND period = {period}
    ORDER BY
      corr_value DESC
    LIMIT
      1000'''
    
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID)
    return df

  def find_bros(date_ref, period):
    '''clique over 3 nodes '''
    df_edgelist = get_corr_pairs_gbq(date_ref, period)
    g = nx.from_pandas_edgelist(df_edgelist, edge_attr=True)
    bros_ = nx.find_cliques(g)
    bros_3 = [bros for bros in bros_ if len(bros) >=3]
    set_bros =  set([i for l_i in bros_3 for i in l_i])
    g_gang = g.subgraph(set_bros)

    df_gangs_edgelist = nx.to_pandas_edgelist(g_gang)
    return df_gangs_edgelist

  def find_gang(date_ref):
    df_gang = pd.DataFrame()
    for period in [20, 40, 60, 90, 120]:
      df_ = find_bros(date, period=period)
      df_gang = df_gang.append(df_)
    return df_gang

  def get_bros_univ(date_ref):

    bros_120 = find_bros(date_ref, 120)
    bros_90 = find_bros(date_ref, 90)
    bros_60 = find_bros(date_ref, 60)
    bros_40 = find_bros(date_ref, 40)
    bros_20 = find_bros(date_ref, 20)

    set_bros_120 =  set([i for l_i in bros_120 for i in l_i ])
    set_bros_90 =  set([i for l_i in bros_90 for i in l_i ])
    set_bros_60 =  set([i for l_i in bros_60 for i in l_i ])
    set_bros_40 =  set([i for l_i in bros_40 for i in l_i ])
    set_bros_20 =  set([i for l_i in bros_20 for i in l_i ])

    s_univ = (
             set_bros_40 | set_bros_20 | set_bros_120 | set_bros_60 | set_bros_90)

    return list(s_univ)
  
  # jobs
  dates = get_krx_on_dates_n_days_ago(date_ref=today, n_days=n_days)
  df_bros = pd.DataFrame()
  for date in dates:
    df = find_gang(date_ref=date)  
    df_bros = df_bros.append(df)

  df_bros.to_csv(bros_univ_dataset.path)

  return 'OK'

@component()
def get_features():
  pass

@component()
def get_target():
  pass

@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_univ_for_price(
  # date_ref: str,
  base_item_dataset: Input[Dataset],
  bros_dataset: Input[Dataset],
  univ_dataset: Output[Dataset],
):
  import pandas as pd
  import pickle
  import logging
  import json
  logger = logging.getLogger(__name__)
  logger.setLevel(logging.DEBUG)
  # console handler
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # create formatter
  formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  ch.setFormatter(formatter)
  # add ch to logger
  logger.addHandler(ch)

  df_top30s = pd.read_csv(base_item_dataset.path)
  df_ed = pd.read_csv(bros_dataset.path, index_col=0).reset_index(drop=True)

  df_ed_r = df_ed.copy() 
  print(f'test : {df_ed_r.columns}')
  df_ed_r.columns = ['target', 'source', 'date', 'corr_value','period']
  df_ed2 = df_ed.append(df_ed_r)
  df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

  dic_univ = {}
  for date, df in df_top30s.groupby('날짜'):
    logger.debug(f'date: {date}')
    l_top30 = df.종목코드.to_list()

    l_bro = df_ed2[(df_ed2.date == date) & 
                  (df_ed2.source.isin(l_top30))].source.unique().tolist()

    dic_univ[date] = list(set(l_top30 + l_bro ))

  with open(univ_dataset.path, 'w', encoding='utf8') as f:
    json.dump(dic_univ, f)

@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
    # packages_to_install = ["tables", "pandas_gbq", "finance-datareader", "bs4", "pickle5"]   # add 20210715 FIX pipeline
)
def get_adj_prices(
  today: str,
  dic_univ_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ) -> str:
  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd
  # with open(dic_univ_dataset.path, 'rb') as f:
  #   dic_univ = pickle.load(f)
  with open(dic_univ_dataset.path, 'r') as f:
    dic_univ = json.load(f)

  codes_stock = []
  for v in dic_univ.values():
    codes_stock.extend(v)

  def get_price_adj(code, start, end):
    return fdr.DataReader(code, start=start, end=end)

  def get_price(l_univ, date_start, date_end):
    df_price = pd.DataFrame()
    for code in l_univ :
      df_ = get_price_adj(code, date_start, date_end)
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  ae_log.debug(f'codes_stock {codes_stock.__len__()}')
  date_start = '20210101'
  date_end = today
  df_adj_price = get_price(codes_stock, date_start=date_start, date_end=today)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)

  return 'good'

@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_target():
  pass
  # import os
  # import pandas as pd
  # import FinanceDataReader as fdr
  # import numpy as np


  # # Preference
  # date_start = '20210101'

  # # functions
  # #-----------------------------------------------------------------------------

  # def make_target(df):

  #   df_ = df.copy()

  #   df_.sort_values(by='date', inplace=True)
  #   df_['high_p1'] = df_.high.shift(-1)
  #   df_['high_p2'] = df_.high.shift(-2)
  #   df_['high_p3'] = df_.high.shift(-3)

  #   df_['close_p1'] = df_.close.shift(-1)
  #   df_['close_p2'] = df_.close.shift(-2)
  #   df_['close_p3'] = df_.close.shift(-3)

  #   df_['change_p1'] = (df_.close_p1 - df_.close) / df_.close
  #   df_['change_p2'] = (df_.close_p2 - df_.close) / df_.close
  #   df_['change_p3'] = (df_.close_p3 - df_.close) / df_.close

  #   df_['change_p1_over5'] = df_['change_p1'] > 0.05
  #   df_['change_p2_over5'] = df_['change_p2'] > 0.05
  #   df_['change_p3_over5'] = df_['change_p3'] > 0.05

  #   df_['change_p1_over10'] = df_['change_p1'] > 0.1
  #   df_['change_p2_over10'] = df_['change_p2'] > 0.1
  #   df_['change_p3_over10'] = df_['change_p3'] > 0.1

  #   df_['close_high_1'] = (df_.high_p1 - df_.close) / df_.close
  #   df_['close_high_2'] = (df_.high_p2 - df_.close) / df_.close
  #   df_['close_high_3'] = (df_.high_p3 - df_.close) / df_.close

  #   df_['close_high_1_over10'] = df_['close_high_1'] > 0.1
  #   df_['close_high_2_over10'] = df_['close_high_2'] > 0.1
  #   df_['close_high_3_over10'] = df_['close_high_3'] > 0.1

  #   df_['close_high_1_over5'] = df_['close_high_1'] > 0.05
  #   df_['close_high_2_over5'] = df_['close_high_2'] > 0.05
  #   df_['close_high_3_over5'] = df_['close_high_3'] > 0.05
    
  #   df_['target_over10'] = np.logical_or.reduce([
  #                                 df_.close_high_1_over10,
  #                                 df_.close_high_2_over10,
  #                                 df_.close_high_3_over10])
    
  #   df_['target_over5'] = np.logical_or.reduce([
  #                                 df_.close_high_1_over5,
  #                                 df_.close_high_2_over5,
  #                                 df_.close_high_3_over5])
    
  #   df_['target_close_over_10'] = np.logical_or.reduce([
  #                                 df_.change_p1_over10,
  #                                 df_.change_p2_over10,
  #                                 df_.change_p3_over10])  
    
  #   df_['target_close_over_5'] = np.logical_or.reduce([
  #                                 df_.change_p1_over5,
  #                                 df_.change_p2_over5,
  #                                 df_.change_p3_over5])  
                                  
  #   df_['target_mclass_close_over10_under5'] = \
  #       np.where(df_['change_p1'] > 0.1, 
  #               1,  np.where(df_['change_p1'] > -0.05, 0, -1))                               

  #   df_['target_mclass_close_p2_over10_under5'] = \
  #       np.where(df_['change_p2'] > 0.1, 
  #               1,  np.where(df_['change_p2'] > -0.05, 0, -1))                               
                
  #   df_['target_mclass_close_p3_over10_under5'] = \
  #       np.where(df_['change_p3'] > 0.1, 
  #               1,  np.where(df_['change_p3'] > -0.05, 0, -1))                               
  #   df_.dropna(subset=['high_p3'], inplace=True)                               
    
  #   return df_
  # #-----------------------------------------------------------------------------

  # #-----------------------------------------------------------------------------
  # def get_target_df(s_univ, date_start):

  #   df_price = get_price(s_univ, date_start)
    
  #   df_price.reset_index(inplace=True)
  #   df_price.columns = df_price.columns.str.lower()

  #   df_target = df_price.groupby('code').apply(lambda df: make_target(df))
  #   df_target = df_target.reset_index(drop=True)
  #   df_target['date'] = df_target.date.dt.strftime('%Y%m%d')

  #   return df_target
  # #-----------------------------------------------------------------------------

  
  # if today != 'holiday' :
  #   print(today)
    
  #   file_path = "/gcs/pipeline-dots-stock/s_univ_top30_theDay_and_bros"
  #   file_name = f"s_univ_top30_theDay_and_bros_{today}.pickle"
  #   full_path = os.path.join(file_path, file_name)

  #   with open(full_path, 'rb') as f:
  #     dict_s_univ = pickle.load(f)

  #   s_univ_all = set()

  #   for s_univ in  dict_s_univ.values():

  #     s_univ_all = s_univ_all | s_univ

  #   print("s_univ_all size: ",s_univ_all.__len__())

  #   df_target = get_target_df(s_univ_all, date_start)
    
  #   print('df_target_code_size', set(df_target.code).__len__())


  #   file_path_target = "/gcs/pipeline-dots-stock/df_target_v01"
  #   file_name_target = f"df_target_v01_{today}.pkl"
  #   full_path_target = os.path.join(file_path_target, file_name_target)

  #   df_target.to_pickle(full_path_target)

  #   return today

  # else :


job_file_name='market-data.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def intro_pipeline():
  str_today = '20210811'
  period = 2
  get_market_info_op = get_market_info(today=str_today, n_days=period)
  op_get_base_item = get_base_item(get_market_info_op.outputs['market_info_dataset'])
  op_get_bros = get_bros(today=str_today, n_days=period)
  op_get_univ = get_univ_for_price(
    op_get_base_item.outputs['base_item_dataset'],
    op_get_bros.outputs['bros_univ_dataset'],
  )
  op_get_adj_price = get_adj_prices(
    today=str_today,
    dic_univ_dataset=op_get_univ.outputs['univ_dataset'],
    )
  
# 
compiler.Compiler().compile(
  pipeline_func=intro_pipeline,
  package_path=job_file_name
)

api_client = AIPlatformClient(
    project_id=PROJECT_ID,
    region=REGION,
)

response = api_client.create_run_from_job_spec(
  job_spec_path=job_file_name,
  enable_caching= True,
  pipeline_root=PIPELINE_ROOT
)