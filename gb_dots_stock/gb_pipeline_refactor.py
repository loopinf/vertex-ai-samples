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
    top30_univ_dataset: Output[Dataset], 
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
  project_id = 'dots-stock'
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

  def get_top30_list(df_market):
      cols_out = ['날짜','종목코드','종목명']
      return (df_market
              .sort_values(['날짜','등락률'], ascending=False)
              .groupby('날짜')
              .head(30)[cols_out]
      )
  
  df_market = get_markets_aws(date_ref=today, n_days=n_days)
  df_top30s = get_top30_list(df_market)

  df_market.to_csv(market_info_dataset.path)
  df_top30s.to_csv(top30_univ_dataset.path)

  return today

job_file_name='market-data.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def intro_pipeline():
  str_today = '20210811'
  get_market_info(today=str_today, n_days=2)
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