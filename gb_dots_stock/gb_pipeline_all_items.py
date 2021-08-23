# -*- coding: utf-8 -*-
import sys
import os
# import pandas as pd

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
  base_image='gcr.io/dots-stock/py38-pandas-cal'
)
def set_defaults()-> NamedTuple(
  'Outputs',
  [
    ('date_ref',str),
    ('n_days', int)
  ]):

  import pandas as pd
  from trading_calendars import get_calendar

  today = pd.Timestamp.now('Asia/Seoul').strftime('%Y%m%d')
  # today = '20210809'
  period_to_train = 20
  n_days = period_to_train + 20

  cal_KRX = get_calendar('XKRX')

  def get_krx_on_dates_start_end(start, end):

      return [date.strftime('%Y%m%d')
              for date in pd.bdate_range(start=start, 
          end=end, freq='C', 
          holidays=cal_KRX.precomputed_holidays)
      ]

  dates_krx_on = get_krx_on_dates_start_end('20210104', today)

  if today in dates_krx_on :
    date_ref = today
  else :
    date_ref = dates_krx_on[-1]
  return (date_ref, n_days)

##############################
# get market info ############
##############################

@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_market_info(
    market_info_dataset: Output[Dataset],
    date_ref: str,
    n_days: int
):
  import pandas as pd
  from trading_calendars import get_calendar
  cal_KRX = get_calendar('XKRX')

  from sqlalchemy import create_engine  

  import logging
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

  AWS_DB_ID = 'gb_master'
  AWS_DB_PWD = 'qwert12345'
  AWS_DB_ADDRESS = 'kwdb-daily.cf6e7v8fhede.ap-northeast-2.rds.amazonaws.com'
  AWS_DB_PORT = '3306'
  DB_DATABASE_NAME_daily_naver = 'daily_naver'
  db_daily_naver_con = create_engine('mysql+pymysql://{0}:{1}@{2}:{3}/{4}?charset=utf8'
                                      .format(AWS_DB_ID, AWS_DB_PWD, AWS_DB_ADDRESS, AWS_DB_PORT, DB_DATABASE_NAME_daily_naver),
                                      encoding='utf8',
                                      echo=False)

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

  def get_krx_on_dates_n_days_ago(date_ref, n_days):
    return [date.strftime('%Y%m%d')
            for date in pd.bdate_range(
        end=date_ref, freq='C', periods=n_days,
        holidays=cal_KRX.precomputed_holidays) ]
  
  def get_markets_aws(date_ref, n_days):
    dates_n_days_ago = get_krx_on_dates_n_days_ago(date_ref, n_days)
    df_market = pd.DataFrame()
    for date in dates_n_days_ago:
        df_ = get_market_from_naver_aws(date)
        logger.debug(f'date : {date} and df_.shape {df_.shape}' )
        df_market  = df_market.append(df_)
    return df_market
  
  df_market = get_markets_aws(date_ref=date_ref, n_days=n_days)

  df_market.to_csv(market_info_dataset.path)

#######################
# get bros ############
#######################
@component(
   base_image="gcr.io/dots-stock/python-img-v5.2"
)
def get_bros(
    date_ref: str,
    n_days: int, 
    bros_univ_dataset: Output[Dataset]
) -> str :
  
  import pandas as pd
  import pandas_gbq
  import networkx as nx
  from trading_calendars import get_calendar
  cal_KRX = get_calendar('XKRX') 

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

    PROJECT_ID = 'dots-stock'
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
  
  # jobs
  dates = get_krx_on_dates_n_days_ago(date_ref=date_ref, n_days=n_days)
  df_bros = pd.DataFrame()
  for date in dates:
    df = find_gang(date_ref=date)  
    df_bros = df_bros.append(df)

  df_bros.to_csv(bros_univ_dataset.path)

  return 'OK'

###############################
# get adj price 01 ############
###############################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_adj_prices_01(
  market_info_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ):

  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

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
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[0:600]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)

###############################
# get adj price 02 ############
###############################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_adj_prices_02(
  market_info_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ):

  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

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
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[600:1200]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)

###############################
# get adj price 03 ############
###############################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_adj_prices_03(
  market_info_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ):

  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

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
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[1200:1800]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)

###############################
# get adj price 04 ############
###############################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_adj_prices_04(
  market_info_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ):

  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

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
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[1800:2400]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)

###############################
# get adj price 05 ############
###############################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_adj_prices_05(
  market_info_dataset: Input[Dataset],
  adj_price_dataset: Output[Dataset]
  ):

  import json
  import FinanceDataReader as fdr
  from ae_module.ae_logger import ae_log
  import pandas as pd

  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

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
      df_['code'] = code
      # df_['price'] = df_['Close'] / df_.Close.iloc[0]
      df_price = df_price.append(df_)
    return df_price

  codes = codes_stock[2400:]
  ae_log.debug(f'codes_stock {codes.__len__()}')

  df_adj_price = get_price(codes, date_start=date_start, date_end=date_ref)

  df_adj_price.to_csv(adj_price_dataset.path)

  ae_log.debug(df_adj_price.shape)
  
###############################
# get full adj     ############
###############################
@component(
  #  base_image="gcr.io/dots-stock/python-img-v5.2"
  packages_to_install=['pandas']
)
def get_full_adj_prices(
  adj_price_dataset01: Input[Dataset],
  adj_price_dataset02: Input[Dataset],
  adj_price_dataset03: Input[Dataset],
  adj_price_dataset04: Input[Dataset],
  adj_price_dataset05: Input[Dataset],
  full_adj_prices_dataset: Output[Dataset]
):

  import pandas as pd

  df_adj_price_01 = pd.read_csv(adj_price_dataset01.path,                          
                          ).reset_index(drop=True)
  df_adj_price_02 = pd.read_csv(adj_price_dataset02.path,                          
                          ).reset_index(drop=True)
  df_adj_price_03 = pd.read_csv(adj_price_dataset03.path,
                          ).reset_index(drop=True)                      
  df_adj_price_04 = pd.read_csv(adj_price_dataset04.path,
                          ).reset_index(drop=True)
  df_adj_price_05 = pd.read_csv(adj_price_dataset05.path,
                          ).reset_index(drop=True)
  
  df_full_adj_prices = pd.concat([df_adj_price_01, df_adj_price_02, df_adj_price_03,df_adj_price_04, df_adj_price_05])

  df_full_adj_prices.to_csv(full_adj_prices_dataset.path)

###############################
# get target       ############
###############################
@component(
    # base_image="gcr.io/deeplearning-platform-release/sklearn-cpu"
    base_image="amancevice/pandas:1.3.2-slim"
)
def get_target(
  df_price_dataset: Input[Dataset],
  df_target_dataset: Output[Dataset]
):
  import pandas as pd
  import numpy as np

  def make_target(df):

    df_ = df.copy()

    df_.sort_values(by='date', inplace=True)
    df_['high_p1'] = df_.high.shift(-1)
    df_['high_p2'] = df_.high.shift(-2)
    df_['high_p3'] = df_.high.shift(-3)

    df_['close_p1'] = df_.close.shift(-1)
    df_['close_p2'] = df_.close.shift(-2)
    df_['close_p3'] = df_.close.shift(-3)

    df_['change_p1'] = (df_.close_p1 - df_.close) / df_.close
    df_['change_p2'] = (df_.close_p2 - df_.close) / df_.close
    df_['change_p3'] = (df_.close_p3 - df_.close) / df_.close

    df_['change_p1_over5'] = df_['change_p1'] > 0.05
    df_['change_p2_over5'] = df_['change_p2'] > 0.05
    df_['change_p3_over5'] = df_['change_p3'] > 0.05

    df_['change_p1_over10'] = df_['change_p1'] > 0.1
    df_['change_p2_over10'] = df_['change_p2'] > 0.1
    df_['change_p3_over10'] = df_['change_p3'] > 0.1

    df_['close_high_1'] = (df_.high_p1 - df_.close) / df_.close
    df_['close_high_2'] = (df_.high_p2 - df_.close) / df_.close
    df_['close_high_3'] = (df_.high_p3 - df_.close) / df_.close

    df_['close_high_1_over10'] = df_['close_high_1'] > 0.1
    df_['close_high_2_over10'] = df_['close_high_2'] > 0.1
    df_['close_high_3_over10'] = df_['close_high_3'] > 0.1

    df_['close_high_1_over5'] = df_['close_high_1'] > 0.05
    df_['close_high_2_over5'] = df_['close_high_2'] > 0.05
    df_['close_high_3_over5'] = df_['close_high_3'] > 0.05
    
    df_['target_over10'] = np.logical_or.reduce([
                                  df_.close_high_1_over10,
                                  df_.close_high_2_over10,
                                  df_.close_high_3_over10])
    
    df_['target_over5'] = np.logical_or.reduce([
                                  df_.close_high_1_over5,
                                  df_.close_high_2_over5,
                                  df_.close_high_3_over5])
    
    df_['target_close_over_10'] = np.logical_or.reduce([
                                  df_.change_p1_over10,
                                  df_.change_p2_over10,
                                  df_.change_p3_over10])  
    
    df_['target_close_over_5'] = np.logical_or.reduce([
                                  df_.change_p1_over5,
                                  df_.change_p2_over5,
                                  df_.change_p3_over5])  
                                  
    df_['target_mclass_close_over10_under5'] = \
        np.where(df_['change_p1'] > 0.1, 
                1,  np.where(df_['change_p1'] > -0.05, 0, -1))                               

    df_['target_mclass_close_p2_over10_under5'] = \
        np.where(df_['change_p2'] > 0.1, 
                1,  np.where(df_['change_p2'] > -0.05, 0, -1))                               
                
    df_['target_mclass_close_p3_over10_under5'] = \
        np.where(df_['change_p3'] > 0.1, 
                1,  np.where(df_['change_p3'] > -0.05, 0, -1))                               
    df_.dropna(subset=['high_p3'], inplace=True)                               
    return df_

  def get_target_df(df_price):
    df_price.reset_index(inplace=True)
    df_price.columns = df_price.columns.str.lower()
    df_target = df_price.groupby('code').apply(lambda df: make_target(df))
    df_target = df_target.reset_index(drop=True)
    # df_target['date'] = df_target.date.str.replace('-', '')
    return df_target

  df_price = pd.read_csv(df_price_dataset.path)
  df_target = get_target_df(df_price=df_price)

  df_target.to_csv(df_target_dataset.path)

###############################
# get tech indicator ##########
###############################
@component(
    base_image="gcr.io/deeplearning-platform-release/sklearn-cpu",
    packages_to_install=["stockstats"]
)
def get_tech_indi(
  df_price_dataset: Input[Dataset],
  df_techini_dataset: Output[Dataset]
):
  TECHNICAL_INDICATORS_LIST = ['macd',
  'boll_ub',
  'boll_lb',
  'rsi_30',
  'dx_30',
  'close_30_sma',
  'close_60_sma']
  from stockstats import StockDataFrame as Sdf
  from sklearn.preprocessing import MaxAbsScaler
  import pandas as pd
  class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
      self,
      use_technical_indicator=True,
      tech_indicator_list=TECHNICAL_INDICATORS_LIST,
      user_defined_feature=False,
  ):
      self.use_technical_indicator = use_technical_indicator
      self.tech_indicator_list = tech_indicator_list
      self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df):
      """main method to do the feature engineering
      @:param config: source dataframe
      @:return: a DataMatrices object
      """
      #clean data
      df = self.clean_data(df)
      
      # add technical indicators using stockstats
      if self.use_technical_indicator == True:
        df = self.add_technical_indicator(df)
        print("Successfully added technical indicators")

      # add user defined feature
      if self.user_defined_feature == True:
        df = self.add_user_defined_feature(df)
        print("Successfully added user defined features")

      # fill the missing values at the beginning and the end
      df = df.fillna(method="bfill").fillna(method="ffill")
      return df
    
    def clean_data(self, data):
      """
      clean the raw data
      deal with missing values
      reasons: stocks could be delisted, not incorporated at the time step 
      :param data: (df) pandas dataframe
      :return: (df) pandas dataframe
      """
      df = data.copy()
      df=df.sort_values(['date','tic'],ignore_index=True)
      df.index = df.date.factorize()[0]
      merged_closes = df.pivot_table(index = 'date',columns = 'tic', values = 'close')
      merged_closes = merged_closes.dropna(axis=1)
      tics = merged_closes.columns
      df = df[df.tic.isin(tics)]
      return df

    def add_technical_indicator(self, data):
      """
      calculate technical indicators
      use stockstats package to add technical inidactors
      :param data: (df) pandas dataframe
      :return: (df) pandas dataframe
      """
      df = data.copy()
      df = df.sort_values(by=['tic','date'])
      stock = Sdf.retype(df.copy())
      unique_ticker = stock.tic.unique()

      for indicator in self.tech_indicator_list:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
          try:
            temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
            temp_indicator = pd.DataFrame(temp_indicator)
            temp_indicator['tic'] = unique_ticker[i]
            temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
            indicator_df = indicator_df.append(
                temp_indicator, ignore_index=True
            )
          except Exception as e:
            print(e)
        df = df.merge(indicator_df[['tic','date',indicator]],on=['tic','date'],how='left')
      df = df.sort_values(by=['date','tic'])
      return df

    def add_user_defined_feature(self, data):
        """
        add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        df['bb_u_ratio'] = df.boll_ub / df.close
        df['bb_l_ratio'] = df.boll_lb / df.close
        df['max_scale_MACD'] = MaxAbsScaler().fit_transform(df[['macd']])
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df
  
  df_price = pd.read_csv(df_price_dataset.path)
  df_price.columns = df_price.columns.str.lower()
  df_price.rename(columns={'code':'tic'}, inplace=True)
  fe = FeatureEngineer(user_defined_feature=True)
  df_process = fe.preprocess_data(df_price)
  df_process.rename(columns={'tic':'code'}, inplace=True)

  df_process.to_csv(df_techini_dataset.path)

###############################
# get full tech indi ##########
###############################
@component(
  #  base_image="gcr.io/dots-stock/python-img-v5.2"
  packages_to_install=['pandas']
)
def get_full_tech_indi(
  tech_indi_dataset01: Input[Dataset],
  tech_indi_dataset02: Input[Dataset],
  tech_indi_dataset03: Input[Dataset],
  tech_indi_dataset04: Input[Dataset],
  tech_indi_dataset05: Input[Dataset],
  full_tech_indi_dataset: Output[Dataset]
):

  import pandas as pd

  df_01 = pd.read_csv(tech_indi_dataset01.path,                          
                          ).reset_index(drop=True)
  df_02 = pd.read_csv(tech_indi_dataset02.path,                          
                          ).reset_index(drop=True)
  df_03 = pd.read_csv(tech_indi_dataset03.path,
                          ).reset_index(drop=True)                      
  df_04 = pd.read_csv(tech_indi_dataset04.path,
                          ).reset_index(drop=True)
  df_05 = pd.read_csv(tech_indi_dataset05.path,
                          ).reset_index(drop=True)
  
  df_full = pd.concat([df_01, df_02, df_03,df_04, df_05])
  df_full.to_csv(full_tech_indi_dataset.path)

#########################################
# get feature ###########################
#########################################
@component(
    base_image="gcr.io/dots-stock/python-img-v5.2",
)
def get_features(
  market_info_dataset: Input[Dataset],
  bros_dataset: Input[Dataset],
  features_dataset: Output[Dataset]
  ):
  
  import pandas as pd
  import numpy as np
  from collections import Counter

  #df_market_info 가져오기
  df_market = pd.read_csv(market_info_dataset.path,
                          index_col=0,
                          dtype={'날짜':str}
                          ).reset_index(drop=True)

  dates_in_set = df_market.날짜.unique().tolist()
  dates_on_train = df_market.날짜.unique().tolist()[-20:]

  # 등락률 -1 
  df_market = df_market.sort_values('날짜')
  df_market['return_-1'] = df_market.groupby('종목코드').등락률.shift(1)

  #df_ed 가져오기
  df_ed = pd.read_csv(bros_dataset.path, index_col=0).reset_index(drop=True)
  df_ed_r = df_ed.copy() 
  df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
  df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
  df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

  cols = ['종목코드', '날짜', '순위_상승률']
  df_mkt_ = df_market[cols]

  cols_market = [ '종목코드','날짜','등락률','return_-1']
  cols_bro = ['source','target','period','date']

  # merge
  df_ed2_1 = ( df_ed2[cols_bro]
                  .merge(df_market[cols_market], 
                      left_on=['target','date'],
                      right_on=['종목코드','날짜'])
                  .rename(columns={'등락률':'target_return',
                  'return_-1':'target_return_-1'}))
  df_ed2_1 = df_ed2_1[['source', 'target', 'period', 'date', 
                      'target_return', 'target_return_-1']]
  
  df_tmp = df_mkt_.merge(df_ed2_1, 
          left_on=['날짜','종목코드'], 
          right_on=['date', 'source'], 
          suffixes=('','_x'),
          how='left')
  df_tmp.drop(columns=['종목코드','날짜'], inplace=True)
  df_tmp.dropna(subset=['target'], inplace=True)

  def get_upbro_ratio(df):
      '''df : '''
      return (
              sum(df.target_return > 0) /
              df.shape[0], # 그날 상승한 친구들의 비율
              df.shape[0], # 그날 친구들 수
              df.target_return.mean(), # 그날 모든 친구들 상승률의 평균
              df[df.target_return > 0].target_return.mean(), # 그날 오른 친구들의 평균
              df['target_return_-1'].mean(),# 전날 친구들 평균상승률
              sum(df['target_return_-1'] > 0) / df.shape[0],# 전날 상승한 친구들 비율
              df[df['target_return_-1'] > 0]['target_return_-1'].mean(),# 전날 상승한 친구들 평균
              )

  bro_up_ratio = (df_tmp.groupby(['date','source','period'])
      .apply(lambda df: get_upbro_ratio(df))
      .reset_index()
      .rename(columns={0:'bro_up_ratio'})
      )
  
  bro_up_ratio[['bro_up_ratio','n_bros', 'all_bro_rtrn_mean', 'up_bro_rtrn_mean',
                  'all_bro_rtrn_mean_ystd', 'bro_up_ratio_ystd', 'up_bro_rtrn_mean_ystd']] = \
      pd.DataFrame(bro_up_ratio.bro_up_ratio.tolist(), index=bro_up_ratio.index) 
  
  # Features related with Rank

  df_rank = df_mkt_.copy()

  df_rank['in_top30'] = df_rank.순위_상승률 <= 30
  df_rank['rank_mean_10'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                              lambda x : x.rolling(10, min_periods=1).mean()
                          )

  df_rank['rank_mean_5'] = df_rank.groupby('종목코드')['순위_상승률'].transform(
                              lambda x : x.rolling(5, min_periods=1).mean()
                          )

  df_rank['in_top_30_5'] = df_rank.groupby('종목코드')['in_top30'].transform(
                              lambda x : x.rolling(5, min_periods=1).sum()
                          )

  df_rank['in_top_30_10'] = df_rank.groupby('종목코드')['in_top30'].transform(
                              lambda x : x.rolling(10, min_periods=1).sum()
                          )

  df_tmp = df_tmp.merge(bro_up_ratio, on=['date','source','period'], how='left')
  df_tmp['up_bro_ratio_20'] = df_tmp[df_tmp.period == 20].bro_up_ratio
  df_tmp['up_bro_ratio_40'] = df_tmp[df_tmp.period == 40].bro_up_ratio
  df_tmp['up_bro_ratio_60'] = df_tmp[df_tmp.period == 60].bro_up_ratio
  df_tmp['up_bro_ratio_90'] = df_tmp[df_tmp.period == 90].bro_up_ratio
  df_tmp['up_bro_ratio_120'] = df_tmp[df_tmp.period == 120].bro_up_ratio

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 bro_up_ratio를 0으로 만들기
  df_tmp.drop(columns=['bro_up_ratio'], inplace=True)

  df_tmp['n_bro_20'] = df_tmp[df_tmp.period == 20].n_bros
  df_tmp['n_bro_40'] = df_tmp[df_tmp.period == 40].n_bros
  df_tmp['n_bro_60'] = df_tmp[df_tmp.period == 60].n_bros
  df_tmp['n_bro_90'] = df_tmp[df_tmp.period == 90].n_bros
  df_tmp['n_bro_120'] = df_tmp[df_tmp.period == 120].n_bros

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['n_bros'], inplace=True)

  df_tmp['all_bro_rtrn_mean_20'] = df_tmp[df_tmp.period == 20].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_40'] = df_tmp[df_tmp.period == 40].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_60'] = df_tmp[df_tmp.period == 60].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_90'] = df_tmp[df_tmp.period == 90].all_bro_rtrn_mean
  df_tmp['all_bro_rtrn_mean_120'] = df_tmp[df_tmp.period == 120].all_bro_rtrn_mean

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['all_bro_rtrn_mean'], inplace=True)

  df_tmp['up_bro_rtrn_mean_20'] = df_tmp[df_tmp.period == 20].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_40'] = df_tmp[df_tmp.period == 40].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_60'] = df_tmp[df_tmp.period == 60].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_90'] = df_tmp[df_tmp.period == 90].up_bro_rtrn_mean
  df_tmp['up_bro_rtrn_mean_120'] = df_tmp[df_tmp.period == 120].up_bro_rtrn_mean

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['up_bro_rtrn_mean'], inplace=True)

  df_tmp['all_bro_rtrn_mean_ystd_20'] = df_tmp[df_tmp.period == 20].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_40'] = df_tmp[df_tmp.period == 40].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_60'] = df_tmp[df_tmp.period == 60].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_90'] = df_tmp[df_tmp.period == 90].all_bro_rtrn_mean_ystd
  df_tmp['all_bro_rtrn_mean_ystd_120'] = df_tmp[df_tmp.period == 120].all_bro_rtrn_mean_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['all_bro_rtrn_mean_ystd'], inplace=True)

  df_tmp['bro_up_ratio_ystd_20'] = df_tmp[df_tmp.period == 20].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_40'] = df_tmp[df_tmp.period == 40].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_60'] = df_tmp[df_tmp.period == 60].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_90'] = df_tmp[df_tmp.period == 90].bro_up_ratio_ystd
  df_tmp['bro_up_ratio_ystd_120'] = df_tmp[df_tmp.period == 120].bro_up_ratio_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['bro_up_ratio_ystd'], inplace=True)

  df_tmp['up_bro_rtrn_mean_ystd_20'] = df_tmp[df_tmp.period == 20].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_40'] = df_tmp[df_tmp.period == 40].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_60'] = df_tmp[df_tmp.period == 60].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_90'] = df_tmp[df_tmp.period == 90].up_bro_rtrn_mean_ystd
  df_tmp['up_bro_rtrn_mean_ystd_120'] = df_tmp[df_tmp.period == 120].up_bro_rtrn_mean_ystd

  df_tmp.fillna(0, inplace=True) #친구가 없는 종목의 n_bros를 0으로 만들기
  df_tmp.drop(columns=['up_bro_rtrn_mean_ystd'], inplace=True)

  # Merge DataFrames
  cols_rank = ['종목코드', '날짜', 'in_top30', 'rank_mean_10', 'rank_mean_5', 'in_top_30_5', 'in_top_30_10']
  df_merged = df_tmp.merge(df_rank[cols_rank],
                      left_on=['source', 'date'],
                      right_on=['종목코드', '날짜'])

  df_merged.fillna(0, inplace=True)
  df_merged.drop(columns=['종목코드', '날짜'], inplace=True)

  df_feats = df_merged.drop_duplicates(subset=['source', 'date'])
  # df_feats = df_merged[df_merged.date.isin(dates_on_train)]
  
  df_feats.to_csv(features_dataset.path)

@component(
  packages_to_install=['pandas']
)
def get_ml_dataset(
  features_dataset : Input[Dataset],
  target_dataset : Input[Dataset],
  tech_indi_dataset : Input[Dataset],
  ml_dataset : Output[Dataset]
):

  import pandas as pd

  df_feats = pd.read_csv(features_dataset.path,
                        index_col=0,
                        dtype={'date':str},
                              ).reset_index(drop=True)

  df_target = pd.read_csv(target_dataset.path,
                          index_col=0,
                          dtype={'code':str},
                              ).reset_index(drop=True)
  df_target['date'] = pd.to_datetime(df_target.date).dt.strftime('%Y%m%d')

  df_tech = pd.read_csv(tech_indi_dataset.path,
                          index_col=0,
                              ).reset_index(drop=True)
  df_tech['date'] = pd.to_datetime(df_tech.date).dt.strftime('%Y%m%d')

  df_ml_dataset = (df_feats.merge(df_target,
                              left_on=['source', 'date'],
                              right_on=['code', 'date']))

  df_ml_dataset = (df_ml_dataset.merge(df_tech,
                              left_on=['source', 'date'],
                              right_on=['code', 'date']))

  df_ml_dataset.to_csv(ml_dataset.path)

#########################################
# create pipeline #######################
#########################################
job_file_name='ml-with-all-items.json'
@dsl.pipeline(
  name=job_file_name.split('.json')[0],
  pipeline_root=PIPELINE_ROOT
)    
def create_awesome_pipeline():
  op_set_defaults = set_defaults()

  op_get_bros = get_bros(
    date_ref=op_set_defaults.outputs['date_ref'],
    n_days=op_set_defaults.outputs['n_days']
  )

  op_get_market_info = get_market_info(
    date_ref=op_set_defaults.outputs['date_ref'],
    n_days=op_set_defaults.outputs['n_days']
  )

  op_get_adj_prices_01 = get_adj_prices_01(
    market_info_dataset = op_get_market_info.outputs['market_info_dataset']
  )
  op_get_adj_prices_02 = get_adj_prices_02(
    market_info_dataset = op_get_market_info.outputs['market_info_dataset']
  )
  op_get_adj_prices_03 = get_adj_prices_03(
    market_info_dataset = op_get_market_info.outputs['market_info_dataset']
  )
  op_get_adj_prices_04 = get_adj_prices_04(
    market_info_dataset = op_get_market_info.outputs['market_info_dataset']
  )
  op_get_adj_prices_05 = get_adj_prices_05(
    market_info_dataset = op_get_market_info.outputs['market_info_dataset']
  )

  op_get_full_adj_prices = get_full_adj_prices(
      adj_price_dataset01= op_get_adj_prices_01.outputs['adj_price_dataset'],
      adj_price_dataset02= op_get_adj_prices_02.outputs['adj_price_dataset'],
      adj_price_dataset03= op_get_adj_prices_03.outputs['adj_price_dataset'],
      adj_price_dataset04= op_get_adj_prices_04.outputs['adj_price_dataset'],
      adj_price_dataset05= op_get_adj_prices_05.outputs['adj_price_dataset']
    )

  op_get_features = get_features(
    market_info_dataset= op_get_market_info.outputs['market_info_dataset'], 
    bros_dataset= op_get_bros.outputs['bros_univ_dataset']
  )

  op_get_target = get_target(
    op_get_full_adj_prices.outputs['full_adj_prices_dataset']
  )
  op_get_techindi_01 = get_tech_indi(
    op_get_adj_prices_01.outputs['adj_price_dataset'])
  op_get_techindi_02 = get_tech_indi(
    op_get_adj_prices_02.outputs['adj_price_dataset'])
  op_get_techindi_03 = get_tech_indi(
    op_get_adj_prices_03.outputs['adj_price_dataset'])
  op_get_techindi_04 = get_tech_indi(
    op_get_adj_prices_04.outputs['adj_price_dataset'])
  op_get_techindi_05 = get_tech_indi(
    op_get_adj_prices_05.outputs['adj_price_dataset'])
  
  op_get_full_tech_indi = get_full_tech_indi(
    tech_indi_dataset01 = op_get_techindi_01.outputs['df_techini_dataset'],
    tech_indi_dataset02 = op_get_techindi_02.outputs['df_techini_dataset'],
    tech_indi_dataset03 = op_get_techindi_03.outputs['df_techini_dataset'],
    tech_indi_dataset04 = op_get_techindi_04.outputs['df_techini_dataset'],
    tech_indi_dataset05 = op_get_techindi_05.outputs['df_techini_dataset']

  )

  get_ml_dataset(
    features_dataset= op_get_features.outputs['features_dataset'],
    target_dataset= op_get_target.outputs['df_target_dataset'],
    tech_indi_dataset= op_get_full_tech_indi.outputs['full_tech_indi_dataset']
  )

  
  
# 
compiler.Compiler().compile(
  pipeline_func=create_awesome_pipeline,
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





######################



# @component(
#     base_image="amancevice/pandas:1.3.2-slim"
# )
# def get_univ_for_price(
#   # date_ref: str,
#   base_item_dataset: Input[Dataset],
#   bros_dataset: Input[Dataset],
#   univ_dataset: Output[Dataset],
# ):
#   import pandas as pd
#   import logging
#   import json
#   logger = logging.getLogger(__name__)
#   FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
#   logging.basicConfig(format=FORMAT)
#   logger.setLevel(logging.DEBUG)

#   # base item
#   df_top30s = pd.read_csv(base_item_dataset.path, 
#                        index_col=0, 
#                        dtype={'날짜': str}).reset_index(drop=True)

#   # load edge_list to make bros
#   df_ed = pd.read_csv(bros_dataset.path, index_col=0).reset_index(drop=True)
#   df_ed_r = df_ed.copy() 
#   df_ed_r.rename(columns={'target':'source', 'source':'target'}, inplace=True)
#   df_ed2 = df_ed.append(df_ed_r, ignore_index=True)
#   df_ed2['date'] = pd.to_datetime(df_ed2.date).dt.strftime('%Y%m%d')

#   dic_univ = {}
#   for date, df in df_top30s.groupby('날짜'):
#     logger.debug(f'date: {date}')
#     l_top30 = df.종목코드.to_list()
#     l_bro = df_ed2[(df_ed2.date == date) & 
#                   (df_ed2.source.isin(l_top30))].target.unique().tolist()

#     dic_univ[date] = list(set(l_top30 + l_bro ))

#   with open(univ_dataset.path, 'w', encoding='utf8') as f:
#     json.dump(dic_univ, f)
