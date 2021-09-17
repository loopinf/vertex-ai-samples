from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)

from kfp.components import OutputPath

def get_market_info(
    market_info_dataset: Output[Dataset],
    # market_info_dataset_path : OutputPath('DataFrame'),
    date_ref: str,
    n_days: int
):

  import pandas as pd
  import pickle
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

  df_market.to_pickle(market_info_dataset.path)