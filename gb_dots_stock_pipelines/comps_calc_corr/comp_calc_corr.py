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
  date_ref : str,
  corr_rolling5_dataset : Output[Dataset] 
  ):

  # import FinanceDataReader as fdr
  # from ae_module.ae_logger import ae_log  
  from multiprocessing import Pool
  # from trading_calendars import get_calendar
  # cal_KRX = get_calendar('XKRX')

  import pandas as pd
  import numpy as np
  import pandas_gbq
  import requests
  import itertools

  df_adj_price = pd.read_pickle(adj_price_dataset.path)
  
  calc_period = 120 

  # l_code = df_adj_price.code.unique()

  import json

  def get_snapshot_markets(dates):
    '''market date
    Args: dates 
    '''
    
    global get_krx_marketcap
    def get_krx_marketcap(date_str):
      url = 'http://data.krx.co.kr/comm/bldAttendant/executeForResourceBundle.cmd?baseName=krx.mdc.i18n.component&key=B128.bld'
      j = json.loads(requests.get(url).text)
      # date_str = j['result']['output'][0]['max_work_dt']
      
      url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'
      data = {
          'bld': 'dbms/MDC/STAT/standard/MDCSTAT01501',
          'mktId': 'ALL',
          'trdDd': date_str,
          'share': '1',
          'money': '1',
          'csvxls_isNo': 'false',
      }
      j = json.loads(requests.post(url, data).text)
      df = pd.json_normalize(j['OutBlock_1'])
      df = df.replace(',', '', regex=True)
      numeric_cols = ['CMPPREVDD_PRC', 'FLUC_RT', 'TDD_OPNPRC', 'TDD_HGPRC', 'TDD_LWPRC', 
                      'ACC_TRDVOL', 'ACC_TRDVAL', 'MKTCAP', 'LIST_SHRS'] 
      df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
      
      df = df.sort_values('MKTCAP', ascending=False)
      cols_map = {'ISU_SRT_CD':'Code', 'ISU_ABBRV':'Name', 
                  'TDD_CLSPRC':'Close', 'SECT_TP_NM': 'Dept', 'FLUC_TP_CD':'ChangeCode', 
                  'CMPPREVDD_PRC':'Changes', 'FLUC_RT':'ChagesRatio', 'ACC_TRDVOL':'Volume', 
                  'ACC_TRDVAL':'Amount', 'TDD_OPNPRC':'Open', 'TDD_HGPRC':'High', 'TDD_LWPRC':'Low',
                  'MKTCAP':'Marcap', 'LIST_SHRS':'Stocks', 'MKT_NM':'Market', 'MKT_ID': 'MarketId' }
      df = df.rename(columns=cols_map)
      df.index = np.arange(len(df)) + 1
      # df['date'] = date_str
      return df.assign(date=date_str, 
                      rank_pct = 
                        lambda df: df.ChagesRatio
                                      .rank(method='first', ascending=False)
                                      .astype(int), 
                      in_top30 = 
                        lambda df: df.rank_pct <= 30
                          )
      # return df

    with Pool(20) as pool:
      result = pool.map(get_krx_marketcap, dates)
    df_market_ = pd.concat(result)

    return df_market_

  df_market = get_snapshot_markets([date_ref])

  l_code = (df_market
  [lambda df: df.Market.isin(['KOSPI','KOSDAQ'])]
  [lambda df: ~df.Dept.str.contains('투자주의|SPAC|관리종목')]
  .Code.unique().tolist()
  )  

  # global get_df_corr
  def get_df_corr(codes):

    df_corr = (df_adj_price #[lambda df: df.code.isin(codes)]
                .reset_index()
                .rename(columns=lambda x: x.lower())
                .pivot_table(values='close', index='date', columns='code')
                .iloc[-55:, :]
                .rolling(5)
                .mean()
                .iloc[4:,]
                .dropna(axis=1, how='any')
                .corr()
              )

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

    return sr_corr_

  # c_of_codes = list(itertools.combinations(l_code, 2))

  # with Pool(100) as pool:
  #   df_r = pool.map(get_df_corr, c_of_codes)

  # sr_corr = pd.concat(df_r)

  # sr_corr = pd.DataFrame()
  # for codes in samples:
  #   sr_corr_ = get_df_corr(codes)
  #   sr_corr = sr_corr.append(sr_corr_)

  # sub_codes = l_code[0:10]

  sr_corr = get_df_corr(l_code)

  def send_to_gbq(df_corr):
    table_schema = [{'name':'date','type':'DATE'}]
    pandas_gbq.to_gbq(df_corr, 
                    'krx_dataset.corr_ohlc_roll_mean_5_120days', 
                    project_id='dots-stock', 
                    if_exists='append',
                    table_schema=table_schema)
  print(f'size : {sr_corr.shape}')
  sr_corr.to_pickle(corr_rolling5_dataset.path)

