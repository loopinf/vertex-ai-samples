from datetime import date
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def calc_cos_similar(
  adj_price_dataset: Input[Dataset],
  date_ref : str,
  cos_similars : Output[Dataset] 
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

  import torch
  from torch import nn
  from torch.nn import functional as F

  # df_adj_price = pd.read_pickle(adj_price_dataset.path)

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

  df_markets = get_snapshot_markets([date_ref])

  # df_markets_filterd =  관리종목, 투자유의 제외, 코넥스 제외
  df_markets_filtered = (df_markets
    .loc[lambda df: ~df.Dept.str.contains('관리종목|투자주의')]
    .loc[lambda df: df.Market.isin(['KOSPI','KOSDAQ'])]
    .sort_values('date')
    )

  l_codes = df_markets_filtered.Code.unique().tolist()
  l_codes_temp = l_codes[0 : 10]
  df_markets_filtered = df_markets_filtered[df_markets_filtered.Code.isin(l_codes_temp)]

  print(f'shpe of df_markets : {df_markets_filtered.shape}')

  #df_oholoccc
  df_oholoccc = (df_markets_filtered
    .assign(Close = 
            lambda df: df.Close.astype(int))
    .assign(oh=
            lambda df: (df.High - df.Open)/df.Open,
            ol=
            lambda df: (df.Low - df.Open)/df.Open,
            oc=
            lambda df: (df.Close - df.Open)/df.Open,
            cc=
            lambda df: df.ChagesRatio/100
            )
    .loc[:, ['Code', 'Name','date','Volume','oh','ol','oc','cc']]        
    )
  
  df_oh = (
    df_oholoccc
    .pivot_table(values='oh', index='date', columns='Code'))
  df_ol = (
    df_oholoccc
    .pivot_table(values='ol', index='date', columns='Code'))
  df_oc = (
    df_oholoccc
    .pivot_table(values='oc', index='date', columns='Code'))
  df_cc = (
    df_oholoccc
    .pivot_table(values='cc', index='date', columns='Code'))

  # shape ... 종목, NONE, 날짜
  def _get_simil(df, code, kernel_size ):

    input = torch.from_numpy(
      df.transpose().values[:, np.newaxis, :] )

    unfolded_ = input.unfold(2, kernel_size, 1)

    # sample 의 기준 날짜는 df 의 제일 최종 날짜와 동일함
    sample_filter = \
    (df.loc[:, code]
    .iloc[-kernel_size:]
    .values
    )

    sample_filter = torch.from_numpy(
        sample_filter[np.newaxis, np.newaxis, :] #쎄미시스코 9/30
    )
    # sample_filter.shape #torch.Size([1, 1, 50])  batch, channel, filter_length
    similarity_result = F.cosine_similarity(unfolded_, sample_filter, dim=-1)
    similarity_result.shape
    df_simil = similarity_result.squeeze()
    similarity_result.shape
    
    df_simil = pd.DataFrame(
        similarity_result.squeeze().permute(1,0).numpy(),
        index=df.index[kernel_size-1:],
        columns = df.columns
    )
    return df_simil
  
  def get_xx_simil(code: str, length: int):
    oc_sim = _get_simil(df_oc, code, length)
    oh_sim = _get_simil(df_oh, code, length)
    ol_sim = _get_simil(df_ol, code, length)
    cc_sim = _get_simil(df_cc, code, length)
    xx_sim = (2*oc_sim + oh_sim + ol_sim + cc_sim)
    return xx_sim

  def get_simil(code, date_ref, length_simi):
    a = (
      get_xx_simil(code, length=length_simi) 
      .where(lambda x: (4.5< x))
      .stack()
      .sort_values(ascending=False)
      .to_frame()
      .rename(columns={0:'similarity'})
      .assign(source_code=code,
              source_date=date_ref)
      .reset_index()
        )
    return a

  df_simil_gbq = pd.DataFrame()

  for code in l_codes:
    try:
      _df = get_simil(code, date_ref, 10)
    except Exception as e:
      print(code)
      print(e)
      continue
    df_simil_gbq = df_simil_gbq.append(_df)


  def send_to_gbq(df_corr):
    table_schema = [{'name':'date','type':'DATE'}]
    pandas_gbq.to_gbq(df_corr, 
                    'krx_dataset.corr_ohlc_roll_mean_5_120days', 
                    project_id='dots-stock', 
                    if_exists='append',
                    table_schema=table_schema)

  print(f'size : {df_simil_gbq.shape}')
  df_simil_gbq.to_pickle(cos_similars.path)

