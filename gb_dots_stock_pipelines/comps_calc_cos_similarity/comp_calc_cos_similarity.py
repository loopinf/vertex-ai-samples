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
  df_markets: Input[Dataset],
  date_ref : str,
  cos_similars : Output[Dataset] 
  ):

  # from trading_calendars import get_calendar
  # cal_KRX = get_calendar('XKRX')
  today = date_ref

  import pandas as pd
  import numpy as np
  import pandas_gbq

  import torch
  from torch import nn
  from torch.nn import functional as F

  df_markets = pd.read_pickle(df_markets.path)

  #### filter df_markets --> df_markets_filtered
  df_markets_filtered = \
  (df_markets
    .loc[lambda df: ~df.Dept.str.contains('관리종목|투자주의')]
    .loc[lambda df: df.Market.isin(['KOSPI','KOSDAQ'])]
    .sort_values('date')
  )

  l_code = \
  (df_markets_filtered
    .Code.unique()
  )
  ####
  kernel_size = 6  # 10days to check
  weights_ratio = {'oc':2.5,'cc':2.5,'oh':1,'ol':1}

  # def get_cosi_(df_markets_filtered, code, kernel_size):
  df_oholoccc = (df_markets
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
    .pivot_table(values='oh', index='date', columns='Code', dropna=False))
  df_ol = (
    df_oholoccc
    .pivot_table(values='ol', index='date', columns='Code', dropna=False))
  df_oc = (
    df_oholoccc
    .pivot_table(values='oc', index='date', columns='Code', dropna=False))
  df_cc = (
    df_oholoccc
    .pivot_table(values='cc', index='date', columns='Code', dropna=False))

  oh_tensor = (torch.from_numpy(df_oh.transpose().values)
    [:, None, :]
  .unfold(dimension=2, size=kernel_size, step=1))
  ol_tensor = (torch.from_numpy(df_ol.transpose().values)
    [:, None, :]
  .unfold(dimension=2, size=kernel_size, step=1))
  oc_tensor = (torch.from_numpy(df_oc.transpose().values)
    [:, None, :]
  .unfold(dimension=2, size=kernel_size, step=1))
  cc_tensor = (torch.from_numpy(df_cc.transpose().values)
    [:, None, :]
  .unfold(dimension=2, size=kernel_size, step=1))
  # (oc_tensor *2).shape
  # (ol_tensor *2).shape
  input_tensor = torch.cat([
                            weights_ratio['oh'] *oh_tensor, 
                            weights_ratio['oc'] *oc_tensor, 
                            weights_ratio['ol'] *ol_tensor, 
                            weights_ratio['cc'] *cc_tensor, 
                            ], dim=1)

  # input_tensor.shape  # torch.Size([2663, 4, 1437, 10])   종목갯수, 4개 (oh ol oh cc), 날짜묶음 갯수, 필터길이

  input_tensor1 = (input_tensor.permute(0, 2, 1, 3)
  .flatten(start_dim=2, end_dim=3)
  )
  # .shape # torch.Size([2663, 1437, 40])   종목갯수(2663), 날짜묶음 갯수(1437), 필터길이 x (oh ol oh cc)(40), 

  cols_code = df_oc.columns
  cols_date = df_oc.index

  def get_vector_from_code(input_tensor, code):
    '''
    Args:
      input_tensor : Tensor  shape  ex) torch.Size([2663, 1437, 40]) 
      code  : "000020"  
      '''
    index_code = cols_code.get_loc(code)
    filter = input_tensor[index_code][-1]  # -1 means last date
    return filter

  unfolded = input_tensor1.cuda(0)

  def _get_co_si(unfolded, code, kernel_size):
    # get sample 
    tensor_code = get_vector_from_code(unfolded, code)

    # calc cosine similarity
    similarity = F.cosine_similarity(unfolded, tensor_code, dim=-1) 

    # 
    df_simil = pd.DataFrame(
        similarity.cpu().numpy().transpose(),
        index = cols_date[kernel_size-1: ],
        columns = cols_code
    )
    return df_simil
    
  def get_simil(unfolded, code, date_ref, kernel_size):
    df = (
      _get_co_si(unfolded, code, kernel_size
                ) 
      .where(lambda x: (.85< x))
      .stack()
      .sort_values(ascending=False)
      .to_frame()
      .rename(columns={0:'similarity'})
      .assign(source_code=code,
              source_date=date_ref)
      .reset_index()
        )
    return df

  # takes 3:53s
  ## 11-23껄로 해보자
  l_df = []
  for code in l_code[10]:
    try:
      _df = get_simil(unfolded, code, today, kernel_size)
    except Exception as e:
      print(code)
      print(e)
    l_df.append(_df.head(100))

  df_simil_gbq = pd.concat(l_df)

  df_simil_gbq['date'] = pd.to_datetime(df_simil_gbq.date).dt.strftime('%Y-%m-%d')
  df_simil_gbq['source_date'] = pd.to_datetime(df_simil_gbq.source_date).dt.strftime('%Y-%m-%d')

  def send_to_gbq(df_simil_gbq):
    table_schema = [{'name':'date','type':'DATE'},
                {'name':'source_date','type':'DATE'},
                {'name':'similarity','type':'FLOAT64'},
                {'name':'Code','type':'STRING'},
                {'name':'source_code','type':'STRING'},
                ]
    pandas_gbq.to_gbq(df_simil_gbq, 
                  f'red_lion.pattern_{kernel_size}_{today}',
                    project_id='dots-stock', 
                    if_exists='append',
                    table_schema=table_schema
                  )

  send_to_gbq(df_simil_gbq)
  print(f'size : {df_simil_gbq.shape}')
  # df_simil_gbq.to_pickle(cos_similars.path)


