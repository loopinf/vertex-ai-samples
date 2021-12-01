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
    .Code.unique().tolist()
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
  for code in l_code:
    print(f'code : {code}')
    try:
      _df = get_simil(unfolded, code, today, kernel_size)
      l_df.append(_df.head(100))
    except Exception as e:
      print(code)
      print(e)

  df_simil_gbq = pd.concat(l_df)

  df_simil_gbq['date'] = pd.to_datetime(df_simil_gbq.date)#.dt.strftime('%Y-%m-%d')
  df_simil_gbq['source_date'] = pd.to_datetime(df_simil_gbq.source_date)#.dt.strftime('%Y-%m-%d')

  ###### table create

  from google.cloud import bigquery
  project_id='dots-stock' 
  client = bigquery.Client(project_id)
  # TODO(developer): Set table_id to the ID of the table to create.
  table_id = f"{project_id}.red_lion.pattern_{kernel_size}_{today}"
  # date	Code	similarity	source_code	source_date	
  schema = [
      bigquery.SchemaField("date", "DATE"),
      bigquery.SchemaField("source_date", "DATE"),
      bigquery.SchemaField("similarity", "FLOAT64"),
      bigquery.SchemaField("Code", "STRING"),
      bigquery.SchemaField("source_code", "STRING"),
  ]

  table = bigquery.Table(table_id, schema=schema)
  table.clustering_fields = ["source_code"] 
  try:
    table = client.create_table(table)  # Make an API request.
    print(
      "Created clustered table {}.{}.{}".format(
          table.project, table.dataset_id, table.table_id)
    )
  except Exception as e:
    print(e)
    print("table already exists")


  # upload table
  def upload_to_gbq(df_simil_gbq):
    table_schema =[
      {'name':'date','type':'DATE'},
      {'name':'source_date','type':'DATE'},
      {'name':'similarity','type':'FLOAT'},
      {'name':'Code','type':'STRING'},
      {'name':'source_code','type':'STRING'}, ]
    pandas_gbq.to_gbq(df_simil_gbq, 
                  f'{table.dataset_id}.{table.table_id}',
                    project_id='dots-stock', 
                    # if_exists='replace', # 있는 경우 삭제하는 것에는 permission이 필요함
                    if_exists='append',
                    table_schema=table_schema
                  )

  upload_to_gbq(df_simil_gbq)


# #######
#   table_id = f'red_lion.pattern_{kernel_size}_{today}',
#   errors = client.insert_rows_from_dataframe(table_id, df_simil_gbq)
#   for chunk in errors:
#     print(f"encountered {len(chunk)} errors: {chunk}")

#   # def send_to_gbq(df_simil_gbq):
#   #   table_schema = [{'name':'date','type':'DATE'},
#   #               {'name':'source_date','type':'DATE'},
#   #               {'name':'similarity','type':'FLOAT'},
#   #               {'name':'Code','type':'STRING'},
#   #               {'name':'source_code','type':'STRING'},
#   #               ]
#   #   pandas_gbq.to_gbq(df_simil_gbq, 
#   #                 f'red_lion.pattern_{kernel_size}_{today}',
#   #                   project_id='dots-stock', 
#   #                   if_exists='append',
#   #                   table_schema=table_schema
#   #                 )

#   # send_to_gbq(df_simil_gbq)
#   print(f'size : {df_simil_gbq.shape}')
#   # df_simil_gbq.to_pickle(cos_similars.path)


