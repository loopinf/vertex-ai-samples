from datetime import date
import logging
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def calc_cos_similar_occc(
  date_ref : str,
	kernel_size : int,
  comp_result : str,
  cos_similars : Output[Dataset] 
  ):
  # return 'finished'
  import logging
  print('calc_cos_similar')
  logging.basicConfig(level=logging.DEBUG)
  logging.debug('calc_cos_similar')
  print(cos_similars.path)

# def calc_cos_similar_occc(
#   # df_markets: Input[Dataset],
#   date_ref : str,
# 	kernel_size : int,
#   comp_result : str,
#   ) -> str:

#   # from trading_calendars import get_calendar
#   # cal_KRX = get_calendar('XKRX')

#   import pandas as pd
#   import numpy as np
#   import pandas_gbq # type: ignore
#   import logging
#   logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

#   import torch # type: ignore
#   from torch.nn import functional as F # type: ignore


#   ######## load data ########
#   # testing_url_markets = '/gcs/red-lion/df_market/df_markets_snap_latest.pkl'
#   # df_markets = pd.read_pickle(df_markets.path)
#   # df_markets = pd.read_pickle(testing_url_markets)
#   def get_df_markets(date_ref):
#     date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
#     ### Volume is not used in this model
#     sql = f'''
#       SELECT
#         Code,
#         Open,
#         High,
#         Low,
#         Close,
#         ChagesRatio,
#         Dept,
#         Market,
#         date,
#         Name
#       FROM
#         `dots-stock.red_lion.df_markets_clust_parti`
#       WHERE
#         date <= "{date_ref_}"
#       ORDER BY
#         date
#       ''' 
#     PROJECT_ID = 'dots-stock'
#     df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
#     return df

#   def check_count_of_df_market(date_ref):
#     date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
#     sql = f'''
#       SELECT
#         count(*)
#       FROM
#         `dots-stock.red_lion.df_markets_clust_parti`
#       WHERE
#         date = "{date_ref_}"
#       LIMIT
#         1000
#       ''' 
#     PROJECT_ID = 'dots-stock'
#     df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
#     result = df.iloc[0,0]
#     return result
  
#   assert check_count_of_df_market(date_ref) > 2000 , f'df_markets on {date_ref} is not available'

#   df_markets = get_df_markets(date_ref)
#   assert df_markets.duplicated(subset=['date','Code']).sum() == 0
#   #### filter df_markets --> df_markets_filtered
#   l_code  = \
#   (df_markets
#     .loc[lambda df: ~df.Dept.str.contains('관리종목|투자주의')]
#     .loc[lambda df: df.Market.isin(['KOSPI','KOSDAQ'])]
#     .sort_values('date')
#     .Code.unique().tolist()
#   )

#   # l_code_1, l_code_2, l_code_3, l_code_4 = np.array_split(l_code, 4)

#   ####
#   # kernel_size = 6  # 10days to check
#   weights_ratio = {'oc':2.5,'cc':2.5,'oh':1,'ol':1}

#   # def get_cosi_(df_markets_filtered, code, kernel_size):
#   df_oholoccc = (df_markets
#     .assign(Close = 
#             lambda df: df.Close.astype(int))
#     .assign(
#             # oh=
#             # lambda df: (df.High - df.Open)/df.Open,
#             # ol=
#             # lambda df: (df.Low - df.Open)/df.Open,
#             oc=
#             lambda df: (df.Close - df.Open)/df.Open,
#             cc=
#             lambda df: df.ChagesRatio/100
#             )
#     .loc[:, ['Code', 'Name','date','oc','cc']]        
#   )

#   df_oc = (
#     df_oholoccc
#     .pivot_table(values='oc', index='date', columns='Code', dropna=False))
#   df_cc = (
#     df_oholoccc
#     .pivot_table(values='cc', index='date', columns='Code', dropna=False))

#   oc_tensor = (torch.from_numpy(df_oc.transpose().values)
#     [:, None, :]
#   .unfold(dimension=2, size=kernel_size, step=1))
#   cc_tensor = (torch.from_numpy(df_cc.transpose().values)
#     [:, None, :]
#   .unfold(dimension=2, size=kernel_size, step=1))
#   # (oc_tensor *2).shape
#   # (ol_tensor *2).shape
#   input_tensor = torch.cat([
#                             weights_ratio['oc'] *oc_tensor, 
#                             weights_ratio['cc'] *cc_tensor, 
#                             ], dim=1)

#   # input_tensor.shape  # torch.Size([2663, 4, 1437, 10])   종목갯수, 4개 (oh ol oh cc), 날짜묶음 갯수, 필터길이

#   input_tensor1 = (input_tensor.permute(0, 2, 1, 3)
#   .flatten(start_dim=2, end_dim=3)
#   )
#   # .shape # torch.Size([2663, 1437, 40])   종목갯수(2663), 날짜묶음 갯수(1437), 필터길이 x (oh ol oh cc)(40), 
#   logging.debug(f'input_tensor1.shape: {input_tensor1.shape}')

#   cols_code = df_oc.columns
#   cols_date = df_oc.index

#   def get_vector_from_code(input_tensor, code):
#     '''
#     Args:
#       input_tensor : Tensor  shape  ex) torch.Size([2663, 1437, 40]) 
#       code  : "000020"  
#       '''
#     index_code = cols_code.get_loc(code)
#     filter = input_tensor[index_code][-1]  # -1 means last date
#     return filter

#   unfolded = input_tensor1.cuda(0)

#   def _get_co_si(unfolded, code, kernel_size):
#     # get sample 
#     tensor_code = get_vector_from_code(unfolded, code)

#     # calc cosine similarity
#     similarity = F.cosine_similarity(unfolded, tensor_code, dim=-1) 

#     # 
#     df_simil = pd.DataFrame(
#         similarity.cpu().numpy().transpose(),
#         index = cols_date[kernel_size-1: ],
#         columns = cols_code
#     )
#     return df_simil
    
#   def get_simil(unfolded, code, date_ref, kernel_size):
#     df = (
#       _get_co_si(unfolded, code, kernel_size
#                 ) 
#       .where(lambda x: (.65< x))
#       .stack()
#       .sort_values(ascending=False)
#       .to_frame()
#       .rename(columns={0:'similarity'})
#       .assign(source_code=code,
#               source_date=date_ref)
#       .reset_index()
#         )
#     return df

#   # takes 3:53s
#   ## 11-23껄로 해보자
#   l_df = []

#   for code in l_code:
#     try:
#       _df = get_simil(unfolded, code, date_ref, kernel_size)
#       l_df.append(_df.head(100))
#     except Exception as e:
#       logging.error(f'error on code : {code}')
#   logging.debug(f'loop done , l_df : {len(l_df)}')

#   df_simil_gbq = pd.concat(l_df)

#   df_simil_gbq['date'] = pd.to_datetime(df_simil_gbq.date)#.dt.strftime('%Y-%m-%d')
#   df_simil_gbq['source_date'] = pd.to_datetime(df_simil_gbq.source_date)#.dt.strftime('%Y-%m-%d')

#   # save similarity result to gcs
#   df_count = (
#               df_simil_gbq
#               .where(df_simil_gbq.similarity >= 0.9) # first threshold
#               .dropna()
#               .groupby('source_code')['source_code'].count().reset_index(name='count')
#               .where(lambda df : df['count'] > 10) # second threshold
#               .dropna()
#               )   

#   code_list = df_count['source_code'].tolist()
#   save_path = f'/gcs/red-lion/similarity_result/similarity_kernel_size_{kernel_size}_{date_ref}.json'

#   import json
#   with open(save_path, 'w') as f:
#     json.dump(code_list, f)

#   ###### table create

#   from google.cloud import bigquery
#   project_id='dots-stock' 
#   client = bigquery.Client(project_id)
#   # TODO(developer): Set table_id to the ID of the table to create.
#   table_id = f"{project_id}.red_lion.pattern_oc_cc_{kernel_size}_{date_ref}"
#   # date	Code	similarity	source_code	source_date	
#   schema = [
#       bigquery.SchemaField("date", "DATE"),
#       bigquery.SchemaField("source_date", "DATE"),
#       bigquery.SchemaField("similarity", "FLOAT64"),
#       bigquery.SchemaField("Code", "STRING"),
#       bigquery.SchemaField("source_code", "STRING"),
#   ]

#   table = bigquery.Table(table_id, schema=schema)
#   table.clustering_fields = ["source_code"] 
#   try:
#     table = client.create_table(table)  # Make an API request.
#     print(
#       "Created clustered table {}.{}.{}".format(
#           table.project, table.dataset_id, table.table_id)
#     )
#   except Exception as e:
#     print(e)
#     print("table already exists")


#   # upload table
#   def upload_to_gbq(df_simil_gbq):
#     table_schema =[
#       {'name':'date','type':'DATE'},
#       {'name':'source_date','type':'DATE'},
#       {'name':'similarity','type':'FLOAT'},
#       {'name':'Code','type':'STRING'},
#       {'name':'source_code','type':'STRING'}, ]
#     pandas_gbq.to_gbq(df_simil_gbq, 
#                   f'{table.dataset_id}.{table.table_id}',
#                     project_id='dots-stock', 
#                     # if_exists='replace', # 있는 경우 삭제하는 것에는 permission이 필요함
#                     if_exists='append',
#                     table_schema=table_schema
#                   )

#   upload_to_gbq(df_simil_gbq)

#   return 'finish'
