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

  import pandas as pd
  import numpy as np
  import deepgraph as dg
  import pandas_gbq
  import os
  # from ae_module.ae_logger import ae_log
  import pickle
  import requests

  from multiprocessing import Pool

  df_adj_price = pd.read_pickle(adj_price_dataset.path)

  df_adj_roll5 = (df_adj_price
                  .pivot_table(values='close', index='date', columns='code')
                  .rolling(5)
                  .mean()
                  .iloc[-120:, :]
                  # .stack()
                  # .to_frame()
                  # .reset_index()
                  # .rename(columns={0:'mean'})
                  )

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

  global create_ei

  class ae_calc_corr_deepgraph():
    def __init__(self):
        self.date_ref = ""

        os.makedirs(PATH_CORRELATIONS, exist_ok=True)
        os.makedirs(PATH_TEMP, exist_ok=True)

    def create_calc_corr(self, N_date_corr, df_date_base):
        df_date_oc = df_date_base.tail(N_date_corr)
        X = df_date_oc.values

        X = X.T
        # # whiten variables for fast parallel computation later on
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
        print('X Looks like', X)
        # load samples as memory-map
        # X = np.load('samples.npy', mmap_mode='r')

        # create node table that stores references to the mem-mapped samples
        v = pd.DataFrame({'index': range(X.shape[0])})

        return X, v

    def store_correlation_values(self):
        # store correlation values
        file_list = os.listdir(PATH_CORRELATIONS)
        file_list.sort()
        
        # store = pd.HDFStore(os.path.join(PATH_TEMP, 'e.h5'), mode='w')
        # for f in files:
        #   et = pd.read_pickle(os.path.join(PATH_CORRELATIONS, f'{f}'))
        #   store.append('e', et, format='t', data_columns=True, index=False)
        # store.close()

        e_value = pd.DataFrame()
        if len(file_list) > 0:
          for f in file_list:
              et = pd.read_pickle(os.path.join(PATH_CORRELATIONS, f'{f}'))
              e_value = e_value.append(et)

        return e_value

        # connector function to compute pairwise pearson correlations

  # parallel computation
  def create_ei(i):

      # initiate DeepGraph
      g = dg.DeepGraph(g_V)

      # create edges
      g.create_edges(connectors=corr, step_size=STEP_SIZE)

      # store edge table
      et = g.e
      et_trimmed = (
          et[et > THRESHOLD_CORR].dropna()
      )
     
      if len(et_trimmed) == 0: 
        return (
          et_trimmed
            .to_pickle(os.path.join(PATH_CORRELATIONS, f'{str(i).zfill(3)}.pickle'))
        )

  def corr(index_s, index_t):
      features_s = g_X[index_s]    # source instance 
      features_t = g_X[index_t]
      corr = np.einsum('ij,ij->i', features_s, features_t) / g_samples
      return corr

  def drop_fill_ticker_w_na(df_oc, n_base):
      '''na 갯수가 얼마 이하인 경우 제외 1/10 '''
      num_na_stocks = df_oc.isna().sum()
      codes_filtered = num_na_stocks.loc[num_na_stocks <= n_base // 10].index
      return df_oc.loc[:,codes_filtered]

  def parallel_calc(indices):     
      # Pool().imap(create_ei, indices)
      with Pool(10) as pool:
          pool.imap(create_ei, indices)

  # parameters (change these to control RAM usage)
  STEP_SIZE = 1e4
  N_PROCESS = 5
  BASE_DIR = os.path.dirname(os.path.abspath(''))
  # BASE_DIR = '/gcs/pipeline-dots-stock/'
  PATH_TEMP = os.path.join(BASE_DIR, 'temp')
  PATH_CORRELATIONS = os.path.join(BASE_DIR, 'correlations')
  THRESHOLD_CORR = 0.6

  n_base_data_set = [60]

  # global values for multiprocess
  global g_X
  global g_V
  global g_features
  global g_samples
  global g_pos_array
  g_X = None         #source?
  g_V = None
  g_features = 0
  g_samples = 0
  g_pos_array = None

  ae_deepgraph = ae_calc_corr_deepgraph()  

  dic_edge = {}

  for n_base_data in n_base_data_set:

    g_X, g_V = ae_deepgraph.create_calc_corr(n_base_data, df_adj_roll5)

    g_features = g_X.shape[0]
    g_samples = g_X.shape[1]

    # index array for parallelization
    g_pos_array = np.array(np.linspace(0, g_features * (g_features - 1) // 2, N_PROCESS), dtype=int)

    indices = np.arange(0, N_PROCESS - 1)

    # parallel_calc(indices)
    create_ei(1)

    e = ae_deepgraph.store_correlation_values()

    

    # e_ox = sum(e_result) / 3
    e_ox = e
    print(e_ox)

    df_date_oc = df_adj_roll5.tail(n_base_data)
    dic_map_종목코드 = dict(enumerate(df_date_oc.columns))

    # drop corr under 0.6
    df_edge = e_ox.where(abs(e_ox) > 0.7, np.nan)
    df_edge.dropna(axis=0, inplace=True)

    df = df_edge.reset_index()    
    print(df.columns)

    # df['s'] = df['s'].map(dic_map_종목코드)
    # df['t'] = df['t'].map(dic_map_종목코드)

    # dic_edge[f'corr_{date_ref}_N{n_base_data}'] = df
  
  df.to_pickle(corr_rolling5_dataset.path)

  edge_dict_dir = '/gcs/pipeline-dots-stock/edge_dict'#"/root/Data/edge_dict"

  DIC_PICKLE_NAME = f'{date_ref}_dic_edge_rolling5.pkl'
  CORR_DICT_FILE = os.path.join(edge_dict_dir, DIC_PICKLE_NAME)

  pickle.dump(dic_edge, open(CORR_DICT_FILE, 'wb'))

  # def get_df_edgelist(dic_edge):
  #   # dic_edge = pickle.load(open(path_dic, 'rb'))
  #   df_ed = pd.DataFrame()
  #   for k,df in dic_edge.items():    
  #     _, date_ref, periods = k.split('_')
  #     df['date'] = date_ref
  #     df['period'] = int(periods.strip('N'))
  #     df.rename(columns={'s':'source', 't':'target', 0:'corr_value'}
  #               , inplace=True)
  #     df_ed = df_ed.append(df)
  #   return df_ed

  # df_edges = get_df_edgelist(dic_edge)
  # df_edges['date'] = pd.to_datetime(df_edges.date).dt.date
  # table_schema = [{'name':'date','type':'DATE'}]
  # pandas_gbq.to_gbq(df_edges, 
  #                   'krx_dataset.corr_ohlc_part1', 
  #                   project_id='dots-stock', 
  #                   if_exists='append',
  #                   table_schema=table_schema)
  # # pandas_gbq.to_gbq(df_edges, 'krx_dataset.corr_ohlc', project_id='dots-stock', if_exists='append')


