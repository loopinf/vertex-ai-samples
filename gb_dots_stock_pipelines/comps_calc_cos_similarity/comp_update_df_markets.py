
def update_df_markets(
  date_ref: str,
) -> str :
  """ Update df_markets with the latest market data.
  create market watch data 
  market_snapshot include 신호등 top30 
  """
  import json
  import logging
  logging.basicConfig(level=logging.DEBUG)
  import numpy as np
  from multiprocessing import Pool
  import pandas as pd
  import requests
  import pandas_gbq # type: ignore

  from trading_calendars import get_calendar
  cal_krx = get_calendar('XKRX')
  from pandas.tseries.offsets import CustomBusinessDay
  cbday = CustomBusinessDay(holidays = cal_krx.adhoc_holidays)

  project_id = 'dots-stock'
  from google.cloud import bigquery
  client = bigquery.Client(project_id)

  def get_snapshot_markets(dates):

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
    # if len(dates) == 1:
    #   return get_krx_marketcap(dates[0])
    with Pool(2) as pool:
      result = pool.map(get_krx_marketcap, dates)
    df_market_ = pd.concat(result)

    return df_market_

  df_markets_date_ref = get_snapshot_markets([date_ref])

  ########## custom business day 

  # code and name to gbq
  # df_markets_date_ref[['Code','Name','Market','Dept','Marcap']].to_gbq(
  #   f'red_lion.market_snapshot_{date_ref}', 'dots-stock', if_exists='replace')


  def push_data_to_gbq(df_markets):
      table_id = 'dots-stock.red_lion.df_markets_clust_parti'
      
      schema = [
          bigquery.SchemaField("Code", "STRING"),
          bigquery.SchemaField("Name", "STRING"),
          bigquery.SchemaField("Market", "STRING"),
          bigquery.SchemaField("Dept", "STRING"),
          # bigquery.SchemaField("MarketId", "STRING"),
          bigquery.SchemaField("Close", "INTEGER"),
          bigquery.SchemaField("Open", "INTEGER"),
          bigquery.SchemaField("High", "INTEGER"),
          bigquery.SchemaField("Low", "INTEGER"),
          bigquery.SchemaField("Volume", "INTEGER"),
          bigquery.SchemaField("Amount", "INTEGER"),
          # bigquery.SchemaField("ChangeCode", "INTEGER"),
          bigquery.SchemaField("Changes", "INTEGER"),
          bigquery.SchemaField("ChagesRatio", "FLOAT"),
          bigquery.SchemaField("Marcap", "INTEGER"),
          bigquery.SchemaField("Stocks", "INTEGER"),
          bigquery.SchemaField("date", "DATE"),
          bigquery.SchemaField("rank_pct", "INTEGER"),
          bigquery.SchemaField("in_top30", "BOOL"),
      ]
      
      columns_to_gbq = df_markets.columns.to_list()
      columns_to_gbq.remove('ChangeCode')
      columns_to_gbq.remove('MarketId')
      
      df_markets_gbq = \
        (df_markets
        .assign(date=lambda df: pd.to_datetime(df.date))
        [columns_to_gbq]
        .assign(Close=lambda df: df.Close.astype(int))
        )
      
      job_config = bigquery.LoadJobConfig(schema=schema)
      print(f'table_id : {table_id}')
      job = client.load_table_from_dataframe(
          df_markets_gbq, table_id, job_config=job_config
      )

  try :
    push_data_to_gbq(df_markets_date_ref)
    logging.debug(f'daily market data {date_ref} pushed to gbq ')
  except :
    print('Something Wrong')
    logging.error('Something Wrong on push market daily data to gbq')
    raise

  return 'finished' 