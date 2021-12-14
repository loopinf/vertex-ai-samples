from kfp.v2.dsl import (Dataset, Input, Output)

def calc_market_watch(
  date_ref: str,
  comp_result : str,
  ):
  import pandas as pd
  import numpy as np
  import pandas_gbq # type: ignore
  import time
  from trading_calendars import get_calendar
  cal_krx = get_calendar('XKRX')

  from pandas.tseries.offsets import CustomBusinessDay
  cbday = CustomBusinessDay(holidays=cal_krx.adhoc_holidays)
    

  def get_df_market(date_ref, n_before):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    date_ref_b = (pd.Timestamp(date_ref) - pd.Timedelta(n_before, 'd')).strftime('%Y-%m-%d')
    sql = f'''
      SELECT
        *
      FROM
        `dots-stock.red_lion.df_markets_clust_parti`
      WHERE
        date between "{date_ref_b}" and "{date_ref_}"
      ''' 
    PROJECT_ID = 'dots-stock'
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    return df

  df_markets_1 =get_df_market(date_ref, 20)

  def get_n_day_straight_up(NN):
  
    df_markets_ = (df_markets_1
    [lambda df: df.date >= pd.Timestamp(date_ref) - (NN-1) * cbday ]
    .sort_values('date', ascending=True)
    )
    l_N_d_up = (df_markets_
    [lambda df: df.Open != 0]  # Open 가격이 0 인 경우 그날 거래 없었던 것
    .assign(
        oc=lambda df: (df.Close - df.Open)/df.Open,
    )
    [lambda df: df.oc > 0]
    [lambda df: df.ChagesRatio > 0]
    .groupby(['Name'])
    [['Code']].agg('count')
    .rename(columns={'Code':'count_Nd_up'})
    [lambda df: df.count_Nd_up == NN]
    ).index.to_list()
    return l_N_d_up

  def get_n_day_straight_dn(NN):
    
    df_markets_ = (df_markets_1
    [lambda df: df.date >= pd.Timestamp(date_ref) - (NN-1) * cbday ]
    .sort_values('date', ascending=True)
    )
    l_N_d_up = (df_markets_
    [lambda df: df.Open != 0]  # Open 가격이 0 인 경우 그날 거래 없었던 것
    .assign(
        oc=lambda df: (df.Close - df.Open)/df.Open,
    )
    [lambda df: df.oc < 0]
    [lambda df: df.ChagesRatio < 0]
    .groupby(['Name'])
    [['Code']].agg('count')
    .rename(columns={'Code':'count_Nd_up'})
    [lambda df: df.count_Nd_up == NN]
    ).index.to_list()
    
    return l_N_d_up

  def get_n_day_straight_up_last_dn(NN):
    '''연속 몇일 오르고 마지막 내린 종목
    Return : list 종목명
    '''
    df_markets_ = (df_markets_1
    [lambda df: df.date >= pd.Timestamp(date_ref) - (NN-1) * cbday ]
    .sort_values('date', ascending=True)
    )
    l_Nd_dn_last_up = (df_markets_
      [lambda df: df.Open != 0]  # Open 가격이 0 인 경우 그날 거래 없었던 것
      .assign(
          oc=lambda df: (df.Close - df.Open)/df.Open,
          last_day=lambda df: df['date'] == pd.Timestamp(date_ref),
          last_day_down = 
            lambda df: (df.last_day ==  True) & (df.oc < 0),
          rest_day_up = 
            lambda df: (df.last_day ==  False) & (df.oc > 0),
          both_met = 
            lambda df: (df.last_day_down | df.rest_day_up),
      )
      # filter 조건 맞는 경우만
      .loc[lambda df: df.both_met == True]
      # groupby 해서 조건맞는 경우가 종목당 6개 인지 확인
      .groupby('Name')
      [['Code']].agg('count')
      .rename(columns={'Code':'count_Nd_up'})
      [lambda df: df.count_Nd_up == NN]
      # [lambda df: df['Code'] == NN]
    ).index.to_list()
    return l_Nd_dn_last_up

  def get_n_day_straight_dn_last_up(NN):
    '''연속 몇일 내리고 마지막 오른 종목
    Return : list 종목명
    '''
    df_markets_ = (df_markets_1
    [lambda df: df.date >= pd.Timestamp(date_ref) - (NN-1) * cbday ]
    .sort_values('date', ascending=True)
    )
    l_Nd_dn_last_up = (df_markets_
      [lambda df: df.Open != 0]  # Open 가격이 0 인 경우 그날 거래 없었던 것
      .assign(
          oc=lambda df: (df.Close - df.Open)/df.Open,
          last_day=lambda df: df['date'] == pd.Timestamp(date_ref),
          last_day_down = 
            lambda df: (df.last_day ==  True) & (df.oc > 0),
          rest_day_up = 
            lambda df: (df.last_day ==  False) & (df.oc < 0),
          both_met = 
            lambda df: (df.last_day_down | df.rest_day_up),
      )
      # filter 조건 맞는 경우만
      .loc[lambda df: df.both_met == True]
      # groupby 해서 조건맞는 경우가 종목당 6개 인지 확인
      .groupby('Name')
      [['Code']].agg('count')
      .rename(columns={'Code':'count_Nd_up'})
      [lambda df: df.count_Nd_up == NN]
      # [lambda df: df['Code'] == NN]
    ).index.to_list()
    return l_Nd_dn_last_up

  def make_df_with_func_2(func1, date_ref):
    df_market_watch = \
      pd.DataFrame(columns=['date_ref','Ndays', 'codes', 'num_codes'])
    NN = 4
    
    while(True):
      # print(NN)
      # l_code = get_n_day_straight_up_last_dn(NN)
      l_code = func1(NN)
      num_codes = len(l_code)
    
      if num_codes == 0:
        break
      
      df_market_watch = \
      (df_market_watch.append(
          {'date_ref': pd.to_datetime(date_ref).strftime('%Y-%m-%d'), 
          'Ndays': NN, 
          'codes': l_code,
          'num_codes': num_codes}, 
          ignore_index=True) )
      NN += 1
    return df_market_watch

  df_계속내리다가_오른목록 = make_df_with_func_2(get_n_day_straight_dn_last_up, date_ref)
  df_계속내리다가_오른목록['categ'] = 'DNDN_UP'
  df_계속오르다가_내린목록 = make_df_with_func_2(get_n_day_straight_up_last_dn, date_ref)
  df_계속오르다가_내린목록['categ'] = 'UPUP_DN'
  df_계속_내린목록 = make_df_with_func_2(get_n_day_straight_dn, date_ref)
  df_계속_내린목록['categ'] = 'DNDNDN'
  df_계속_오른목록 = make_df_with_func_2(get_n_day_straight_up, date_ref)
  df_계속_오른목록['categ'] = 'UPUPUP'

  df_market_watch = pd.concat([
    df_계속내리다가_오른목록,
    df_계속오르다가_내린목록,
    df_계속_오른목록,
    df_계속_내린목록 
    ])

  from google.cloud import bigquery
  project_id = 'dots-stock'
  client = bigquery.Client(project_id)

  def to_gbq_table(df, date_ref, table_name, if_exists='append'):
    schema = [
        bigquery.SchemaField(name="date_ref", field_type="DATE"),
        bigquery.SchemaField(name="Ndays", field_type="INT64"),
        bigquery.SchemaField(name="codes", field_type="STRING", mode='REPEATED'),
        bigquery.SchemaField(name="num_codes", field_type="INT64"),
        bigquery.SchemaField(name="categ", field_type="STRING"),

    ]
    table_id = f'red_lion.{table_name}_{date_ref}'
    table = bigquery.Table(
        f"{project_id}.{table_id}",
        schema=schema
    )
    print(table)
    if if_exists=='replace':
      client.query(f'DROP TABLE IF EXISTS {table_id}')
      
    try:
      client.create_table(table)
    except Exception as e:
      print(e)
      if ('Already Exists' in e.args[0]): # and if_exists=='replace' : 
        pass
      else: raise
    time.sleep (0.5)
    errors = client.insert_rows_from_dataframe(table, df)
    for chunk in errors:
      print(f"encountered {len(chunk)} errors: {chunk}")
      if len(errors) > 1: raise
    print(len(errors))

  to_gbq_table(df_market_watch, date_ref, 'market_watch')