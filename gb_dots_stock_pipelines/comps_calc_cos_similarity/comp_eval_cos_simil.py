from kfp.v2.dsl import (Dataset, Input, Output)
from typing import NamedTuple

from numpy import left_shift

def eval_cos_simil(
  date_ref: str,
  # calc_cos_simil_1: str,
  # calc_cos_simil_2: str,
  # calc_cos_simil_3: str,
  # calc_cos_simil_4: str,
  # df_markets_update: Output[Dataset],
) -> str :
  # print(calc_cos_simil_1, calc_cos_simil_2,calc_cos_simil_3, calc_cos_simil_4)
  import itertools
  import numpy as np
  import pandas_gbq # type: ignore 
  import pandas as pd
  from trading_calendars import get_calendar
  cal_krx = get_calendar('XKRX')

  from pandas.tseries.offsets import CustomBusinessDay
  cbday = CustomBusinessDay(holidays=cal_krx.adhoc_holidays)

  PROJECT_ID = 'dots-stock' 


  ########## get price ### TODO: check run after update_df_prices
  def get_price(date_ref):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    table_id = f"{PROJECT_ID}.red_lion.adj_price_{date_ref}"
    sql = f'''
      SELECT
        *
      FROM
        `{table_id}`
      # WHERE
      #   date = "{date_ref_}"
      ORDER BY
        date
      ''' 
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    return df

  df_price = get_price(date_ref)

  def check_min_max_date_codes():
    print(
        df_price.Date.min(), '\n',
        df_price.Date.max(), '\n',
        df_price.code.unique().__len__(), '\n',
    )
  check_min_max_date_codes()

  df_price_ = df_price.set_index('Date')
  df_open = df_price_.loc[:, ['code', 'Open']]
  df_open = df_open.pivot_table(values='Open', columns='code', index=df_open.index)
  df_open.index += pd.Timedelta('9h')
  df_close = df_price_.loc[:,['code', 'Close']]
  df_close = df_close.pivot_table(values='Close', columns='code', index=df_close.index)
  df_close.index += pd.Timedelta('15h30m')
  price = pd.concat([df_open, df_close], axis=0).sort_index()

  price1 = \
    (price.stack()
    .reset_index()
    .rename(columns={0:'adj_p'})
    )

  ############## get pattern #############

  def make_report_table_ohlc(date_ref_pattern, kernel_size):

    def get_pattern(date_ref, kernel_size):
      date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
      table_id = f"{PROJECT_ID}.red_lion.pattern_v2_{kernel_size}_{date_ref}"
      sql = f'''
        SELECT
          *
        FROM
          `{table_id}`
        WHERE
          source_date = "{date_ref_}"
        ORDER BY
          date
        ''' 

      df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
      return df

    def get_pattern_date_filtered(date_ref, kernel_size):
      return (get_pattern(date_ref, kernel_size)
        [lambda df: df.date != df.source_date] # not same date
        [lambda df: df.similarity > .95]
        .rename(columns=lambda col: col.lower()) )
    df = get_pattern_date_filtered(date_ref_pattern, kernel_size)

    N_points = 10
    for days in range(0, N_points): #TODO: improve speed, it took so long
      df[f'd{days}_1'] = df.date + pd.Timedelta('9h') + days * cbday     # 해당일 시가
      df[f'd{days}_2'] = df.date + pd.Timedelta('15h30m') + days * cbday # 해당일 종가
    cols_1 = ['date','code','source_code','source_date',]
    cols_d_op = list(
        itertools.chain.from_iterable(
            [(f'd{i}_1',f'd{i}_2') for i in range(0,N_points)])
    )
    cols_all = cols_1 + cols_d_op
    df_ = (df
      .rename(columns=lambda col: col.lower())
      .drop_duplicates(['code','date'])    # must to do  한 종목이 여러 소스종목과 동일한 경우가 있음 
      .melt(id_vars=cols_1, value_vars=cols_d_op)
      .sort_values(['code','variable'])
      .merge(price1, 
              left_on=['code', 'value'],
              right_on=['code','Date'], 
              how='left')
      .sort_values(['code','date','value'])
    )

    df_1 = (df_
      .assign(
          # base_p=lambda df: df.Date-df.date,
          # base_on=lambda df: df.base_p == pd.Timedelta('15h30m'),
          base_on = lambda df: df.variable == 'd0_2',
          adj_p_base=lambda df: df.adj_p.where(df.base_on),
          adj_p_base2=lambda df: df.adj_p_base.bfill(limit=1), #한 번 뒤로 밀어서 채우고,
          adj_p_base3=lambda df: df.adj_p_base2.ffill(),       # 나머지는 ffill 로 채움
          ret_p_d0_2 = lambda df: df.adj_p / df.adj_p_base3,
      )
      .drop([
              'base_on',
              'adj_p_base', 
              'adj_p_base2',
              'adj_p_base3', 
              ], axis=1
        )
      .assign(
          # base_on=lambda df: df.base_p == pd.Timedelta('1d9h'), #하루가 cbday  
          base_on = lambda df: df.variable == 'd1_1',
          adj_p_base=lambda df: df.adj_p.where(df.base_on),
          adj_p_base2=lambda df: df.adj_p_base.bfill(limit=2), #두번 뒤로 밀어서 채우고,
          adj_p_base3=lambda df: df.adj_p_base2.ffill(),       # 나머지는 ffill 로 채움
          ret_p_d1_1 = lambda df: df.adj_p / df.adj_p_base3
      )
      .drop(['base_on',
              'adj_p_base', 
              'adj_p_base2',
              'adj_p_base3',
              ], axis=1
        )
      )

    df_report_1 = (
        df_1
        .groupby(['source_code','variable'])['ret_p_d0_2', 'ret_p_d1_1']
        .agg(['mean','max','count'])
        )
    df_report_1.columns = df_report_1.columns.to_flat_index().str.join('_')

    df_report_3 = (df_1
        .sort_values(['source_code','variable'], ascending=True)
        .groupby(['source_code','variable'])['ret_p_d0_2', 'ret_p_d1_1']
        .apply(lambda col: (col > 1).sum())  # col > 1 상승한 경우 count
        )

    _df_report = \
        (pd.concat([
                  df_report_1,
                  df_report_3
                  ], axis=1
                  )
        .assign(
            ud_r_d0_2=
            lambda df: df.ret_p_d0_2 / df.ret_p_d0_2_count,
            ud_r_d1_1=
            lambda df: df.ret_p_d1_1 / df.ret_p_d1_1_count,
        ).reset_index()
        )
    df_to_gbq = (_df_report
      .assign(kernel_size=kernel_size,
              date_ref=date_ref)
      [lambda df: ~df.variable.isin(['d0_1','d0_2'])]
      [lambda df: df.ret_p_d1_1_count > 20]
      .sort_values('ud_r_d1_1', ascending=False)
      )
    
    return df_to_gbq


  def make_report_table_occc(date_ref_pattern, kernel_size):

    def get_pattern(date_ref, kernel_size):
      date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
      table_id = f"{PROJECT_ID}.red_lion.pattern_oc_cc_{kernel_size}_{date_ref}"
      sql = f'''
        SELECT
          *
        FROM
          `{table_id}`
        WHERE
          source_date = "{date_ref_}"
        ORDER BY
          date
        ''' 

      df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
      return df

    def get_pattern_date_filtered(date_ref, kernel_size):
      return (get_pattern(date_ref, kernel_size)
        [lambda df: df.date != df.source_date] # not same date
        [lambda df: df.similarity > .95]
        .rename(columns=lambda col: col.lower()) )
    df = get_pattern_date_filtered(date_ref_pattern, kernel_size)

    N_points = 10
    for days in range(0, N_points): #TODO: improve speed, it took so long
      df[f'd{days}_1'] = df.date + pd.Timedelta('9h') + days * cbday     # 해당일 시가
      df[f'd{days}_2'] = df.date + pd.Timedelta('15h30m') + days * cbday # 해당일 종가
    cols_1 = ['date','code','source_code','source_date',]
    cols_d_op = list(
        itertools.chain.from_iterable(
            [(f'd{i}_1',f'd{i}_2') for i in range(0,N_points)])
    )
    cols_all = cols_1 + cols_d_op
    df_ = (df
      .rename(columns=lambda col: col.lower())
      .drop_duplicates(['code','date'])    # must to do  한 종목이 여러 소스종목과 동일한 경우가 있음 
      .melt(id_vars=cols_1, value_vars=cols_d_op)
      .sort_values(['code','variable'])
      .merge(price1, 
              left_on=['code', 'value'],
              right_on=['code','Date'], 
              how='left')
      .sort_values(['code','date','value'])
    )

    df_1 = (df_
      .assign(
          # base_p=lambda df: df.Date-df.date,
          # base_on=lambda df: df.base_p == pd.Timedelta('15h30m'),
          base_on = lambda df: df.variable == 'd0_2',
          adj_p_base=lambda df: df.adj_p.where(df.base_on),
          adj_p_base2=lambda df: df.adj_p_base.bfill(limit=1), #한 번 뒤로 밀어서 채우고,
          adj_p_base3=lambda df: df.adj_p_base2.ffill(),       # 나머지는 ffill 로 채움
          ret_p_d0_2 = lambda df: df.adj_p / df.adj_p_base3,
      )
      .drop([
              'base_on',
              'adj_p_base', 
              'adj_p_base2',
              'adj_p_base3', 
              ], axis=1
        )
      .assign(
          # base_on=lambda df: df.base_p == pd.Timedelta('1d9h'), #하루가 cbday  
          base_on = lambda df: df.variable == 'd1_1',
          adj_p_base=lambda df: df.adj_p.where(df.base_on),
          adj_p_base2=lambda df: df.adj_p_base.bfill(limit=2), #두번 뒤로 밀어서 채우고,
          adj_p_base3=lambda df: df.adj_p_base2.ffill(),       # 나머지는 ffill 로 채움
          ret_p_d1_1 = lambda df: df.adj_p / df.adj_p_base3
      )
      .drop(['base_on',
              'adj_p_base', 
              'adj_p_base2',
              'adj_p_base3',
              ], axis=1
        )
      )

    df_report_1 = (
        df_1
        .groupby(['source_code','variable'])['ret_p_d0_2', 'ret_p_d1_1']
        .agg(['mean','max','count'])
        )
    df_report_1.columns = df_report_1.columns.to_flat_index().str.join('_')

    df_report_3 = (df_1
        .sort_values(['source_code','variable'], ascending=True)
        .groupby(['source_code','variable'])['ret_p_d0_2', 'ret_p_d1_1']
        .apply(lambda col: (col > 1).sum())  # col > 1 상승한 경우 count
        )

    _df_report = \
        (pd.concat([
                  df_report_1,
                  df_report_3
                  ], axis=1
                  )
        .assign(
            ud_r_d0_2=
            lambda df: df.ret_p_d0_2 / df.ret_p_d0_2_count,
            ud_r_d1_1=
            lambda df: df.ret_p_d1_1 / df.ret_p_d1_1_count,
        ).reset_index()
        )
    df_to_gbq = (_df_report
      .assign(kernel_size=kernel_size,
              date_ref=date_ref)
      [lambda df: ~df.variable.isin(['d0_1','d0_2'])]
      [lambda df: df.ret_p_d1_1_count > 20]
      .sort_values('ud_r_d1_1', ascending=False)
      )
    
    return df_to_gbq

  l_df_to_gbq = []
  for kernel_size in [3, 6]:
    l_df_to_gbq.append(
      make_report_table_ohlc(date_ref, kernel_size)
      )
  
  for kernel_size in [10,20]:
    l_df_to_gbq.append(
      make_report_table_occc(date_ref, kernel_size)
      )
  df_to_gbq = pd.concat(l_df_to_gbq)
  df_to_gbq['date_ref'] = pd.to_datetime(df_to_gbq.date_ref)#.dt.strftime('%Y-%m-%d')

  ########### send to gbq
  from google.cloud import bigquery
  project_id = 'dots-stock'
  client = bigquery.Client(project_id)

  # TODO(developer): Set table_id to the ID of the table to create.
  table_id = f"{project_id}.red_lion.pattern_eval_{date_ref}"

  def push_data_to_gbq(df_to_gbq):
    schema = [
      bigquery.SchemaField("source_code", "STRING"),
      bigquery.SchemaField("variable", "STRING"),
      bigquery.SchemaField("ret_p_d0_2_mean", "FLOAT"),
      bigquery.SchemaField("ret_p_d0_2_max", "FLOAT"),
      bigquery.SchemaField("ret_p_d0_2_count", "INTEGER"),
      bigquery.SchemaField("ret_p_d1_1_mean", "FLOAT"),
      bigquery.SchemaField("ret_p_d1_1_max", "FLOAT"),
      bigquery.SchemaField("ret_p_d1_1_count", "INTEGER"),
      bigquery.SchemaField("ret_p_d0_2", "INTEGER"),
      bigquery.SchemaField("ret_p_d1_1", "INTEGER"),
      bigquery.SchemaField("ud_r_d0_2", "FLOAT"),
      bigquery.SchemaField("ud_r_d1_1", "FLOAT"),
      bigquery.SchemaField("kernel_size", "INTEGER"),
      bigquery.SchemaField("date_ref", "DATE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.clustering_fields = [ "kernel_size", "variable", "date_ref"]

    job_config = bigquery.LoadJobConfig(schema=schema)

    try :
      table = client.create_table(table)  # Make an API request.

    except Exception as e:
      if 'Already Exists' in e.args[0] :
        print('Table already exists')

    try :
      job = client.load_table_from_dataframe(
          df_to_gbq, table_id, job_config=job_config
      )
    except Exception as e:
      print(e)
      raise
  push_data_to_gbq(df_to_gbq=df_to_gbq)

        