from kfp.v2.dsl import (Dataset, Input, Output)


def add_price_on_pattern(date_ref, kernel_size):

  import logging
  FORMAT = "[%(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
  logging.basicConfig(format=FORMAT, level=logging.DEBUG)
  import pandas_gbq  # type: ignore
  from trading_calendars import get_calendar
  cal_krx = get_calendar('XKRX')
  import pandas as pd
  import numpy as np
  from pandas.tseries.offsets import CustomBusinessDay
  cbday = CustomBusinessDay(holidays=cal_krx.adhoc_holidays)

  import multiprocessing
  from multiprocessing import Pool
  N_cpu = multiprocessing.cpu_count()
  logging.debug(f'cpu_count : {N_cpu}')

  import functools

  PROJECT_ID = 'dots-stock'
  from google.cloud import bigquery
  client = bigquery.Client(PROJECT_ID)

  N_prev = 20
  N_next = 5
  N_all = N_prev + N_next

  # get pattern
  def get_pattern(date_ref, kernel_size):
    logging.debug(f'{date_ref, kernel_size}')
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    if kernel_size in (3, 6):
      table_id = f"{PROJECT_ID}.red_lion.pattern_v2_{kernel_size}_{date_ref}"
    elif kernel_size in (10, 20):
      table_id = f"{PROJECT_ID}.red_lion.pattern_oc_cc_{kernel_size}_{date_ref}"
    else:
      raise

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

  df_pattern_raw = get_pattern(date_ref, kernel_size)
  def get_price(date_ref):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    table_id = f"{PROJECT_ID}.red_lion.adj_price_{date_ref}"
    logging.debug(f'{date_ref}: {table_id}')
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

  df_pattern_dedup = (
    df_pattern_raw.drop_duplicates(subset=['date','Code'])
  )
  _df = df_pattern_dedup

  @functools.lru_cache(maxsize=None)
  def get_krx_range(x):
      return pd.bdate_range(
        pd.Timestamp(x) - (N_prev - 1) * cbday,
        periods=N_all,
        freq='C',
        holidays = cal_krx.adhoc_holidays
      )

  # it took 29s
  _df1 = _df.assign(
    date_ohlcv = _df[['date']].applymap(
      get_krx_range
    )
  )[['Code','date','date_ohlcv']]

  l_df_spl = np.array_split(_df1, N_cpu)

  global mp_split
  def mp_split(_df1_spl):
    _df1_spl = _df1_spl.copy()
    _df1_split = (_df1_spl
    .date_ohlcv
    .apply(
        pd.Series           # it took 1m 12s
        # transform_to_series # it took 1m 34s
        ))
    return _df1_split

  with Pool(N_cpu) as pool:
    result = pool.map(mp_split, l_df_spl)  # 1m 17s
    # result = pool.imap(mp_split, l_df_spl)  # 훨 씬 오래 걸림

  _df1_split = \
  (pd.concat(result)
  .rename(columns=lambda x: f'd{x:02d}')
  )
  _df11 = pd.concat( 
      [_df1.loc[:,['Code','date']],  _df1_split ],
      axis=1
  )
  _df2 = \
  (_df11
  .set_index(['Code','date'], drop=True)
  .stack()
  .reset_index()
  .rename(columns={0:'date_price','level_2':'n_step'})
  )
  #### it took 41s
  _df3 = \
  (_df2
  .merge(df_price,
          left_on=['Code','date_price'],
          right_on=['code','Date'],
          how='left'
          )
  .drop(['Date','Change'], axis=1)
  .assign(date_price=
          lambda df: df.date_price.dt.strftime('%Y-%m-%d'))
  #  .replace([np.nan], [None]) # to GBQ
  #  .replace([np.nan], [0]) # to GBQ
  .fillna(0)
  )

  # !pip install multiprocesspandas
  # from multiprocesspandas import applyparallel
  df_Open = (_df3
  .groupby(['Code','date'])
  ['Open']
  .apply(list)
  ).to_frame()
  
  df_Close = (_df3
  .groupby(['Code','date'])
  ['Close']
  .apply(list)
  ).to_frame()
  
  df_High = (_df3
  .groupby(['Code','date'])
  ['High']
  .apply(list)
  ).to_frame()
  
  df_Low = (_df3
  .groupby(['Code','date'])
  ['Low']
  .apply(list)
  ).to_frame()
  
  df_Volume = (_df3
  .groupby(['Code','date'])
  ['Volume']
  .apply(list)
  ).to_frame()

  df_date_price = (_df3
  .groupby(['Code','date'])
  ['date_price']
  .apply(list)
  ).to_frame()
  
  _df4 = pd.concat([
            df_Open,
            df_High,
            df_Low,
            df_Close,
            df_Volume,
            df_date_price],
            axis=1
  ).reset_index() 
  _df5 = \
  (pd.merge(
      df_pattern_raw,
      _df4,
      how='left',
      left_on=['date','Code'],
      right_on = ['date', 'Code']  )
  )
  _df6 = \
  (_df5.assign(
      date = lambda df: df.date.dt.strftime('%Y-%m-%d'),
      source_date = lambda df: df.source_date.dt.strftime('%Y-%m-%d'),
  ))

  def to_gbq_table_pattern(df, date_ref): 
    schema = [
        bigquery.SchemaField(name="date", field_type="DATE"),
        bigquery.SchemaField(name="source_date", field_type="DATE"),
        bigquery.SchemaField(name="similarity", field_type="FLOAT"),
        bigquery.SchemaField(name="source_code", field_type="STRING"),
        bigquery.SchemaField(name="Code", field_type="STRING"),
        bigquery.SchemaField(name="Open", field_type="INT64", mode="REPEATED"),
        bigquery.SchemaField(name="High", field_type="INT64", mode="REPEATED"),
        bigquery.SchemaField(name="Low", field_type="INT64", mode="REPEATED"),
        bigquery.SchemaField(name="Close", field_type="INT64", mode="REPEATED"),
        bigquery.SchemaField(name="Volume", field_type="INT64", mode="REPEATED"),
        bigquery.SchemaField(name="date_price", field_type="STRING", mode="REPEATED")
    ]
    # table_id = f'red_lion.{table_name}_testing_{date_ref}'
    if kernel_size in (3, 6):
      table_id = f"{PROJECT_ID}.red_lion.pattern_v2_price_{kernel_size}_{date_ref}"
    elif kernel_size in (10, 20):
      table_id = f"{PROJECT_ID}.red_lion.pattern_oc_cc_price_{kernel_size}_{date_ref}"
    else:
      logging.error(f'Check kernel_size: {kernel_size}')  
    table = bigquery.Table(
        table_id,
        schema=schema
    )
    table.clustering_fields = ["source_code"] 
    print(table)
    try:
      client.create_table(table)
    except Exception as e:
      print(e)
      if ('Already Exists' in e.args[0]): # and if_exists=='replace' : 
        table = client.get_table(table_id)
      else: raise

    errors = client.insert_rows_from_dataframe(table, df)
    for chunk in errors:
      print(f"encountered {len(chunk)} errors: {chunk}")
      # if len(errors) > 1: raise
    print(len(errors))
  to_gbq_table_pattern(_df6, date_ref )