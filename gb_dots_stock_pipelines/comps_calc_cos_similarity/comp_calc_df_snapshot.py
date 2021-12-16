
def calc_df_snapshot(
  date_ref: str,
)-> str:
  ############## top30 신호등
  import logging
  logging.basicConfig(level=logging.DEBUG)
  import pandas_gbq # type: ignore
  import pandas as pd
  import numpy as np
  from google.cloud import bigquery 

  from trading_calendars import get_calendar
  from pandas.tseries.offsets import CustomBusinessDay
  cal_krx = get_calendar('XKRX')
  cbday = CustomBusinessDay(holidays = cal_krx.adhoc_holidays)

  N_days = 20
  PROJECT_ID = 'dots-stock'
  client = bigquery.Client(PROJECT_ID)

  def get_df_market(date_ref, n_before):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    date_ref_b = (pd.Timestamp(date_ref) - cbday * (n_before -1)).strftime('%Y-%m-%d')
    sql = f'''
      SELECT
        Code,Name,Market,Dept,Marcap,in_top30, date
      FROM
        `dots-stock.red_lion.df_markets_clust_parti`
      WHERE
        date between "{date_ref_b}" and "{date_ref_}"
      LIMIT
        1000000
      ''' 
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID, use_bqstorage_api=True)
    df = df.drop_duplicates()
    return df

  df_markets_1 = get_df_market(date_ref, N_days)

  logging.debug(f'date_ref : {date_ref}')
  logging.debug(f'df_markets_1 downloaded : {df_markets_1.shape}')
  logging.debug(f'df_markets_1 unique date length : {df_markets_1.date.unique().__len__()}')
  assert df_markets_1.date.unique().shape[0] == N_days
  
  ###
  df_to_gbq = (df_markets_1
    [lambda df: df.date == df.date.max()]
    )
  # make dataframe of top30 true and false
  _tmp = (df_markets_1
    .sort_values('date')
    .pivot_table(index='date', columns='Code', values='in_top30')
    .replace(False, 'NoTop30')
    .replace(True,'Top30')
    .fillna('Missing')
    )
  # make dictionary : code -> list of top30 
  dic_top30_list = {item[0]:list(item[1]) for item in _tmp.items()} 

  df_to_gbq = df_to_gbq.assign(
      in_top30_list = lambda df: df.Code.map(dic_top30_list)
  )

  def to_gbq_snapshot(df_to_gbq, date_ref):
    schema = [
      bigquery.SchemaField("Code", "STRING"),
      bigquery.SchemaField("Name", "STRING"),
      bigquery.SchemaField("Market", "STRING"),
      bigquery.SchemaField("Dept", "STRING"),
      bigquery.SchemaField("Marcap", "INTEGER"),
      bigquery.SchemaField("in_top30", "BOOL"),
      bigquery.SchemaField(name="in_top30_list", field_type="STRING", mode="REPEATED"),
    ]
    table_id = f'{PROJECT_ID}.red_lion.market_snapshot_top30_test_{date_ref}'
    table = bigquery.Table(table_id, schema=schema)
    try:
      table = client.create_table(table) 
    except:
      table = client.get_table(table_id)

    errors = client.insert_rows_from_dataframe(table, df_to_gbq)
    for chunk in errors:
      print(chunk)

  to_gbq_snapshot(df_to_gbq, date_ref)

  return 'finish'