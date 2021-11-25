from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)
from kfp.components import InputPath, OutputPath

def get_adj_prices_daily(
  market_info_dataset: Input[Dataset],
  date_ref : str,
  adj_price_dataset: Output[Dataset],  
  ):

  import FinanceDataReader as fdr
  import pandas as pd
  from multiprocessing import Pool

  df_market = pd.read_pickle(market_info_dataset.path)

  start_date = '20110101'

  print(f'dates : {df_market.날짜.unique().tolist()}')

  l_code = df_market.종목코드.unique().tolist() 

  global get_price

  def get_price(code):
    return (
        fdr.DataReader(code, start=start_date, end=date_ref)
        .assign(code=code)
    )

  with Pool(15) as pool:
    result = pool.map(get_price, l_code)

  df_adj_price = pd.concat(result)
  # df_adj_price = df_adj_price.reset_index()
  # df_adj_price.columns = df_adj_price.columns.str.lower()
  # df_adj_price['date'] = df_adj_price.date.dt.strftime('%Y%m%d')


  # df_adj_price.to_pickle(adj_price_dataset.path)

  from google.cloud import bigquery

  project_id = 'dots-stock'
  client = bigquery.Client(project_id)

  # TODO(developer): Set table_id to the ID of the table to create.
  table_id = f"{project_id}.red_lion.adj_price_{date_ref}"

  schema = [
      bigquery.SchemaField("code", "STRING"),
      bigquery.SchemaField("Date", "DATE"),
      bigquery.SchemaField("Open", "INTEGER"),
      bigquery.SchemaField("High", "INTEGER"),
      bigquery.SchemaField("Low", "INTEGER"),
      bigquery.SchemaField("Close", "INTEGER"),
      bigquery.SchemaField("Volume", "INTEGER"),
      bigquery.SchemaField("Change", "FLOAT"),
  ]

  table = bigquery.Table(table_id, schema=schema)
  table.clustering_fields = ["code"] 

  # set table to expire 5 days from now
  import datetime
  expiration = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
      days=10
  )
  table.expires = expiration

  def push_data_to_gbq():

    job_config = bigquery.LoadJobConfig(schema=schema)

    job = client.load_table_from_dataframe(
        df_adj_price, table_id, job_config=job_config
    )

  try :

    table = client.create_table(table)  # Make an API request.
    push_data_to_gbq()

  except Exception as e:
    if 'Already Exists' in e.args[0] :
      print('Table already exists')
      tableExist = client.get_table(table_id)
        
      


  

