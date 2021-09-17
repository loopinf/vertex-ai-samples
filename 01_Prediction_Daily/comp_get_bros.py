from kfp.components import OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def get_bros(
    date_ref: str,
    n_days: int, 
    # bros_univ_dataset_path: OutputPath('DataFrame')
    bros_univ_dataset: Output[Dataset] 
):
  
  import pandas as pd
  import pickle
  import pandas_gbq
  import networkx as nx
  from trading_calendars import get_calendar
  cal_KRX = get_calendar('XKRX') 

  def get_krx_on_dates_n_days_ago(date_ref, n_days=20):
    return [date.strftime('%Y%m%d')
            for date in pd.bdate_range(
        end=date_ref, freq='C', periods=n_days,
        holidays=cal_KRX.precomputed_holidays) ]

  def get_corr_pairs_gbq(date_ref, period):
    date_ref_ = pd.Timestamp(date_ref).strftime('%Y-%m-%d')
    sql = f'''
    SELECT
      DISTINCT source,
      target,
      corr_value,
      period,
      date
    FROM
      `dots-stock.krx_dataset.corr_ohlc_part1`
    WHERE
      date = "{date_ref_}"
      AND period = {period}
    ORDER BY
      corr_value DESC
    LIMIT
      1000'''

    PROJECT_ID = 'dots-stock'
    df = pandas_gbq.read_gbq(sql, project_id=PROJECT_ID)
    return df

  def find_bros(date_ref, period):
    '''clique over 3 nodes '''
    df_edgelist = get_corr_pairs_gbq(date_ref, period)
    g = nx.from_pandas_edgelist(df_edgelist, edge_attr=True)
    bros_ = nx.find_cliques(g)
    bros_3 = [bros for bros in bros_ if len(bros) >=3]
    set_bros =  set([i for l_i in bros_3 for i in l_i])
    g_gang = g.subgraph(set_bros)

    df_gangs_edgelist = nx.to_pandas_edgelist(g_gang)
    return df_gangs_edgelist

  def find_gang(date_ref):
    df_gang = pd.DataFrame()
    for period in [20, 40, 60, 90, 120]:
      df_ = find_bros(date, period=period)
      df_gang = df_gang.append(df_)
    return df_gang
  
  # jobs
  dates = get_krx_on_dates_n_days_ago(date_ref=date_ref, n_days=n_days)
  df_bros = pd.DataFrame()
  for date in dates:
    df = find_gang(date_ref=date)  
    df_bros = df_bros.append(df)

  df_bros.to_pickle(bros_univ_dataset.path)