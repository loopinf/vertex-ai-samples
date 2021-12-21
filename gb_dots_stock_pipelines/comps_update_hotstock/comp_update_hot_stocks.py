from kfp.v2.dsl import (Dataset, Input, Output)


def update_hot_stocks():
  import numpy as np
  import pandas as pd
  import pandas_gbq
  import json

  today = pd.Timestamp.now('Asia/Seoul').strftime('%Y%m%d')


  url_hot_naver = 'https://finance.naver.com/sise/lastsearch2.naver'
  dfs = pd.read_html(url_hot_naver, encoding='euc-kr')
  df_hot = \
  (dfs[1].dropna()
  .rename(columns={'종목명': 'name', '검색비율': 'search_portion'})
  .assign(
  source_info='네이버_검색상위',
  scrape_time=pd.Timestamp.now('Asia/Seoul').strftime('%Y%m%d%H%M')
  )
    .loc[:, ['name', 'source_info', 'search_portion', 'scrape_time']]
  )

  def send_to_gbq(df):
    pandas_gbq.to_gbq(df,
    f'red_lion.hot_stocks_naver_search_{today}',
    project_id='dots-stock',
    # if_exists='replace',
    if_exists='append',
    )
  send_to_gbq(df_hot)
