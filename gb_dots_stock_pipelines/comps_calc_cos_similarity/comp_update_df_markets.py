from kfp.v2.dsl import (Dataset, Input, Output)

def update_df_markets(
  date_ref: str,
  df_markets_update: Output[Dataset],
):
  #@markdown Define some functions - validate data
  #@markdown Define funcs
  import json
  import numpy as np
  from multiprocessing import Pool
  import pandas as pd
  import requests

  # create calendar: cal_krx 
  from trading_calendars import get_calendar
  cal_krx = get_calendar('XKRX')

  today = date_ref

  #get_snapshot_markets(dates): with Pool
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
      return df
    # if len(dates) == 1:
    #   return get_krx_marketcap(dates[0])
    with Pool(2) as pool:
      result = pool.map(get_krx_marketcap, dates)
    df_market_ = pd.concat(result)

    return df_market_

  df_markets_today = get_snapshot_markets([today])

  #### Check dates missing
  def check_dates_missing_dupl(df_markets):
    # get dates from df_markets -- ex 20160104
    dates_market = df_markets.date.unique()
    
    # check min max dates
    print('Min date', 'Latest date',
    df_markets.date.min(),
    df_markets.date.max()
        
    )
    # get dates from calendar
    start_date = dates_market.min()
    start_date = '-'.join([start_date[:4], start_date[4:6], start_date[6:]])
    end_date = dates_market.max()
    end_date  = '-'.join([end_date[:4], end_date[4:6], end_date[6:]])
    
    dates_krx_open = cal_krx.all_sessions
    dates_open = dates_krx_open[
                        (start_date <= dates_krx_open) & (dates_krx_open <= end_date)]
    dates_open = [date.strftime('%Y%m%d') for date in dates_open]
    
    # compare 2 sets  / difference 
    krx_dates = set(dates_open)
    markets_dates = set(dates_market)
    
    print(f'krx_dates ^ markets_dates : {krx_dates ^ markets_dates}')
    assert len(krx_dates ^ markets_dates) == 0

    # Check duplicates
    assert df_markets.duplicated(subset=['date','Code']).sum() == 0
    print('check duplicates : Done')

  ############# Load latestest
  # url_markets = '/content/drive/Shareddrives/HBong-Stock/03_datasets/df_markets_snap_latest.pkl'
  url_markets = '/gcs/red-lion/df_market/df_markets_snap_latest.pkl'
  df_markets = pd.read_pickle(url_markets)

  ########## Check dates , missing dates, 
  check_dates_missing_dupl(df_markets)

  ########## Merge with df_markets_today
  df_markets = pd.concat([df_markets, df_markets_today])
  df_markets = (df_markets
  .sort_values(['date', 'Code'])
  )
  df_markets.drop_duplicates(subset=['date','Code'], inplace=True)

  ########## Check dates , missing dates, 
  check_dates_missing_dupl(df_markets)

  ########## save to recent and backup
  # df_markets.to_pickle(url_markets) #TODO: uncomment this line

  ########## 
  df_markets.to_pickle(df_markets_update.path)
