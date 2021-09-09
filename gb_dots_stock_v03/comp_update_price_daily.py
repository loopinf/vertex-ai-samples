
def update_price_daily(
):

  import pandas as pd
  import FinanceDataReader as fdr
  # import pickle
  import os

  folder = "/gcs/pipeline-dots-stock/bong_price_updated"  

  file_list = os.listdir(folder)

  cols_to_keep = ['name', 'code', 'date', 'Prediction', 'Proba01', 'Proba02',
                  # 'high', 'low', 'volume', 
                  'change', 'c_1', 'c_2', 'c_3', 'close'
                  ]

  for f in file_list:
    print(f)
    path = os.path.join(folder, f)

    df_ = pd.read_pickle(path)
    # df_ = df_[cols_to_keep]
    df_ = df_.reset_index(drop=True)

    try : # for the datasets, not updated ever before
     
      df_ = df_[cols_to_keep]

      l_dates = df_.date.unique().tolist()
      l_dates_to_update = l_dates[-5:]

      df_to_hold = df_[~df_.date.isin(l_dates_to_update)]
      df_to_update = df_[df_.date.isin(l_dates_to_update)]

    except :
      print('newbie')
      l_dates = df_.date.unique().tolist()
      l_dates_to_update = l_dates#[-5:]
      # df_to_hold = df_[~df_.date.isin(l_dates_to_update)]
      df_to_update = df_[df_.date.isin(l_dates_to_update)]

    

    codes_to_update = df_to_update.code.unique().tolist()

    print(f'codes to update : {codes_to_update}')

    def get_price_adj(code, start):
      return fdr.DataReader(code, start=start)    

    def get_price(codes, date_start):

        df_price = pd.DataFrame()
        for code in codes :      
            df_ = get_price_adj(code, date_start)
            df_['code'] = code
            df_price = df_price.append(df_)

            # print(df_price.shape, code)      
        return df_price

    date_start = l_dates_to_update[0]
    df_price = get_price(codes_to_update, date_start)
    df_price.reset_index(inplace=True)
    df_price.columns = df_price.columns.str.lower()
    df_price['date'] = df_price.date.dt.strftime('%Y%m%d')

    def get_price_tracked(df):

      df_ = df.copy()
      df_.sort_values(by='date', inplace=True)
      df_['c_1'] = df_.close.shift(-1)
      df_['c_2'] = df_.close.shift(-2)
      df_['c_3'] = df_.close.shift(-3)

      return df_

    df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
    df_price_updated = df_price_updated[['date', 'code', 'c_1', 'c_2', 'c_3', 'close']]
    df_price_updated = df_price_updated.reset_index(drop=True)

    df_to_update.drop(columns=['c_1', 'c_2', 'c_3', 'close'], inplace=True)

    df_to_update = df_to_update.merge(
                          df_price_updated,
                          left_on=['date', 'code'],
                          right_on=['date', 'code'] )

    df_to_update.fillna(0, inplace=True)

    try :
      df_updated = df_to_hold.append(df_to_update)
      df_updated = df_updated[cols_to_keep]
    except :
      print('newbie')
      df_updated = df_to_update
      df_updated = df_updated[cols_to_keep]

    df_updated.to_pickle(path)
    
