from kfp.components import InputPath, OutputPath

def update_price(
  predictions_path : InputPath('DataFrame'),
):

  import pandas as pd
  import FinanceDataReader as fdr
  import pickle
  import os

  ver = '10r'
  df_pred_result = pd.read_pickle(predictions_path)

  # check date

  l_dates = df_pred_result.date.unique().tolist()

  dates_to_update = l_dates #[-4:]

  # df_to_hold = df_pred_result[~df_pred_result.date.isin(dates_to_update)]
  df_to_update = df_pred_result[df_pred_result.date.isin(dates_to_update)]

  codes_to_update = df_to_update.code.unique().tolist()

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

  date_start = dates_to_update[0]
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
  df_price_updated = df_price_updated.reset_index(drop=True)

  try :
    df_to_update.drop(columns=['c_1', 'c_2', 'c_3'], inplace=True)
  except :
      pass

  df_to_update = df_to_update.merge(
                          df_price_updated,
                          left_on=['date', 'code'],
                          right_on=['date', 'code'] )
  df_to_update.fillna(0, inplace=True)

  dir = "/gcs/pipeline-dots-stock/bong_price_updated"  
  
  file_name = f'bong_{ver}.pkl'
  path = os.path.join(dir, file_name)

  with open(path, 'wb') as f:
    pickle.dump(df_to_update, f)

  # df_to_update.to_pickle(updated_result_02.path)