from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def update_pred_result_reg(
  ver : str,
  market_info_dataset: Input[Dataset],
  predict_dataset : Input[Dataset],
  pred_w_price_dataset : Output[Dataset]
):

  import pandas as pd
  import FinanceDataReader as fdr

  # Prediction result / All period / Comes from prev comp
  df_preded = pd.read_pickle(predict_dataset.path) # comes from prev comp.

  df_preded['c_1'] = 0
  df_preded['c_2'] = 0
  df_preded['c_3'] = 0

  cols_to_kepp = ['name', 'code', 'date',
                  'Prediction',
                  'c_1', 'c_2', 'c_3', 'close', 'change'] 


  def add_close(df, l_dates):

    df_mkt = pd.read_pickle(market_info_dataset.path)

    df_mkt_ = df_mkt[df_mkt.날짜.isin(l_dates)]
    df_mkt__ = df_mkt_[['날짜', '종목코드', '현재가', '등락률']]  

    df_to_add = df[df.date.isin(l_dates)]
    df_to_add = (df_to_add.merge(
                                  df_mkt__,
                                  left_on=['code', 'date'],
                                  right_on = ['종목코드', '날짜'],
                                  how='left',        
                                  ))

    df_to_add.drop(columns=['종목코드', '날짜'], inplace=True)
    df_to_add.rename(columns={'현재가' : 'close', '등락률':'change'}, inplace=True)
    df_to_add = df_to_add[cols_to_kepp]

    return df_to_add

  if df_preded.shape[0] == 0: # 추천 종목이 없는 경우
    exit()
  
  else :


    # 추천 종목이 있음!
    l_dates_of_pred = sorted(df_preded.date.unique()) 

    try :
      # 기존 추천 종목이 있는 경우
      df_pred_w_price_stored = pd.read_pickle(f'/gcs/pipeline-dots-stock/bong_price_updated/bong_{ver}.pkl')
      print(f'bong_{ver}.pkl : loaded')
      l_dates_of_pred_w_price_stored = df_pred_w_price_stored.date.unique().tolist()

      s_dates_to_add = set(l_dates_of_pred) - set(l_dates_of_pred_w_price_stored)

      df_to_add =add_close(df_preded, s_dates_to_add)

      df_pred_w_price_new = pd.concat([df_pred_w_price_stored, df_to_add], join="inner")
      df_pred_w_price_new.fillna(0, inplace=True)
      df_pred_w_price_new.reset_index(drop=True, inplace=True)

      if s_dates_to_add.__len__() == 0: # 기존 추천 종목이 이미 최신인 경우
        exit()

    except :
      # 기존 추천 종목이 없는 경우 df_preded에 컬럼 몇 개를 추가해 줘야 함
      
      s_dates_to_add = l_dates_of_pred

      df_pred_w_price_new =add_close(df_preded, s_dates_to_add)

      
  df_pred_w_price_new.to_pickle(f'/gcs/pipeline-dots-stock/bong_price_updated/bong_{ver}.pkl')
  df_pred_w_price_new.to_pickle(pred_w_price_dataset.path)
  