from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def update_pred_result(
  ver : str,
  market_info_dataset: Input[Dataset],
  predict_dataset : Input[Dataset],
  pred_w_price_dataset : Output[Dataset]
):

  import pandas as pd
  import FinanceDataReader as fdr

  cols_to_kepp = ['name', 'code', 'date',
                  'Prediction', 'Proba01', 'Proba02',
                  'c_1', 'c_2', 'c_3', 'close', 'change']  

  # Prediction result / All period / Comes from prev comp
  df_preded = pd.read_pickle(predict_dataset.path) # comes from prev comp.
  l_dates_of_pred = sorted(df_preded.date.unique()) 



  try: # 이미 저장되어 있는 가격정보 포함 결과가 있는 경우
    df_pred_w_price_stored = pd.read_pickle(f'/gcs/pipeline-dots-stock/bong_price_updated/bong_{ver}.pkl')
    l_dates_of_pred_w_price_stored = df_pred_w_price_stored.date.unique().tolist()

    s_dates_to_add = set(l_dates_of_pred) - set(l_dates_of_pred_w_price_stored)

    df_mkt = pd.read_pickle(market_info_dataset.path)
    df_mkt_ = df_mkt[df_mkt.날짜.isin(s_dates_to_add)]
    df_mkt__ = df_mkt_[['날짜', '종목코드', '현재가', '등락률']]

    df_preded_to_add = df_preded[df_preded.date.isin(s_dates_to_add)]
    df_preded_to_add = (df_preded_to_add.merge(
                                              df_mkt__,
                                              left_on=['code', 'date'],
                                              right_on = ['종목코드', '날짜'],
                                              how='left',        
                                              ))
    df_preded_to_add.rename(columns={'현재가' : 'close', '등락률':'change'}, inplace=True)
    df_preded_to_add = df_preded_to_add[cols_to_kepp]

    df_pred_w_price_new = pd.concat([df_pred_w_price_stored, df_preded_to_add], join="inner")
    df_pred_w_price_new.fillna(0, inplace=True)
    df_pred_w_price_new.reset_index(drop=True, inplace=True)

  except: # 최초로 추천 종목이 생겼을 경우
    print('first', df_preded.head())
    df_mkt = pd.read_pickle(market_info_dataset.path)
    df_mkt_ = df_mkt[['날짜', '종목코드', '현재가', '등락률']]
    df_mkt__ = df_mkt_[df_mkt_.날짜.isin(df_preded.date.to_list())]

    df_preded['c_1'] = 0
    df_preded['c_2'] = 0
    df_preded['c_3'] = 0

    print('first', df_preded.columns)
    df_preded_ = (df_preded.merge(
                                df_mkt__,
                                left_on=['code', 'date'],
                                right_on = ['종목코드', '날짜'],
                                how='left',
                                ))

    df_preded_.rename(columns={'현재가':'close', '등락률':'change'}, inplace=True)
    df_preded_ = df_preded_[cols_to_kepp]
    df_pred_w_price_new = df_preded_
    df_pred_w_price_new.reset_index(drop=True, inplace=True)

  try : 
    df_pred_w_price_new.sort_values(by=['date', 'Proba02'], ascending=[True, False], inplace=True)
  except :
    print('Maybe, Empty dataframe')

  df_pred_w_price_new.to_pickle(f'/gcs/pipeline-dots-stock/bong_price_updated/bong_{ver}.pkl')
  df_pred_w_price_new.to_pickle(pred_w_price_dataset.path)
  