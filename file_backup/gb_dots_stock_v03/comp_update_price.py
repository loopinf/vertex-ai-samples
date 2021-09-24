from kfp.components import InputPath, OutputPath
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics)


def update_price(
  ver : str,
  predict_dataset : Input[Dataset],
):

  import pandas as pd
  import FinanceDataReader as fdr
  
  # Preset : cols to keep
  cols_to_keep = ['name', 'code', 'date', 'Prediction', 'Prob01', 'Prob02']

  # # Prediction result / All period / Comes from prev comp
  # df_preded = pd.read_pickle(predict_dataset.path) # comes from prev comp.
  # l_dates_of_pred = df_preded.date.unique().tolist() 

  # try: # 이미 저장되어 있는 가격정보 포함 결과가 있는 경우
  #   df_pred_w_price_stored = pd.read_pickle(f'/gcs/pipeline-dots-stock/bong_price_updated/bong_{ver}.pkl')
  #   l_dates_of_pred_w_price_stored = df_pred_w_price_stored.date.unique().tolist()

  #   s_dates_to_add = set(l_dates_of_pred) - set(l_dates_of_pred_w_price_stored)

  #   df_preded_to_add = df_preded[df_preded.date.isin(s_dates_to_add)]
  #   df_pred_w_price_new = pd.concat([df_pred_w_price_stored, df_preded_to_add], join="inner")
  #   df_pred_w_price_new.fillna(0, inplace=True)

  # except: # 최초로 추천 종목이 생겼을 경우
  #   df_pred_w_price_new = df_preded

  # # 가격 정보 업데이트할 날짜와 코드 정보를 가져와 보자

  # # 일단은 최근 5일 간의 날짜와 시작일을 가져오자 
  # l_dates_to_update = sorted(df_pred_w_price_new.date.unique())[-5:]
  # date_start = l_dates_to_update[0]

  # # 위의 기간에 해당하는 code를 가져오자
  # codes_to_update = df_pred_w_price_new[df_pred_w_price_new.date.isin(l_dates_to_update)].code.unique().tolist()
  
  # # 수정주가 반영된 가격 정보 가져오기 함수들
  # def get_price_adj(code, start):
  #     return fdr.DataReader(code, start=start)    

  # def get_price(codes, date_start):

  #     df_price = pd.DataFrame()
  #     for code in codes :      
  #         df_ = get_price_adj(code, date_start)
  #         df_['code'] = code
  #         df_price = df_price.append(df_)

  #         # print(df_price.shape, code)      
  #     return df_price

  # df_to_update = df_pred_w_price_new[df_pred_w_price_new.date.isin(l_dates_to_update)]

  #   #########
  # if s_dates_to_update.__len__() :
  # # df_to_hold = df_pred_result[~df_pred_result.date.isin(dates_to_update)]
  # df_to_update = df_pred_result[df_pred_result.date.isin(dates_to_update)]

  # codes_to_update = df_to_update.code.unique().tolist()

  # def get_price_adj(code, start):
  #     return fdr.DataReader(code, start=start)    

  # def get_price(codes, date_start):

  #     df_price = pd.DataFrame()
  #     for code in codes :      
  #         df_ = get_price_adj(code, date_start)
  #         df_['code'] = code
  #         df_price = df_price.append(df_)

  #         # print(df_price.shape, code)      
  #     return df_price

  # date_start = dates_to_update[0]
  # df_price = get_price(codes_to_update, date_start)

  # df_price.reset_index(inplace=True)
  # df_price.columns = df_price.columns.str.lower()
  # df_price['date'] = df_price.date.dt.strftime('%Y%m%d')

  # def get_price_tracked(df):

  #   df_ = df.copy()
  #   df_.sort_values(by='date', inplace=True)
  #   df_['c_1'] = df_.close.shift(-1)
  #   df_['c_2'] = df_.close.shift(-2)
  #   df_['c_3'] = df_.close.shift(-3)

  #   return df_
  
  # df_price_updated  = df_price.groupby('code').apply(lambda df: get_price_tracked(df))
  # df_price_updated = df_price_updated.reset_index(drop=True)

  # try :
  #   df_to_update.drop(columns=['c_1', 'c_2', 'c_3'], inplace=True)
  # except :
  #     pass

  # df_to_update = df_to_update.merge(
  #                         df_price_updated,
  #                         left_on=['date', 'code'],
  #                         right_on=['date', 'code'] )
  # df_to_update.fillna(0, inplace=True)

  # dir = "/gcs/pipeline-dots-stock/bong_price_updated"  
  
  # file_name = f'bong_{ver}.pkl'
  # path = os.path.join(dir, file_name)

  # with open(path, 'wb') as f:
  #   pickle.dump(df_to_update, f)

  # # df_to_update.to_pickle(updated_result_02.path)