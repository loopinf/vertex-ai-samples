from kfp.v2.components import create_component_from_func_v2
# from kfp.components import InputPath, OutputPath, create_component_from_func_v2
from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component)

# @component(
#   base_image="gcr.io/dots-stock/python-img-v5.2",
#   packages_to_install=['catboost', 'scikit-learn', 'ipywidgets']
# )
def create_model_and_prediction_02(
  ml_dataset : Input[Dataset],
  prediction_result_01 : Output[Dataset],
  model_01_artifact: Output[Model] ,
):

  import pandas as pd
  import numpy as np
  from pykrx import stock
  import FinanceDataReader as fdr

  df_dataset = pd.read_pickle(ml_dataset.path)

  df_preP = df_dataset.copy()

  print(f'size01 {df_preP.shape}')

  # drop duplicated column
  cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'change_x']
  cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'change_y']
  df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

  print(f'size02 {df_preP.shape}')

  # drop SPACs
  stock_names = pd.Series(df_preP.name.unique())
  stock_names_SPAC = stock_names[ stock_names.str.contains('스팩')].tolist()

  df_preP = df_preP.where( 
            lambda df : ~df.name.isin(stock_names_SPAC)
            ).dropna(subset=['name'])

  print(f'size03 {df_preP.shape}')

  # Remove administrative items
  krx_adm = fdr.StockListing('KRX-ADMINISTRATIVE') # 관리종목
  df_preP = df_preP.merge(krx_adm[['Symbol','DesignationDate']], 
         left_on='code', right_on='Symbol', how='left')
  
  print(f'size04 {df_preP.shape}')

  df_preP['date'] = pd.to_datetime(df_preP.date)
  df_preP['admin_stock'] = df_preP.DesignationDate <= df_preP.date
  df_preP = (
        df_preP.where(
            lambda df: df.admin_stock == 0
        ).dropna(subset=['admin_stock'])
        ) 
  print(f'size05 {df_preP.shape}')
  # Add day of week
  df_preP['dayofweek'] = pd.to_datetime(df_preP.date.astype('str')).dt.dayofweek.astype('category')

  # Add market_cap categotu
  df_preP['mkt_cap_cat'] = pd.cut(
                            df_preP['mkt_cap'],
                            bins=[0, 1000, 5000, 10000, 50000, np.inf],
                            include_lowest=True,
                            labels=['A', 'B', 'C', 'D', 'E'])

  # Set Target & Features
  target_col = ['target_close_over_10']
  cols_indicator = [
                    'code',
                    'name',
                    'date',
                    ]
  features = [
                #  'code',
                #  'name',
                #  'date',
                'rank',
                'mkt_cap',
                'mkt_cap_cat',
                'in_top30',
                'rank_mean_10',
                'rank_mean_5',
                'in_top_30_5',
                'in_top_30_10',
                'in_top_30_20',
                'up_bro_ratio_20',
                'up_bro_ratio_40',
                'up_bro_ratio_60',
                'up_bro_ratio_90',
                'up_bro_ratio_120',
                'n_bro_20',
                'n_bro_40',
                'n_bro_60',
                'n_bro_90',
                'n_bro_120',
                'all_bro_rtrn_mean_20',
                'all_bro_rtrn_mean_40',
                'all_bro_rtrn_mean_60',
                'all_bro_rtrn_mean_90',
                'all_bro_rtrn_mean_120',
                'up_bro_rtrn_mean_20',
                'up_bro_rtrn_mean_40',
                'up_bro_rtrn_mean_60',
                'up_bro_rtrn_mean_90',
                'up_bro_rtrn_mean_120',
                'all_bro_rtrn_mean_ystd_20',
                'all_bro_rtrn_mean_ystd_40',
                'all_bro_rtrn_mean_ystd_60',
                'all_bro_rtrn_mean_ystd_90',
                'all_bro_rtrn_mean_ystd_120',
                'bro_up_ratio_ystd_20',
                'bro_up_ratio_ystd_40',
                'bro_up_ratio_ystd_60',
                'bro_up_ratio_ystd_90',
                'bro_up_ratio_ystd_120',
                'up_bro_rtrn_mean_ystd_20',
                'up_bro_rtrn_mean_ystd_40',
                'up_bro_rtrn_mean_ystd_60',
                'up_bro_rtrn_mean_ystd_90',
                'up_bro_rtrn_mean_ystd_120',
                #  'index',
                #  'open_x',
                #  'high_x',
                #  'low_x',
                #  'close_x',
                #  'volume_x',
                #  'change_x',
                #  'high_p1',
                #  'high_p2',
                #  'high_p3',
                #  'close_p1',
                #  'close_p2',
                #  'close_p3',
                #  'change_p1',
                #  'change_p2',
                #  'change_p3',
                #  'change_p1_over5',
                #  'change_p2_over5',
                #  'change_p3_over5',
                #  'change_p1_over10',
                #  'change_p2_over10',
                #  'change_p3_over10',
                #  'close_high_1',
                #  'close_high_2',
                #  'close_high_3',
                #  'close_high_1_over10',
                #  'close_high_2_over10',
                #  'close_high_3_over10',
                #  'close_high_1_over5',
                #  'close_high_2_over5',
                #  'close_high_3_over5',
                #  'open_y',
                #  'high_y',
                #  'low_y',
                #  'close_y',
                #  'volume_y',
                #  'change_y',
                #  'macd',
                #  'boll_ub',
                #  'boll_lb',
                'rsi_30',
                'dx_30',
                #  'close_30_sma',
                #  'close_60_sma',
                #  'daily_return',
                'return_lag_1',
                'return_lag_2',
                'return_lag_3',
                'bb_u_ratio',
                'bb_l_ratio',
                'max_scale_MACD',
                'volume_change_wrt_10max',
                'volume_change_wrt_10mean',
                'close_ratio_wrt_10max',
                'close_ratio_wrt_10min',
                'oh_ratio',
                'oc_ratio',
                'ol_ratio',
                #  'Symbol',
                #  'DesignationDate',
                #  'admin_stock',
                'dayofweek']        

  # Change datetime format to str
  df_preP['date'] = df_preP.date.dt.strftime('%Y%m%d')

  # Split Dataset into For Training & For Prediction
  l_dates = df_preP.date.unique().tolist()
  print(f'size06 {l_dates.__len__()}')

  dates_for_train = l_dates[-23:-3] # 며칠전까지 볼것인가!! 20일만! 일단은
  print(f'size07 {dates_for_train.__len__()}')

  date_for_pred = l_dates[-1]  # prediction date
  print(f'size08 {date_for_pred}')

  df_train = df_preP[df_preP.date.isin(dates_for_train)]
  df_train = df_train.dropna(axis=0, subset=target_col)   # target 없는 날짜 제외

  df_pred = df_preP[df_preP.date == date_for_pred] 

  print(f'size of train {df_train.shape} size of pred {df_pred.shape}')

  # ML Model
  from catboost import CatBoostClassifier
  from sklearn.model_selection import train_test_split
  from catboost import Pool
  from catboost.utils import get_roc_curve, get_confusion_matrix
  import sklearn
  # from sklearn import metrics

  X = df_train[features + cols_indicator]
  y = df_train[target_col].astype('float')
  X['in_top30'] = X.in_top30.astype('int')
  df_pred['in_top30'] = df_pred.in_top30.astype('int')

  # Run prediction 3 times
  df_pred_final_01 = pd.DataFrame()
  for _ in range(3):

    # Set Model
    model_01 = CatBoostClassifier(
            # random_seed = 42,
            # task_type = 'GPU',
            # iterations=3000,
            iterations=2000,
            train_dir = '/tmp',
            verbose=500
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_train_indictor = X_train[cols_indicator]
    X_test_indictor = X_test[cols_indicator]

    X_train = X_train[features]
    X_test = X_test[features]

    print('X Train Size : ', X_train.shape, 'Y Train Size : ', y_train.shape)
    print('No. of true : ', y.sum() )

    model_01.fit(X_train, y_train,
              # , verbose=200
              # , plot=True, 
              cat_features=['in_top30','dayofweek', 'mkt_cap_cat'])

    print(f'model score : {model_01.score(X_test, y_test)}')

    model_01.save_model(model_01_artifact.path)

    # Prediction
    pred_stocks_01 = model_01.predict(df_pred[features])    
    pred_proba_01 = model_01.predict_proba(df_pred[features])
    
    df_pred_stocks_01 = pd.DataFrame(pred_stocks_01, columns=['Prediction']).reset_index(drop=True)
    df_pred_proba_01 = pd.DataFrame(pred_proba_01, columns=['Proba01', 'Proba02']).reset_index(drop=True)

    df_pred_name_code = df_pred[cols_indicator].reset_index(drop=True)

    print('results', df_pred_stocks_01, df_pred_stocks_01.shape)
    print('results', df_pred_proba_01, df_pred_proba_01.shape)
    print('result', df_pred_name_code.head(), df_pred_name_code.shape)

    df_pred_r_01 = pd.concat(
                    [
                      df_pred_name_code,
                      df_pred_stocks_01,
                      df_pred_proba_01
                      ],
                      axis=1)

    df_pred_r_01 = df_pred_r_01[df_pred_r_01.Prediction > 0]
    print('results', df_pred_r_01.code)
    df_pred_final_01 = df_pred_final_01.append(df_pred_r_01)

  print(f'columns of df : {df_pred_final_01.columns}' )
  df_pred_final_01 = df_pred_final_01.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
  df_pred_final_01 = df_pred_final_01.reset_index()
  df_pred_final_01 = df_pred_final_01.sort_values(by='Proba02', ascending=False) # high probability first

  df_pred_final_01.drop_duplicates(subset=['code', 'date'], inplace=True) # remove duplicates
  df_pred_final_01.to_pickle(prediction_result_01.path) # save 


if __name__ == '__main__':
    create_component_from_func_v2(
        # test, 
        create_model_and_prediction_02,
        output_component_file='ml_com03.yaml',
        base_image="gcr.io/dots-stock/python-img-v5.2",
        packages_to_install=['catboost', 'scikit-learn', 'ipywidgets'],
    )
    # print(__file__)