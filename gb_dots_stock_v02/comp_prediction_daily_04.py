# Writer : D.S. Kim / S.H. Kim
# Date : 2021. 09. 07.

from datetime import date
from kfp.components import InputPath, OutputPath
# from typing import NamedTuple

def get_prediction_04(
  ml_dataset_path : InputPath('DataFrame'),
  bros_univ_dataset_path: InputPath('DataFrame'),
  predictions_path : OutputPath('DataFrame')
):

    import pandas as pd
    import numpy as np
    from pykrx import stock
    import FinanceDataReader as fdr
    import os
    # import pickle

    ver='04'
    file_path = f'/gcs/pipeline-dots-stock/bong_predictions/bong_{ver}.pkl'

    # Load feature_dataset
    df_dataset = pd.read_pickle(ml_dataset_path)
    df_preP = df_dataset.copy()

    df_bros = pd.read_pickle(bros_univ_dataset_path)
    df_bros = df_bros[df_bros.period.isin(['60', '90', '120'])]

    # drop duplicated column
    cols_ohlcv_x = ['open_x', 'high_x', 'low_x', 'close_x', 'volume_x', 'change_x']
    cols_ohlcv_y = ['open_y', 'high_y', 'low_y', 'close_y', 'volume_y', 'change_y']
    df_preP = df_preP.drop(columns=cols_ohlcv_x+cols_ohlcv_y)

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
            # 'rank',
            'mkt_cap',
            # 'mkt_cap_cat',
            'in_top30',
            # 'rank_mean_10',
            # 'rank_mean_5',
            'in_top_30_5',
            'in_top_30_10',
            'in_top_30_20',
            # 'up_bro_ratio_20',
            # 'up_bro_ratio_40',
            'up_bro_ratio_60',
            'up_bro_ratio_90',
            'up_bro_ratio_120',
            # 'n_bro_20',
            # 'n_bro_40',
            'n_bro_60',
            'n_bro_90',
            'n_bro_120',
            # 'all_bro_rtrn_mean_20',
            # 'all_bro_rtrn_mean_40',
            'all_bro_rtrn_mean_60',
            'all_bro_rtrn_mean_90',
            'all_bro_rtrn_mean_120',
            # 'up_bro_rtrn_mean_20',
            # 'up_bro_rtrn_mean_40',
            'up_bro_rtrn_mean_60',
            'up_bro_rtrn_mean_90',
            'up_bro_rtrn_mean_120',
            # 'all_bro_rtrn_mean_ystd_20',
            # 'all_bro_rtrn_mean_ystd_40',
            'all_bro_rtrn_mean_ystd_60',
            'all_bro_rtrn_mean_ystd_90',
            'all_bro_rtrn_mean_ystd_120',
            # 'bro_up_ratio_ystd_20',
            # 'bro_up_ratio_ystd_40',
            'bro_up_ratio_ystd_60',
            'bro_up_ratio_ystd_90',
            'bro_up_ratio_ystd_120',
            # 'up_bro_rtrn_mean_ystd_20',
            # 'up_bro_rtrn_mean_ystd_40',
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
            # 'rsi_30',
            # 'dx_30',
             'close_30_sma',
             'close_60_sma',
            #  'daily_return',
            # 'return_lag_1',
            # 'return_lag_2',
            # 'return_lag_3',
            'bb_u_ratio',
            'bb_l_ratio',
            # 'max_scale_MACD',
            'volume_change_wrt_10max',
            # 'volume_change_wrt_5max',
            # 'volume_change_wrt_20max',
            'volume_change_wrt_10mean',
            # 'volume_change_wrt_5mean',
            # 'volume_change_wrt_20mean',
            'close_ratio_wrt_10max',
            'close_ratio_wrt_10min',
            'oh_ratio',
            'oc_ratio',
            'ol_ratio',
            # 'ch_ratio',
            #  'Symbol',
            #  'DesignationDate',
            #  'admin_stock',
            # 'dayofweek'
            ]        

    # Change datetime format to str
    df_preP['date'] = df_preP.date.dt.strftime('%Y%m%d')

    # Split Dataset into For Training & For Prediction
    l_dates = df_preP.date.unique().tolist()

    # Filtering function
    def get_univ_bh01(df, l_dates): # input dataframe : top30s in the period

        # l_dates = df.date.unique().tolist()
        print(f'length of l_date : {l_dates.__len__()}')
        df_univ = pd.DataFrame()
        for date in l_dates :
            df_of_the_day = df[df.date == date]
            df_of_the_day = df_of_the_day.sort_values(by='rank', ascending=True)
            # print('df_of_the_date', df_of_the_day)
            df_top30_in_date = df_of_the_day.head(30)
            l_top30s_in_date = df_top30_in_date.code.to_list()
            # print(f'size of top30 in the date : {df_top30_in_date.shape}')
            
            df_bros_in_date = df_bros[df_bros.date == date]
            l_bros_of_top30s = df_bros_in_date[\
                    df_bros_in_date.source.isin(l_top30s_in_date)].target.unique().tolist()
            df_bros_of_top30 = df_of_the_day[df_of_the_day.code.isin(l_bros_of_top30s)]
            # print(f'size of top30 + friends in the date : {df_bros_of_top30.shape}')

            df_ = df_top30_in_date.append(df_bros_of_top30)
            # print('size of the day : ', df_.shape)
            df_univ = df_univ.append(df_)
        # print('size of returned :', df_univ.shape)
        return df_univ
    
    dates_for_pred = l_dates[-10:]  # prediction date
    date_ref = dates_for_pred[-1]
    yesterday = dates_for_pred[-2]
    print(f'date_ref : {date_ref}, yesterday : {yesterday}')

    df_pred = get_univ_bh01(df_preP, dates_for_pred)
    df_pred['date'] = date_ref

    # ML Model
    from catboost import CatBoostClassifier
    from sklearn.model_selection import train_test_split
    from catboost import Pool

    df_pred['in_top30'] = df_pred.in_top30.astype('int')

    # Run prediction 3 times
    df_pred_final_01 = pd.DataFrame()
    for iter_n in range(3):

        path = f'/gcs/pipeline-dots-stock/bong_model/bong_model_{ver}_{yesterday}_{iter_n}'

        model = CatBoostClassifier()
        model.load_model(path)

        # Prediction
        pred_stocks_01 = model.predict(df_pred[features])    
        pred_proba_01 = model.predict_proba(df_pred[features])
        
        df_pred_stocks_01 = pd.DataFrame(pred_stocks_01, columns=['Prediction']).reset_index(drop=True)
        df_pred_proba_01 = pd.DataFrame(pred_proba_01, columns=['Proba01', 'Proba02']).reset_index(drop=True)

        df_pred_name_code = df_pred[cols_indicator].reset_index(drop=True)

        df_pred_r_01 = pd.concat(
                        [
                        df_pred_name_code,
                        df_pred_stocks_01,
                        df_pred_proba_01
                        ],
                        axis=1)

        df_pred_r_01 = df_pred_r_01[df_pred_r_01.Prediction > 0]
        # print('prediction results', df_pred_r_01)
        df_pred_final_01 = df_pred_final_01.append(df_pred_r_01)

    # print(f'size of df_pred_final_01_1 : {df_pred_final_01.shape}' )
    df_pred_final_01 = df_pred_final_01.groupby(['name', 'code', 'date']).mean() # apply mean to duplicated recommends
    df_pred_final_01 = df_pred_final_01.reset_index()
    df_pred_final_01 = df_pred_final_01.sort_values(by='Proba02', ascending=False) # high probability first
    # print(f'size of df_pred_final_01_2 : {df_pred_final_01.shape}' )

    df_pred_final_01.drop_duplicates(subset=['code', 'date'], inplace=True) # remove duplicates
    # print(f'size of df_pred_final_01_3 : {df_pred_final_01.shape}' )

    # Load stored prediction result file   
    df_pred_stored = pd.read_pickle(file_path)

    # Check the loaded(stored) dataframe has today data
    l_dates_stored = df_pred_stored.date.to_list() 

    if date_ref not in l_dates_stored:
         
        df_pred_new = df_pred_stored.append(df_pred_final_01)
        df_pred_new.reset_index(drop=True, inplace=True)
        df_pred_new.to_pickle(file_path)
        print(f'{date_ref} -  prediction result added & stored')
    
    else :

        df_pred_new = df_pred_stored

    df_pred_new.to_pickle(predictions_path)
    print(f'{date_ref} - result in already')

